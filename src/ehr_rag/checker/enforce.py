"""Enforcement: lines that fail the checker are rewritten once (``rewrite``) or dropped (``filter``).

Per failing line, ``rewrite`` mode prompts the same model with the line, its cited units
(only those) and the checker's messages (``prompts/<checker.rewrite_prompt>.txt``). The
answer's first non-blank line is re-checked; it must end with evidence citations. Outcomes:

  kept       the line passed (or had no checkable content) and is unchanged
  repaired   rewritten, passes, and keeps >= as many checkable numbers AND names
  retreated  rewritten, passes, with fewer checkable numbers or names
  dropped    failed twice (rewrite), failed once (filter), or the rewrite was unusable

A section whose lines were all dropped gets ``Insufficient evidence. [NONE]`` (logged as
``sections_emptied``); abstention lines are kept as they are. The enforced report is
``enforced.txt``; ``parsed.json`` then points to it (``raw_file``), keeping the original in
``pre_enforcement_raw_file``; ``enforcement.json`` logs every line.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from ehr_rag.checker.rules import AdmissionContext, Checker, LineCheck
from ehr_rag.config import SECTIONS
from ehr_rag.eval.parse import ABSTENTION_TEXT, Sentence, parse_line, parse_report
from ehr_rag.generation.generate import check_format
from ehr_rag.generation.prompt import evidence_line

MODES: tuple[str, ...] = ("rewrite", "filter")
OUTCOMES: tuple[str, ...] = ("kept", "repaired", "retreated", "dropped")
ABSTENTION_LINE = f"{ABSTENTION_TEXT} [NONE]"
Rewriter = Callable[[str], tuple[str, dict[str, Any]]]


def load_rewrite_template(prompts_dir: Path, version: str) -> str:
    """Rewrite prompt template with placeholders {sentence}, {errors}, {units}."""
    p = prompts_dir / f"{version}.txt"
    text = p.read_text(encoding="utf-8")
    missing = [ph for ph in ("{sentence}", "{errors}", "{units}") if ph not in text]
    if missing:
        raise ValueError(f"{p.name}: missing placeholder(s) {missing}")
    return text


def rewrite_prompt(template: str, lc: LineCheck, ctx: AdmissionContext) -> str:
    """The rewrite prompt for one failing line: the line, its errors, and only its cited units."""
    ids = list(dict.fromkeys(i for it in lc.items for i in it.ids if i in ctx.shown))
    units = "\n".join(evidence_line(i, ctx.shown[i].unit_type, ctx.shown[i].text) for i in ids) or "(none)"
    return (template.replace("{sentence}", lc.line).replace("{errors}", "\n".join(f"- {m}" for m in lc.messages()))
            .replace("{units}", units))


def parse_rewrite(text: str, section: str, index: int) -> tuple[Sentence | None, str]:
    """First non-blank line of a rewrite answer as a sentence, or (None, reason)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, "empty rewrite"
    line = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", lines[0]).strip().strip('"').strip()
    s = parse_line(line, section, index, 0)
    if s.is_abstention:
        return None, "rewrite abstained"
    if not s.evidence_ids:
        return None, "rewrite has no trailing evidence citation"
    return s, ""


def _counts(lc: LineCheck | None) -> tuple[int, int]:
    return (lc.n_numbers, lc.n_names) if lc is not None else (0, 0)


def enforce_report(
    raw: str, ctx: AdmissionContext, checker: Checker, mode: str, rewrite: Rewriter | None = None,
    template: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Enforce one report. Returns (enforced text, log with one record per line and a summary)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "rewrite" and (rewrite is None or template is None):
        raise ValueError("rewrite mode needs a rewriter and a template")
    parsed = parse_report(raw)
    records: list[dict[str, Any]] = []
    out_sections: dict[str, list[str]] = {}
    emptied: list[str] = []
    for sec in SECTIONS:
        kept: list[str] = []
        sents = parsed.sections.get(sec, [])
        for s in sents:
            if s.is_abstention:
                kept.append(s.raw.strip())
                continue
            lc = checker.check_line(s, ctx)
            rec: dict[str, Any] = {"section": sec, "index": s.index, "line": lc.line, "status": lc.status,
                                   "error_types": lc.error_types, "messages": lc.messages(),
                                   "n_numbers": lc.n_numbers, "n_names": lc.n_names}
            if lc.status != "fail":
                rec.update(outcome="kept", final_line=lc.line, final_n_numbers=lc.n_numbers,
                           final_n_names=lc.n_names)
                kept.append(lc.line)
            elif mode == "filter":
                rec.update(outcome="dropped", reason="failed check (filter mode)", final_line=None,
                           final_n_numbers=0, final_n_names=0)
            else:
                prompt = rewrite_prompt(template, lc, ctx)            # type: ignore[arg-type]
                answer, meta = rewrite(prompt)                         # type: ignore[misc]
                new, reason = parse_rewrite(answer, sec, s.index)
                lc2 = checker.check_line(new, ctx) if new is not None else None
                rec.update(rewrite_prompt=prompt, rewrite_raw=answer, rewrite_call=meta,
                           rewrite_line=new.raw.strip() if new is not None else None,
                           rewrite_status=lc2.status if lc2 else None,
                           rewrite_error_types=lc2.error_types if lc2 else [],
                           rewrite_messages=lc2.messages() if lc2 else [])
                if lc2 is None or lc2.status == "fail":
                    rec.update(outcome="dropped", reason=reason or "rewrite failed the check", final_line=None,
                               final_n_numbers=0, final_n_names=0)
                else:
                    n2, m2 = _counts(lc2)
                    rec.update(outcome="repaired" if n2 >= lc.n_numbers and m2 >= lc.n_names else "retreated",
                               final_line=lc2.line, final_n_numbers=n2, final_n_names=m2)
                    kept.append(lc2.line)
            records.append(rec)
        if not kept:
            emptied.append(sec)
            kept = [ABSTENTION_LINE]
        out_sections[sec] = kept
    text = "\n".join(f"### {sec}\n" + "\n".join(lines) for sec, lines in out_sections.items()) + "\n"
    log = {"mode": mode, "lines": records, "sections_emptied": emptied,
           "dropped_preamble_or_extra_lines": len(parsed.preamble) + len(parsed.extra_lines),
           "summary": summarize_lines(records)}
    return text, log


def summarize_lines(records: list[dict[str, Any]]) -> dict[str, int]:
    """Counts of statuses, outcomes and checkable numbers/names before and after enforcement."""
    out: dict[str, int] = {"n_lines": len(records)}
    for st in ("pass", "fail", "unchecked"):
        out[f"n_{st}"] = sum(1 for r in records if r["status"] == st)
    for o in OUTCOMES:
        out[f"n_{o}"] = sum(1 for r in records if r["outcome"] == o)
    out["n_lines_after"] = len(records) - out["n_dropped"]
    out["n_numbers_before"] = sum(r["n_numbers"] for r in records)
    out["n_numbers_after"] = sum(r["final_n_numbers"] for r in records)
    out["n_names_before"] = sum(r["n_names"] for r in records)
    out["n_names_after"] = sum(r["final_n_names"] for r in records)
    out["n_rewrite_calls"] = sum(1 for r in records if "rewrite_prompt" in r)
    return out


def enforce_dir(
    report_dir: Path, ctx: AdmissionContext, checker: Checker, mode: str, rewrite: Rewriter | None = None,
    template: str | None = None, checker_meta: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Enforce the report in ``report_dir`` in place (see module docstring). Returns its summary."""
    parsed_p = report_dir / "parsed.json"
    meta = json.loads(parsed_p.read_text(encoding="utf-8"))
    raw_file = meta.get("pre_enforcement_raw_file", meta["raw_file"])
    raw = (report_dir / raw_file).read_text(encoding="utf-8")
    text, log = enforce_report(raw, ctx, checker, mode, rewrite, template)
    log["checker"] = checker_meta
    log["source_raw_file"] = raw_file
    rewrites = report_dir / "rewrites"
    if rewrites.exists():
        shutil.rmtree(rewrites)
    for r in log["lines"]:
        if "rewrite_prompt" in r:
            stem = f"{r['section']}_{r['index']}"
            _write(rewrites / f"{stem}_prompt.txt", r["rewrite_prompt"])
            _write(rewrites / f"{stem}_raw.txt", r["rewrite_raw"])
    _write(report_dir / "enforced.txt", text)
    _write(report_dir / "enforcement.json", json.dumps(log, indent=2, ensure_ascii=False) + "\n")
    _, problems = check_format(text)
    meta.update(pre_enforcement_raw_file=raw_file, raw_file="enforced.txt", enforcement_mode=mode,
                enforced_format_problems=problems, enforcement_summary=log["summary"])
    _write(parsed_p, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return log["summary"]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def add_summaries(total: dict[str, int], s: dict[str, int]) -> dict[str, int]:
    """Element-wise sum of two summaries."""
    for k, v in s.items():
        total[k] = total.get(k, 0) + v
    return total


def frozen_units(shown: dict[str, Any], units: list[Any]) -> list[Any]:
    """Units in evidence_shown order, checked identical (id, type, text) to the current unit files."""
    by_id = {u.unit_id: u for u in units}
    out = []
    for e in shown["units"]:
        u = by_id.get(e["unit_id"])
        if u is None or u.unit_type != e["unit_type"] or u.text != e["text"]:
            raise ValueError(f"hadm {shown['hadm_id']} unit {e['unit_id']} differs from the evidence the report "
                             "was generated with; units were rebuilt?")
        out.append(u)
    return out


def filter_generation_runs(
    cfg: dict[str, Any], gen_dirs: list[Path], out_dir: Path, conditions: list[str],
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, int]]:
    """Filter-only enforcement (no model): copy each report of ``conditions`` from the generation
    run(s) to ``out_dir/<condition>_filtered/<hadm_id>/`` and drop its failing lines.
    Returns summed summaries per filtered condition."""
    from ehr_rag.checker.rules import checker_hashes
    from ehr_rag.config import resolve_path
    from ehr_rag.evidence.build_units import units_dir
    from ehr_rag.evidence.schema import read_units

    checker = Checker.from_config(cfg)
    meta = checker_hashes(cfg)
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    totals: dict[str, dict[str, int]] = {}
    for gdir in gen_dirs:
        for cond in conditions:
            cdir = gdir / cond
            if not cdir.is_dir():
                continue
            for d in sorted((p for p in cdir.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
                name = f"{cond}_filtered"
                dest = out_dir / name / d.name
                if dest.exists():
                    raise ValueError(f"{cond} hadm {d.name} appears in more than one generation run")
                shutil.copytree(d, dest)
                shown = json.loads((dest / "evidence_shown.json").read_text(encoding="utf-8"))
                shown["condition"] = name
                _write(dest / "evidence_shown.json", json.dumps(shown, indent=2, ensure_ascii=False) + "\n")
                units = read_units(udir / f"{d.name}.jsonl")
                ctx = checker.admission(frozen_units(shown, units), units)
                s = enforce_dir(dest, ctx, checker, "filter", checker_meta=meta)
                add_summaries(totals.setdefault(name, {}), s)
                log(f"  {name} hadm {d.name}: kept {s['n_kept']}, dropped {s['n_dropped']} of {s['n_lines']}")
    return totals
