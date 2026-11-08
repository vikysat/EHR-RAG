"""Checker statistics for generation runs (``check-citations``). Separate from evaluation.

For every report of a base condition (no_retrieval, ehr_rag, full_context) the checker is
run on the GENERATED text (before any enforcement). Enforced / filtered conditions are
not re-checked (their pre-enforcement text is the base report, and their final text was
selected by the checker); their ``enforcement.json`` logs give the enforcement table.

Outputs (in the check-citations run folder):
  checker_lines.jsonl      every checked line: status, error types, messages, counts, items
  checker_profile.csv      per condition x scope (section, History+Findings,
                           Impression+Recommendations, ALL): lines, pass / fail / unchecked,
                           checkable share, lines with each error type, numbers, names
  enforcement_summary.csv  per enforced / filtered condition: kept / repaired / retreated /
                           dropped lines, share of lines and of numbers and names kept
  checker_review.csv       for tuning the lexicon by hand: every failing line plus a seeded
                           sample of passing and unchecked lines, with the cited unit texts and
                           blank columns human_checker_correct (yes/no) and notes
  checker_summary.md       the tables above, with the checker's file hashes
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from ehr_rag.checker.enforce import add_summaries, frozen_units
from ehr_rag.checker.rules import ERROR_TYPES, Checker, checker_hashes
from ehr_rag.config import ALL_CONDITIONS, CONDITIONS, SECTIONS, resolve_path
from ehr_rag.eval.parse import parse_report

SCOPES: dict[str, tuple[str, ...]] = {
    **{s: (s,) for s in SECTIONS},
    "History+Findings": ("History", "Findings"),
    "Impression+Recommendations": ("Impression", "Recommendations"),
    "ALL": SECTIONS,
}
REVIEW_COLUMNS = ("condition", "hadm_id", "section", "index", "status", "error_types", "messages", "line",
                  "cited_texts", "human_checker_correct", "notes")


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _report_dirs(gen_dirs: list[Path]) -> list[tuple[str, Path]]:
    out = []
    for g in gen_dirs:
        for cond in ALL_CONDITIONS:
            cdir = g / cond
            if cdir.is_dir():
                out.extend((cond, d) for d in sorted((p for p in cdir.iterdir() if p.is_dir()), key=lambda p: int(p.name)))
    seen: set[tuple[str, str]] = set()
    for cond, d in out:
        if (cond, d.name) in seen:
            raise ValueError(f"{cond} hadm {d.name} appears in more than one generation run")
        seen.add((cond, d.name))
    return out


def check_runs(cfg: dict[str, Any], gen_dirs: list[Path]) -> dict[str, Any]:
    """Check every base-condition report; collect enforcement logs. Returns lines, enforcement totals."""
    from ehr_rag.evidence.build_units import units_dir
    from ehr_rag.evidence.schema import read_units

    checker = Checker.from_config(cfg)
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    lines: list[dict[str, Any]] = []
    texts: dict[tuple[str, int], dict[str, str]] = {}
    enforcement: dict[str, dict[str, int]] = {}
    emptied: dict[str, int] = {}
    units_cache: dict[int, list[Any]] = {}
    for cond, d in _report_dirs(gen_dirs):
        h = int(d.name)
        if cond not in CONDITIONS:
            log = _read_json(d / "enforcement.json")
            add_summaries(enforcement.setdefault(cond, {}), log["summary"])
            emptied[cond] = emptied.get(cond, 0) + len(log["sections_emptied"])
            continue
        meta = _read_json(d / "parsed.json")
        raw = (d / meta.get("pre_enforcement_raw_file", meta["raw_file"])).read_text(encoding="utf-8")
        shown = _read_json(d / "evidence_shown.json")
        if h not in units_cache:
            units_cache[h] = read_units(udir / f"{h}.jsonl")
        ctx = checker.admission(frozen_units(shown, units_cache[h]), units_cache[h])
        texts[(cond, h)] = {u["unit_id"]: u["text"] for u in shown["units"]}
        for lc in checker.check_report(parse_report(raw), ctx):
            lines.append({"condition": cond, "hadm_id": h, **lc.to_dict()})
    for cond in enforcement:
        enforcement[cond]["n_sections_emptied"] = emptied.get(cond, 0)
    return {"lines": lines, "texts": texts, "enforcement": enforcement}


def _share(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def profile(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The numeric error profile: rows per condition x scope."""
    rows = []
    for cond in [c for c in CONDITIONS if any(ln["condition"] == c for ln in lines)]:
        for scope, secs in SCOPES.items():
            ls = [ln for ln in lines if ln["condition"] == cond and ln["section"] in secs]
            n = len(ls)
            row: dict[str, Any] = {"condition": cond, "scope": scope, "n_lines": n}
            for st in ("pass", "fail", "unchecked"):
                row[f"n_{st}"] = sum(1 for ln in ls if ln["status"] == st)
            row["checkable_share"] = _share(n - row["n_unchecked"], n)
            row["pass_share_of_checkable"] = _share(row["n_pass"], n - row["n_unchecked"])
            for et in ERROR_TYPES:
                row[et] = sum(1 for ln in ls if et in ln["error_types"])
            row["n_numbers"] = sum(ln["n_numbers"] for ln in ls)
            row["n_names"] = sum(ln["n_names"] for ln in ls)
            rows.append(row)
    return rows


def enforcement_rows(enforcement: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """Repair vs retreat and coverage after enforcement, per enforced / filtered condition."""
    rows = []
    for cond in [c for c in ALL_CONDITIONS if c in enforcement]:
        t = enforcement[cond]
        rows.append({
            "condition": cond, "n_lines": t["n_lines"], "n_pass": t["n_pass"], "n_unchecked": t["n_unchecked"],
            "n_fail": t["n_fail"], "kept": t["n_kept"], "repaired": t["n_repaired"], "retreated": t["n_retreated"],
            "dropped": t["n_dropped"], "rewrite_calls": t["n_rewrite_calls"],
            "sections_emptied": t.get("n_sections_emptied", 0),
            "lines_kept_share": _share(t["n_lines_after"], t["n_lines"]),
            "numbers_kept_share": _share(t["n_numbers_after"], t["n_numbers_before"]),
            "names_kept_share": _share(t["n_names_after"], t["n_names_before"]),
            "n_numbers_before": t["n_numbers_before"], "n_numbers_after": t["n_numbers_after"],
        })
    return rows


def review_sample(lines: list[dict[str, Any]], texts: dict[tuple[str, int], dict[str, str]], n_per_status: int,
                  seed: int) -> list[dict[str, Any]]:
    """All failing lines plus up to ``n_per_status`` passing and unchecked lines per condition (seeded)."""
    out = [ln for ln in lines if ln["status"] == "fail"]
    for cond in sorted({ln["condition"] for ln in lines}):
        for st in ("pass", "unchecked"):
            pool = [ln for ln in lines if ln["condition"] == cond and ln["status"] == st]
            out += random.Random(f"{seed}:{cond}:{st}").sample(pool, min(n_per_status, len(pool)))
    rows = []
    for ln in out:
        t = texts[(ln["condition"], ln["hadm_id"])]
        ids = list(dict.fromkeys(i for it in ln["items"] for i in it["ids"]))
        rows.append({"condition": ln["condition"], "hadm_id": ln["hadm_id"], "section": ln["section"],
                     "index": ln["index"], "status": ln["status"], "error_types": " ".join(ln["error_types"]),
                     "messages": "\n".join(f"{e['type']}: {e['message']}" for e in ln["errors"]),
                     "line": ln["line"],
                     "cited_texts": "\n".join(f"[{i}] {t.get(i, '<not in evidence>')}" for i in ids),
                     "human_checker_correct": "", "notes": ""})
    return rows


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def _table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + [
        "| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]


def summary_markdown(prof: list[dict[str, Any]], enf: list[dict[str, Any]], runs: list[str],
                     hashes: dict[str, Any]) -> str:
    """``checker_summary.md``: checker statistics only (never mixed with NLI or manual-label metrics)."""
    lines = [f"# Citation checker statistics: {' + '.join(runs)}", "",
             "Deterministic checker; independent of `evaluate` (NLI and manual labels are reported there).",
             f"Rules sha256 `{hashes['rules_sha256']}`, lexicon sha256 `{hashes['lexicon_sha256']}`, "
             f"rewrite prompt sha256 `{hashes['rewrite_prompt_sha256']}`.", "",
             "## Status and coverage (generated text, before enforcement)", ""]
    lines += _table([r for r in prof if r["scope"] in ("ALL", "History+Findings", "Impression+Recommendations")],
                    ["condition", "scope", "n_lines", "n_pass", "n_fail", "n_unchecked", "checkable_share",
                     "pass_share_of_checkable"])
    lines += ["", "## Error profile (lines with each error type; per section in checker_profile.csv)", ""]
    lines += _table([r for r in prof if r["scope"] in SECTIONS or r["scope"] == "ALL"],
                    ["condition", "scope", "n_lines", *ERROR_TYPES, "n_numbers", "n_names"])
    if enf:
        lines += ["", "## Enforcement: repair vs retreat, and what was kept", ""]
        lines += _table(enf, ["condition", "n_lines", "kept", "repaired", "retreated", "dropped", "rewrite_calls",
                              "sections_emptied", "lines_kept_share", "numbers_kept_share", "names_kept_share"])
        lines += ["", "repaired = rewritten, passes, keeps >= as many checkable numbers and names; retreated = "
                      "passes with fewer; dropped = failed twice (or once in filter mode)."]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, res: dict[str, Any], runs: list[str], hashes: dict[str, Any],
                  review_n: int, seed: int) -> dict[str, Any]:
    """Write every check-citations output; returns the profile and enforcement rows."""
    prof, enf = profile(res["lines"]), enforcement_rows(res["enforcement"])
    with (out_dir / "checker_lines.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for ln in res["lines"]:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    for name, rows in (("checker_profile.csv", prof), ("enforcement_summary.csv", enf)):
        with (out_dir / name).open("w", encoding="utf-8", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    review = review_sample(res["lines"], res["texts"], review_n, seed)
    with (out_dir / "checker_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(review)
    (out_dir / "checker_summary.md").write_text(summary_markdown(prof, enf, runs, hashes), encoding="utf-8",
                                               newline="\n")
    return {"profile": prof, "enforcement": enf, "n_review": len(review)}


def run_check(cfg: dict[str, Any], gen_dirs: list[Path], out_dir: Path, review_n: int, seed: int) -> dict[str, Any]:
    """``check-citations`` end to end."""
    res = check_runs(cfg, gen_dirs)
    return write_outputs(out_dir, res, [d.name for d in gen_dirs], checker_hashes(cfg), review_n, seed)
