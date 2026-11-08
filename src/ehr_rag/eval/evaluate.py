"""Evaluation of one generation run (``evaluate`` command).

Reads ``runs/<generation_run>/<condition>/<hadm_id>/``: ``parsed.json`` (which raw
file was used: ``raw.txt`` or, after a format retry, ``raw_retry.txt``) and
``evidence_shown.json`` (the source of truth for valid ids and unit texts). Reports
with ``format_error`` status are INCLUDED (parsed best-effort) and counted.

Writes into the evaluation run: ``sentences.jsonl`` (every sentence with its NLI
result), ``metrics.json``, ``metrics_by_section.csv``, ``metrics_summary.md``,
``manual_review.csv``, ``nli_too_long.csv`` (every pair labeled ``too_long``).
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ehr_rag.config import CONDITIONS
from ehr_rag.eval.citations import GROUPS, SentenceEval, aggregate, evaluate_report, metrics_table, premise_text
from ehr_rag.eval.manual_review import sample_sentences, write_manual_review
from ehr_rag.eval.nli import NLI
from ehr_rag.eval.parse import parse_report

METRICS = ("cited_rate", "invalid_citation_rate", "citation_recall", "citation_precision",
           "citation_precision_plain")
TOO_LONG_COLUMNS = ("condition", "hadm_id", "section", "index", "claim_index", "premise_ids", "kind", "n_tokens",
                    "max_length")
SUMMARY_NOTE = (
    "Entailment-based recall/precision are most meaningful for History+Findings; Impression+Recommendations "
    "contain clinical reasoning that a sentence-level NLI verifier is not expected to capture. "
    "Abstention lines ('Insufficient evidence. [NONE]') are excluded from all rates and counted separately."
)


@dataclass
class ReportInput:
    """One generated report and what the model saw."""

    condition: str
    hadm_id: int
    status: str
    raw_file: str
    raw: str
    evidence: list[dict[str, Any]]


def load_generation_run(run_dir: Path) -> list[ReportInput]:
    """All reports of a generation run, in condition order then hadm_id."""
    out: list[ReportInput] = []
    for cond in CONDITIONS:
        cdir = run_dir / cond
        if not cdir.is_dir():
            continue
        for d in sorted((p for p in cdir.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
            parsed_p = d / "parsed.json"
            if not parsed_p.exists():
                raise FileNotFoundError(f"incomplete report directory (no parsed.json): {d}")
            meta = json.loads(parsed_p.read_text(encoding="utf-8"))
            shown = json.loads((d / "evidence_shown.json").read_text(encoding="utf-8"))
            out.append(ReportInput(
                condition=cond, hadm_id=int(d.name), status=meta["status"], raw_file=meta["raw_file"],
                raw=(d / meta["raw_file"]).read_text(encoding="utf-8"), evidence=shown["units"],
            ))
    if not out:
        raise FileNotFoundError(f"no reports found under {run_dir}")
    return out


def load_generation_runs(run_dirs: list[Path]) -> list[ReportInput]:
    """Reports of one or more generation runs (e.g. the shards of one job array), in run order.

    Raises ValueError if the same (condition, admission) appears in two runs.
    """
    out: list[ReportInput] = []
    seen: dict[tuple[str, int], str] = {}
    for d in run_dirs:
        for r in load_generation_run(d):
            key = (r.condition, r.hadm_id)
            if key in seen:
                raise ValueError(f"{r.condition} hadm {r.hadm_id} is in both {seen[key]} and {d.name}")
            seen[key] = d.name
            out.append(r)
    return out


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def summary_markdown(rows: list[dict[str, Any]], format_counts: dict[str, dict[str, int]],
                     gen_run: str, nli_info: dict[str, Any]) -> str:
    """``metrics_summary.md``."""
    lines = [f"# Metrics summary: generation run `{gen_run}`", "",
             f"NLI verifier: `{nli_info.get('model')}` (revision `{nli_info.get('revision')}`); "
             f"{nli_info.get('n_too_long_pairs', 0)} premise/claim pair(s) longer than "
             f"{nli_info.get('max_length', 'the model limit')} tokens were labeled `too_long` (not entailed; "
             "listed in `nli_too_long.csv`); nothing was truncated.", "",
             "Precision: `citation_precision` is ALCE (a citation is irrelevant if it does not entail the claim "
             "alone and removing it does not change whether the rest entail it); `citation_precision_plain` is "
             "the share of citations that entail their claim on their own.", "",
             "| condition | scope | sentences | claims | abstentions | cited_rate | invalid_citation_rate | "
             "citation_recall | citation_precision (ALCE) | citation_precision_plain |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["scope"] not in GROUPS:
            continue
        lines.append(f"| {r['condition']} | {r['scope']} | {r['n_sentences']} | {r['n_claims']} | "
                     f"{r['n_abstentions']} | " + " | ".join(_fmt(r[m]) for m in METRICS) + " |")
    lines += ["", "Format status of the reports (from generation):", "",
              "| condition | ok | ok_after_retry | format_error |", "|---|---|---|---|"]
    for c, n in format_counts.items():
        lines.append(f"| {c} | {n.get('ok', 0)} | {n.get('ok_after_retry', 0)} | {n.get('format_error', 0)} |")
    lines += ["", SUMMARY_NOTE, "", "Per-section values: `metrics_by_section.csv`. Definitions: README "
              "'Metric definitions'.", ""]
    return "\n".join(lines)


def run_evaluation(
    gen_run_dirs: Path | list[Path], out_dir: Path, run_id: str, nli: NLI, manual_review_n: int, seed: int
) -> dict[str, Any]:
    """Evaluate every report of one or more generation runs (shards) and write all outputs to ``out_dir``.

    Reports are ordered by condition then hadm_id, as within a single run, so shards
    give the same metrics and manual-review sample as one unsharded run.
    """
    dirs = [gen_run_dirs] if isinstance(gen_run_dirs, Path) else list(gen_run_dirs)
    reports = sorted(load_generation_runs(dirs), key=lambda r: (CONDITIONS.index(r.condition), r.hadm_id))
    gen_name = " + ".join(d.name for d in dirs)
    sents: list[SentenceEval] = []
    per_report: list[dict[str, Any]] = []
    texts: dict[tuple[str, int], dict[str, str]] = {}
    format_counts: dict[str, dict[str, int]] = {}
    for rep in reports:
        parsed = parse_report(rep.raw)
        ev = evaluate_report(parsed, rep.evidence, rep.condition, rep.hadm_id, nli)
        sents.extend(ev)
        texts[(rep.condition, rep.hadm_id)] = {u["unit_id"]: u["text"] for u in rep.evidence}
        fc = format_counts.setdefault(rep.condition, {})
        fc[rep.status] = fc.get(rep.status, 0) + 1
        per_report.append({"condition": rep.condition, "hadm_id": rep.hadm_id, "status": rep.status,
                           "raw_file": rep.raw_file, "parse_flags": parsed.flags(), **aggregate(ev)})
    conditions = [c for c in CONDITIONS if c in format_counts]
    rows = metrics_table(sents, conditions)

    with (out_dir / "sentences.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for s in sents:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    with (out_dir / "metrics_by_section.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    too_long = too_long_rows(sents, texts, nli)
    with (out_dir / "nli_too_long.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TOO_LONG_COLUMNS)
        w.writeheader()
        w.writerows(too_long)
    info = nli.info()
    metrics = {
        "generation_run": dirs[0].name if len(dirs) == 1 else [d.name for d in dirs], "nli": info,
        "format_status": format_counts,
        "conditions": {c: {r["scope"]: {k: v for k, v in r.items() if k not in ("condition", "scope")}
                           for r in rows if r["condition"] == c} for c in conditions},
        "per_report": per_report,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8", newline="\n")
    (out_dir / "metrics_summary.md").write_text(
        summary_markdown(rows, format_counts, gen_name, info), encoding="utf-8", newline="\n")
    sample = sample_sentences(sents, manual_review_n, seed)
    write_manual_review(out_dir / "manual_review.csv", run_id, sample, texts)
    return {"rows": rows, "format_counts": format_counts, "sentences": sents, "texts": texts,
            "n_manual_review": len(sample), "nli": info, "reports": reports, "n_too_long_rows": len(too_long)}


def too_long_rows(sents: list[SentenceEval], texts: dict[tuple[str, int], dict[str, str]],
                  nli: NLI) -> list[dict[str, Any]]:
    """One row per (claim, premise) pair labeled ``too_long``: ``kind`` is all / single / without_one."""
    tokens = getattr(nli, "pair_tokens", {})
    max_length = getattr(nli, "max_length", None)
    out = []
    for s in sents:
        for ids in s.too_long:
            kind = "all" if ids == s.cited_ids else "single" if len(ids) == 1 else "without_one"
            premise = premise_text(ids, texts[(s.condition, s.hadm_id)])
            out.append({"condition": s.condition, "hadm_id": s.hadm_id, "section": s.section, "index": s.index,
                        "claim_index": s.claim_index, "premise_ids": " ".join(ids), "kind": kind,
                        "n_tokens": tokens.get((premise, s.sentence)), "max_length": max_length})
    return out


def gate_sample(sents: list[SentenceEval], texts: dict[tuple[str, int], dict[str, str]], n: int,
                seed: int) -> list[dict[str, Any]]:
    """``n`` random non-abstention sentences with citations, cited texts and NLI label (Gate 4 check)."""
    pool = [s for s in sents if not s.is_abstention]
    pick = random.Random(seed).sample(pool, min(n, len(pool)))
    return [{"condition": s.condition, "hadm_id": s.hadm_id, "section": s.section, "sentence": s.sentence,
             "line": s.line if s.n_claims > 1 else None, "cited_ids": s.cited_ids, "invalid_ids": s.invalid_ids,
             "cited_texts": [texts[(s.condition, s.hadm_id)][i] for i in s.cited_ids],
             "nli_label": s.nli_label, "irrelevant_ids": s.irrelevant_ids} for s in pick]
