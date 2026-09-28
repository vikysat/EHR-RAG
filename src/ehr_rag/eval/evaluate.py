"""Evaluation of one generation run (``evaluate`` command).

Reads ``runs/<generation_run>/<condition>/<hadm_id>/``: ``parsed.json`` (which raw
file was used: ``raw.txt`` or, after a format retry, ``raw_retry.txt``) and
``evidence_shown.json`` (the source of truth for valid ids and unit texts). Reports
with ``format_error`` status are INCLUDED (parsed best-effort) and counted.

Writes into the evaluation run: ``sentences.jsonl`` (every sentence with its NLI
result), ``metrics.json``, ``metrics_by_section.csv``, ``metrics_summary.md``,
``manual_review.csv``.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ehr_rag.config import CONDITIONS
from ehr_rag.eval.citations import GROUPS, SentenceEval, aggregate, evaluate_report, metrics_table
from ehr_rag.eval.manual_review import sample_sentences, write_manual_review
from ehr_rag.eval.nli import NLI
from ehr_rag.eval.parse import parse_report

METRICS = ("cited_rate", "invalid_citation_rate", "citation_recall", "citation_precision")
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


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def summary_markdown(rows: list[dict[str, Any]], format_counts: dict[str, dict[str, int]],
                     gen_run: str, nli_info: dict[str, Any]) -> str:
    """``metrics_summary.md``."""
    lines = [f"# Metrics summary: generation run `{gen_run}`", "",
             f"NLI verifier: `{nli_info.get('model')}` (revision `{nli_info.get('revision')}`); "
             f"premise truncated for {nli_info.get('n_truncated_pairs', 0)} pair(s).", "",
             "| condition | scope | sentences | abstentions | cited_rate | invalid_citation_rate | "
             "citation_recall | citation_precision |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["scope"] not in GROUPS:
            continue
        lines.append(f"| {r['condition']} | {r['scope']} | {r['n_sentences']} | {r['n_abstentions']} | "
                     + " | ".join(_fmt(r[m]) for m in METRICS) + " |")
    lines += ["", "Format status of the reports (from generation):", "",
              "| condition | ok | ok_after_retry | format_error |", "|---|---|---|---|"]
    for c, n in format_counts.items():
        lines.append(f"| {c} | {n.get('ok', 0)} | {n.get('ok_after_retry', 0)} | {n.get('format_error', 0)} |")
    lines += ["", SUMMARY_NOTE, "", "Per-section values: `metrics_by_section.csv`. Definitions: README "
              "'Metric definitions'.", ""]
    return "\n".join(lines)


def run_evaluation(
    gen_run_dir: Path, out_dir: Path, run_id: str, nli: NLI, manual_review_n: int, seed: int
) -> dict[str, Any]:
    """Evaluate every report of a generation run and write all outputs to ``out_dir``."""
    reports = load_generation_run(gen_run_dir)
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
    info = nli.info()
    metrics = {
        "generation_run": gen_run_dir.name, "nli": info, "format_status": format_counts,
        "conditions": {c: {r["scope"]: {k: v for k, v in r.items() if k not in ("condition", "scope")}
                           for r in rows if r["condition"] == c} for c in conditions},
        "per_report": per_report,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8", newline="\n")
    (out_dir / "metrics_summary.md").write_text(
        summary_markdown(rows, format_counts, gen_run_dir.name, info), encoding="utf-8", newline="\n")
    sample = sample_sentences(sents, manual_review_n, seed)
    write_manual_review(out_dir / "manual_review.csv", run_id, sample, texts)
    return {"rows": rows, "format_counts": format_counts, "sentences": sents, "texts": texts,
            "n_manual_review": len(sample), "nli": info, "reports": reports}


def gate_sample(sents: list[SentenceEval], texts: dict[tuple[str, int], dict[str, str]], n: int,
                seed: int) -> list[dict[str, Any]]:
    """``n`` random non-abstention sentences with citations, cited texts and NLI label (Gate 4 check)."""
    pool = [s for s in sents if not s.is_abstention]
    pick = random.Random(seed).sample(pool, min(n, len(pool)))
    return [{"condition": s.condition, "hadm_id": s.hadm_id, "section": s.section, "sentence": s.sentence,
             "cited_ids": s.cited_ids, "invalid_ids": s.invalid_ids,
             "cited_texts": [texts[(s.condition, s.hadm_id)][i] for i in s.cited_ids],
             "nli_label": s.nli_label, "irrelevant_ids": s.irrelevant_ids} for s in pick]
