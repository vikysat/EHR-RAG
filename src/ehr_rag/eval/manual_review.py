"""Manual-review export: ``manual_review.csv`` (SPEC Phase 5).

Sampling, as implemented: abstention lines are excluded. Strata are
(condition, section), sorted. Allocation is equal: one sentence at a time is given
to each stratum in turn (round-robin) until ``eval.manual_review_n`` sentences are
allocated or every stratum is exhausted. Within a stratum, sentences are drawn with
``random.Random(eval.seed)`` (a separate generator per stratum, seeded with
``f"{seed}:{condition}:{section}"``), so the sample does not depend on the other strata.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from ehr_rag.eval.citations import SentenceEval

COLUMNS: tuple[str, ...] = (
    "run_id", "condition", "hadm_id", "section", "sentence", "cited_ids", "cited_texts", "nli_label",
    "human_support", "human_error_type", "notes", "sentence_key", "full_line",
)
HUMAN_SUPPORT_VALUES: tuple[str, ...] = ("supported", "partial", "unsupported", "not_checkable")


def sentence_key(s: SentenceEval) -> str:
    """Stable id of a claim within an evaluation run (``.<k>`` added for claims of a split line)."""
    key = f"{s.condition}|{s.hadm_id}|{s.section}|{s.index}"
    return f"{key}.{s.claim_index}" if s.n_claims > 1 else key


def sample_sentences(sents: list[SentenceEval], n: int, seed: int) -> list[SentenceEval]:
    """Stratified (condition x section) equal-allocation sample."""
    strata: dict[tuple[str, str], list[SentenceEval]] = {}
    for s in sents:
        if not s.is_abstention:
            strata.setdefault((s.condition, s.section), []).append(s)
    shuffled = {}
    for k in sorted(strata):
        items = list(strata[k])
        random.Random(f"{seed}:{k[0]}:{k[1]}").shuffle(items)
        shuffled[k] = items
    taken = {k: 0 for k in shuffled}
    out: list[SentenceEval] = []
    while len(out) < n:
        progressed = False
        for k in sorted(shuffled):
            if len(out) >= n:
                break
            if taken[k] < len(shuffled[k]):
                out.append(shuffled[k][taken[k]])
                taken[k] += 1
                progressed = True
        if not progressed:
            break
    return out


def write_manual_review(
    path: Path, run_id: str, sample: list[SentenceEval], texts: dict[tuple[str, int], dict[str, str]]
) -> None:
    """Write the CSV; ``texts[(condition, hadm_id)]`` maps unit id -> text shown to the model."""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for s in sample:
            t = texts[(s.condition, s.hadm_id)]
            w.writerow({
                "run_id": run_id, "condition": s.condition, "hadm_id": s.hadm_id, "section": s.section,
                "sentence": s.sentence, "cited_ids": " ".join(s.cited_ids + s.invalid_ids),
                "cited_texts": "\n".join(f"[{i}] {t[i]}" if i in t else f"[{i}] <not in evidence>"
                                         for i in s.cited_ids + s.invalid_ids),
                "nli_label": s.nli_label, "human_support": "", "human_error_type": "", "notes": "",
                "sentence_key": sentence_key(s), "full_line": s.line if s.n_claims > 1 else "",
            })


def read_manual_review(path: Path) -> list[dict[str, Any]]:
    """Read a (possibly labeled) manual_review.csv."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in ("nli_label", "human_support") if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    return rows
