"""Reference-based text metrics (ROUGE-L, BERTScore). They REQUIRE a reference file.

The MIMIC-IV Demo has no reference reports and this code never generates any. The
reference file is a CSV with columns ``hadm_id,reference`` (one reference report per
admission). ROUGE-L is implemented here (LCS over lowercased word tokens, F1).
BERTScore uses the optional ``bert-score`` package (not in requirements.txt; install
it only if you have real references); if it is missing, the command stops.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

NO_REFERENCES_MESSAGE = (
    "No reference reports provided (--references). The MIMIC-IV Clinical Database Demo has no "
    "reference reports, and this project never generates references. ROUGE-L / BERTScore were not computed."
)


class ReferenceFileError(RuntimeError):
    """Missing or invalid reference file."""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def rouge_l(candidate: str, reference: str) -> dict[str, float]:
    """ROUGE-L precision, recall and F1 from the longest common subsequence of word tokens."""
    c, r = _tokens(candidate), _tokens(reference)
    if not c or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    prev = [0] * (len(r) + 1)
    for ct in c:
        cur = [0] * (len(r) + 1)
        for j, rt in enumerate(r, 1):
            cur[j] = prev[j - 1] + 1 if ct == rt else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    p, rec = lcs / len(c), lcs / len(r)
    return {"precision": p, "recall": rec, "f1": 0.0 if lcs == 0 else 2 * p * rec / (p + rec)}


def load_references(path: Path | None) -> dict[int, str]:
    """Read ``hadm_id,reference`` CSV. Raises ReferenceFileError if absent or malformed."""
    if path is None:
        raise ReferenceFileError(NO_REFERENCES_MESSAGE)
    if not path.exists():
        raise ReferenceFileError(f"reference file not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or set(rows[0]) != {"hadm_id", "reference"}:
        raise ReferenceFileError(f"{path}: expected columns hadm_id,reference")
    refs = {int(r["hadm_id"]): r["reference"] for r in rows}
    if any(not v.strip() for v in refs.values()):
        raise ReferenceFileError(f"{path}: empty reference text")
    return refs


def bertscore(candidates: list[str], references: list[str], lang: str = "en") -> list[float]:
    """BERTScore F1 via the ``bert-score`` package (default model for ``lang``)."""
    try:
        from bert_score import score
    except ImportError as e:
        raise ReferenceFileError("BERTScore needs the optional package bert-score (pip install bert-score)") from e
    _, _, f1 = score(candidates, references, lang=lang)
    return [float(x) for x in f1]


def text_metrics(reports: dict[int, str], references: dict[int, str], with_bertscore: bool) -> list[dict[str, Any]]:
    """Per-admission ROUGE-L (and optionally BERTScore) for admissions that have a reference."""
    ids = [h for h in reports if h in references]
    rows = [{"hadm_id": h, **{f"rougeL_{k}": v for k, v in rouge_l(reports[h], references[h]).items()}}
            for h in ids]
    if with_bertscore and ids:
        for row, f1 in zip(rows, bertscore([reports[h] for h in ids], [references[h] for h in ids])):
            row["bertscore_f1"] = f1
    return rows
