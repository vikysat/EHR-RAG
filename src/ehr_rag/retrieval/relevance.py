"""Relevance-judgment export and pooled retrieval metrics.

Export: for ``retrieval.relevance_export_pairs`` (admission, query) pairs drawn
with ``random.Random(eval.seed)`` from all (selected admission, filled query)
pairs, write the pooled, deduplicated top-10 of each method to
``relevance_to_label.csv`` (rows sorted by unit number, so the method is not
revealed). Each method's top-10 ranking is saved in ``relevance_rankings.json``
next to it.

Metrics (per method, averaged over pairs that have >= 1 relevant unit in the pool):
  Recall@5 = |relevant in method top-5| / |relevant in pool|
  MRR      = mean of 1 / rank of the first relevant unit in the method's top-10
             (0 if none).
Relevance is pooled: units outside the pool are unjudged, so recall is relative
to the pool, not to all relevant units.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

POOL_DEPTH = 10
RECALL_K = 5
CSV_COLUMNS = ["hadm_id", "query", "unit_id", "unit_text", "relevant"]
POOLING_NOTE = (
    "NOTE: relevance judgments are pooled (union of each method's top-10). "
    "Recall@5 is relative to the relevant units found in the pool, not to all relevant units."
)


def sample_pairs(pairs: list[tuple[int, str]], n: int, seed: int) -> list[tuple[int, str]]:
    """Deterministic sample of (hadm_id, query) pairs (all pairs if n >= len)."""
    pairs = sorted(pairs)
    if n >= len(pairs):
        return pairs
    return random.Random(seed).sample(pairs, n)


def _unit_num(uid: str) -> int:
    return int(uid[1:])


def write_export(
    out_dir: Path,
    items: list[dict[str, Any]],
    unit_texts: dict[int, dict[str, str]],
) -> tuple[Path, Path]:
    """Write the CSV to label and the rankings sidecar.

    ``items``: [{"hadm_id", "query", "rankings": {method: [unit_id, ...top-10]}}].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "relevance_to_label.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for it in items:
            pool = sorted({d for r in it["rankings"].values() for d in r[:POOL_DEPTH]}, key=_unit_num)
            for d in pool:
                w.writerow([it["hadm_id"], it["query"], d, unit_texts[it["hadm_id"]][d], ""])
    side = out_dir / "relevance_rankings.json"
    side.write_text(json.dumps({"pool_depth": POOL_DEPTH, "pairs": items}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return csv_path, side


def read_labels(csv_path: Path) -> dict[tuple[int, str], dict[str, int]]:
    """Read a labeled CSV. ``relevant`` must be 0 or 1 on every row."""
    labels: dict[tuple[int, str], dict[str, int]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path}: missing columns {missing}")
        for line, row in enumerate(reader, start=2):
            v = (row["relevant"] or "").strip()
            if v not in ("0", "1"):
                raise ValueError(f"{csv_path}:{line}: 'relevant' must be 0 or 1, got {v!r}")
            labels.setdefault((int(row["hadm_id"]), row["query"]), {})[row["unit_id"]] = int(v)
    return labels


def compute_metrics(
    labels: dict[tuple[int, str], dict[str, int]], rankings: dict[str, Any]
) -> dict[str, Any]:
    """Recall@5 and MRR per method over pairs with at least one relevant pooled unit."""
    per_method: dict[str, dict[str, list[float]]] = {}
    skipped: list[dict[str, Any]] = []
    for it in rankings["pairs"]:
        key = (int(it["hadm_id"]), it["query"])
        if key not in labels:
            raise ValueError(f"pair {key} has no labels in the CSV")
        lab = labels[key]
        pool = {d for r in it["rankings"].values() for d in r[:POOL_DEPTH]}
        unlabeled = pool - set(lab)
        if unlabeled:
            raise ValueError(f"pair {key}: pooled units without labels {sorted(unlabeled)}")
        rel = {d for d, v in lab.items() if v == 1}
        if not rel:
            skipped.append({"hadm_id": key[0], "query": key[1]})
            continue
        for method, ranking in it["rankings"].items():
            top = ranking[:POOL_DEPTH]
            recall = len(rel & set(top[:RECALL_K])) / len(rel)
            rr = next((1.0 / (i + 1) for i, d in enumerate(top) if d in rel), 0.0)
            m = per_method.setdefault(method, {"recall@5": [], "mrr": []})
            m["recall@5"].append(recall)
            m["mrr"].append(rr)
    summary = {
        method: {
            "recall@5": sum(v["recall@5"]) / len(v["recall@5"]),
            "mrr": sum(v["mrr"]) / len(v["mrr"]),
            "n_pairs": len(v["mrr"]),
        }
        for method, v in per_method.items()
    }
    return {"methods": summary, "skipped_no_relevant": skipped, "note": POOLING_NOTE}
