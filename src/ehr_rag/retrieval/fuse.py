"""Reciprocal Rank Fusion."""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Fuse ranked id lists with RRF: score(d) = sum over lists of 1 / (k + rank), rank from 1.

    Ids are deduplicated (only the first occurrence in each list counts). Ties are
    broken by best single rank, then by first appearance across the input lists.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    pos = 0
    for ranking in rankings:
        seen: set[str] = set()
        for r, doc in enumerate(ranking, start=1):
            if doc in seen:
                continue
            seen.add(doc)
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r)
            best_rank[doc] = min(best_rank.get(doc, r), r)
            if doc not in first_seen:
                first_seen[doc] = pos
                pos += 1
    order = sorted(scores, key=lambda d: (-scores[d], best_rank[d], first_seen[d]))
    return [(d, scores[d]) for d in order]
