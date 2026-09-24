"""Sparse retrieval with ``rank_bm25.BM25Okapi``."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

# Lowercased alphanumeric runs; a decimal part is kept attached so numbers
# such as "5.9" or "n17.9" stay one token.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization that keeps numbers (including decimals) intact."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 over a fixed list of documents identified by ``doc_ids``."""

    def __init__(self, doc_ids: list[str], texts: list[str]) -> None:
        if len(doc_ids) != len(texts) or not doc_ids:
            raise ValueError("doc_ids and texts must be non-empty and the same length")
        self.doc_ids = list(doc_ids)
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def rank(self, query: str) -> list[tuple[str, float]]:
        """All documents sorted by BM25 score (desc); ties keep document order."""
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        order = np.argsort(-scores, kind="stable")
        return [(self.doc_ids[i], float(scores[i])) for i in order]
