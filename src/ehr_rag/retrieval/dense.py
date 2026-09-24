"""Dense retrieval with MedCPT encoders and a per-admission FAISS index.

Pooling follows the MedCPT model cards: the [CLS] vector of
``last_hidden_state`` (``[:, 0, :]``). Queries use max_length 64; articles are
encoded as ``[title, abstract]`` pairs with max_length 512. Embeddings are then
L2-normalized (per SPEC) so inner product = cosine similarity.

Embeddings are cached on disk in ``<cache_dir>/embeddings/<model-slug>.npz``,
keyed by sha256 of (model name, title, text).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import faiss
import numpy as np

QUERY_MAX_LEN = 64
ARTICLE_MAX_LEN = 512


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def cache_key(model_name: str, title: str, text: str) -> str:
    """sha256 over model name, title and text."""
    h = hashlib.sha256()
    for part in (model_name, title, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class EmbeddingCache:
    """A simple npz-backed ``key -> vector`` store for one model."""

    def __init__(self, cache_dir: Path, model_name: str) -> None:
        self.path = cache_dir / "embeddings" / f"{_slug(model_name)}.npz"
        self._data: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as z:
                self._data = {k: z[k] for k in z.files}
        self._dirty = False

    def get(self, key: str) -> np.ndarray | None:
        """Cached vector or None."""
        return self._data.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        """Store a vector (saved on ``flush``)."""
        self._data[key] = vec.astype(np.float32)
        self._dirty = True

    def flush(self) -> None:
        """Atomically write the cache file if anything changed."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, **self._data)
        os.replace(tmp, self.path)
        self._dirty = False


class MedCPTEncoder:
    """Lazily loaded MedCPT encoder (query or article). Fails loudly if the model cannot load."""

    def __init__(self, model_name: str, kind: str, cache_dir: Path, batch_size: int = 16) -> None:
        if kind not in ("query", "article"):
            raise ValueError("kind must be 'query' or 'article'")
        self.model_name = model_name
        self.kind = kind
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_dir, model_name)
        self._model = None
        self._tok = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  (import here so tests without models stay fast)
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def _encode_batch(self, items: list[tuple[str, str]] | list[str]) -> np.ndarray:
        import torch

        self._load()
        with torch.no_grad():
            if self.kind == "query":
                enc = self._tok(items, truncation=True, padding=True, return_tensors="pt",
                                max_length=QUERY_MAX_LEN)
            else:
                titles = [t for t, _ in items]  # type: ignore[misc]
                texts = [a for _, a in items]  # type: ignore[misc]
                enc = self._tok(titles, texts, truncation=True, padding=True, return_tensors="pt",
                                max_length=ARTICLE_MAX_LEN)
            out = self._model(**enc).last_hidden_state[:, 0, :]
        return out.cpu().numpy().astype(np.float32)

    def encode(self, items: list[str] | list[tuple[str, str]]) -> np.ndarray:
        """Encode queries (``str``) or articles (``(title, text)``); returns L2-normalized rows."""
        keys = [
            cache_key(self.model_name, "", it) if isinstance(it, str) else cache_key(self.model_name, it[0], it[1])
            for it in items
        ]
        missing = [i for i, k in enumerate(keys) if self.cache.get(k) is None]
        for start in range(0, len(missing), self.batch_size):
            idx = missing[start:start + self.batch_size]
            vecs = self._encode_batch([items[i] for i in idx])  # type: ignore[misc]
            for i, v in zip(idx, vecs):
                self.cache.put(keys[i], v)
        self.cache.flush()
        mat = np.stack([self.cache.get(k) for k in keys]).astype(np.float32)
        faiss.normalize_L2(mat)
        return mat


class DenseIndex:
    """FAISS ``IndexFlatIP`` over one admission's unit embeddings."""

    def __init__(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        if len(doc_ids) != embeddings.shape[0] or not doc_ids:
            raise ValueError("doc_ids and embeddings must be non-empty and aligned")
        self.doc_ids = list(doc_ids)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    def rank(self, query_vec: np.ndarray) -> list[tuple[str, float]]:
        """All documents sorted by inner product (desc)."""
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        scores, idx = self.index.search(q, len(self.doc_ids))
        return [(self.doc_ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


class MedCPTCrossEncoder:
    """Optional MedCPT cross-encoder reranker (loaded only when used)."""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tok = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance logits for (query, text) pairs, following the model card."""
        import torch

        self._load()
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                enc = self._tok([[query, t] for t in batch], truncation=True, padding=True,
                                return_tensors="pt", max_length=ARTICLE_MAX_LEN)
                logits = self._model(**enc).logits.squeeze(dim=1)
                out.extend(float(x) for x in logits.cpu().numpy().reshape(-1))
        return out
