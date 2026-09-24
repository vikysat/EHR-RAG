"""Phase 3: retrieval (fake units, fake deterministic encoder; no model downloads)."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from ehr_rag.config import load_config
from ehr_rag.data.load import load_tables
from ehr_rag.evidence.build_units import build_units_for_admission
from ehr_rag.evidence.schema import EvidenceUnit
from ehr_rag.retrieval.bm25 import tokenize
from ehr_rag.retrieval.fuse import rrf
from ehr_rag.retrieval.relevance import compute_metrics, read_labels, sample_pairs, write_export
from ehr_rag.retrieval.retrieve import AdmissionRetriever, Encoders, interleave, retrieve_admission


class FakeEncoder:
    """Deterministic bag-of-hashed-tokens encoder standing in for MedCPT in tests."""

    dim = 64

    def encode(self, items: list) -> np.ndarray:
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for i, it in enumerate(items):
            text = it if isinstance(it, str) else f"{it[0]} {it[1]}"
            for tok in tokenize(text):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        out += 1e-6
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _fake_encoders() -> Encoders:
    return Encoders(query=FakeEncoder(), doc=FakeEncoder(), cross=None)  # type: ignore[arg-type]


def _units(fake_config: Path) -> dict[int, list[EvidenceUnit]]:
    cfg = load_config(fake_config)
    t = load_tables(cfg)
    return {h: build_units_for_admission(t, h, cfg["units"])[0] for h in (12, 21)}


def test_tokenize_keeps_numbers() -> None:
    assert tokenize("Potassium 5.9 [HIGH] ref 3.3-5.1 ICD-10 N17.9") == [
        "potassium", "5.9", "high", "ref", "3.3", "5.1", "icd", "10", "n17.9"
    ]


def test_rrf_and_dedupe() -> None:
    fused = rrf([["a", "b", "c", "a"], ["b", "a", "d"]], k=60)
    ids = [d for d, _ in fused]
    assert ids[:2] in (["a", "b"], ["b", "a"]) and sorted(ids) == ["a", "b", "c", "d"]
    assert len(ids) == len(set(ids))
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_interleave_by_rank() -> None:
    assert interleave([["a", "b", "c"], ["d", "a", "e"], ["f"]], cap=5) == ["a", "d", "f", "b", "c"]


def test_retrieval_isolation(fake_config: Path) -> None:
    units = _units(fake_config)
    enc = _fake_encoders()
    cfg = load_config(fake_config)
    for h, us in units.items():
        own = {u.unit_id for u in us}
        r = AdmissionRetriever(us, enc, rrf_k=60)
        for q in ["Fakeium", "FakeDrugA 20 mg", "Fake condition three", "prior admission"]:
            res = r.search(q)
            for ranking in (res.bm25, res.dense, res.fused):
                assert {d for d, _ in ranking} <= own
        out = retrieve_admission(us, enc, cfg["retrieval"])
        assert out.hadm_id == h
        assert all(us_id in own for us_id in out.evidence)
        # Every unit returned really belongs to this admission (not just an id collision).
        by_id = {u.unit_id: u for u in us}
        assert {by_id[d].hadm_id for d in out.evidence} == {h}


def test_retriever_rejects_mixed_admissions(fake_config: Path) -> None:
    units = _units(fake_config)
    with pytest.raises(ValueError, match="more than one admission"):
        AdmissionRetriever(units[12] + units[21], _fake_encoders(), rrf_k=60)


def test_evidence_order_and_caps(fake_config: Path) -> None:
    cfg = load_config(fake_config)
    r = dict(cfg["retrieval"], top_k_per_query=2, max_units_per_section=3, max_units_total=5)
    out = retrieve_admission(_units(fake_config)[12], _fake_encoders(), r)
    assert all(len(v) <= 3 for v in out.sections.values())
    assert len(out.evidence) <= 5 and len(set(out.evidence)) == len(out.evidence)
    expected: list[str] = []
    for s in ("History", "Findings", "Impression", "Recommendations"):
        expected += [d for d in out.sections[s] if d not in expected]
    assert out.evidence == expected[:5]
    assert out.truncated_by_total_cap == expected[5:]
    assert "Fake acute condition two" in out.queries["Impression"][0].query


def test_relevance_export_and_metrics(tmp_path: Path) -> None:
    items = [{"hadm_id": 1, "query": "q", "rankings": {"bm25": ["E1", "E2"], "dense": ["E3", "E1"],
                                                        "fused": ["E1", "E3"]}}]
    csv_path, side = write_export(tmp_path, items, {1: {"E1": "t1", "E2": "t2", "E3": "t3"}})
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert [r["unit_id"] for r in rows] == ["E1", "E2", "E3"]
    assert list(rows[0].keys()) == ["hadm_id", "query", "unit_id", "unit_text", "relevant"]
    with pytest.raises(ValueError, match="must be 0 or 1"):
        read_labels(csv_path)
    for r in rows:
        r["relevant"] = "1" if r["unit_id"] == "E3" else "0"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    import json

    m = compute_metrics(read_labels(csv_path), json.loads(side.read_text(encoding="utf-8")))["methods"]
    assert m["bm25"] == {"recall@5": 0.0, "mrr": 0.0, "n_pairs": 1}
    assert m["dense"]["mrr"] == 1.0 and m["fused"]["mrr"] == 0.5 and m["fused"]["recall@5"] == 1.0


def test_sample_pairs_deterministic() -> None:
    pairs = [(h, f"q{i}") for h in (1, 2, 3) for i in range(5)]
    assert sample_pairs(pairs, 4, 13) == sample_pairs(list(reversed(pairs)), 4, 13)


class FakeCross:
    """Scores a text by how many times it contains 'Fake' (stands in for the MedCPT cross-encoder)."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(t.count("Fake")) for t in texts]


def test_rerank_path_and_lazy_model(fake_config: Path) -> None:
    cfg = load_config(fake_config)
    assert Encoders.from_config(cfg, Path("unused")).cross is None  # rerank: false -> no cross-encoder
    enc = Encoders(query=FakeEncoder(), doc=FakeEncoder(), cross=FakeCross())  # type: ignore[arg-type]
    us = _units(fake_config)[12]
    res = AdmissionRetriever(us, enc, rrf_k=60).search("anything")
    assert res.reranked is not None and res.final() == [d for d, _ in res.reranked]
    scores = [s for _, s in res.reranked]
    assert scores == sorted(scores, reverse=True)
    assert {d for d, _ in res.reranked} == {u.unit_id for u in us}


def test_commit_hash_required() -> None:
    from types import SimpleNamespace

    from ehr_rag.retrieval.dense import _commit_hash

    assert _commit_hash(SimpleNamespace(config=SimpleNamespace(_commit_hash="abc123")), "m") == "abc123"
    with pytest.raises(RuntimeError, match="revision hash"):
        _commit_hash(SimpleNamespace(config=SimpleNamespace()), "m")
