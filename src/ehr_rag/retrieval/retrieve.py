"""Per-admission hybrid retrieval and section-wise evidence assembly.

Isolation: an ``AdmissionRetriever`` is built from the units of exactly one
admission and can only ever return ids from that set.

Assembly (as implemented):
  1. For each section query, rank this admission's units with BM25 and dense,
     fuse with RRF (BM25 results with score 0, i.e. no shared term, are excluded
     from fusion) (optionally rerank the whole fused list with the
     cross-encoder), and keep the top ``top_k_per_query``.
  2. Section list: interleave the section's query results by rank (rank 1 of
     each query in query order, then rank 2, ...), deduped, capped at
     ``max_units_per_section``.
  3. Evidence: concatenate sections in order History, Findings, Impression,
     Recommendations, deduped by first appearance, capped at ``max_units_total``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ehr_rag.config import SECTIONS
from ehr_rag.evidence.schema import EvidenceUnit
from ehr_rag.retrieval.bm25 import BM25Index
from ehr_rag.retrieval.dense import DenseIndex, MedCPTCrossEncoder, MedCPTEncoder
from ehr_rag.retrieval.fuse import rrf


def unit_title(u: EvidenceUnit) -> str:
    """Title passed to the MedCPT article encoder: lab label, drug name or ICD long title."""
    f = u.fields
    if u.unit_type == "lab":
        return str(f["label"])
    if u.unit_type == "medication":
        return str(f["drug"])
    return str(f["long_title"])


def primary_dx(units: list[EvidenceUnit]) -> str:
    """Long title of the seq_num = 1 diagnosis of the target admission."""
    hits = [u for u in units if u.unit_type == "diagnosis" and u.fields["seq_num"] == 1]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one seq_num=1 diagnosis unit, found {len(hits)}")
    return str(hits[0].fields["long_title"])


def section_queries(query_cfg: dict[str, list[str]], primary: str) -> dict[str, list[str]]:
    """Fill ``{primary_dx}`` in the configured query templates."""
    return {s: [q.replace("{primary_dx}", primary) for q in query_cfg[s]] for s in SECTIONS}


@dataclass
class Encoders:
    """Shared encoders (one instance per process)."""

    query: MedCPTEncoder
    doc: MedCPTEncoder
    cross: MedCPTCrossEncoder | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], cache_dir: Path) -> "Encoders":
        """Build encoders; the cross-encoder only if ``retrieval.rerank`` is true."""
        r = cfg["retrieval"]
        cross = MedCPTCrossEncoder(r["rerank_model"]) if r["rerank"] else None
        return cls(
            query=MedCPTEncoder(r["dense_query_model"], "query", cache_dir),
            doc=MedCPTEncoder(r["dense_doc_model"], "article", cache_dir),
            cross=cross,
        )

    def revisions(self) -> dict[str, str]:
        """Model name -> resolved Hugging Face commit hash, for every model in use."""
        out = {self.query.model_name: self.query.revision(), self.doc.model_name: self.doc.revision()}
        if self.cross is not None:
            out[self.cross.model_name] = self.cross.revision()
        return out


@dataclass
class QueryResult:
    """Full rankings for one query."""

    query: str
    bm25: list[tuple[str, float]]
    dense: list[tuple[str, float]]
    fused: list[tuple[str, float]]
    reranked: list[tuple[str, float]] | None = None

    def final(self) -> list[str]:
        """Ranking used for assembly (reranked if available, else fused)."""
        src = self.reranked if self.reranked is not None else self.fused
        return [d for d, _ in src]


class AdmissionRetriever:
    """Hybrid retriever restricted to the units of ONE admission."""

    def __init__(self, units: list[EvidenceUnit], encoders: Encoders, rrf_k: int) -> None:
        if not units:
            raise ValueError("no units")
        hadm = {u.hadm_id for u in units}
        if len(hadm) != 1:
            raise ValueError(f"units from more than one admission: {sorted(hadm)}")
        ids = [u.unit_id for u in units]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate unit ids")
        self.hadm_id = hadm.pop()
        self.units = {u.unit_id: u for u in units}
        self.unit_ids = ids
        self.encoders = encoders
        self.rrf_k = rrf_k
        self.bm25 = BM25Index(ids, [u.text for u in units])
        emb = encoders.doc.encode([(unit_title(u), u.text) for u in units])
        self.dense = DenseIndex(ids, emb)

    def _own(self, ranking: list[tuple[str, float]]) -> list[tuple[str, float]]:
        bad = [d for d, _ in ranking if d not in self.units]
        if bad:
            raise RuntimeError(f"retriever for hadm {self.hadm_id} produced foreign ids {bad[:5]}")
        return ranking

    def search(self, query: str) -> QueryResult:
        """Rank all of this admission's units for ``query``."""
        bm = self._own(self.bm25.rank(query))
        qv = self.encoders.query.encode([query])[0]
        de = self._own(self.dense.rank(qv))
        # Zero-score BM25 results share no term with the query; their order is arbitrary
        # (document order), so they are excluded from fusion.
        bm_matched = [d for d, s in bm if s > 0]
        fu = self._own(rrf([bm_matched, [d for d, _ in de]], self.rrf_k))
        rr = None
        if self.encoders.cross is not None:
            ids = [d for d, _ in fu]
            scores = self.encoders.cross.score(query, [self.units[d].text for d in ids])
            order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
            rr = self._own([(ids[i], scores[i]) for i in order])
        return QueryResult(query=query, bm25=bm, dense=de, fused=fu, reranked=rr)


def interleave(lists: list[list[str]], cap: int) -> list[str]:
    """Round-robin by rank across lists, deduped, capped."""
    out: list[str] = []
    seen: set[str] = set()
    depth = max((len(x) for x in lists), default=0)
    for r in range(depth):
        for lst in lists:
            if r < len(lst) and lst[r] not in seen:
                seen.add(lst[r])
                out.append(lst[r])
                if len(out) >= cap:
                    return out
    return out


@dataclass
class AdmissionRetrieval:
    """Retrieval result for one admission."""

    hadm_id: int
    primary_dx: str
    queries: dict[str, list[QueryResult]]
    sections: dict[str, list[str]]
    evidence: list[str]
    truncated_by_total_cap: list[str] = field(default_factory=list)

    def to_json(self, top_n: int = 10) -> dict[str, Any]:
        """JSON-able summary (rankings truncated to ``top_n``)."""
        def cut(r: list[tuple[str, float]] | None) -> list[list[Any]] | None:
            return None if r is None else [[d, round(s, 6)] for d, s in r[:top_n]]

        return {
            "hadm_id": self.hadm_id,
            "primary_dx": self.primary_dx,
            "queries": {
                s: [{"query": q.query, "bm25": cut(q.bm25), "dense": cut(q.dense),
                     "fused": cut(q.fused), "reranked": cut(q.reranked)} for q in qs]
                for s, qs in self.queries.items()
            },
            "sections": self.sections,
            "evidence": self.evidence,
            "truncated_by_total_cap": self.truncated_by_total_cap,
        }


def retrieve_admission(
    units: list[EvidenceUnit], encoders: Encoders, retrieval_cfg: dict[str, Any]
) -> AdmissionRetrieval:
    """Run all section queries for one admission and assemble the evidence list."""
    r = retrieval_cfg
    retriever = AdmissionRetriever(units, encoders, r["rrf_k"])
    pdx = primary_dx(units)
    qmap = section_queries(r["queries"], pdx)
    results: dict[str, list[QueryResult]] = {}
    sections: dict[str, list[str]] = {}
    for s in SECTIONS:
        results[s] = [retriever.search(q) for q in qmap[s]]
        tops = [q.final()[: r["top_k_per_query"]] for q in results[s]]
        sections[s] = interleave(tops, r["max_units_per_section"])
    evidence: list[str] = []
    seen: set[str] = set()
    dropped: list[str] = []
    for s in SECTIONS:
        for d in sections[s]:
            if d in seen:
                continue
            seen.add(d)
            if len(evidence) < r["max_units_total"]:
                evidence.append(d)
            else:
                dropped.append(d)
    return AdmissionRetrieval(
        hadm_id=retriever.hadm_id, primary_dx=pdx, queries=results, sections=sections,
        evidence=evidence, truncated_by_total_cap=dropped,
    )
