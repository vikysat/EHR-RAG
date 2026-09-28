"""Citation metrics (SPEC Phase 5), as implemented.

Unit of evaluation: every sentence line in the four sections of a parsed report,
EXCEPT abstention lines (``Insufficient evidence. [NONE]``), which are counted
separately (``n_abstentions``). Only trailing citations count (see ``parse.py``).
Citation ids are de-duplicated within a sentence. ``[NONE]`` is not a citation.

  valid citation   an E-id present in the report's ``evidence_shown.json``
  cited_rate       sentences with >= 1 valid citation / sentences
  invalid_citation_rate
                   E-id citations not in evidence_shown / all E-id citations
  citation_recall  (ALCE, Gao et al. 2023) sentences whose valid cited units, concatenated
                   in citation order (joined by a single space), entail the sentence
                   under NLI (premise = units, hypothesis = sentence) / sentences.
                   Sentences without a valid citation are unsupported.
  citation_precision
                   (ALCE) relevant citations / all E-id citations. A valid citation c of a
                   sentence with valid set C is IRRELEVANT iff (a) NLI(c alone) is not
                   entailment AND (b) entail(C without c) == entail(C), where
                   entail(empty set) = False. Invalid (fabricated) ids are irrelevant.
Aggregates are micro-averages (pooled over sentences / citations) per condition,
per section, and for the groups History+Findings and Impression+Recommendations.
A rate with a zero denominator is reported as null.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ehr_rag.config import SECTIONS
from ehr_rag.eval.nli import NLI
from ehr_rag.eval.parse import ParsedReport, Sentence

GROUPS: dict[str, tuple[str, ...]] = {
    "History+Findings": ("History", "Findings"),
    "Impression+Recommendations": ("Impression", "Recommendations"),
    "ALL": SECTIONS,
}
NO_VALID_CITATION = "no_valid_citation"


@dataclass
class SentenceEval:
    """Evaluation of one sentence."""

    condition: str
    hadm_id: int
    section: str
    index: int
    sentence: str
    cited_ids: list[str]          # valid ids, citation order
    invalid_ids: list[str]
    cites_none: bool
    is_abstention: bool
    flags: list[str]
    nli_label: str = NO_VALID_CITATION   # joint label, or no_valid_citation
    supported: bool = False
    irrelevant_ids: list[str] = field(default_factory=list)

    @property
    def n_citations(self) -> int:
        return len(self.cited_ids) + len(self.invalid_ids)

    @property
    def n_relevant(self) -> int:
        return len(self.cited_ids) - len(self.irrelevant_ids)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["n_citations"] = self.n_citations
        return d


def _premise(ids: Iterable[str], texts: dict[str, str]) -> str:
    return " ".join(texts[i] for i in ids)


def _pairs_for(s: SentenceEval, texts: dict[str, str]) -> list[tuple[str, str]]:
    """Every (premise, hypothesis) pair needed for recall and precision of ``s``."""
    ids = s.cited_ids
    if not ids:
        return []
    pairs = [(_premise(ids, texts), s.sentence)]
    for c in ids:
        pairs.append((texts[c], s.sentence))
        rest = [i for i in ids if i != c]
        if rest:
            pairs.append((_premise(rest, texts), s.sentence))
    return pairs


def evaluate_report(
    parsed: ParsedReport, evidence: list[dict[str, Any]], condition: str, hadm_id: int, nli: NLI
) -> list[SentenceEval]:
    """Sentence-level evaluation of one report against its ``evidence_shown`` units."""
    texts = {u["unit_id"]: u["text"] for u in evidence}
    out: list[SentenceEval] = []
    for s in parsed.sentences():
        ev = [i for i in s.evidence_ids]
        out.append(SentenceEval(
            condition=condition, hadm_id=hadm_id, section=s.section, index=s.index, sentence=s.text,
            cited_ids=[i for i in ev if i in texts], invalid_ids=[i for i in ev if i not in texts],
            cites_none="NONE" in s.citations, is_abstention=s.is_abstention, flags=list(s.flags),
        ))
    scored = [e for e in out if not e.is_abstention and e.cited_ids]
    needed = [p for e in scored for p in _pairs_for(e, texts)]
    labels = dict(zip(needed, nli.labels(needed))) if needed else {}
    ent = nli.entailment_label

    def entails(ids: list[str], hyp: str) -> bool:
        return bool(ids) and labels[(_premise(ids, texts), hyp)] == ent

    for e in scored:
        e.nli_label = labels[(_premise(e.cited_ids, texts), e.sentence)]
        e.supported = e.nli_label == ent
        for c in e.cited_ids:
            rest = [i for i in e.cited_ids if i != c]
            if not entails([c], e.sentence) and entails(rest, e.sentence) == e.supported:
                e.irrelevant_ids.append(c)
    return out


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def aggregate(sents: list[SentenceEval]) -> dict[str, Any]:
    """Micro-averaged metrics over a set of sentence evaluations."""
    scored = [s for s in sents if not s.is_abstention]
    n = len(scored)
    n_cit = sum(s.n_citations for s in scored)
    n_invalid = sum(len(s.invalid_ids) for s in scored)
    n_relevant = sum(s.n_relevant for s in scored)
    return {
        "n_sentences": n,
        "n_abstentions": len(sents) - n,
        "n_citations": n_cit,
        "n_invalid_citations": n_invalid,
        "cited_rate": _rate(sum(1 for s in scored if s.cited_ids), n),
        "invalid_citation_rate": _rate(n_invalid, n_cit),
        "citation_recall": _rate(sum(1 for s in scored if s.supported), n),
        "citation_precision": _rate(n_relevant, n_cit),
        "n_mid_sentence_citation": sum(1 for s in sents if "mid_sentence_citation" in s.flags),
        "n_multiple_sentences": sum(1 for s in sents if "multiple_sentences" in s.flags),
    }


def metrics_table(sents: list[SentenceEval], conditions: list[str]) -> list[dict[str, Any]]:
    """Rows (condition, scope) for every section, group and ALL."""
    rows = []
    for c in conditions:
        cs = [s for s in sents if s.condition == c]
        for scope in (*SECTIONS, *GROUPS):
            secs = GROUPS.get(scope, (scope,))
            rows.append({"condition": c, "scope": scope, **aggregate([s for s in cs if s.section in secs])})
    return rows
