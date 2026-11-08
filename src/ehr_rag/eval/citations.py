"""Citation metrics (SPEC Phase 5), as implemented.

Unit of evaluation: every CLAIM of every sentence line in the four sections of a
parsed report, EXCEPT abstention lines (``Insufficient evidence. [NONE]``), which are
counted separately (``n_abstentions``). A line is one claim unless it has citation
groups inside it (a list such as ``Na 130 [E1], K 5.9 [E2], and Cr 2.1 [E3].``),
which ``parse.split_claims`` splits into one claim per item, each with its own ids.
Citation ids are de-duplicated within a claim. ``[NONE]`` is not a citation.

  valid citation   an E-id present in the report's ``evidence_shown.json``
  premise          the texts of a claim's valid cited units only, in citation order,
                   joined by a single space (never other evidence)
  entail(S)        NLI(premise of S, claim) is the entailment label; entail(empty) = False.
                   A pair longer than the NLI model's limit gets ``too_long`` and is
                   therefore NOT entailed (``nli.TOO_LONG``); such pairs are logged.
  cited_rate       claims with >= 1 valid citation / claims
  invalid_citation_rate
                   E-id citations not in evidence_shown / all E-id citations
  citation_recall  (ALCE, Gao et al. 2023) claims with entail(valid ids) / claims.
                   Claims without a valid citation are unsupported.
  citation_precision
                   (ALCE) relevant citations / all E-id citations. A valid citation c of a
                   claim with valid set C is IRRELEVANT iff (a) NOT entail({c}) AND
                   (b) entail(C without c) == entail(C). Invalid (fabricated) ids are irrelevant.
  citation_precision_plain
                   citations that support their claim on their own, entail({c}), / all
                   E-id citations. Invalid ids never support.
Aggregates are micro-averages (pooled over claims / citations) per condition, per
section, and for the groups History+Findings and Impression+Recommendations. A rate
with a zero denominator is reported as null. ``n_sentences`` counts lines,
``n_claims`` the scored claims (the rates' denominator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ehr_rag.config import SECTIONS
from ehr_rag.eval.nli import NLI, TOO_LONG
from ehr_rag.eval.parse import ParsedReport, split_claims

GROUPS: dict[str, tuple[str, ...]] = {
    "History+Findings": ("History", "Findings"),
    "Impression+Recommendations": ("Impression", "Recommendations"),
    "ALL": SECTIONS,
}
NO_VALID_CITATION = "no_valid_citation"


@dataclass
class SentenceEval:
    """Evaluation of one claim (a whole line unless the line was split)."""

    condition: str
    hadm_id: int
    section: str
    index: int                    # line position within the section
    sentence: str                 # claim text (the NLI hypothesis)
    cited_ids: list[str]          # valid ids, citation order
    invalid_ids: list[str]
    cites_none: bool
    is_abstention: bool
    flags: list[str]
    claim_index: int = 0
    n_claims: int = 1             # claims of this line
    line: str = ""                # the whole line's text (citations removed)
    nli_label: str = NO_VALID_CITATION   # joint label, or no_valid_citation
    supported: bool = False
    irrelevant_ids: list[str] = field(default_factory=list)
    supporting_ids: list[str] = field(default_factory=list)   # entail({c}) alone
    too_long: list[list[str]] = field(default_factory=list)   # premises (id lists) labeled too_long

    @property
    def n_citations(self) -> int:
        return len(self.cited_ids) + len(self.invalid_ids)

    @property
    def n_relevant(self) -> int:
        return len(self.cited_ids) - len(self.irrelevant_ids)

    @property
    def line_key(self) -> tuple[str, int, str, int]:
        return (self.condition, self.hadm_id, self.section, self.index)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["n_citations"] = self.n_citations
        return d


def premise_text(ids: Iterable[str], texts: dict[str, str]) -> str:
    """NLI premise: the given units' texts in order, joined by a space."""
    return " ".join(texts[i] for i in ids)


def premise_sets(ids: list[str]) -> list[list[str]]:
    """Every premise needed for recall and both precisions: all ids, each id alone, all but each id."""
    out: list[list[str]] = [list(ids)] if ids else []
    for c in ids:
        rest = [i for i in ids if i != c]
        for p in ([c], rest):
            if p and p not in out:
                out.append(p)
    return out


def evaluate_report(
    parsed: ParsedReport, evidence: list[dict[str, Any]], condition: str, hadm_id: int, nli: NLI
) -> list[SentenceEval]:
    """Claim-level evaluation of one report against its ``evidence_shown`` units."""
    texts = {u["unit_id"]: u["text"] for u in evidence}
    out: list[SentenceEval] = []
    for s in parsed.sentences():
        claims = split_claims(s)
        for k, (text, ids) in enumerate(claims):
            ev = [i for i in ids if i != "NONE"]
            flags = list(s.flags) + (["split_claim"] if len(claims) > 1 else [])
            out.append(SentenceEval(
                condition=condition, hadm_id=hadm_id, section=s.section, index=s.index, sentence=text,
                cited_ids=[i for i in ev if i in texts], invalid_ids=[i for i in ev if i not in texts],
                cites_none="NONE" in ids, is_abstention=s.is_abstention, flags=flags,
                claim_index=k, n_claims=len(claims), line=s.text,
            ))
    scored = [e for e in out if not e.is_abstention and e.cited_ids]
    needed = list(dict.fromkeys((premise_text(p, texts), e.sentence)
                                for e in scored for p in premise_sets(e.cited_ids)))
    labels = dict(zip(needed, nli.labels(needed))) if needed else {}
    ent = nli.entailment_label

    def label(ids: list[str], hyp: str) -> str:
        return labels[(premise_text(ids, texts), hyp)]

    def entails(ids: list[str], hyp: str) -> bool:
        return bool(ids) and label(ids, hyp) == ent

    for e in scored:
        e.nli_label = label(e.cited_ids, e.sentence)
        e.supported = e.nli_label == ent
        e.too_long = [p for p in premise_sets(e.cited_ids) if label(p, e.sentence) == TOO_LONG]
        for c in e.cited_ids:
            alone = entails([c], e.sentence)
            if alone:
                e.supporting_ids.append(c)
            rest = [i for i in e.cited_ids if i != c]
            if not alone and entails(rest, e.sentence) == e.supported:
                e.irrelevant_ids.append(c)
    return out


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def aggregate(sents: list[SentenceEval]) -> dict[str, Any]:
    """Micro-averaged metrics over a set of claim evaluations."""
    scored = [s for s in sents if not s.is_abstention]
    n = len(scored)
    n_cit = sum(s.n_citations for s in scored)
    n_invalid = sum(len(s.invalid_ids) for s in scored)
    lines = {s.line_key: s for s in sents}

    def n_lines(flag: str) -> int:
        return sum(1 for s in lines.values() if flag in s.flags)

    return {
        "n_sentences": len({s.line_key for s in scored}),
        "n_claims": n,
        "n_abstentions": len({s.line_key for s in sents if s.is_abstention}),
        "n_citations": n_cit,
        "n_invalid_citations": n_invalid,
        "cited_rate": _rate(sum(1 for s in scored if s.cited_ids), n),
        "invalid_citation_rate": _rate(n_invalid, n_cit),
        "citation_recall": _rate(sum(1 for s in scored if s.supported), n),
        "citation_precision": _rate(sum(s.n_relevant for s in scored), n_cit),
        "citation_precision_plain": _rate(sum(len(s.supporting_ids) for s in scored), n_cit),
        "n_split_sentences": n_lines("split_claim"),
        "n_too_long_claims": sum(1 for s in scored if s.too_long),
        "n_mid_sentence_citation": n_lines("mid_sentence_citation"),
        "n_multiple_sentences": n_lines("multiple_sentences"),
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
