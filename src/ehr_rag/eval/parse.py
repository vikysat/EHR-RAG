"""Report parser (SPEC Phase 5).

As implemented:
  - Sections are split on the exact header lines ``### History``, ``### Findings``,
    ``### Impression``, ``### Recommendations`` (trailing whitespace ignored). Any other
    line starting with ``#`` starts an *extra* section; its lines are kept but are not
    part of the four report sections. Non-blank lines before the first header are
    kept as ``preamble``.
  - Each non-blank line inside a section is one sentence.
  - Only the TRAILING citation group counts as the sentence's citations. It is one or
    more bracket groups at the end of the line: ``[E3]``, ``[E3, E7]``, ``[E3][E7]``,
    ``[NONE]``. Ids are de-duplicated in order of appearance.
  - Citation-like brackets elsewhere in the line are removed from the stored sentence
    text, recorded in ``mid_citations`` and flagged ``mid_sentence_citation``; they
    do NOT count as citations.
  - A line is flagged ``multiple_sentences`` if its text contains a sentence
    boundary: ``.``, ``!`` or ``?`` followed by whitespace and an uppercase letter
    (decimals such as ``5.9`` are not boundaries).
  - ``Insufficient evidence. [NONE]`` (exact text, [NONE] only) is an abstention line.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ehr_rag.config import SECTIONS

_HEADER_LINES = {f"### {s}": s for s in SECTIONS}
_CITE_GROUP = r"\[\s*(?:NONE|E\d+(?:\s*,\s*E\d+)*)\s*\]"
TRAILING_RE = re.compile(rf"(?:\s*{_CITE_GROUP})+\s*$")
CITE_GROUP_RE = re.compile(_CITE_GROUP)
CITE_ID_RE = re.compile(r"E\d+|NONE")
_BOUNDARY_RE = re.compile(r"[.!?]\s+[A-Z]")
ABSTENTION_TEXT = "Insufficient evidence."


@dataclass
class Sentence:
    """One report line."""

    section: str
    index: int                  # position within the section (0-based)
    line_no: int                # 1-based line number in the raw text
    raw: str
    text: str                   # sentence without citation brackets
    citations: list[str]        # trailing group ids, e.g. ["E3", "E7"] or ["NONE"]
    mid_citations: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def evidence_ids(self) -> list[str]:
        """Trailing citation ids other than NONE."""
        return [c for c in self.citations if c != "NONE"]

    @property
    def is_abstention(self) -> bool:
        """``Insufficient evidence. [NONE]``."""
        return self.text == ABSTENTION_TEXT and self.citations == ["NONE"]

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict."""
        return asdict(self)


@dataclass
class ParsedReport:
    """Parsed report with structural flags."""

    sections: dict[str, list[Sentence]]
    missing_sections: list[str]
    duplicate_sections: list[str]
    extra_sections: list[str]
    out_of_order: bool
    preamble: list[str]
    extra_lines: list[str]

    def sentences(self) -> list[Sentence]:
        """All sentences of the four sections, in section order."""
        return [s for sec in SECTIONS for s in self.sections.get(sec, [])]

    def flags(self) -> dict[str, Any]:
        """Report-level flags and counts of sentence-level flags."""
        sents = self.sentences()
        return {
            "missing_sections": self.missing_sections,
            "duplicate_sections": self.duplicate_sections,
            "extra_sections": self.extra_sections,
            "out_of_order": self.out_of_order,
            "n_preamble_lines": len(self.preamble),
            "n_uncited_lines": sum(1 for s in sents if not s.citations),
            "n_mid_sentence_citation": sum(1 for s in sents if "mid_sentence_citation" in s.flags),
            "n_multiple_sentences": sum(1 for s in sents if "multiple_sentences" in s.flags),
        }


def _dedupe(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def parse_line(raw: str, section: str, index: int, line_no: int) -> Sentence:
    """Parse one sentence line."""
    line = raw.strip()
    m = TRAILING_RE.search(line)
    citations = _dedupe(CITE_ID_RE.findall(m.group(0))) if m else []
    body = line[: m.start()] if m else line
    mids = [i for g in CITE_GROUP_RE.findall(body) for i in CITE_ID_RE.findall(g)]
    text = re.sub(r"\s{2,}", " ", CITE_GROUP_RE.sub("", body)).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    flags: list[str] = []
    if not citations:
        flags.append("no_trailing_citation")
    if mids:
        flags.append("mid_sentence_citation")
    if _BOUNDARY_RE.search(text):
        flags.append("multiple_sentences")
    if re.match(r"^(?:[-*•]|\d+[.)])\s+", line):
        flags.append("bullet")
    return Sentence(section, index, line_no, raw, text, citations, _dedupe(mids), flags)


def parse_report(text: str) -> ParsedReport:
    """Parse raw model output into sections and sentences."""
    sections: dict[str, list[Sentence]] = {}
    order: list[str] = []
    duplicates: list[str] = []
    extra: list[str] = []
    preamble: list[str] = []
    extra_lines: list[str] = []
    current: str | None = None
    in_extra = False
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line in _HEADER_LINES:
            name = _HEADER_LINES[line]
            if name in sections:
                duplicates.append(name)
            else:
                sections[name] = []
            order.append(name)
            current, in_extra = name, False
            continue
        if line.lstrip().startswith("#"):
            extra.append(line.strip())
            current, in_extra = None, True
            continue
        if in_extra:
            extra_lines.append(line)
        elif current is None:
            preamble.append(line)
        else:
            sections[current].append(parse_line(line, current, len(sections[current]), i))
    first_seen = [s for k, s in enumerate(order) if s not in order[:k]]
    present = [s for s in SECTIONS if s in sections]
    return ParsedReport(
        sections={s: sections[s] for s in present},
        missing_sections=[s for s in SECTIONS if s not in sections],
        duplicate_sections=duplicates,
        extra_sections=extra,
        out_of_order=first_seen != present,
        preamble=preamble,
        extra_lines=extra_lines,
    )
