"""Perturbation target selection (SPEC Phase 6) and the numeric matcher.

Numeric matching (as implemented): dates (``2150-03-01``, optionally with a time) and
clock times (``08:00``) are removed first. A number is ``-?\\d+(\\.\\d+)?`` (or a
comma-grouped integer such as ``1,200``) that is not preceded by a letter, digit or
``.`` and not followed by a digit or ``.<digit>``. Numbers compare by exact decimal
value, so 5.9 == 5.90, but 5.9 never matches inside 15.9 or 5.95. A leading ``-``
counts as a sign only when it is not attached to a preceding word or number (in
``3.3-5.1`` the second number is 5.1).

Candidates, from the ORIGINAL ehr_rag reports of one generation run: every
(sentence, cited unit, number) where the sentence's trailing citations include a lab
unit (numeric labs only) or a medication unit (numeric dose only) and the sentence
contains a number equal to one of that unit's values: lab points first/max/min/last,
medication dose_val_rx. A lab candidate is NOT eligible when the number equals points
from different measurements (different times): the sentence could refer to either.
Eligible candidates are de-duplicated by (hadm_id, unit_id, value), keeping the first
sentence in report order. Targets: shuffle eligible candidates with
``random.Random(perturbation.seed)`` and take them in order, at most
``max_targets_per_admission`` per admission, up to ``n_targets``.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from ehr_rag.eval.parse import ParsedReport

LAB_POINTS: tuple[str, ...] = ("first", "max", "min", "last")
_DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)")
_NUMBER_RE = re.compile(r"(?<![\w.])(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?!\d|\.\d|,\d{3})")


def to_decimal(value: Any) -> Decimal | None:
    """Decimal of a numeric string, or None."""
    try:
        d = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def numbers_in(text: str) -> list[Decimal]:
    """All standalone numbers in ``text`` (dates/times removed)."""
    cleaned = _DATETIME_RE.sub(" ", text)
    out = []
    for m in _NUMBER_RE.finditer(cleaned):
        tok = m.group(1)
        start = m.start(1)
        if tok.startswith("-") and start > 0 and not (cleaned[start - 1].isspace() or cleaned[start - 1] in "(["):
            tok = tok[1:]
        d = to_decimal(tok)
        if d is not None:
            out.append(d)
    return out


def mentions_value(text: str, value: str) -> bool:
    """True if ``text`` states the number ``value`` (formatting-tolerant, never inside larger numbers)."""
    v = to_decimal(value)
    return v is not None and any(n == v for n in numbers_in(text))


@dataclass
class Candidate:
    """One (sentence, unit, value) pairing."""

    hadm_id: int
    section: str
    sentence_index: int
    sentence: str
    sentence_line: str
    unit_id: str
    unit_type: str
    label: str
    value: str
    fields_matched: list[str]
    eligible: bool
    reason: str = ""
    candidate_id: str = ""
    target_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fields_matched"] = " ".join(self.fields_matched)
        return d


def unit_label(unit_type: str, fields: dict[str, Any]) -> str:
    """Lab label or drug name."""
    return str(fields["label"] if unit_type == "lab" else fields["drug"])


def candidates_for_report(
    hadm_id: int, parsed: ParsedReport, units: dict[str, Any], shown_ids: set[str]
) -> list[Candidate]:
    """All candidates of one report. ``units``: unit_id -> EvidenceUnit (current unit files)."""
    out: list[Candidate] = []
    for s in parsed.sentences():
        if s.is_abstention:
            continue
        nums = numbers_in(s.text)
        if not nums:
            continue
        for uid in s.evidence_ids:
            if uid not in shown_ids:
                continue
            u = units[uid]
            f = u.fields
            if u.unit_type == "lab" and f.get("numeric"):
                values: dict[Decimal, list[str]] = {}
                for name in LAB_POINTS:
                    d = to_decimal(f[name]["value"])
                    if d is not None:
                        values.setdefault(d, []).append(name)
            elif u.unit_type == "medication" and f.get("dose_numeric"):
                d = to_decimal(f["dose_val_rx"])
                values = {d: ["dose_val_rx"]} if d is not None else {}
            else:
                continue
            seen: set[Decimal] = set()
            for n in nums:
                if n not in values or n in seen:
                    continue
                seen.add(n)
                names = values[n]
                if u.unit_type == "lab":
                    raw = f[names[0]]["value"]
                    times = {f[p]["time"] for p in names}
                    eligible = len(times) == 1
                    reason = "" if eligible else "value matches different measurements (ambiguous)"
                else:
                    raw, eligible, reason = f["dose_val_rx"], True, ""
                out.append(Candidate(hadm_id, s.section, s.index, s.text, s.raw.strip(), uid, u.unit_type,
                                     unit_label(u.unit_type, f), raw, names, eligible, reason))
    return out


def select_targets(
    candidates: list[Candidate], n_targets: int, max_per_admission: int, seed: int
) -> list[Candidate]:
    """De-duplicate eligible candidates and pick targets with a fixed seed."""
    for i, c in enumerate(candidates, 1):
        c.candidate_id = f"C{i:04d}"
    seen: set[tuple[int, str, Decimal]] = set()
    pool: list[Candidate] = []
    for c in candidates:
        key = (c.hadm_id, c.unit_id, to_decimal(c.value))
        if c.eligible and key not in seen:
            seen.add(key)
            pool.append(c)
    order = list(pool)
    random.Random(seed).shuffle(order)
    per: dict[int, int] = {}
    chosen: list[Candidate] = []
    for c in order:
        if len(chosen) >= n_targets:
            break
        if per.get(c.hadm_id, 0) >= max_per_admission:
            continue
        per[c.hadm_id] = per.get(c.hadm_id, 0) + 1
        chosen.append(c)
    chosen.sort(key=lambda c: (c.hadm_id, c.candidate_id))
    for i, c in enumerate(chosen, 1):
        c.target_id = f"T{i:02d}"
    return chosen

