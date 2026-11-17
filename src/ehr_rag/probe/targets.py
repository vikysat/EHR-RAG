"""Probe units and their contexts (``probe-targets``, CPU).

Unit: in each plain ehr_rag report, the FIRST direction claim about a lab: a direction word
attached to a lab name ("elevated potassium") or a lab-derived lexicon term ("hyperkalemia"),
found with the citation checker (``Checker.direction_claim_spans`` on each claim's text, as the
checker reads list lines, so a direction word never attaches across a citation). "First" is per checker lab
group: a claim is a unit candidate only if none of its lab groups had a direction claim earlier in
the report (report order). The candidate becomes a unit if it is not negated and the claim's
citation group (the first citation run at or after it, as in ``parse.split_claims``) cites a shown
numeric lab unit of a matching group (fluid included for terms) with a two-sided reference range:
the cited unit. Otherwise it is logged in ``probe_skips.csv`` with the reason.

Number: the first clinical number of the same claim that matches a value of the cited unit
(``Checker.number_matches``); ``order`` is number_first / term_first / no_number.

Contexts (same retrieved set and prompt; only the evidence changes):
  original      the evidence the report was generated with
  cited_edit    the cited unit edited across the reference range (``audit_edit``, as in the audit),
                values clamped to ``probe.physiologic_bounds``
  within_flag   the cited unit's values changed, markers unchanged (``within_flag_edit``)
  matched_null  ``audit_edit`` of a shown lab unit cited NOWHERE in the report whose status
                (high/low/mixed/normal) equals the cited unit's, shuffled with
                ``f"{seed}:{hadm}:{cited unit}:null"``; missing if there is none
  ablation      every corroborator removed (no renumbering): other shown labs of the claim's
                groups, diagnosis/history/procedure units whose title contains a form of a lexicon
                term over those groups with the claim's direction, and medications in the
                APPROXIMATE ``probe.related_drugs`` list; only for corroborated units

Alternative (scored against the original term): consistent with the cited edit. The edit's new
state is normal (to_normal), high or low. Direction word -> the first word of that direction in
``checker.direction`` (elevated / low / normal). Term -> "<normal word> <first form of its first
lab group>" ("normal potassium") for a normal state, else the lexicon term over the same labs with
the new direction (hypokalemia -> hyperkalemia). A unit whose edit leaves the claim true, or whose
term has no alternative, is skipped. The alternative is capitalized only at the start of a line.
"""

from __future__ import annotations

import dataclasses
import random
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

from ehr_rag.eval.parse import CITE_GROUP_RE, CITE_ID_RE, ParsedReport
from ehr_rag.evidence.schema import EvidenceUnit
from ehr_rag.perturb.edit import EditResult, SkipEdit, audit_edit, fmt, physiologic_bounds, within_flag_edit
from ehr_rag.perturb.edit import decimal_places
from ehr_rag.perturb.select_targets import _NUMBER_RE, LAB_POINTS, lab_status, range_skip_reason, to_decimal

CONTEXTS: tuple[str, ...] = ("original", "cited_edit", "within_flag", "matched_null", "ablation")
NEW_STATE = {"to_normal": "normal", "to_high": "high", "to_low": "low"}
DX_TYPES = ("diagnosis", "history_diagnosis", "procedure")
_LEAD_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*")

UNIT_COLUMNS = ["probe_id", "hadm_id", "section", "sentence_index", "line_no", "claim_kind", "surface", "direction",
                "groups", "negated", "cited_unit", "status", "edit_type", "term_start", "term_end", "term_cut",
                "term_alt", "number", "number_start", "number_end", "number_cut", "number_alt_cited",
                "number_alt_within", "order", "corroborators", "corroborated", "null_unit", "contexts"]


@dataclass
class ProbeUnit:
    """One probed claim (character offsets are into the report's final raw text)."""

    hadm_id: int
    section: str
    sentence_index: int
    line_no: int
    claim_kind: str              # term | word
    surface: str
    direction: str               # high | low | normal
    groups: list[str]
    negated: bool
    cited_unit: str
    status: str                  # cited unit: high | low | mixed | normal
    edit_type: str               # to_normal | to_high | to_low
    term_start: int
    term_end: int
    term_cut: int                # teacher forcing stops here (before the whitespace preceding the term)
    term_alt: str
    number: str = ""
    number_start: int = -1
    number_end: int = -1
    number_cut: int = -1
    number_alt_cited: str = ""
    number_alt_within: str = ""
    order: str = "no_number"     # number_first | term_first | no_number
    corroborators: list[str] = field(default_factory=list)
    null_unit: str = ""
    contexts: list[str] = field(default_factory=list)
    probe_id: str = ""

    @property
    def corroborated(self) -> bool:
        return bool(self.corroborators)

    def row(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(groups=" | ".join(self.groups), corroborators=" ".join(self.corroborators),
                 corroborated=self.corroborated, contexts=" ".join(self.contexts))
        return {k: d[k] for k in UNIT_COLUMNS}


def _cut(text: str, start: int, line_start: int) -> int:
    """Where teacher forcing stops for a span starting at ``start``: before the whitespace that precedes
    it on the same line (so a word-initial space stays with the scored continuation)."""
    i = start
    while i > line_start and text[i - 1] in " \t":
        i -= 1
    return i


def _cite_runs(line: str) -> list[tuple[int, int, list[str]]]:
    """(start, end, ids) of adjacent citation groups, as in ``parse.split_claims``."""
    runs: list[list[Any]] = []
    for m in CITE_GROUP_RE.finditer(line):
        ids = CITE_ID_RE.findall(m.group(0))
        if runs and not line[runs[-1][1]: m.start()].strip():
            runs[-1][1] = m.end()
            runs[-1][2].extend(ids)
        else:
            runs.append([m.start(), m.end(), ids])
    return [(a, b, [i for i in ids if i != "NONE"]) for a, b, ids in runs]


def claim_segments(line: str) -> list[tuple[int, int]]:
    """(start, end) of each claim's text: up to each citation run, plus any text after the last one."""
    out, prev = [], 0
    for a, b, _ in _cite_runs(line):
        if line[prev:a].strip():
            out.append((prev, a))
        prev = b
    if line[prev:].strip(" .!?"):
        out.append((prev, len(line)))
    return out


def claim_citation(line: str, pos: int) -> tuple[int, int, list[str]]:
    """(claim segment start, segment end, cited ids) of the claim containing ``pos``."""
    prev = 0
    for a, b, ids in _cite_runs(line):
        if a >= pos:
            return prev, a, ids
        prev = b
    return prev, len(line), []


def _cap(alt: str, at_line_start: bool) -> str:
    return alt[:1].upper() + alt[1:] if at_line_start else alt


def alternative(checker: Any, words: dict[str, str], kind: str, direction: str, term: dict[str, Any] | None,
                edit_type: str) -> str | None:
    """Edit-consistent alternative (module docstring), lowercase; None if there is none."""
    new = NEW_STATE[edit_type]
    if new == direction:
        return None
    if kind == "word":
        return words[new]
    assert term is not None
    if new == "normal":
        return f"{words['normal']} {checker.lab_groups[term['labs'][0]][0]}"
    for t in checker.terms.values():
        if set(t["labs"]) == set(term["labs"]) and t["direction"] == new:
            return str(t["forms"][0])
    return None


def corroborators(checker: Any, ctx: Any, shown: list[EvidenceUnit], cited_id: str, groups: list[str],
                  direction: str, term: dict[str, Any] | None, related_drugs: dict[str, Any]) -> list[str]:
    """Unit ids that could support the claim besides the cited unit (module docstring)."""
    from ehr_rag.checker.rules import _contains, norm

    gs = set(groups)
    if term is not None:
        forms = {norm(f) for f in term["forms"]}
    else:
        forms = {norm(f) for t in checker.terms.values() if set(t["labs"]) & gs and t["direction"] == direction
                 for f in t["forms"]}
    drugs = [d.lower() for g in groups for d in related_drugs.get(g, {}).get(direction, [])]
    out = []
    for u in shown:
        if u.unit_id == cited_id:
            continue
        if u.unit_type == "lab" and ctx.concept(u) in gs:
            out.append(u.unit_id)
        elif u.unit_type in DX_TYPES and any(_contains(checker.unit_name(u), f) for f in forms):
            out.append(u.unit_id)
        elif u.unit_type == "medication" and any(d in str(u.fields.get("drug") or "").lower() for d in drugs):
            out.append(u.unit_id)
    return out


def _bounds(checker: Any, u: EvidenceUnit, table: dict[str, Any]) -> Any:
    return physiologic_bounds(table, checker.lab_group(str(u.fields.get("label") or "")), u.fields.get("uom"))


def _first_number(checker: Any, line: str, seg: tuple[int, int], unit: EvidenceUnit
                  ) -> tuple[int, int, str, list[str]] | None:
    """(start, end, stated text, matched point names) of the claim's first number matching the unit."""
    masked = checker.mask(line)
    pts = {p: to_decimal(unit.fields[p]["value"]) for p in LAB_POINTS if unit.fields.get(p)}
    for m in _NUMBER_RE.finditer(masked, seg[0], seg[1]):
        tok = m.group(1).lstrip("-")
        d = to_decimal(tok)
        if d is None:
            continue
        names = [p for p, v in pts.items() if v is not None and checker.number_matches(d, [v])]
        if names:
            start = m.start(1) + (len(m.group(1)) - len(tok))
            return start, m.end(1), tok, names
    return None


def _stated(value: str, stated: str) -> str:
    """``value`` written with the stated number's decimals."""
    return fmt(Decimal(value), decimal_places(stated))


@dataclass
class ReportProbe:
    units: list[ProbeUnit]
    skips: list[dict[str, Any]]
    specs: dict[tuple[int, str], dict[str, Any]]    # (index in units, context) -> evidence change


def probe_report(hadm_id: int, raw: str, parsed: ParsedReport, shown: list[EvidenceUnit], cited_counts: dict[str, int],
                 checker: Any, ctx: Any, cfg: dict[str, Any], words: dict[str, str]) -> ReportProbe:
    """Units, skips and context specs of one report (module docstring)."""
    pc, ac = cfg["probe"], cfg["audit"]
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))
    shown_by_id = {u.unit_id: u for u in shown}
    seen: set[str] = set()
    out = ReportProbe([], [], {})

    def skip(s: Any, surface: str, reason: str) -> None:
        out.skips.append({"hadm_id": hadm_id, "section": s.section, "sentence_index": s.index, "surface": surface,
                          "reason": reason})

    def edit(u: EvidenceUnit) -> EditResult:
        return audit_edit(u, checker.lab_group(str(u.fields.get("label") or "")), ac["outside_fraction"],
                          ac["below_range_groups"], _bounds(checker, u, pc["physiologic_bounds"]))

    for s in parsed.sentences():
        if s.is_abstention:
            continue
        line = lines[s.line_no - 1].rstrip("\r\n")
        base = offsets[s.line_no - 1]
        claims = []
        for s0, s1 in claim_segments(line):          # per claim, as the checker reads list lines
            seg = line[s0:s1]
            claims += [(*c[:5], c[5] + s0, c[6] + s0) for c in checker.direction_claim_spans(seg, ctx.mentions(seg))]
        claims.sort(key=lambda c: c[5])
        for _, m, direction, negated, word, a, b in claims:
            kind = "term" if m.kind == "term" else "word"
            groups = list(m.term["labs"]) if kind == "term" else [m.key]
            first = not (set(groups) & seen)
            seen.update(groups)
            if not first:
                continue
            if negated:
                skip(s, word, "negated claim")
                continue
            seg_a, seg_b, ids = claim_citation(line, a)
            cited = [shown_by_id[i] for i in ids if i in shown_by_id]
            if kind == "term":
                units = checker._lab_units(set(groups), m.term.get("fluid"), cited, ctx)
            else:
                units = checker._lab_units({m.key}, None, cited, ctx)
            if not units:
                skip(s, word, "claim does not cite a matching lab unit")
                continue
            ranged = [u for u in units if not range_skip_reason(u)]
            if not ranged:
                skip(s, word, "cited lab unit has no two-sided reference range")
                continue
            u = ranged[0]
            try:
                ce = edit(u)
            except SkipEdit as e:
                skip(s, word, f"cited edit: {e}")
                continue
            alt = alternative(checker, words, kind, direction, m.term, ce.edit_type)
            if alt is None:
                skip(s, word, "the cited edit leaves the claim true or the term has no alternative")
                continue
            at_start = _LEAD_RE.match(line).end() == a   # type: ignore[union-attr]
            pu = ProbeUnit(hadm_id, s.section, s.index, s.line_no, kind, word, direction, groups, negated, u.unit_id,
                           lab_status(u.fields), ce.edit_type, base + a, base + b, base + _cut(line, a, 0),
                           _cap(alt, at_start))
            specs: dict[str, dict[str, Any]] = {"original": {}, "cited_edit": {"replace": [ce.to_dict()]}}
            try:
                wf = within_flag_edit(u, pc["within_flag_percent"], f"{pc['seed']}:{hadm_id}:{u.unit_id}:within",
                                      _bounds(checker, u, pc["physiologic_bounds"]))
                specs["within_flag"] = {"replace": [wf.to_dict()]}
            except SkipEdit as e:
                wf = None
                skip(s, word, f"within-flag context missing: {e}")
            num = _first_number(checker, line, (seg_a, seg_b), u)
            if num:
                ns, ne, tok, names = num
                pu.number, pu.number_start, pu.number_end = tok, base + ns, base + ne
                pu.number_cut = base + _cut(line, ns, 0)
                pu.number_alt_cited = _stated(ce.edited_fields[names[0]]["value"], tok)   # type: ignore[index]
                if wf is not None:
                    pu.number_alt_within = _stated(wf.edited_fields[names[0]]["value"], tok)  # type: ignore[index]
                pu.order = "number_first" if ns < a else "term_first"
            nulls = [x for x in shown if x.unit_type == "lab" and x.unit_id not in cited_counts
                     and not range_skip_reason(x) and lab_status(x.fields) == pu.status]
            random.Random(f"{pc['seed']}:{hadm_id}:{u.unit_id}:null").shuffle(nulls)
            for x in nulls:
                try:
                    specs["matched_null"] = {"replace": [edit(x).to_dict()]}
                    pu.null_unit = x.unit_id
                    break
                except SkipEdit:
                    continue
            if not pu.null_unit:
                skip(s, word, "matched-null context missing: no editable uncited lab with the same status")
            pu.corroborators = corroborators(checker, ctx, shown, u.unit_id, groups, direction, m.term,
                                             pc["related_drugs"])
            if pu.corroborators:
                specs["ablation"] = {"remove": list(pu.corroborators)}
            pu.contexts = [c for c in CONTEXTS if c in specs]
            k = len(out.units)
            out.units.append(pu)
            for c, sp in specs.items():
                out.specs[(k, c)] = sp
    return out


def apply_spec(evidence: list[EvidenceUnit], spec: dict[str, Any]) -> list[EvidenceUnit]:
    """The evidence list with a context's edits applied (ids never renumbered)."""
    rep = {e["unit_id"]: e for e in spec.get("replace", [])}
    drop = set(spec.get("remove", []))
    return [dataclasses.replace(u, fields=rep[u.unit_id]["edited_fields"], text=rep[u.unit_id]["edited_text"])
            if u.unit_id in rep else u for u in evidence if u.unit_id not in drop]
