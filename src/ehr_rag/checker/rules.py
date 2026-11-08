"""Deterministic citation checker: each sentence is tested against ONLY the units it cites.

Rules and vocabulary come from two frozen files (``checker.rules`` and ``checker.lexicon``
in the config); their SHA-256 hashes are recorded in every run (``checker_hashes``).

Unit of checking: every non-abstention line of the four sections. A list line (citation
groups inside it, see ``parse.split_claims``) is checked per item, and every item after
the first is prefixed with the list's opening words (the first item's text before its
first number or name: ``Labs showed elevated Na 150 [E1], K 5.9 [E2]`` gives the items
``Labs showed elevated Na 150`` [E1] and ``Labs showed elevated K 5.9`` [E2]).

Rules, per item (claim):
  invalid_id          a cited id is not in the evidence shown to the model for this admission
  number_mismatch     a clinical number is not a value in any cited unit. Clinical numbers
                      are ``numbers_in`` (ISO dates and clock times removed) after masking the
                      configured ignore patterns (ages, other dates, section/list numbers,
                      durations, ICD versions). A number matches a unit value if equal, or if
                      the value rounded half-up to the stated decimals equals it.
  name_missing        a named lab, drug, diagnosis or lab-derived term does not appear in any
                      cited unit. Names are found with the admission's own unit vocabulary
                      (lab labels, drug names, diagnosis/procedure titles and their simple
                      variants) plus the lexicon's synonyms/abbreviations; a name is supported
                      if a cited unit's name (lab label, drug, title) contains any synonym, or
                      for a lab, if a cited lab unit belongs to the same lab group. A term
                      (hyperkalemia, ...) is supported by a cited lab of its lab group or a
                      cited diagnosis/history/procedure title containing one of its forms.
  direction_mismatch  a direction word (elevated/low/normal ...) within ``window_tokens`` of a
                      lab name, or a lab-derived term, disagrees with the cited lab's markers
                      ([HIGH]/[LOW] from ``render.lab_marker``). If the item states a number
                      equal to one of the lab's points, that point decides; otherwise high/low
                      hold if any shown point is HIGH/LOW, normal if every cited unit's last
                      point is unmarked. A negation cue before the claim flips it ("no
                      leukocytosis" holds if no WBC point is HIGH). Labs without a reference
                      range are not checkable for direction. A lab name supported only by a
                      non-lab unit (e.g. the drug Potassium Chloride) with a direction word fails.
  trend_mismatch      a trend word (rising/falling/improving/worsening/trending) next to a lab
                      name needs >= 2 cited measurements of that lab at different times (the
                      shown first/max/min/last points of all cited units of that lab); the
                      latest minus the earliest value must have the word's sign
                      (improving/worsening use ``worse_direction``; unknown labs and
                      undirected words need only the two measurements).
  too_many_citations  more than ``max_citations`` ids on one claim

A line FAILS if any item has an error, PASSES if no errors and some item has checkable
content (a clinical number, a name, a direction or trend claim), and is UNCHECKED
otherwise ("no checkable content"; reported separately to measure coverage).
"""

from __future__ import annotations

import bisect
import re
from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import yaml

from ehr_rag.config import resolve_path
from ehr_rag.eval.parse import ParsedReport, Sentence, split_claims
from ehr_rag.evidence.render import effective_ref, lab_marker
from ehr_rag.evidence.schema import EvidenceUnit
from ehr_rag.perturb.select_targets import _DATETIME_RE, LAB_POINTS, numbers_in, to_decimal

ERROR_TYPES: tuple[str, ...] = (
    "invalid_id", "number_mismatch", "name_missing", "direction_mismatch", "trend_mismatch", "too_many_citations",
)
STATUSES: tuple[str, ...] = ("pass", "fail", "unchecked")
_KIND_PRIORITY = {"term": 0, "lab": 1, "drug": 2, "dx": 3}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")
_DIGIT_RE = re.compile(r"\d")


class CheckerError(ValueError):
    """Invalid checker rules or lexicon."""


# --------------------------------------------------------------------------- config files and hashes


def checker_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Resolved paths of the rules file and the lexicon."""
    c = cfg["checker"]
    return {"rules": resolve_path(cfg, c["rules"]), "lexicon": resolve_path(cfg, c["lexicon"])}


def checker_hashes(cfg: dict[str, Any]) -> dict[str, Any]:
    """Hashes of the rules, lexicon and rewrite prompt (``runs.checker_hashes``)."""
    from ehr_rag.runs import checker_hashes as file_hashes

    return file_hashes(cfg) or {}


# --------------------------------------------------------------------------- text helpers


def norm(text: str) -> str:
    """Lowercase alphanumeric tokens (decimals and a trailing % kept) joined by single spaces."""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _contains(haystack_norm: str, needle_norm: str) -> bool:
    return bool(needle_norm) and f" {needle_norm} " in f" {haystack_norm} "


def form_pattern(form: str, plural: bool = False) -> re.Pattern[str]:
    """Regex for a name form in a sentence: tokens separated by any punctuation/space, word-bounded.

    Forms of at most 4 characters (``+`` ignored) containing a capital letter are case-sensitive.
    """
    raw = form.strip()
    case_sensitive = len(raw.replace("+", "")) <= 4 and any(c.isupper() for c in raw)
    toks = re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?%?", raw)
    body = r"[\W_]+".join(re.escape(t) for t in toks)
    if case_sensitive:      # short abbreviations never touch a '/': the K in "K/uL" is a unit, not potassium
        return re.compile(rf"(?<![A-Za-z0-9/]){body}(?![A-Za-z0-9/])")
    if plural:
        body += "(?:s|es)?"
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


def _phrase_regex(words: list[str]) -> re.Pattern[str]:
    bad = [w for w in words if not isinstance(w, str)]
    if bad:
        raise CheckerError(f"word lists must contain strings only, got {bad} (quote yes/no/on/off in YAML)")
    alts = sorted({w.strip().lower() for w in words if w.strip()}, key=len, reverse=True)
    body = "|".join(r"[\s-]+".join(re.escape(p) for p in re.split(r"[\s-]+", a)) for a in alts)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", re.IGNORECASE)


@dataclass
class Mention:
    """A name found in an item."""

    start: int
    end: int
    surface: str
    kind: str                       # lab | drug | dx | term
    key: str                        # lab group / concept, or term name
    forms: frozenset[str]           # normalized synonyms used for support
    term: dict[str, Any] | None = None


@dataclass
class _Entry:
    pattern: re.Pattern[str]
    kind: str
    key: str
    forms: frozenset[str]
    term: dict[str, Any] | None = None


# --------------------------------------------------------------------------- results


@dataclass
class ItemCheck:
    """One claim of a line."""

    text: str
    ids: list[str]
    n_numbers: int = 0
    n_names: int = 0
    n_directions: int = 0
    n_trends: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def checkable(self) -> bool:
        return bool(self.n_numbers or self.n_names or self.n_directions or self.n_trends)


@dataclass
class LineCheck:
    """Checker result for one line."""

    section: str
    index: int
    line: str                       # raw line as generated
    text: str                       # sentence text without citations
    citations: list[str]
    items: list[ItemCheck]

    @property
    def errors(self) -> list[dict[str, Any]]:
        return [{**e, "item": k} for k, it in enumerate(self.items) for e in it.errors]

    @property
    def error_types(self) -> list[str]:
        return sorted({e["type"] for e in self.errors}, key=ERROR_TYPES.index)

    @property
    def checkable(self) -> bool:
        return any(it.checkable for it in self.items)

    @property
    def status(self) -> str:
        return "fail" if self.errors else "pass" if self.checkable else "unchecked"

    @property
    def n_numbers(self) -> int:
        return sum(it.n_numbers for it in self.items)

    @property
    def n_names(self) -> int:
        return sum(it.n_names for it in self.items)

    def messages(self) -> list[str]:
        """Human-readable error messages (used in the rewrite prompt)."""
        return [f"{e['type']}: {e['message']}" for e in self.errors]

    def to_dict(self) -> dict[str, Any]:
        return {"section": self.section, "index": self.index, "line": self.line, "text": self.text,
                "citations": self.citations, "status": self.status, "checkable": self.checkable,
                "error_types": self.error_types, "errors": self.errors, "n_numbers": self.n_numbers,
                "n_names": self.n_names, "n_items": len(self.items), "items": [asdict(i) for i in self.items]}


# --------------------------------------------------------------------------- the checker


class Checker:
    """Deterministic checker built from the rules and lexicon mappings (see module docstring)."""

    def __init__(self, rules: dict[str, Any], lexicon: dict[str, Any]) -> None:
        try:
            self.max_citations = int(rules["max_citations"])
            num = rules["number"]
            self.ignore = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in num["ignore_patterns"].values()]
            self.allow_rounding = bool(num["allow_rounding"])
            d, n, t = rules["direction"], rules["negation"], rules["trend"]
            self.dir_window, self.neg_window, self.trend_window = (int(d["window_tokens"]), int(n["window_tokens"]),
                                                                   int(t["window_tokens"]))
            self.dir_words = {k: _phrase_regex(d[k]) for k in ("high", "low", "normal")}
            self.neg_re = _phrase_regex(n["cues"])
            self.trend_words = {k: _phrase_regex(t[k]) for k in ("up", "down", "improving", "worsening", "undirected")}
            self.lab_groups: dict[str, list[str]] = {k: list(v) for k, v in lexicon["labs"].items()}
            self.dx_groups: dict[str, list[str]] = {k: list(v) for k, v in lexicon["diagnoses"].items()}
            self.drug_groups: dict[str, list[str]] = {k: list(v) for k, v in lexicon["drugs"].items()}
            self.terms: dict[str, dict[str, Any]] = dict(lexicon["terms"])
            self.worse: dict[str, str] = dict(lexicon["worse_direction"])
            self.stopwords = {norm(w) for w in lexicon["name_stopwords"]}
        except (KeyError, TypeError, re.error) as e:
            raise CheckerError(f"invalid checker rules or lexicon: {e!r}") from e
        for name, t in self.terms.items():
            bad = [g for g in t["labs"] if g not in self.lab_groups]
            if bad or t["direction"] not in ("high", "low"):
                raise CheckerError(f"lexicon term {name!r}: unknown lab group(s) {bad} or direction {t['direction']!r}")
        bad = [g for g in self.worse if g not in self.lab_groups]
        if bad:
            raise CheckerError(f"worse_direction: unknown lab group(s) {bad}")
        self._static = self._static_entries()

    @classmethod
    def from_files(cls, rules_path: Path, lexicon_path: Path) -> "Checker":
        """Load from the two YAML files."""
        return cls(yaml.safe_load(rules_path.read_text(encoding="utf-8")),
                   yaml.safe_load(lexicon_path.read_text(encoding="utf-8")))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Checker":
        """Load the files named in ``cfg['checker']``."""
        p = checker_paths(cfg)
        return cls.from_files(p["rules"], p["lexicon"])

    # ---------------------------------------------------------------- vocabulary

    def _static_entries(self) -> list[_Entry]:
        out: list[_Entry] = []
        for kind, groups in (("lab", self.lab_groups), ("dx", self.dx_groups), ("drug", self.drug_groups)):
            for key, forms in groups.items():
                fs = frozenset(norm(f) for f in forms)
                out.extend(_Entry(form_pattern(f, plural=True), kind, key, fs) for f in forms)
        for key, t in self.terms.items():
            fs = frozenset(norm(f) for f in t["forms"])
            out.extend(_Entry(form_pattern(f), "term", key, fs, t) for f in t["forms"])
        return out

    def lab_group(self, label: str) -> str | None:
        """Lab group of a unit label: the group with the longest form contained in the label."""
        lab = norm(label)
        best, best_len = None, 0
        for key, forms in self.lab_groups.items():
            for f in forms:
                nf = norm(f)
                if len(nf) > best_len and _contains(lab, nf):
                    best, best_len = key, len(nf)
        return best

    def lab_concept(self, u: EvidenceUnit) -> str:
        """Lab group of a lab unit, or ``lab:<label before comma>`` if it has none."""
        label = str(u.fields.get("label") or "")
        return self.lab_group(label) or "lab:" + norm(re.sub(r"\([^)]*\)", " ", label).split(",")[0])

    def _variants(self, raw: str, first_word: bool) -> list[str]:
        """Simple surface variants of a unit name (raw case kept for abbreviations in parentheses)."""
        base = re.sub(r"\([^)]*\)", " ", raw)
        out = [raw, base, base.split(",")[0]]
        out += [p for p in re.findall(r"\(([^)]*)\)", raw) if not _DIGIT_RE.search(p)]
        cleaned = re.sub(r"\b(?:unspecified|not elsewhere classified|nos|other specified)\b", " ", base,
                         flags=re.IGNORECASE)
        out += [cleaned, cleaned.split(",")[0]]
        if first_word:
            w = raw.split()[0] if raw.split() else ""
            if re.fullmatch(r"[A-Za-z][A-Za-z-]{4,}", w):
                out.append(w)
        keep = []
        for v in out:
            nv = norm(v)
            short_abbrev = len(nv) >= 2 and any(c.isupper() for c in v.strip())
            if nv and nv not in self.stopwords and (len(nv) >= 4 or short_abbrev) and v.strip() not in keep:
                keep.append(v.strip())
        return keep

    def unit_name(self, u: EvidenceUnit) -> str:
        """The unit's name used for name support: lab label, drug name or title (normalized)."""
        f = u.fields
        if u.unit_type == "lab":
            return norm(str(f.get("label") or ""))
        if u.unit_type == "medication":
            return norm(str(f.get("drug") or ""))
        return norm(str(f.get("long_title") or ""))

    def admission(self, shown: list[EvidenceUnit], all_units: list[EvidenceUnit]) -> "AdmissionContext":
        """Per-admission context: shown units (valid ids) and the vocabulary built from all its units."""
        entries = list(self._static)
        for u in all_units:
            f = u.fields
            if u.unit_type == "lab":
                key = self.lab_concept(u)
                group_forms = {norm(x) for x in self.lab_groups.get(key, [])}
                for v in self._variants(str(f.get("label") or ""), first_word=False):
                    entries.append(_Entry(form_pattern(v, plural=True), "lab", key, frozenset({norm(v)} | group_forms)))
            elif u.unit_type == "medication":
                for v in self._variants(str(f.get("drug") or ""), first_word=True):
                    entries.append(_Entry(form_pattern(v), "drug", "drug:" + norm(v), frozenset({norm(v)})))
            else:
                for v in self._variants(str(f.get("long_title") or ""), first_word=False):
                    if len(norm(v)) >= 6:
                        entries.append(_Entry(form_pattern(v), "dx", "dx:" + norm(v), frozenset({norm(v)})))
        seen: set[tuple[str, int, str, str]] = set()
        uniq = []
        for e in entries:
            k = (e.pattern.pattern, e.pattern.flags, e.kind, e.key)
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        return AdmissionContext(self, {u.unit_id: u for u in shown}, uniq)

    # ---------------------------------------------------------------- checking

    def check_report(self, parsed: ParsedReport, ctx: "AdmissionContext") -> list[LineCheck]:
        """Every non-abstention line of the four sections."""
        return [self.check_line(s, ctx) for s in parsed.sentences() if not s.is_abstention]

    def items(self, s: Sentence, ctx: "AdmissionContext") -> list[tuple[str, list[str]]]:
        """The claims of one line as (text, evidence ids); list items get the list's opening words."""
        claims = split_claims(s)
        if len(claims) > 1:
            first = claims[0][0]
            lead = first[: self._content_start(first, ctx)].strip()
            claims = [claims[0]] + [((f"{lead} {t}" if lead else t), ids) for t, ids in claims[1:]]
        return [(t, [i for i in ids if i != "NONE"]) for t, ids in claims]

    def check_line(self, s: Sentence, ctx: "AdmissionContext") -> LineCheck:
        """Check one parsed line (split into items for list sentences)."""
        items = [self._check_item(t, ids, ctx) for t, ids in self.items(s, ctx)]
        return LineCheck(s.section, s.index, s.raw.strip(), s.text, list(s.citations), items)

    def _content_start(self, text: str, ctx: "AdmissionContext") -> int:
        starts = [m.start for m in ctx.mentions(text)]
        d = _DIGIT_RE.search(self.mask(text))
        if d:
            starts.append(d.start())
        return min(starts) if starts else 0

    def mask(self, text: str) -> str:
        """``text`` with ISO dates/times, then every ignore-pattern match, blanked out (same length)."""
        text = _DATETIME_RE.sub(lambda m: " " * len(m.group(0)), text)
        for p in self.ignore:
            text = p.sub(lambda m: " " * len(m.group(0)), text)
        return text

    def clinical_numbers(self, text: str) -> list[Decimal]:
        """Numbers that must match a cited unit (see module docstring)."""
        return numbers_in(self.mask(text))

    def number_matches(self, stated: Decimal, values: list[Decimal]) -> bool:
        """Equal, or a unit value rounded half-up to the stated decimals."""
        exp = stated.as_tuple().exponent
        for v in values:
            if v == stated:
                return True
            if self.allow_rounding and isinstance(exp, int) and v.as_tuple().exponent < exp:
                if v.quantize(Decimal(1).scaleb(exp), rounding=ROUND_HALF_UP) == stated:
                    return True
        return False

    def _check_item(self, text: str, ids: list[str], ctx: "AdmissionContext") -> ItemCheck:
        it = ItemCheck(text=text, ids=list(ids))
        cited = [ctx.shown[i] for i in ids if i in ctx.shown]
        for i in ids:
            if i not in ctx.shown:
                it.errors.append({"type": "invalid_id", "message": f"[{i}] is not in the evidence for this admission"})
        if len(ids) > self.max_citations:
            it.errors.append({"type": "too_many_citations",
                              "message": f"{len(ids)} citations on one claim (at most {self.max_citations})"})
        # numbers
        values = [n for u in cited for n in numbers_in(u.text)]
        stated = self.clinical_numbers(text)
        it.n_numbers = len(stated)
        for n in stated:
            if not self.number_matches(n, values):
                where = " ".join(f"[{u.unit_id}]" for u in cited) or "no cited unit"
                it.errors.append({"type": "number_mismatch", "message": f"{n} is not a value in {where}"})
        # names
        mentions = ctx.mentions(text)
        it.n_names = len(mentions)
        supported: dict[int, str] = {}       # mention index -> "lab" | "other"
        for k, m in enumerate(mentions):
            how = ctx.support(m, cited)
            if how is None:
                it.errors.append({"type": "name_missing",
                                  "message": f"'{m.surface}' does not appear in the cited unit(s) "
                                             f"{' '.join('[' + u.unit_id + ']' for u in cited) or '(none)'}"})
            else:
                supported[k] = how
        # directions and trends
        self._check_directions(text, mentions, supported, cited, stated, it, ctx)
        self._check_trends(text, mentions, supported, cited, it, ctx)
        return it

    # ---------------------------------------------------------------- direction / trend helpers

    @staticmethod
    def _token_index(text: str) -> tuple[list[int], Any]:
        starts = [m.start() for m in re.finditer(r"\S+", text)]
        return starts, lambda pos: max(bisect.bisect_right(starts, pos) - 1, 0)

    def _negated(self, text: str, anchors: list[int], tok: Any) -> bool:
        """A negation cue at most ``negation.window_tokens`` tokens before any anchor (the word or the name)."""
        cues = [tok(m.start()) for m in self.neg_re.finditer(text)]
        return any(a - self.neg_window <= c < a for c in cues for a in anchors)

    def _nearest_lab(self, pos_tok: int, labs: list[tuple[int, Mention]], tok: Any, window: int) -> tuple[int, Mention] | None:
        best = None
        for k, m in labs:
            d = abs(tok(m.start) - pos_tok)
            if d <= window and (best is None or d < best[0] or (d == best[0] and m.start > best[2].start)):
                best = (d, k, m)
        return (best[1], best[2]) if best else None

    def _lab_units(self, concepts: set[str], fluid: str | None, cited: list[EvidenceUnit],
                   ctx: "AdmissionContext") -> list[EvidenceUnit]:
        return [u for u in cited if u.unit_type == "lab" and u.fields.get("numeric")
                and ctx.concept(u) in concepts and (fluid is None or u.fields.get("fluid") == fluid)]

    def _points(self, u: EvidenceUnit) -> list[tuple[str | None, Decimal, str]]:
        f = u.fields
        out, seen = [], set()
        for name in LAB_POINTS:
            p = f.get(name)
            if not p:
                continue
            key = (p.get("time"), p.get("value"))
            if key in seen:
                continue
            seen.add(key)
            v = to_decimal(p.get("value"))
            if v is not None:
                out.append((p.get("time"), v, lab_marker(p["value"], p.get("flag"), f.get("ref_low"), f.get("ref_high"))))
        return out

    def _direction_holds(self, units: list[EvidenceUnit], direction: str, stated: list[Decimal]) -> bool | None:
        """True / False, or None if not checkable (no unit with a reference range)."""
        usable = [u for u in units if effective_ref(u.fields.get("ref_low"), u.fields.get("ref_high")) != (None, None)]
        if not usable:
            return None
        pts = [(u, p) for u in usable for p in self._points(u)]
        matched = [p for _, p in pts if stated and self.number_matches_any(p[1], stated)]
        if direction in ("high", "low"):
            use = matched or [p for _, p in pts]
            return any(p[2] == f"[{direction.upper()}]" for p in use)
        if matched:
            return all(p[2] == "" for p in matched)
        return all(lab_marker(u.fields["last"]["value"], u.fields["last"].get("flag"), u.fields.get("ref_low"),
                              u.fields.get("ref_high")) == "" for u in usable)

    def number_matches_any(self, value: Decimal, stated: list[Decimal]) -> bool:
        """True if any stated number matches this unit value (equal or rounded)."""
        return any(self.number_matches(s, [value]) for s in stated)

    def direction_claims(self, text: str, mentions: list[Mention]) -> list[tuple[int, Mention, str, bool, str]]:
        """Direction claims in ``text``: (mention index, mention, high|low|normal, negated, word).

        A direction word is attached to the nearest lab name within ``direction.window_tokens``;
        every lab-derived term is a claim of its own direction.
        """
        _, tok = self._token_index(text)
        labs = [(k, m) for k, m in enumerate(mentions) if m.kind == "lab"]
        spans = [(m.start, m.end) for m in mentions]
        claims: list[tuple[int, Mention, str, bool, str]] = []
        for direction, rx in self.dir_words.items():
            for w in rx.finditer(text):
                if any(a <= w.start() < b for a, b in spans):
                    continue
                near = self._nearest_lab(tok(w.start()), labs, tok, self.dir_window)
                if near:
                    k, m = near
                    anchors = [tok(w.start()), tok(m.start)]
                    claims.append((k, m, direction, self._negated(text, anchors, tok), w.group(0)))
        for k, m in enumerate(mentions):
            if m.kind == "term":
                claims.append((k, m, m.term["direction"], self._negated(text, [tok(m.start)], tok), m.surface))
        return claims

    def direction_holds(self, units: list[EvidenceUnit], direction: str, stated: list[Decimal]) -> bool | None:
        """Whether ``direction`` (high/low/normal) holds for these lab units; None if not checkable."""
        return self._direction_holds(units, direction, stated)

    def _check_directions(self, text: str, mentions: list[Mention], supported: dict[int, str],
                          cited: list[EvidenceUnit], stated: list[Decimal], it: ItemCheck,
                          ctx: "AdmissionContext") -> None:
        claims = [c for c in self.direction_claims(text, mentions)
                  if c[1].kind != "term" or supported.get(c[0]) == "lab"]
        for k, m, direction, negated, word in claims:
            if k not in supported:
                continue                                  # name_missing already reported
            if m.kind == "term":
                units = self._lab_units(set(m.term["labs"]), m.term.get("fluid"), cited, ctx)
            else:
                units = self._lab_units({m.key}, None, cited, ctx)
            if not units:
                it.n_directions += 1
                it.errors.append({"type": "direction_mismatch",
                                  "message": f"'{word}' about {m.surface} needs a cited {m.surface} lab unit"})
                continue
            holds = self._direction_holds(units, direction, stated)
            if holds is None:
                continue
            it.n_directions += 1
            if holds == negated:
                ids = " ".join(f"[{u.unit_id}]" for u in units)
                claim = f"{'not ' if negated else ''}{direction}"
                it.errors.append({"type": "direction_mismatch",
                                  "message": f"'{word}' ({claim}) for {m.surface} disagrees with the markers of {ids}"})

    def _check_trends(self, text: str, mentions: list[Mention], supported: dict[int, str],
                      cited: list[EvidenceUnit], it: ItemCheck, ctx: "AdmissionContext") -> None:
        _, tok = self._token_index(text)
        labs = [(k, m) for k, m in enumerate(mentions) if m.kind in ("lab", "term")]
        spans = [(m.start, m.end) for m in mentions]
        for kind, rx in self.trend_words.items():
            for w in rx.finditer(text):
                if any(a <= w.start() < b for a, b in spans):
                    continue
                near = self._nearest_lab(tok(w.start()), labs, tok, self.trend_window)
                if not near or near[0] not in supported:
                    continue
                k, m = near
                concepts = set(m.term["labs"]) if m.kind == "term" else {m.key}
                units = self._lab_units(concepts, m.term.get("fluid") if m.term else None, cited, ctx)
                it.n_trends += 1
                pts = sorted({(p[0], p[1]) for u in units for p in self._points(u) if p[0]})
                if len({t for t, _ in pts}) < 2:
                    it.errors.append({"type": "trend_mismatch",
                                      "message": f"'{w.group(0)}' for {m.surface} needs at least two cited "
                                                 "measurements at different times"})
                    continue
                delta = pts[-1][1] - pts[0][1]
                sign = "up" if delta > 0 else "down" if delta < 0 else "flat"
                want = None
                if kind in ("up", "down"):
                    want = kind
                elif kind in ("improving", "worsening") and len(concepts) == 1:
                    worse = self.worse.get(next(iter(concepts)))
                    if worse:
                        want = worse if kind == "worsening" else ("down" if worse == "up" else "up")
                if want is None:
                    continue
                negated = self._negated(text, [tok(w.start()), tok(m.start)], tok)
                if (sign == want) == negated:
                    it.errors.append({"type": "trend_mismatch",
                                      "message": f"'{w.group(0)}' for {m.surface}: cited values go {sign} "
                                                 f"({pts[0][1]} -> {pts[-1][1]})"})


class AdmissionContext:
    """Shown units and the name vocabulary of one admission."""

    def __init__(self, checker: Checker, shown: dict[str, EvidenceUnit], entries: list[_Entry]) -> None:
        self.checker = checker
        self.shown = shown
        self.entries = entries
        self._concepts: dict[int, str] = {}

    def concept(self, u: EvidenceUnit) -> str:
        """Lab concept of a lab unit (memoized)."""
        k = id(u)
        if k not in self._concepts:
            self._concepts[k] = self.checker.lab_concept(u)
        return self._concepts[k]

    def mentions(self, text: str) -> list[Mention]:
        """Non-overlapping names in ``text``: longest first, then term > lab > drug > diagnosis."""
        cands = []
        for e in self.entries:
            for m in e.pattern.finditer(text):
                cands.append((-(m.end() - m.start()), _KIND_PRIORITY[e.kind], m.start(), m.end(), m.group(0), e))
        cands.sort(key=lambda c: c[:3])
        taken: list[Mention] = []
        for _, _, a, b, surface, e in cands:
            if all(b <= t.start or a >= t.end for t in taken):
                taken.append(Mention(a, b, surface, e.kind, e.key, e.forms, e.term))
        return sorted(taken, key=lambda m: m.start)

    def support(self, m: Mention, cited: list[EvidenceUnit]) -> str | None:
        """``"lab"`` if a cited lab unit supports the name, ``"other"`` if a cited unit's name contains it, else None."""
        ck = self.checker
        if m.kind == "term":
            if ck._lab_units(set(m.term["labs"]), m.term.get("fluid"), cited, self):
                return "lab"
            return "other" if any(u.unit_type != "lab" and any(_contains(ck.unit_name(u), f) for f in m.forms)
                                  for u in cited) else None
        if m.kind == "lab" and any(u.unit_type == "lab" and self.concept(u) == m.key for u in cited):
            return "lab"
        return "other" if any(_contains(ck.unit_name(u), f) for u in cited for f in m.forms) else None
