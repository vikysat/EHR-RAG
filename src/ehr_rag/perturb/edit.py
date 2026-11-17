"""Perturbation edits (SPEC Phase 6). Every edit changes ``fields`` and re-renders with
``render``; unit text is never edited directly. Unit ids are never renumbered.

As implemented:
  lab_minor  The target measurement (all of first/max/min/last that are that one
             measurement) changes by 10-20 %. Candidate changes (+/-10 %, +/-11 %, ...
             +/-20 %) are tried in an order shuffled with the target's seed. The first one
             that, after rounding to the original decimal places, differs from the
             original, is still 10-20 % away, keeps the same marker ([HIGH]/[LOW]/none;
             for labs without a reference range the flag-based marker, which a value
             change cannot alter), and keeps the unit consistent, is used.
  lab_major  Needs a reference range. Normal -> 1.5 x ref_high if there is an upper
             bound, else 0.7 x ref_low. Abnormal ([HIGH]/[LOW]) -> middle of the normal
             range (needs both bounds). The marker must actually change. ``n_abnormal`` is
             adjusted by the change of the edited measurement's status.
  med_dose   dose_val_rx is DOUBLED (halving could need more decimal places than the
             original; doubling never does), keeping the original decimal places.
  delete     The unit is removed from the evidence block; nothing else changes.

Consistency: after a lab edit, ``max`` must be >= and ``min`` <= every shown point
(first, max, min, last). Otherwise the edit is skipped (``SkipEdit``) and logged.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any

from ehr_rag.evidence.render import effective_ref, lab_marker, render
from ehr_rag.evidence.schema import EvidenceUnit
from ehr_rag.perturb.select_targets import LAB_POINTS, to_decimal

EDITS_FOR_TYPE: dict[str, tuple[str, ...]] = {
    "lab": ("lab_minor", "lab_major", "delete"),
    "medication": ("med_dose", "delete"),
}


class SkipEdit(Exception):
    """This edit cannot be applied to this target (reason in the message)."""


@dataclass
class EditResult:
    """Everything logged for one edit."""

    edit_type: str
    unit_id: str
    original_fields: dict[str, Any]
    edited_fields: dict[str, Any] | None   # None for delete
    original_text: str
    edited_text: str | None                # None for delete
    original_value: str
    edited_value: str | None               # None for delete

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def decimal_places(value: str) -> int:
    """Number of digits after the decimal point in the original string."""
    s = str(value).strip()
    return len(s.split(".", 1)[1]) if "." in s else 0


def fmt(x: Decimal, places: int) -> str:
    """Format with exactly ``places`` decimals (half-up rounding)."""
    q = Decimal(1).scaleb(-places)
    return str(x.quantize(q, rounding=ROUND_HALF_UP))


def _marker(fields: dict[str, Any], point: dict[str, Any], value: str) -> str:
    return lab_marker(value, point.get("flag"), fields.get("ref_low"), fields.get("ref_high"))


def _consistent(fields: dict[str, Any]) -> bool:
    vals = {p: Decimal(fields[p]["value"]) for p in LAB_POINTS}
    return vals["max"] == max(vals.values()) and vals["min"] == min(vals.values())


def _with_value(fields: dict[str, Any], names: list[str], new: str) -> dict[str, Any]:
    f = copy.deepcopy(fields)
    for p in names:
        f[p]["value"] = new
    return f


def lab_minor(fields: dict[str, Any], names: list[str], seed: str) -> dict[str, Any]:
    """Edited fields for lab_minor (see module docstring)."""
    point = fields[names[0]]
    orig = point["value"]
    v = Decimal(orig)
    if v == 0:
        raise SkipEdit("original value is 0; a relative change is undefined")
    places = decimal_places(orig)
    m0 = _marker(fields, point, orig)
    options = [(sign, pct) for pct in range(10, 21) for sign in (1, -1)]
    random.Random(seed).shuffle(options)
    for sign, pct in options:
        new = fmt(v * (1 + Decimal(sign * pct) / 100), places)
        dv = Decimal(new)
        if dv == v:
            continue
        rel = abs(dv - v) / abs(v)
        if not (Decimal("0.10") <= rel <= Decimal("0.20")):
            continue
        if _marker(fields, point, new) != m0:
            continue
        f = _with_value(fields, names, new)
        if _consistent(f):
            return f
    raise SkipEdit("no 10-20% change keeps the same status and a consistent unit at the original precision")


def lab_major(fields: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Edited fields for lab_major (see module docstring)."""
    point = fields[names[0]]
    orig = point["value"]
    places = decimal_places(orig)
    lo, hi = effective_ref(fields.get("ref_low"), fields.get("ref_high"))
    if lo is None and hi is None:
        raise SkipEdit("no reference range; normal/abnormal cannot be moved")
    m0 = _marker(fields, point, orig)
    if m0 == "":
        if hi is not None:
            new = fmt(Decimal(hi) * Decimal("1.5"), places)
        else:
            new = fmt(Decimal(lo) * Decimal("0.7"), places)
    else:
        if lo is None or hi is None:
            raise SkipEdit("abnormal value but the reference range has only one bound; no normal middle")
        new = fmt((Decimal(lo) + Decimal(hi)) / 2, places)
    m1 = _marker(fields, point, new)
    if m1 == m0:
        raise SkipEdit(f"edited value {new} does not change the status ({m0 or 'normal'})")
    f = _with_value(fields, names, new)
    if not _consistent(f):
        raise SkipEdit(f"edited value {new} makes the unit inconsistent (max/min no longer extreme)")
    if "n_abnormal" in f:
        f["n_abnormal"] = int(f["n_abnormal"]) + (1 if m1 else 0) - (1 if m0 else 0)
    return f


def med_dose(fields: dict[str, Any]) -> dict[str, Any]:
    """Edited fields for med_dose: dose doubled at the original precision."""
    if not fields.get("dose_numeric"):
        raise SkipEdit("dose_val_rx is not a single number")
    orig = fields["dose_val_rx"]
    f = copy.deepcopy(fields)
    f["dose_val_rx"] = fmt(Decimal(orig) * 2, decimal_places(orig))
    return f


def apply_edit(unit: EvidenceUnit, edit_type: str, value: str, point_names: list[str], seed: str) -> EditResult:
    """Apply one edit type to ``unit`` for a target stating ``value``. Raises SkipEdit."""
    if edit_type not in EDITS_FOR_TYPE.get(unit.unit_type, ()):
        raise SkipEdit(f"{edit_type} does not apply to a {unit.unit_type} unit")
    if unit.text != render(unit.unit_type, unit.fields):
        raise SkipEdit("unit text does not equal render(fields); refusing to edit")
    if edit_type == "delete":
        return EditResult(edit_type, unit.unit_id, copy.deepcopy(unit.fields), None, unit.text, None, value, None)
    if edit_type == "lab_minor":
        f = lab_minor(unit.fields, point_names, seed)
        new = f[point_names[0]]["value"]
    elif edit_type == "lab_major":
        f = lab_major(unit.fields, point_names)
        new = f[point_names[0]]["value"]
    else:
        f = med_dose(unit.fields)
        new = f["dose_val_rx"]
    if to_decimal(new) == to_decimal(value):
        raise SkipEdit("edited value equals the original")
    return EditResult(edit_type, unit.unit_id, copy.deepcopy(unit.fields), f, unit.text,
                      render(unit.unit_type, f), value, new)


# --------------------------------------------------------------------------- counterfactual audit edits
#
# ``audit_edit`` moves EVERY shown measurement of a lab unit (first/max/min/last) to one value:
#   to_normal  unit with a [HIGH]/[LOW] point -> midpoint of the reference range; flags cleared,
#              n_abnormal 0
#   to_high    normal unit -> ref_high + audit.outside_fraction * (ref_high - ref_low); flags set to
#   to_low     "abnormal", n_abnormal = n_measurements. to_low (ref_low - fraction * width) for the
#              lab groups in audit.below_range_groups; skipped if that is not > 0
# The value keeps the unit's original precision (most decimals among its points). The edit is
# skipped if, after rounding, the markers are not all '' (to_normal) or all [HIGH]/[LOW].


Bounds = tuple["Decimal | None", "Decimal | None"]


def physiologic_bounds(table: dict[str, Any], group: str | None, uom: str | None) -> Bounds:
    """(low, high) plausibility bounds of a lab group from ``probe.physiologic_bounds``.

    An entry is ``[low, high]`` or a mapping ``uom -> [low, high]`` (labs reported in several units).
    The low bound is never below 0 (no negative values); a group or uom without an entry gets (0, None).
    """
    e = table.get(group) if group else None
    if isinstance(e, dict):
        e = e.get(uom or "")
    if not e:
        return Decimal(0), None
    lo, hi = (None if x is None else Decimal(str(x)) for x in e)
    return max(lo if lo is not None else Decimal(0), Decimal(0)), hi


def clamp(x: Decimal, bounds: Bounds | None) -> Decimal:
    """``x`` moved into ``bounds`` (None: unchanged)."""
    if bounds is None:
        return x
    lo, hi = bounds
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x


def audit_edit(unit: EvidenceUnit, group: str | None, outside_fraction: float,
               below_groups: list[str], bounds: Bounds | None = None) -> EditResult:
    """Audit edit of one lab unit (see above). Raises SkipEdit.

    ``bounds`` (probe only; the audit passes None): the new value is clamped into these physiologic
    bounds before rounding; the edit is skipped if the clamped value no longer gives the wanted marker.
    """
    from ehr_rag.perturb.select_targets import lab_status, range_skip_reason

    reason = range_skip_reason(unit)
    if reason:
        raise SkipEdit(reason)
    if unit.text != render(unit.unit_type, unit.fields):
        raise SkipEdit("unit text does not equal render(fields); refusing to edit")
    f0 = unit.fields
    points = [p for p in LAB_POINTS if f0.get(p)]
    places = max(decimal_places(f0[p]["value"]) for p in points)
    lo, hi = (Decimal(x) for x in effective_ref(f0["ref_low"], f0["ref_high"]))
    width = hi - lo
    if lab_status(f0) != "normal":
        edit_type, new, flag, want = "to_normal", (lo + hi) / 2, None, ""
    elif group in below_groups:
        edit_type, new, flag, want = "to_low", lo - Decimal(str(outside_fraction)) * width, "abnormal", "[LOW]"
        if new <= 0:
            raise SkipEdit(f"below-range value {new} is not positive")
    else:
        edit_type, new, flag, want = "to_high", hi + Decimal(str(outside_fraction)) * width, "abnormal", "[HIGH]"
    new = clamp(new, bounds)
    value = fmt(new, places)
    f = copy.deepcopy(f0)
    for p in points:
        f[p]["value"] = value
        f[p]["flag"] = flag
    f["n_abnormal"] = 0 if edit_type == "to_normal" else int(f.get("n_measurements", len(points)))
    marks = {_marker(f, f[p], value) for p in points}
    if marks != {want}:
        raise SkipEdit(f"{value} at the original precision does not give marker {want or 'none'} ({sorted(marks)})")
    original = " ".join(dict.fromkeys(f0[p]["value"] for p in points))
    return EditResult(edit_type, unit.unit_id, copy.deepcopy(f0), f, unit.text, render("lab", f), original, value)


# --------------------------------------------------------------------------- probe: within-flag edit
#
# ``within_flag_edit`` (citation-dependence probe) multiplies EVERY shown measurement of a lab unit
# by the same factor 1 +/- pct/100, pct in ``probe.within_flag_percent`` [low, high] (whole
# percents), tried in an order shuffled with ``seed``. Each scaled value is rounded to its point's
# original precision and then clamped to stay STRICTLY within its band, the side of the reference
# range it was on: normal -> ref_low < v < ref_high, high -> v > ref_high, low -> 0 < v < ref_low,
# intersected with the physiologic bounds (all at the point's precision). If any point's band holds
# no value at its precision, the edit is skipped. The first factor for which every point changes
# and the unit stays consistent is used. Flags and n_abnormal are unchanged.


def in_band_limits(fields: dict[str, Any], point: str, bounds: Bounds | None) -> tuple[Decimal, Decimal | None] | None:
    """(lowest, highest or None) value at the point's precision strictly inside its band; None if empty."""
    value = fields[point]["value"]
    places = decimal_places(value)
    ulp = Decimal(1).scaleb(-places)
    lo, hi = (Decimal(x) for x in effective_ref(fields["ref_low"], fields["ref_high"]))
    m = _marker(fields, fields[point], value)
    a: Decimal
    b: Decimal | None
    q = ulp
    above_lo = lo.quantize(q, rounding=ROUND_FLOOR) + ulp      # smallest value at this precision > ref_low
    below_hi = hi.quantize(q, rounding=ROUND_CEILING) - ulp    # largest value at this precision < ref_high
    if m == "[HIGH]":
        a, b = hi.quantize(q, rounding=ROUND_FLOOR) + ulp, None
    elif m == "[LOW]":
        a, b = ulp, lo.quantize(q, rounding=ROUND_CEILING) - ulp
    else:
        a, b = above_lo, below_hi
    a = max(a, ulp)                                   # strictly positive
    if bounds is not None:
        if bounds[0] is not None:
            a = max(a, bounds[0].quantize(q, rounding=ROUND_CEILING))
        if bounds[1] is not None:
            top = bounds[1].quantize(q, rounding=ROUND_FLOOR)
            b = top if b is None else min(b, top)
    if b is not None and a > b:
        return None
    return a, b


def within_flag_edit(unit: EvidenceUnit, percent: list[int], seed: str, bounds: Bounds | None = None) -> EditResult:
    """Within-flag edit of one lab unit (see above). Raises SkipEdit."""
    from ehr_rag.perturb.select_targets import range_skip_reason

    reason = range_skip_reason(unit)
    if reason:
        raise SkipEdit(reason)
    if unit.text != render(unit.unit_type, unit.fields):
        raise SkipEdit("unit text does not equal render(fields); refusing to edit")
    f0 = unit.fields
    points = [p for p in LAB_POINTS if f0.get(p)]
    if any(Decimal(f0[p]["value"]) == 0 for p in points):
        raise SkipEdit("a shown value is 0; a relative change is undefined")
    limits = {p: in_band_limits(f0, p, bounds) for p in points}
    empty = [p for p, lim in limits.items() if lim is None]
    if empty:
        raise SkipEdit(f"no valid in-band value for {', '.join(empty)} at its precision")
    lo_pct, hi_pct = int(percent[0]), int(percent[1])
    options = [(sign, pct) for pct in range(lo_pct, hi_pct + 1) for sign in (1, -1)]
    random.Random(seed).shuffle(options)
    for sign, pct in options:
        factor = 1 + Decimal(sign * pct) / 100
        f = copy.deepcopy(f0)
        ok = True
        for p in points:
            v = Decimal(f0[p]["value"])
            places = decimal_places(f0[p]["value"])
            new = clamp(Decimal(fmt(v * factor, places)), limits[p])
            if new == v:
                ok = False
                break
            f[p]["value"] = fmt(new, places)
        if ok and _consistent(f):
            if any(_marker(f, f[p], f[p]["value"]) != _marker(f0, f0[p], f0[p]["value"]) for p in points):
                raise AssertionError("within-flag edit changed a marker")      # the bands make this impossible
            original = " ".join(dict.fromkeys(f0[p]["value"] for p in points))
            edited = " ".join(dict.fromkeys(f[p]["value"] for p in points))
            return EditResult("within_flag", unit.unit_id, copy.deepcopy(f0), f, unit.text, render("lab", f),
                              original, edited)
    raise SkipEdit("no within-flag change moves every value inside its band and keeps the unit consistent")
