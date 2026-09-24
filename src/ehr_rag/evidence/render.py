"""Pure rendering of evidence-unit text from structured fields.

``render(unit_type, fields)`` is the ONLY way unit text is produced. It has no
side effects and depends only on its arguments. Abnormality markers are
computed here from the values and reference range in ``fields``; they are
never stored in fields or text elsewhere.

Marker rule for a numeric lab value: if the unit has at least one reference
bound, ``[HIGH]`` when value > ref_high, ``[LOW]`` when value < ref_low. If the
unit has no reference bound, ``[ABNORMAL]`` when that measurement's ``flag``
field is ``"abnormal"`` (MIMIC's flag carries no direction). Otherwise no marker.
"""

from __future__ import annotations

from typing import Any, Callable


def _fmt_time(iso: str | None) -> str:
    """'2150-03-01T08:00:00' -> '2150-03-01 08:00'; date-only stays a date."""
    if not iso:
        return "unknown time"
    s = iso.replace("T", " ")
    return s[:16] if len(s) >= 16 else s[:10]


def _fmt_date(iso: str | None) -> str:
    return iso[:10] if iso else "unknown date"


def lab_marker(value: str, flag: str | None, ref_low: str | None, ref_high: str | None) -> str:
    """Marker for one lab value: '[HIGH]', '[LOW]', '[ABNORMAL]' or ''."""
    if ref_low is not None or ref_high is not None:
        v = float(value)
        if ref_high is not None and v > float(ref_high):
            return "[HIGH]"
        if ref_low is not None and v < float(ref_low):
            return "[LOW]"
        return ""
    if flag is not None and flag.strip().lower() == "abnormal":
        return "[ABNORMAL]"
    return ""


def _ref_text(lo: str | None, hi: str | None) -> str:
    if lo is not None and hi is not None:
        return f"ref {lo}–{hi}"
    if lo is not None:
        return f"ref ≥{lo}"
    if hi is not None:
        return f"ref ≤{hi}"
    return ""


def _lab_head(f: dict[str, Any]) -> str:
    head = f["label"]
    if f.get("fluid"):
        head += f" ({f['fluid']})"
    parts = [head]
    if f.get("uom"):
        parts.append(f["uom"])
    ref = _ref_text(f.get("ref_low"), f.get("ref_high"))
    if ref:
        parts.append(ref)
    return ", ".join(parts)


def _lab_point(name: str, p: dict[str, Any], f: dict[str, Any]) -> str:
    marker = lab_marker(p["value"], p.get("flag"), f.get("ref_low"), f.get("ref_high"))
    val = f"{p['value']} {marker}" if marker else p["value"]
    return f"{name} {val} ({_fmt_time(p['time'])})"


def render_lab(f: dict[str, Any]) -> str:
    """Render a lab unit."""
    head = _lab_head(f)
    if not f["numeric"]:
        last = f["last"]
        n = f["n_measurements"]
        noun = "result" if n == 1 else "results"
        marker = "[ABNORMAL]" if (last.get("flag") or "").strip().lower() == "abnormal" else ""
        val = f"{last['value']} {marker}" if marker else last["value"]
        return f"{head}: {n} non-numeric {noun}. Last {val} ({_fmt_time(last['time'])})."
    n = f["n_measurements"]
    if n == 1:
        return f"{head}: 1 measurement. {_lab_point('Value', f['first'], f)}."
    points = "; ".join([
        _lab_point("First", f["first"], f),
        _lab_point("max", f["max"], f),
        _lab_point("min", f["min"], f),
        _lab_point("last", f["last"], f),
    ])
    return f"{head}: {n} measurements. {points}."


def render_medication(f: dict[str, Any]) -> str:
    """Render a medication unit."""
    parts = [f["drug"]]
    if f.get("dose_val_rx") is not None:
        dose = f["dose_val_rx"]
        if f.get("dose_unit_rx") is not None:
            dose += f" {f['dose_unit_rx']}"
        parts.append(dose)
    elif f.get("dose_unit_rx") is not None:
        parts.append(f"(dose unit {f['dose_unit_rx']}, no dose value)")
    if f.get("route") is not None:
        parts.append(f["route"])
    n = f["n_orders"]
    ordered = "ordered once" if n == 1 else f"ordered {n} times"
    start, stop = f.get("start"), f.get("stop")
    if start and stop:
        span = f"{_fmt_date(start)} to {_fmt_date(stop)}"
    elif start:
        span = f"from {_fmt_date(start)}"
    elif stop:
        span = f"until {_fmt_date(stop)}"
    else:
        span = "dates unknown"
    return f"{' '.join(parts)}, {ordered}, {span}."


def _icd(f: dict[str, Any]) -> str:
    return f"ICD-{f['icd_version']} {f['icd_code']}"


def render_diagnosis(f: dict[str, Any]) -> str:
    """Render a current-admission diagnosis unit."""
    return f"Diagnosis (priority {f['seq_num']} of {f['n_total']}): {f['long_title']} ({_icd(f)})."


def render_procedure(f: dict[str, Any]) -> str:
    """Render a current-admission procedure unit."""
    return (
        f"Procedure (priority {f['seq_num']} of {f['n_total']}, {_fmt_date(f.get('chartdate'))}): "
        f"{f['long_title']} ({_icd(f)})."
    )


def render_history_diagnosis(f: dict[str, Any]) -> str:
    """Render a diagnosis from a prior admission."""
    return f"Prior admission ({_fmt_date(f['prior_admittime'])}): {f['long_title']} ({_icd(f)})."


_RENDERERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "lab": render_lab,
    "medication": render_medication,
    "diagnosis": render_diagnosis,
    "procedure": render_procedure,
    "history_diagnosis": render_history_diagnosis,
}


def render(unit_type: str, fields: dict[str, Any]) -> str:
    """Render unit text from ``fields``. Pure: no side effects, no hidden state."""
    try:
        fn = _RENDERERS[unit_type]
    except KeyError:
        raise ValueError(f"unknown unit_type {unit_type!r}") from None
    return fn(fields)
