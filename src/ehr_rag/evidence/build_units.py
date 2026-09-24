"""Build evidence units and the patient header for target admissions.

Unit construction and ordering follow SPEC §Phase 2. Unit text is produced only
by ``render.render``. Anything that cannot be turned into a unit, or is
truncated/summarized lossy, is recorded in a ``dropped`` log.

Ordering before assigning E1..En:
  type order diagnosis, procedure, lab, medication, history_diagnosis; then
  - diagnosis, procedure: seq_num (priority), then source key;
  - lab, medication, history_diagnosis: timestamp (None last), then source key.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ehr_rag.data.load import Tables
from ehr_rag.evidence.render import lab_marker, render
from ehr_rag.evidence.schema import UNIT_TYPES, EvidenceUnit, write_units

TYPE_ORDER: dict[str, int] = {t: i for i, t in enumerate(UNIT_TYPES)}
DEID_MASK = "___"  # MIMIC de-identification mask


def _iso(ts: Any) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


def _iso_date(ts: Any) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).date().isoformat()


def _s(v: Any) -> str | None:
    """Missing -> None, else the original string."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return str(v)


def _mode(values: list[Any]) -> tuple[Any, int]:
    """Most frequent value (ties -> the latest occurrence) and number of distinct values."""
    c = Counter(values)
    best = max(c.values())
    for v in reversed(values):
        if c[v] == best:
            return v, len(c)
    raise ValueError("empty")  # pragma: no cover


def build_header(tables: Tables, hadm_id: int) -> dict[str, Any]:
    """Non-citable patient/admission header."""
    adm = tables["admissions"]
    a = adm[adm["hadm_id"] == hadm_id]
    if len(a) != 1:
        raise KeyError(f"hadm_id {hadm_id}: expected 1 admission row, found {len(a)}")
    a = a.iloc[0]
    pat = tables["patients"]
    p = pat[pat["subject_id"] == a["subject_id"]]
    if len(p) != 1:
        raise KeyError(f"subject_id {a['subject_id']}: expected 1 patient row, found {len(p)}")
    p = p.iloc[0]
    age = int(p["anchor_age"]) + int(a["admittime"].year) - int(p["anchor_year"])
    los = (a["dischtime"] - a["admittime"]).total_seconds() / 86400
    return {
        "hadm_id": int(hadm_id),
        "subject_id": int(a["subject_id"]),
        "age": age,
        "gender": _s(p["gender"]),
        "admission_type": _s(a["admission_type"]),
        "admission_location": _s(a["admission_location"]),
        "admittime": _iso(a["admittime"]),
        "dischtime": _iso(a["dischtime"]),
        "los_days": round(los, 2),
    }


def _lab_units(tables: Tables, hadm_id: int, drop: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labs = tables["labevents"]
    labs = labs[labs["link_hadm_id"] == hadm_id]
    items = tables["d_labitems"].set_index("itemid")
    out: list[dict[str, Any]] = []
    for itemid, g in labs.groupby("itemid", sort=True):
        itemid = int(itemid)
        g = g.sort_values(["charttime", "labevent_id"])
        if itemid not in items.index or pd.isna(items.loc[itemid, "label"]):
            drop.append({"unit_type": "lab", "reason": "itemid not in d_labitems / no label",
                         "detail": f"itemid={itemid}", "n_rows": len(g)})
            continue
        item = items.loc[itemid]
        num = g[g["valuenum_f"].notna()]
        base = {
            "itemid": itemid,
            "label": _s(item["label"]),
            "fluid": _s(item["fluid"]),
            "category": _s(item["category"]),
            "n_time_window_linked": int((g["linked_by"] == "time_window").sum()),
        }
        if len(num):
            refs = [(_s(r.ref_range_lower), _s(r.ref_range_upper)) for r in num.itertuples()]
            (ref_low, ref_high), n_refs = _mode(refs)
            if n_refs > 1:
                drop.append({"unit_type": "lab", "reason": "reference range varies across rows; most frequent used",
                             "detail": f"itemid={itemid} {base['label']}: {sorted(set(refs), key=str)}",
                             "n_rows": len(num)})
            uoms = [_s(u) for u in num["valueuom"]]
            uom, n_uoms = _mode(uoms)
            if n_uoms > 1:
                drop.append({"unit_type": "lab", "reason": "unit of measure varies across rows; most frequent used",
                             "detail": f"itemid={itemid} {base['label']}: {sorted(set(uoms), key=str)}",
                             "n_rows": len(num)})
            rest = g[g["valuenum_f"].isna()]
            n_empty = int(rest["value"].isna().sum())
            text_vals = rest[rest["value"].notna()]["value"]
            if n_empty:
                drop.append({"unit_type": "lab", "reason": "rows with empty value in a numeric lab (excluded)",
                             "detail": f"itemid={itemid} {base['label']}", "n_rows": n_empty})
            if len(text_vals):
                drop.append({"unit_type": "lab", "reason": "text results in a numeric lab (excluded)",
                             "detail": f"itemid={itemid} {base['label']}: {sorted(set(text_vals))[:5]}",
                             "n_rows": len(text_vals)})

            def point(r: Any) -> dict[str, Any]:
                return {"value": _s(r["valuenum"]), "time": _iso(r["charttime"]), "flag": _s(r["flag"])}

            # min/max ties -> earliest (num is time-sorted; idxmin/idxmax return first occurrence).
            first, last = num.iloc[0], num.iloc[-1]
            mn = num.loc[num["valuenum_f"].idxmin()]
            mx = num.loc[num["valuenum_f"].idxmax()]
            fields = {
                **base, "numeric": True, "uom": uom, "ref_low": ref_low, "ref_high": ref_high,
                "n_measurements": len(num),
                "first": point(first), "max": point(mx), "min": point(mn), "last": point(last),
            }
            fields["n_abnormal"] = sum(
                1 for r in num.itertuples()
                if lab_marker(_s(r.valuenum), _s(r.flag), ref_low, ref_high)
            )
            rows = num
        else:
            g_val = g[g["value"].notna()]
            if not len(g_val):
                drop.append({"unit_type": "lab", "reason": "no value (value and valuenum both null)",
                             "detail": f"itemid={itemid} {base['label']}", "n_rows": len(g)})
                continue
            last = g_val.iloc[-1]
            if str(last["value"]).strip() == DEID_MASK:
                drop.append({"unit_type": "lab", "reason": "text-only lab whose last value is the ___ mask",
                             "detail": f"itemid={itemid} {base['label']}", "n_rows": len(g_val)})
                continue
            uom, _ = _mode([_s(u) for u in g_val["valueuom"]])
            fields = {
                **base, "numeric": False, "uom": uom, "ref_low": None, "ref_high": None,
                "n_measurements": len(g_val),
                "last": {"value": _s(last["value"]), "time": _iso(last["charttime"]), "flag": _s(last["flag"])},
                "n_abnormal": int((g_val["flag"].fillna("").str.lower() == "abnormal").sum()),
            }
            rows = g_val
        out.append({
            "unit_type": "lab", "time_scope": "current", "timestamp": _iso(rows["charttime"].min()),
            "fields": fields, "source_table": "labevents",
            "source_rows": sorted(int(x) for x in rows["labevent_id"]),
            "sort_key": (str(itemid).zfill(10),),
        })
    return out


def _med_units(tables: Tables, hadm_id: int, drop: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rx = tables["prescriptions"]
    rx = rx[rx["hadm_id"] == hadm_id]
    no_drug = rx[rx["drug"].isna()]
    if len(no_drug):
        drop.append({"unit_type": "medication", "reason": "drug name null",
                     "detail": f"pharmacy_id={sorted(int(x) for x in no_drug['pharmacy_id'])[:10]}",
                     "n_rows": len(no_drug)})
    rx = rx[rx["drug"].notna()]
    key_cols = ["drug", "dose_val_rx", "dose_unit_rx", "route"]
    out: list[dict[str, Any]] = []
    for key, g in rx.groupby(key_cols, dropna=False, sort=True):
        drug, dose, unit, route = (_s(k) for k in key)
        dose_numeric = False
        if dose is not None:
            try:
                float(dose)
                dose_numeric = True
            except ValueError:
                pass
        if dose is not None and not dose_numeric:
            drop.append({"unit_type": "medication", "reason": "dose_val_rx not a single number (kept as text; "
                         "not eligible for med_dose perturbation)", "detail": f"{drug}: {dose!r}", "n_rows": len(g)})
        if dose is None:
            drop.append({"unit_type": "medication", "reason": "dose_val_rx null (unit kept without dose)",
                         "detail": f"{drug}", "n_rows": len(g)})
        start = g["starttime"].min()
        stop = g["stoptime"].max()
        fields = {
            "drug": drug, "dose_val_rx": dose, "dose_unit_rx": unit, "route": route,
            "dose_numeric": dose_numeric, "n_orders": len(g),
            "start": _iso(start), "stop": _iso(stop),
        }
        out.append({
            "unit_type": "medication", "time_scope": "current", "timestamp": _iso(start),
            "fields": fields, "source_table": "prescriptions",
            "source_rows": sorted(int(x) for x in g["pharmacy_id"]),
            "sort_key": tuple(k or "" for k in (drug, dose, unit, route)),
        })
    return out


def _icd_units(
    tables: Tables, hadm_id: int, table: str, dict_table: str, unit_type: str, drop: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = tables[table]
    rows = rows[rows["hadm_id"] == hadm_id].sort_values("seq_num")
    titles = tables[dict_table].set_index(["icd_code", "icd_version"])["long_title"]
    admittime = tables["admissions"].set_index("hadm_id").loc[hadm_id, "admittime"]
    n_total = len(rows)
    out: list[dict[str, Any]] = []
    for r in rows.itertuples():
        k = (r.icd_code, int(r.icd_version))
        if k not in titles.index:
            drop.append({"unit_type": unit_type, "reason": f"ICD code not in {dict_table}",
                         "detail": f"ICD-{k[1]} {k[0]} seq_num={r.seq_num}", "n_rows": 1})
            continue
        fields: dict[str, Any] = {
            "seq_num": int(r.seq_num), "n_total": n_total, "icd_code": r.icd_code,
            "icd_version": int(r.icd_version), "long_title": titles.loc[k],
        }
        if unit_type == "procedure":
            fields["chartdate"] = _iso_date(r.chartdate)
            ts = fields["chartdate"]
        else:
            ts = _iso(admittime)
        out.append({
            "unit_type": unit_type, "time_scope": "current", "timestamp": ts, "fields": fields,
            "source_table": table, "source_rows": [{"hadm_id": int(hadm_id), "seq_num": int(r.seq_num)}],
            "sort_key": (f"{int(r.seq_num):05d}", r.icd_code),
        })
    return out


def _history_units(
    tables: Tables, hadm_id: int, units_cfg: dict[str, Any], drop: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    adm = tables["admissions"]
    target = adm[adm["hadm_id"] == hadm_id].iloc[0]
    prior = adm[(adm["subject_id"] == target["subject_id"]) & (adm["dischtime"] < target["admittime"])]
    prior = prior.sort_values(["dischtime", "hadm_id"], ascending=[False, False])
    n_prior = len(prior)
    keep = prior.head(units_cfg["max_prior_admissions"])
    if n_prior > len(keep):
        drop.append({"unit_type": "history_diagnosis", "reason": "older prior admissions beyond max_prior_admissions",
                     "detail": f"hadm_ids={sorted(int(h) for h in prior.iloc[len(keep):]['hadm_id'])}",
                     "n_rows": n_prior - len(keep)})
    dx = tables["diagnoses_icd"]
    titles = tables["d_icd_diagnoses"].set_index(["icd_code", "icd_version"])["long_title"]
    cap = units_cfg["max_history_dx_per_admission"]
    out: list[dict[str, Any]] = []
    for p in keep.itertuples():
        rows = dx[dx["hadm_id"] == p.hadm_id].sort_values("seq_num")
        if len(rows) > cap:
            drop.append({"unit_type": "history_diagnosis", "reason": "diagnoses beyond max_history_dx_per_admission",
                         "detail": f"prior hadm_id={p.hadm_id}", "n_rows": len(rows) - cap})
        for r in rows.head(cap).itertuples():
            k = (r.icd_code, int(r.icd_version))
            if k not in titles.index:
                drop.append({"unit_type": "history_diagnosis", "reason": "ICD code not in d_icd_diagnoses",
                             "detail": f"ICD-{k[1]} {k[0]} prior hadm_id={p.hadm_id}", "n_rows": 1})
                continue
            fields = {
                "prior_hadm_id": int(p.hadm_id), "prior_admittime": _iso(p.admittime),
                "prior_dischtime": _iso(p.dischtime), "seq_num": int(r.seq_num),
                "icd_code": r.icd_code, "icd_version": int(r.icd_version), "long_title": titles.loc[k],
            }
            out.append({
                "unit_type": "history_diagnosis", "time_scope": "prior", "timestamp": _iso(p.admittime),
                "fields": fields, "source_table": "diagnoses_icd",
                "source_rows": [{"hadm_id": int(p.hadm_id), "seq_num": int(r.seq_num)}],
                "sort_key": (f"{int(p.hadm_id):012d}", f"{int(r.seq_num):05d}"),
            })
    return out


def _order_key(u: dict[str, Any]) -> tuple[Any, ...]:
    t = u["unit_type"]
    if t in ("diagnosis", "procedure"):
        return (TYPE_ORDER[t], u["sort_key"])
    ts = u["timestamp"]
    return (TYPE_ORDER[t], ts is None, ts or "", u["sort_key"])


def build_units_for_admission(
    tables: Tables, hadm_id: int, units_cfg: dict[str, Any]
) -> tuple[list[EvidenceUnit], list[dict[str, Any]]]:
    """Build all evidence units for one target admission. Returns (units, dropped log)."""
    hadm_id = int(hadm_id)
    adm = tables["admissions"]
    subject_id = int(adm.loc[adm["hadm_id"] == hadm_id, "subject_id"].iloc[0])
    drop: list[dict[str, Any]] = []
    raw = (
        _icd_units(tables, hadm_id, "diagnoses_icd", "d_icd_diagnoses", "diagnosis", drop)
        + _icd_units(tables, hadm_id, "procedures_icd", "d_icd_procedures", "procedure", drop)
        + _lab_units(tables, hadm_id, drop)
        + _med_units(tables, hadm_id, drop)
        + _history_units(tables, hadm_id, units_cfg, drop)
    )
    raw.sort(key=_order_key)
    units = [
        EvidenceUnit(
            unit_id=f"E{i}", hadm_id=hadm_id, subject_id=subject_id, unit_type=u["unit_type"],
            time_scope=u["time_scope"], timestamp=u["timestamp"], fields=u["fields"],
            text=render(u["unit_type"], u["fields"]), source_table=u["source_table"],
            source_rows=u["source_rows"],
        )
        for i, u in enumerate(raw, start=1)
    ]
    for d in drop:
        d["hadm_id"] = hadm_id
    return units, drop


def units_dir(derived: Path) -> Path:
    """Directory holding per-admission unit files."""
    return derived / "units"


def build_all(
    tables: Tables, hadm_ids: list[int], units_cfg: dict[str, Any], out_dir: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build and write units + headers for all targets. Returns (count table, dropped log)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for h in hadm_ids:
        units, drop = build_units_for_admission(tables, h, units_cfg)
        header = build_header(tables, h)
        write_units(out_dir / f"{h}.jsonl", units)
        (out_dir / f"{h}_header.json").write_text(
            json.dumps(header, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        c = Counter(u.unit_type for u in units)
        counts.append({"hadm_id": h, **{t: c.get(t, 0) for t in UNIT_TYPES}, "total": len(units)})
        dropped.extend(drop)
    return pd.DataFrame(counts), dropped


def read_header(out_dir: Path, hadm_id: int) -> dict[str, Any]:
    """Load a stored header."""
    return json.loads((out_dir / f"{hadm_id}_header.json").read_text(encoding="utf-8"))
