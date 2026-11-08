"""OBVIOUSLY FAKE evidence units for checker tests (built with the real ``render``)."""

from __future__ import annotations

from pathlib import Path

from ehr_rag.checker.rules import AdmissionContext, Checker, LineCheck
from ehr_rag.eval.parse import parse_report
from ehr_rag.evidence.render import render
from ehr_rag.evidence.schema import EvidenceUnit

REPO = Path(__file__).resolve().parents[1]


def checker() -> Checker:
    return Checker.from_files(REPO / "config" / "checker.yaml", REPO / "config" / "checker_lexicon.yaml")


def lab(uid: str, label: str, pts: list[tuple[str, str]], lo: str | None = "3.3", hi: str | None = "5.1",
        fluid: str = "Blood", uom: str = "mmol/L") -> EvidenceUnit:
    """A numeric lab unit from (iso time, value) points."""
    f = {"itemid": 1, "label": label, "fluid": fluid, "category": "Chemistry", "n_time_window_linked": 0,
         "numeric": True, "uom": uom, "ref_low": lo, "ref_high": hi, "n_measurements": len(pts)}
    vals = sorted(pts)
    mx = max(vals, key=lambda p: float(p[1]))
    mn = min(vals, key=lambda p: float(p[1]))
    for name, p in (("first", vals[0]), ("max", mx), ("min", mn), ("last", vals[-1])):
        f[name] = {"value": p[1], "time": p[0], "flag": None}
    f["n_abnormal"] = 0
    return EvidenceUnit(uid, 1, 1, "lab", "current", vals[0][0], f, render("lab", f), "labevents", [])


def med(uid: str, drug: str, dose: str = "40", unit: str = "mg") -> EvidenceUnit:
    f = {"drug": drug, "dose_val_rx": dose, "dose_unit_rx": unit, "route": "IV", "dose_numeric": True,
         "n_orders": 2, "start": "2150-03-01T08:00:00", "stop": "2150-03-03T08:00:00"}
    return EvidenceUnit(uid, 1, 1, "medication", "current", f["start"], f, render("medication", f), "prescriptions", [])


def dx(uid: str, title: str, code: str = "FAKE1", seq: int = 1) -> EvidenceUnit:
    f = {"seq_num": seq, "n_total": 3, "icd_code": code, "icd_version": 10, "long_title": title}
    return EvidenceUnit(uid, 1, 1, "diagnosis", "current", "2150-03-01T00:00:00", f, render("diagnosis", f),
                        "diagnoses_icd", [])


T1, T2, T3 = "2150-03-01T08:00:00", "2150-03-02T08:00:00", "2150-03-03T08:00:00"


def fake_units() -> list[EvidenceUnit]:
    return [
        lab("E1", "Potassium", [(T1, "4.1"), (T2, "5.9"), (T3, "4.3")]),
        lab("E2", "Sodium", [(T1, "130"), (T3, "128")], lo="133", hi="145"),
        lab("E3", "Creatinine", [(T1, "1.2"), (T3, "2.4")], lo="0.5", hi="1.1"),
        med("E4", "Potassium Chloride"),
        med("E5", "Furosemide"),
        lab("E6", "White Blood Cells", [(T1, "7.2"), (T3, "6.8")], lo="4.0", hi="11.0", uom="K/uL"),
        lab("E7", "Hemoglobin", [(T1, "9.1"), (T3, "8.40")], lo="13.7", hi="17.5", uom="g/dL"),
        dx("E8", "Acute kidney failure, unspecified", "N179"),
        lab("E9", "Troponin T", [(T1, "0.02")], lo=None, hi=None, uom="ng/mL"),
        lab("E10", "Glucose", [(T1, "105")], lo="70", hi="100", uom="mg/dL"),
    ]


def check(line: str, units: list[EvidenceUnit] | None = None, section: str = "Findings",
          ck: Checker | None = None) -> LineCheck:
    """Check one fake line against the fake units."""
    units = units if units is not None else fake_units()
    ck = ck or checker()
    ctx: AdmissionContext = ck.admission(units, units)
    s = parse_report(f"### {section}\n{line}\n").sections[section][0]
    return ck.check_line(s, ctx)
