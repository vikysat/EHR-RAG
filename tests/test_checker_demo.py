"""Every checker rule on REAL MIMIC-IV Demo units (open data); skipped if the Demo is not present.

Sentences are written by the test from the units' own fields, so they hold for whatever the
Demo admission contains. Admission 23559586 (selected in config/selected_admissions.yaml) has
AKI, low sodium, anemia, thrombocytopenia and a raised WBC.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ehr_rag.checker.rules import Checker
from ehr_rag.config import load_config
from ehr_rag.eval.parse import parse_report
from ehr_rag.evidence.schema import EvidenceUnit

REPO = Path(__file__).resolve().parents[1]
HADM = 23559586
CFG = load_config(REPO / "config" / "default.yaml")
pytestmark = pytest.mark.skipif(not (Path(CFG["_project_root"]) / CFG["data"]["root"]).is_dir(),
                                reason="MIMIC-IV Demo not present")


@pytest.fixture(scope="module")
def demo() -> dict[str, Any]:
    from ehr_rag.data.load import load_tables
    from ehr_rag.evidence.build_units import build_units_for_admission

    units, _ = build_units_for_admission(load_tables(CFG), HADM, CFG["units"])
    ck = Checker.from_config(CFG)
    ctx = ck.admission(units, units)

    def blood_lab(group: str) -> EvidenceUnit:
        cands = [u for u in units if u.unit_type == "lab" and u.fields.get("numeric") and u.fields["fluid"] == "Blood"
                 and ck.lab_group(u.fields["label"]) == group and u.fields.get("ref_high")
                 and u.fields["n_measurements"] > 1]
        return max(cands, key=lambda u: u.fields["n_measurements"])

    def check(line: str) -> Any:
        s = parse_report(f"### Findings\n{line}\n").sections["Findings"][0]
        return ck.check_line(s, ctx)

    return {"units": units, "ck": ck, "ctx": ctx, "lab": blood_lab, "check": check}


def v(u: EvidenceUnit, point: str) -> str:
    return u.fields[point]["value"]


def test_numbers_on_demo_labs(demo: dict[str, Any]) -> None:
    cr = demo["lab"]("creatinine")
    c = demo["check"]
    assert c(f"Creatinine peaked at {v(cr, 'max')}. [{cr.unit_id}]").status == "pass"
    assert c(f"Creatinine peaked at {v(cr, 'max')} mg/dL on 2137-08-08. [{cr.unit_id}]").status == "pass"
    wrong = Decimal(v(cr, "max")) + 7
    assert c(f"Creatinine peaked at {wrong}. [{cr.unit_id}]").error_types == ["number_mismatch"]
    hgb = demo["lab"]("hemoglobin")
    rounded = Decimal(v(hgb, "min")).quantize(Decimal(1))                 # rounding to the stated precision
    assert "number_mismatch" not in c(f"Hemoglobin fell to {rounded}. [{hgb.unit_id}]").error_types


def test_abbreviations_on_demo_labs(demo: dict[str, Any]) -> None:
    c = demo["check"]
    for group, abbr in (("creatinine", "Cr"), ("potassium", "K"), ("sodium", "Na"), ("hemoglobin", "Hgb"),
                        ("white blood cells", "WBC"), ("platelet", "Plt")):
        u = demo["lab"](group)
        assert c(f"{abbr} was {v(u, 'last')}. [{u.unit_id}]").status == "pass", group
    cr, na = demo["lab"]("creatinine"), demo["lab"]("sodium")
    assert c(f"Cr was {v(cr, 'last')}. [{na.unit_id}]").error_types == ["number_mismatch", "name_missing"]


def test_direction_and_terms_on_demo_labs(demo: dict[str, Any]) -> None:
    c = demo["check"]
    na, hgb, plt, wbc, cr = (demo["lab"](g) for g in ("sodium", "hemoglobin", "platelet", "white blood cells",
                                                      "creatinine"))
    assert c(f"Hyponatremia to {v(na, 'min')}. [{na.unit_id}]").status == "pass"
    assert "direction_mismatch" in c(f"Hypernatremia was present. [{na.unit_id}]").error_types
    assert c(f"There was no hypernatremia. [{na.unit_id}]").status == "pass"            # negation
    assert c(f"Anemia with hemoglobin as low as {v(hgb, 'min')}. [{hgb.unit_id}]").status == "pass"
    assert c(f"Thrombocytopenia was present. [{plt.unit_id}]").status == "pass"
    assert c(f"Leukocytosis to {v(wbc, 'max')}. [{wbc.unit_id}]").status == "pass"
    assert "direction_mismatch" in c(f"There was no leukocytosis. [{wbc.unit_id}]").error_types
    assert c(f"Elevated creatinine of {v(cr, 'max')}. [{cr.unit_id}]").status == "pass"
    assert "direction_mismatch" in c(f"Creatinine was low. [{cr.unit_id}]").error_types


def test_terms_supported_by_diagnosis_or_lab(demo: dict[str, Any]) -> None:
    c = demo["check"]
    aki = next(u for u in demo["units"] if u.unit_type == "diagnosis" and "kidney failure" in u.fields["long_title"])
    cr, na = demo["lab"]("creatinine"), demo["lab"]("sodium")
    assert c(f"The patient had AKI. [{aki.unit_id}]").status == "pass"
    assert c(f"The patient had acute kidney failure. [{aki.unit_id}]").status == "pass"
    assert c(f"The patient had AKI. [{cr.unit_id}]").status == "pass"                  # creatinine HIGH
    assert c(f"The patient had AKI. [{na.unit_id}]").error_types == ["name_missing"]


def test_blood_terms_need_blood_labs(demo: dict[str, Any]) -> None:
    urine_k = next(u for u in demo["units"] if u.unit_type == "lab" and u.fields["label"] == "Potassium, Urine")
    assert demo["check"](f"Hyperkalemia was present. [{urine_k.unit_id}]").error_types == ["name_missing"]


def test_ratio_units_are_not_creatinine(demo: dict[str, Any]) -> None:
    ratio = next(u for u in demo["units"] if u.unit_type == "lab" and "Protein/Creatinine" in u.fields["label"])
    assert demo["ck"].lab_group(ratio.fields["label"]) == "protein/creatinine ratio"
    assert "direction_mismatch" in demo["check"](f"Creatinine was elevated. [{ratio.unit_id}]").error_types


def test_trend_on_demo_labs(demo: dict[str, Any]) -> None:
    cr = demo["lab"]("creatinine")
    first, last = Decimal(v(cr, "first")), Decimal(v(cr, "last"))
    assert first != last
    right, wrong = ("rising", "falling") if last > first else ("falling", "rising")
    assert demo["check"](f"Creatinine was {right}. [{cr.unit_id}]").status == "pass"
    assert demo["check"](f"Creatinine was {wrong}. [{cr.unit_id}]").error_types == ["trend_mismatch"]
    single = next(u for u in demo["units"] if u.unit_type == "lab" and u.fields.get("numeric")
                  and u.fields["n_measurements"] == 1 and demo["ck"].lab_group(u.fields["label"]))
    assert "trend_mismatch" in demo["check"](f"{single.fields['label']} was rising. [{single.unit_id}]").error_types


def test_medication_on_demo_units(demo: dict[str, Any]) -> None:
    med = next(u for u in demo["units"] if u.unit_type == "medication" and u.fields["dose_numeric"]
               and u.fields["drug"].isalpha() and len(u.fields["drug"]) >= 5)
    drug, dose = med.fields["drug"], med.fields["dose_val_rx"]
    other = next(u for u in demo["units"] if u.unit_type == "diagnosis")
    assert demo["check"](f"{drug} {dose} {med.fields['dose_unit_rx']} was given. [{med.unit_id}]").status == "pass"
    assert demo["check"](f"{drug} {Decimal(dose) * 3} was given. [{med.unit_id}]").error_types == ["number_mismatch"]
    assert demo["check"](f"{drug} was given. [{other.unit_id}]").error_types == ["name_missing"]


def test_list_sentence_on_demo_units(demo: dict[str, Any]) -> None:
    cr, k, na = (demo["lab"](g) for g in ("creatinine", "potassium", "sodium"))
    line = (f"Labs showed Cr {v(cr, 'max')} [{cr.unit_id}], K {v(k, 'max')} [{k.unit_id}], "
            f"and Na {v(na, 'min')} [{na.unit_id}].")
    lc = demo["check"](line)
    assert lc.status == "pass" and len(lc.items) == 3 and lc.items[2].text.startswith("Labs showed Na")
    lead = demo["check"](line.replace("Labs showed", "Labs showed elevated"))
    assert [bool(it.errors) for it in lead.items] == [False, False, True]   # Na was low, not elevated


def test_structure_and_no_checkable_content(demo: dict[str, Any]) -> None:
    c = demo["check"]
    ids = [u.unit_id for u in demo["units"][:4]]
    assert c(f"The patient was admitted and treated. [{ids[0]}]").status == "unchecked"
    assert c(f"The patient was admitted and treated. [{', '.join(ids)}]").error_types == ["too_many_citations"]
    assert c("The patient was admitted and treated. [E9999]").error_types == ["invalid_id"]


def test_whole_report_counts(demo: dict[str, Any]) -> None:
    cr, na = demo["lab"]("creatinine"), demo["lab"]("sodium")
    rep = (f"### History\nThe patient was admitted. [{cr.unit_id}]\n"
           f"### Findings\nCreatinine peaked at {v(cr, 'max')}. [{cr.unit_id}]\nSodium was high. [{na.unit_id}]\n"
           "### Impression\nInsufficient evidence. [NONE]\n### Recommendations\nInsufficient evidence. [NONE]\n")
    res = demo["ck"].check_report(parse_report(rep), demo["ctx"])
    assert [r.status for r in res] == ["unchecked", "pass", "fail"]            # abstentions are not checked
