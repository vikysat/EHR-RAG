"""Phase 2: evidence units (fake fixture data)."""

from __future__ import annotations

import copy
from pathlib import Path

from ehr_rag.config import load_config
from ehr_rag.data.load import load_tables
from ehr_rag.evidence.build_units import build_all, build_header, build_units_for_admission
from ehr_rag.evidence.render import render
from ehr_rag.evidence.schema import read_units


def _units(fake_config: Path, hadm: int = 12) -> list:
    cfg = load_config(fake_config)
    units, _ = build_units_for_admission(load_tables(cfg), hadm, cfg["units"])
    return units


def test_unit_ids_identical_across_two_builds(fake_config: Path, tmp_path: Path) -> None:
    cfg = load_config(fake_config)
    a, b = tmp_path / "a", tmp_path / "b"
    build_all(load_tables(cfg), [12, 21], cfg["units"], a)
    build_all(load_tables(cfg), [12, 21], cfg["units"], b)
    for h in (12, 21):
        assert (a / f"{h}.jsonl").read_bytes() == (b / f"{h}.jsonl").read_bytes()
        assert (a / f"{h}_header.json").read_bytes() == (b / f"{h}_header.json").read_bytes()
    assert [u.unit_id for u in read_units(a / "12.jsonl")] == [f"E{i}" for i in range(1, 9)]


def test_type_order_and_seq_num_order(fake_config: Path) -> None:
    units = _units(fake_config)
    assert [u.unit_type for u in units] == [
        "diagnosis", "diagnosis", "procedure", "lab", "lab", "medication", "medication", "history_diagnosis",
    ]
    dx = [u for u in units if u.unit_type == "diagnosis"]
    assert [u.fields["seq_num"] for u in dx] == [1, 2]
    assert dx[0].text == "Diagnosis (priority 1 of 2): Fake acute condition two (ICD-10 FAKE2)."


def test_units_belong_to_target_only(fake_config: Path) -> None:
    for h in (12, 21):
        assert {u.hadm_id for u in _units(fake_config, h)} == {h}


def test_lab_render_and_precision(fake_config: Path) -> None:
    lab = next(u for u in _units(fake_config) if u.unit_type == "lab" and u.fields["numeric"])
    assert lab.text == (
        "Fakeium (Blood), mmol/L, ref 3.3–5.1: 4 measurements. First 4.1 (2100-02-01 09:00); "
        "max 5.9 [HIGH] (2100-02-02 09:00); min 3.9 (2100-02-03 09:00); last 4.30 (2100-02-04 09:00)."
    )
    assert lab.fields["n_abnormal"] == 1
    assert lab.source_rows == [1, 2, 3, 4]
    text_lab = next(u for u in _units(fake_config) if u.unit_type == "lab" and not u.fields["numeric"])
    assert text_lab.fields["numeric"] is False
    assert "POSITIVE [ABNORMAL]" in text_lab.text


def test_medication_and_history(fake_config: Path) -> None:
    units = _units(fake_config)
    meds = [u for u in units if u.unit_type == "medication"]
    a = next(u for u in meds if u.fields["drug"] == "FakeDrugA")
    assert a.text == "FakeDrugA 40 mg IV, ordered 2 times, 2100-02-01 to 2100-02-04."
    b = next(u for u in meds if u.fields["drug"] == "FakeDrugB")
    assert b.fields["dose_numeric"] is False
    hist = [u for u in units if u.unit_type == "history_diagnosis"]
    assert len(hist) == 1 and hist[0].time_scope == "prior"
    assert hist[0].text == "Prior admission (2100-01-01): Fake chronic condition one (ICD-10 FAKE1)."


def test_render_is_pure_and_edit_is_local(fake_config: Path) -> None:
    units = _units(fake_config)
    before = [u.text for u in units]
    snapshot = copy.deepcopy([u.fields for u in units])
    # Rendering twice gives identical text and never mutates fields.
    assert [render(u.unit_type, u.fields) for u in units] == before
    assert [u.fields for u in units] == snapshot
    # Edit one unit's fields (normal -> high) and re-render all units.
    i = next(i for i, u in enumerate(units) if u.unit_type == "lab" and u.fields["numeric"])
    edited = copy.deepcopy(units[i].fields)
    edited["last"]["value"] = "6.30"
    after = [render(u.unit_type, edited if j == i else u.fields) for j, u in enumerate(units)]
    changed = [j for j in range(len(units)) if after[j] != before[j]]
    assert changed == [i]
    assert "last 6.30 [HIGH]" in after[i]
    assert [u.fields for u in units] == snapshot


def test_marker_crosses_reference_range() -> None:
    f = {"label": "Fakeium", "fluid": "Blood", "uom": "mmol/L", "ref_low": "3.3", "ref_high": "5.1",
         "numeric": True, "n_measurements": 1,
         "first": {"value": "4.0", "time": "2100-01-01T00:00:00", "flag": None}}
    assert "[" not in render("lab", f)
    f["first"]["value"] = "2.9"
    assert "2.9 [LOW]" in render("lab", f)
    f["first"]["value"] = "5.2"
    assert "5.2 [HIGH]" in render("lab", f)
    g = dict(f, ref_low=None, ref_high=None)
    g["first"] = {"value": "5.2", "time": "2100-01-01T00:00:00", "flag": "abnormal"}
    assert "5.2 [ABNORMAL]" in render("lab", g)


def test_header(fake_config: Path) -> None:
    h = build_header(load_tables(load_config(fake_config)), 12)
    assert h["age"] == 50 and h["gender"] == "F" and h["admission_type"] == "EW EMER."
    assert h["admittime"] == "2100-02-01T08:00:00" and h["los_days"] == 5.04


def test_zero_zero_range_falls_back_to_flag() -> None:
    f = {"label": "Fakeocytes", "fluid": "Blood", "uom": "%", "ref_low": "0", "ref_high": "0",
         "numeric": True, "n_measurements": 1,
         "first": {"value": "21", "time": "2100-01-01T00:00:00", "flag": None}}
    text = render("lab", f)
    assert "ref" not in text and "[" not in text  # no range shown, no [HIGH]
    f["first"]["flag"] = "abnormal"
    assert "21 [ABNORMAL]" in render("lab", f)


def test_zero_zero_range_in_selection_mask() -> None:
    import pandas as pd

    from ehr_rag.data.select_admissions import lab_abnormal_mask

    labs = pd.DataFrame({
        "valuenum_f": [21.0, 21.0, 6.0],
        "ref_range_lower_f": [0.0, 0.0, 3.3],
        "ref_range_upper_f": [0.0, 0.0, 5.1],
        "flag": [None, "abnormal", None],
    })
    assert lab_abnormal_mask(labs).tolist() == [False, True, True]


def test_masked_text_lab_dropped(fake_config: Path) -> None:
    cfg = load_config(fake_config)
    units, drop = build_units_for_admission(load_tables(cfg), 21, cfg["units"])
    assert not any(u.unit_type == "lab" and u.fields["label"] == "Fake Masked" for u in units)
    assert any(d["reason"] == "text-only lab whose last value is the ___ mask" for d in drop)
