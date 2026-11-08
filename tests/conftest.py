"""Shared test fixtures.

All fixture data here is OBVIOUSLY FAKE (drug "FakeDrugA", lab "Fakeium",
ICD codes "FAKE1"...). It only mimics the MIMIC-IV table shapes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import yaml

HADM = [12, 21]

REPO = Path(__file__).resolve().parents[1]

FAKE_TABLES: dict[str, list[list[Any]]] = {
    "patients": [
        ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
        [1, "F", 50, 2100, "2011 - 2013", ""],
        [2, "M", 60, 2100, "2011 - 2013", ""],
    ],
    "admissions": [
        ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
         "admission_location", "discharge_location"],
        [1, 11, "2100-01-01 10:00:00", "2100-01-05 12:00:00", "URGENT", "FAKE LOCATION", "HOME"],
        [1, 12, "2100-02-01 08:00:00", "2100-02-06 09:00:00", "EW EMER.", "FAKE LOCATION", "HOME"],
        [2, 21, "2100-03-01 08:00:00", "2100-03-03 09:00:00", "URGENT", "FAKE LOCATION", "HOME"],
    ],
    "labevents": [
        ["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "value", "valuenum",
         "valueuom", "ref_range_lower", "ref_range_upper", "flag"],
        [1, 1, 12, 900, "2100-02-01 09:00:00", "4.1", "4.1", "mmol/L", "3.3", "5.1", ""],
        [2, 1, 12, 900, "2100-02-02 09:00:00", "5.9", "5.9", "mmol/L", "3.3", "5.1", "abnormal"],
        [3, 1, 12, 900, "2100-02-03 09:00:00", "3.9", "3.9", "mmol/L", "3.3", "5.1", ""],
        [4, 1, "", 900, "2100-02-04 09:00:00", "4.30", "4.30", "mmol/L", "3.3", "5.1", ""],
        [5, 1, 12, 901, "2100-02-02 10:00:00", "POSITIVE", "", "", "", "", "abnormal"],
        [6, 2, 21, 900, "2100-03-01 09:00:00", "2.9", "2.9", "mmol/L", "3.3", "5.1", "abnormal"],
        [7, 1, 11, 900, "2100-01-02 09:00:00", "4.0", "4.0", "mmol/L", "3.3", "5.1", ""],
        [8, 2, 21, 902, "2100-03-01 10:00:00", "___", "", "", "", "", ""],
    ],
    "d_labitems": [
        ["itemid", "label", "fluid", "category"],
        [900, "Fakeium", "Blood", "Chemistry"],
        [901, "Fake Screen", "Urine", "Urinalysis"],
        [902, "Fake Masked", "Blood", "Chemistry"],
    ],
    "prescriptions": [
        ["subject_id", "hadm_id", "pharmacy_id", "starttime", "stoptime", "drug", "dose_val_rx",
         "dose_unit_rx", "route"],
        [1, 12, 5001, "2100-02-01 12:00:00", "2100-02-02 12:00:00", "FakeDrugA", "40", "mg", "IV"],
        [1, 12, 5002, "2100-02-03 12:00:00", "2100-02-04 12:00:00", "FakeDrugA", "40", "mg", "IV"],
        [1, 12, 5003, "2100-02-01 12:00:00", "", "FakeDrugB", "1-2", "TAB", "PO"],
        [2, 21, 5004, "2100-03-01 12:00:00", "2100-03-02 12:00:00", "FakeDrugA", "20", "mg", "PO"],
    ],
    "diagnoses_icd": [
        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        [1, 11, 1, "FAKE1", 10],
        [1, 12, 2, "FAKE1", 10],
        [1, 12, 1, "FAKE2", 10],
        [2, 21, 1, "FAKE3", 10],
    ],
    "d_icd_diagnoses": [
        ["icd_code", "icd_version", "long_title"],
        ["FAKE1", 10, "Fake chronic condition one"],
        ["FAKE2", 10, "Fake acute condition two"],
        ["FAKE3", 10, "Fake condition three"],
    ],
    "procedures_icd": [
        ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
        [1, 12, 1, "2100-02-02", "FAKEP1", 10],
    ],
    "d_icd_procedures": [
        ["icd_code", "icd_version", "long_title"],
        ["FAKEP1", 10, "Fake procedure one"],
    ],
}


@pytest.fixture()
def fake_config(tmp_path: Path) -> Path:
    """A config pointing at a fake MIMIC-shaped data root under tmp_path."""
    hosp = tmp_path / "fake_mimic" / "hosp"
    hosp.mkdir(parents=True)
    for name, rows in FAKE_TABLES.items():
        with (hosp / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
    cfg = yaml.safe_load((REPO / "config" / "default.yaml").read_text(encoding="utf-8"))
    cfg["data"]["root"] = str(tmp_path / "fake_mimic")
    cfg["data"]["derived"] = str(tmp_path / "derived")
    cfg["checker"]["rules"] = str(REPO / "config" / "checker.yaml")
    cfg["checker"]["lexicon"] = str(REPO / "config" / "checker_lexicon.yaml")
    (tmp_path / "config").mkdir()
    p = tmp_path / "config" / "test.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


@pytest.fixture()
def env(fake_config: Path) -> dict[str, Any]:
    """Fake units + headers on disk, the real prompts copied in, and a saved retrieve run."""
    from ehr_rag.config import load_config
    from ehr_rag.data.load import load_tables
    from ehr_rag.evidence.build_units import build_all, units_dir
    from ehr_rag.evidence.schema import read_units
    from ehr_rag.retrieval.retrieve import Encoders, evidence_record, retrieve_admission
    from ehr_rag.runs import start_run
    from fakes import FakeEncoder

    cfg = load_config(fake_config)
    root = Path(cfg["_project_root"])
    (root / "prompts").mkdir()
    for p in (REPO / "prompts").glob("*.txt"):
        (root / "prompts" / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    out = units_dir(Path(cfg["data"]["derived"]))
    build_all(load_tables(cfg), HADM, cfg["units"], out)
    units = {h: read_units(out / f"{h}.jsonl") for h in HADM}
    enc = Encoders(query=FakeEncoder(), doc=FakeEncoder(), cross=None)  # type: ignore[arg-type]
    rrun = start_run(cfg, "retrieve", command="retrieve")
    evidence = {}
    for h in HADM:
        res = retrieve_admission(units[h], enc, cfg["retrieval"])
        rrun.write_json(f"retrieval/{h}.json", evidence_record(res, units[h]))
        evidence[h] = res.evidence
    return {"cfg": cfg, "root": root, "units": units, "evidence": evidence, "retrieval_run": rrun.run_id,
            "units_dir": out}
