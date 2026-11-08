"""Cohort selection (DuckDB over csv.gz) on an OBVIOUSLY FAKE MIMIC-shaped dataset."""

from __future__ import annotations

import csv
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from ehr_rag.config import ConfigError, load_config
from ehr_rag.data.cohort import (
    CohortError, demo_subject_ids, find_note_file, hash_path, ids_path, load_split_ids, select_cohort, write_split,
)

REPO = Path(__file__).resolve().parents[1]
N_SUBJECTS = 40
DEMO_SUBJECTS = {1, 2, 3}
T0 = datetime(2100, 1, 1, 8, 0, 0)
BASE = 2_000_000  # fake hadm_id = BASE + 10 * subject + k


def _ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _los2(s: int) -> float:
    """Length of stay (days) of each subject's second admission."""
    return {0: 1.0, 1: 20.0}.get(s % 5, 5.0)


def _n_labs2(s: int) -> int:
    return 5 if s % 4 == 0 else 25


def _has_note(s: int) -> bool:
    return s % 3 != 0


def _write_gz(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def fake_tables() -> dict[str, list[list[Any]]]:
    """FAKE tables: subject s has admissions BASE+s*10+1 (3 days) and BASE+s*10+2 (``_los2`` days, later)."""
    t: dict[str, list[list[Any]]] = {
        "patients": [["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"]],
        "admissions": [["subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
                        "admission_location", "discharge_location", "new_full_only_column"]],
        "labevents": [["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "value", "valuenum",
                       "valueuom", "ref_range_lower", "ref_range_upper", "flag"]],
        "d_labitems": [["itemid", "label", "fluid", "category"], [900, "Fakeium", "Blood", "Chemistry"]],
        "prescriptions": [["subject_id", "hadm_id", "pharmacy_id", "starttime", "stoptime", "drug", "dose_val_rx",
                           "dose_unit_rx", "route"]],
        "diagnoses_icd": [["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]],
        "d_icd_diagnoses": [["icd_code", "icd_version", "long_title"], ["FAKE1", 10, "Fake condition, one"]],
        "procedures_icd": [["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"]],
        "d_icd_procedures": [["icd_code", "icd_version", "long_title"], ["FAKEP1", 10, "Fake procedure"]],
    }
    lab_id, rx_id = 1, 1
    for s in range(1, N_SUBJECTS + 1):
        t["patients"].append([s, "F", 50, 2100, "2011 - 2013", ""])
        a1, a2 = T0, T0 + timedelta(days=60)
        for h, start, los in ((BASE + s * 10 + 1, a1, 3.0), (BASE + s * 10 + 2, a2, _los2(s))):
            t["admissions"].append([s, h, _ts(start), _ts(start + timedelta(days=los)), "URGENT", "FAKE", "HOME",
                                    "x"])
            t["diagnoses_icd"].append([s, h, 1, "FAKE1", 10])
            t["prescriptions"].append([s, h, rx_id, _ts(start), _ts(start + timedelta(hours=5)), "FakeDrugA",
                                       "40", "mg", "IV"])
            rx_id += 1
            n_labs = _n_labs2(s) if h % 10 == 2 else 3
            for i in range(n_labs):
                t["labevents"].append([lab_id, s, h, 900, _ts(start + timedelta(hours=i)), f"4.{i % 10}",
                                       f"4.{i % 10}", "mmol/L", "3.3", "5.1", ""])
                lab_id += 1
        # a null-hadm lab inside the second stay (time-window linked by build-units, not counted here)
        t["labevents"].append([lab_id, s, "", 900, _ts(a2 + timedelta(hours=1)), "4.30", "4.30", "mmol/L",
                               "3.3", "5.1", ""])
        lab_id += 1
        t["procedures_icd"].append([s, BASE + s * 10 + 2, 1, a2.strftime("%Y-%m-%d"), "FAKEP1", 10])
    return t


def make_fake_mimic(root: Path) -> tuple[Path, Path, Path]:
    """Write FAKE full-MIMIC (hosp/*.csv.gz), note root and Demo root. Returns (mimic, note, demo)."""
    mimic, note, demo = root / "fake_full", root / "fake_note", root / "fake_demo"
    for name, rows in fake_tables().items():
        _write_gz(mimic / "hosp" / f"{name}.csv.gz", rows)
    notes = [["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"]]
    for s in range(1, N_SUBJECTS + 1):
        if _has_note(s):
            notes.append([f"{s}-DS-1", s, BASE + s * 10 + 2, "DS", 1, "", "",
                          f'FAKE discharge summary {s}, line one.\nLine two with "quotes", commas, 5.9.'])
    notes.append(["4-DS-2", 4, BASE + 42, "DS", 2, "", "", "FAKE addendum for subject 4."])
    _write_gz(note / "note" / "discharge.csv.gz", notes)
    # The FAKE Demo: same headers as the fake full data minus its full-only column (an "earlier release").
    for name, rows in fake_tables().items():
        header = [c for c in rows[0] if c != "new_full_only_column"]
        _write_gz(demo / "hosp" / f"{name}.csv.gz", [header])
    (demo / "demo_subject_id.csv").write_text("subject_id\n" + "".join(f"{s}\n" for s in sorted(DEMO_SUBJECTS)),
                                              encoding="utf-8")
    return mimic, note, demo


PARAMS = {"pool_fraction": 1.0, "seed": 13, "min_los_days": 2, "max_los_days": 14, "min_prior_admissions": 1,
          "min_lab_results": 20, "n_dev": 3, "n_test": 5}


@pytest.fixture()
def fake_mimic(tmp_path: Path) -> tuple[Path, Path, Path]:
    return make_fake_mimic(tmp_path)


def _expected_eligible(note: bool, demo: bool) -> set[int]:
    """Independent statement of the rules for the fake data: only second admissions can qualify."""
    out = set()
    for s in range(1, N_SUBJECTS + 1):
        if demo and s in DEMO_SUBJECTS:
            continue
        if 2 <= _los2(s) <= 14 and _n_labs2(s) >= 20 and (_has_note(s) or not note):
            out.add(BASE + s * 10 + 2)
    return out


def test_select_counts_each_filter(fake_mimic: tuple[Path, Path, Path]) -> None:
    mimic, note, demo = fake_mimic
    lines: list[str] = []
    sel = select_cohort(mimic, find_note_file(note), demo_subject_ids(demo), PARAMS, log=lines.append)
    c = dict(sel.counts)
    n = N_SUBJECTS - len(DEMO_SUBJECTS)
    assert c["subjects in patients"] == N_SUBJECTS
    assert c["pool after removing Demo subjects"] == n
    assert c["admissions of pool subjects"] == 2 * n
    los_ok = sum(1 for s in range(4, N_SUBJECTS + 1) if 2 <= _los2(s) <= 14)
    assert c["hospital stay 2-14 days"] == n + los_ok        # every first stay is 3 days
    assert c[">= 1 prior admission(s)"] == los_ok              # first stays have no prior admission
    elig = _expected_eligible(note=True, demo=True)
    assert c["has a discharge note (MIMIC-IV-Note)"] == len(elig)
    chosen = sel.splits["dev"] + sel.splits["test"]
    assert len(sel.splits["dev"]) == 3 and len(sel.splits["test"]) == 5
    assert len(set(chosen)) == 8 and set(chosen) <= elig
    assert not {(h - BASE) // 10 for h in sel.splits["dev"]} & {(h - BASE) // 10 for h in sel.splits["test"]}
    assert all(any(ch.isdigit() for ch in ln) for ln in lines if not ln.lstrip().startswith("("))


def test_select_is_deterministic_and_seeded(fake_mimic: tuple[Path, Path, Path]) -> None:
    mimic, note, demo = fake_mimic
    args = (mimic, None, None)
    small = {**PARAMS, "pool_fraction": 0.5, "n_dev": 2, "n_test": 2}
    a = select_cohort(*args, small, log=lambda s: None)
    b = select_cohort(*args, small, log=lambda s: None)
    c = select_cohort(*args, {**small, "seed": 14}, log=lambda s: None)
    assert a.splits == b.splits and a.splits != c.splits
    assert dict(a.counts)["seeded pool (0.5 of subjects, seed 13)"] == N_SUBJECTS // 2


def test_skip_flags_and_too_few(fake_mimic: tuple[Path, Path, Path]) -> None:
    mimic, note, demo = fake_mimic
    lines: list[str] = []
    sel = select_cohort(mimic, None, None, {**PARAMS, "n_dev": 0, "n_test": 0}, log=lines.append)
    assert "  (discharge-note filter skipped)" in lines and "  (Demo exclusion skipped)" in lines
    assert dict(sel.counts)["eligible subjects (latest eligible admission each)"] == len(
        _expected_eligible(note=False, demo=False))
    with pytest.raises(CohortError, match="eligible subjects"):
        select_cohort(mimic, None, None, {**PARAMS, "n_test": 1000}, log=lambda s: None)


def _cohort_config(tmp_path: Path, mimic: Path, note: Path, demo: Path) -> Path:
    cfg = yaml.safe_load((REPO / "config" / "default.yaml").read_text(encoding="utf-8"))
    cfg["cohort"].update(mimic_root=str(mimic), note_root=str(note), demo_root=str(demo),
                         out_dir=str(tmp_path / "cohort_out"), pool_fraction=1.0, n_dev=3, n_test=5)
    (tmp_path / "config").mkdir(exist_ok=True)
    p = tmp_path / "config" / "cohort.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_split_hash_roundtrip_and_tamper(tmp_path: Path, fake_mimic: tuple[Path, Path, Path]) -> None:
    cfg = load_config(_cohort_config(tmp_path, *fake_mimic))
    digest = write_split(cfg, "dev", [30, 10, 20])
    assert ids_path(cfg, "dev").read_text(encoding="ascii") == "10\n20\n30\n"
    assert hash_path(cfg, "dev").read_text(encoding="ascii") == f"{digest}  dev_hadm_ids.txt\n"
    assert hash_path(cfg, "dev").parent == tmp_path / "config" / "cohort"
    assert load_split_ids(cfg, "dev") == [10, 20, 30]
    ids_path(cfg, "dev").write_text("10\n20\n31\n", encoding="ascii")
    with pytest.raises(ConfigError, match="SHA-256 differs"):
        load_split_ids(cfg, "dev")
    with pytest.raises(ConfigError, match="not found"):
        load_split_ids(cfg, "test")


def test_select_cohort_cli(tmp_path: Path, fake_mimic: tuple[Path, Path, Path],
                           capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main

    cfg_path = _cohort_config(tmp_path, *fake_mimic)
    assert main(["select-cohort", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out.split("wrote the id lists")[0]
    cfg = load_config(cfg_path)
    dev, test = load_split_ids(cfg, "dev"), load_split_ids(cfg, "test")
    assert len(dev) == 3 and len(test) == 5
    assert not any(str(h) in out for h in dev + test)    # counts only, never ids
    assert main(["select-cohort", "--config", str(cfg_path)]) == 2   # fixed cohort: no silent overwrite
    assert "already exist" in capsys.readouterr().err
    assert main(["select-cohort", "--config", str(cfg_path), "--overwrite"]) == 0
    assert load_split_ids(cfg, "dev") == dev


# --------------------------------------------------------------------------- extraction and schema check


def _build(cfg_path: Path, root: Path, ids: list[int], out: Path) -> list[dict[str, Any]]:
    import copy

    from ehr_rag.data.load import load_tables
    from ehr_rag.evidence.build_units import build_all

    cfg = copy.deepcopy(load_config(cfg_path))
    cfg["data"]["root"] = str(root)
    _, dropped = build_all(load_tables(cfg), ids, cfg["units"], out)
    return dropped


def test_extract_keeps_demo_columns_and_builds_identical_units(
        tmp_path: Path, fake_mimic: tuple[Path, Path, Path]) -> None:
    from ehr_rag.data.cohort import NOTE_COLUMNS, extract_cohort
    from ehr_rag.data.load import find_table_files, read_header
    from ehr_rag.eval.text_metrics import load_references

    mimic, note, demo = fake_mimic
    cfg_path = _cohort_config(tmp_path, mimic, note, demo)
    ids = [BASE + 42, BASE + 52, BASE + 62]   # subject 4: two notes, 5: one note, 6: none
    ev, ref = tmp_path / "ext" / "evidence", tmp_path / "ext" / "reference"
    lines: list[str] = []
    res = extract_cohort(mimic, demo, ids, ev, find_note_file(note), ref, log=lines.append)
    files = find_table_files(ev)
    demo_files = find_table_files(demo)
    assert all(read_header(files[t]) == read_header(demo_files[t]) for t in files)   # Demo columns, Demo order
    assert "new_full_only_column" not in read_header(files["admissions"])
    assert res["n_subjects"] == 3 and res["rows"]["admissions"] == 6          # every admission of the patients
    assert res["rows"]["labevents"] == sum(3 + _n_labs2(s) + 1 for s in (4, 5, 6))  # incl. null-hadm rows
    assert res["rows"]["d_icd_diagnoses"] == 1
    assert res["notes"]["admissions without a note"] == 1 and res["notes"]["admissions with several notes"] == 1
    refs = load_references(ref / "references.csv")
    assert set(refs) == {BASE + 42, BASE + 52}
    assert refs[BASE + 52] == 'FAKE discharge summary 5, line one.\nLine two with "quotes", commas, 5.9.'
    assert refs[BASE + 42].endswith("\n\nFAKE addendum for subject 4.")
    assert set(NOTE_COLUMNS) <= set(read_header(ref / "discharge.parquet"))
    assert not any("FAKE" in ln for ln in lines)                               # counts only, never text
    # the existing builder runs unchanged on the Parquet and gives byte-identical units
    a, b = tmp_path / "units_csv", tmp_path / "units_parquet"
    assert _build(cfg_path, mimic, ids, a) == _build(cfg_path, ev, ids, b)
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir()) and len(names) == 6
    assert all((a / n).read_bytes() == (b / n).read_bytes() for n in names)


def test_schema_report_and_missing_demo_column(tmp_path: Path, fake_mimic: tuple[Path, Path, Path]) -> None:
    from ehr_rag.data.cohort import extract_cohort, schema_ok, schema_report
    from ehr_rag.data.load import SchemaError

    mimic, note, demo = fake_mimic
    rep = schema_report(mimic, demo, None, find_note_file(note))
    assert rep["admissions"]["full_not_in_demo"] == ["new_full_only_column"]
    assert rep["admissions"]["demo_not_in_full"] == [] and schema_ok(rep)
    assert rep["discharge"]["required_not_in_full"] == []
    # a column the Demo has but the full data lacks blocks extraction
    _write_gz(demo / "hosp" / "patients.csv.gz",
              [["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod", "old_col"]])
    rep = schema_report(mimic, demo, None, None)
    assert rep["patients"]["demo_not_in_full"] == ["old_col"] and not schema_ok(rep)
    with pytest.raises(SchemaError, match="old_col"):
        extract_cohort(mimic, demo, [BASE + 42], tmp_path / "e", None, tmp_path / "r", log=lambda s: None)


def test_extract_and_schema_cli(tmp_path: Path, fake_mimic: tuple[Path, Path, Path],
                                capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main
    from ehr_rag.data.cohort import evidence_root, reference_dir

    cfg_path = _cohort_config(tmp_path, *fake_mimic)
    assert main(["select-cohort", "--config", str(cfg_path)]) == 0
    assert main(["extract-cohort", "--config", str(cfg_path), "--splits", "dev"]) == 0
    cfg = load_config(cfg_path)
    assert (evidence_root(cfg) / "hosp" / "labevents.parquet").exists()
    assert (reference_dir(cfg) / "references.csv").exists()
    capsys.readouterr()
    assert main(["cohort-schema-check", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "new_full_only_column" in out and "same as the Demo" in out and "OK:" in out
    ids_path(cfg, "dev").write_text("1\n", encoding="ascii")          # tampered list: refused
    assert main(["extract-cohort", "--config", str(cfg_path), "--splits", "dev"]) == 2
