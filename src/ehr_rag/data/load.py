"""Locate, schema-check and load the MIMIC-IV tables used by this project.

All identifier/code columns are read as strings first (ICD codes have leading
zeros) and only the documented numeric keys are converted. Lab ``value``,
``valuenum`` and reference-range columns are kept as their original strings so
rendering can preserve decimal precision; float copies are added with an
``_f`` suffix for comparisons. Only empty cells are treated as missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ehr_rag.config import resolve_path

MODULE_DIRS: tuple[str, ...] = ("hosp", "core")

EXPECTED_COLUMNS: dict[str, list[str]] = {
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year"],
    "admissions": [
        "subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
        "admission_location", "discharge_location",
    ],
    "labevents": [
        "labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "value", "valuenum",
        "valueuom", "ref_range_lower", "ref_range_upper", "flag",
    ],
    "d_labitems": ["itemid", "label", "fluid", "category"],
    "prescriptions": [
        "subject_id", "hadm_id", "starttime", "stoptime", "drug", "dose_val_rx", "dose_unit_rx", "route",
    ],
    "diagnoses_icd": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    "d_icd_diagnoses": ["icd_code", "icd_version", "long_title"],
    "procedures_icd": ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
    "d_icd_procedures": ["icd_code", "icd_version", "long_title"],
}

# Columns beyond the spec's expected list that we also load. Verified present in
# the real header; used only as primary keys for ``source_rows``.
EXTRA_COLUMNS: dict[str, list[str]] = {
    "prescriptions": ["pharmacy_id"],
}

# Columns converted to integers (nullable Int64) after loading.
_INT_COLUMNS: dict[str, list[str]] = {
    "patients": ["subject_id", "anchor_age", "anchor_year"],
    "admissions": ["subject_id", "hadm_id"],
    "labevents": ["labevent_id", "subject_id", "hadm_id", "itemid"],
    "d_labitems": ["itemid"],
    "prescriptions": ["subject_id", "hadm_id", "pharmacy_id"],
    "diagnoses_icd": ["subject_id", "hadm_id", "seq_num", "icd_version"],
    "d_icd_diagnoses": ["icd_version"],
    "procedures_icd": ["subject_id", "hadm_id", "seq_num", "icd_version"],
    "d_icd_procedures": ["icd_version"],
}

_DATETIME_COLUMNS: dict[str, list[str]] = {
    "admissions": ["admittime", "dischtime"],
    "labevents": ["charttime"],
    "prescriptions": ["starttime", "stoptime"],
    "procedures_icd": ["chartdate"],
}

_FLOAT_COPIES: dict[str, list[str]] = {
    "labevents": ["valuenum", "ref_range_lower", "ref_range_upper"],
}


class SchemaError(RuntimeError):
    """Raised when data files or columns differ from the spec."""


@dataclass
class Tables:
    """All loaded tables, keyed by table name."""

    frames: dict[str, pd.DataFrame]
    files: dict[str, Path]
    lab_link_stats: dict[str, int] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.frames[name]


def data_root(cfg: dict[str, Any]) -> Path:
    """Absolute data root from config."""
    return resolve_path(cfg, cfg["data"]["root"])


def directory_tree(root: Path) -> list[str]:
    """Return an indented listing of ``root`` with file sizes."""
    lines = [f"{root}"]

    def walk(d: Path, depth: int) -> None:
        for p in sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name)):
            indent = "  " * depth
            if p.is_dir():
                lines.append(f"{indent}{p.name}/")
                walk(p, depth + 1)
            else:
                lines.append(f"{indent}{p.name}  ({p.stat().st_size:,} bytes)")

    walk(root, 1)
    return lines


def find_table_files(root: Path) -> dict[str, Path]:
    """Find ``<table>.csv.gz`` or ``<table>.csv`` under one of ``MODULE_DIRS``.

    Raises SchemaError if a table is missing or found in more than one place.
    """
    if not root.is_dir():
        raise SchemaError(f"data root does not exist: {root}")
    found: dict[str, Path] = {}
    problems: list[str] = []
    for table in EXPECTED_COLUMNS:
        hits = [
            root / mod / f"{table}{ext}"
            for mod in MODULE_DIRS
            for ext in (".csv.gz", ".csv")
            if (root / mod / f"{table}{ext}").is_file()
        ]
        if not hits:
            problems.append(f"{table}: not found under {[str(root / m) for m in MODULE_DIRS]}")
        elif len(hits) > 1:
            problems.append(f"{table}: found in multiple places {[str(h) for h in hits]}")
        else:
            found[table] = hits[0]
    if problems:
        raise SchemaError("table files differ from spec:\n  " + "\n  ".join(problems))
    return found


def read_header(path: Path) -> list[str]:
    """Read only the header row of a CSV(.gz)."""
    return list(pd.read_csv(path, nrows=0).columns)


def check_columns(files: dict[str, Path]) -> dict[str, dict[str, list[str]]]:
    """Compare real headers against ``EXPECTED_COLUMNS``.

    Returns ``{table: {"missing": [...], "extra": [...]}}``.
    """
    report: dict[str, dict[str, list[str]]] = {}
    for table, path in files.items():
        header = read_header(path)
        expected = EXPECTED_COLUMNS[table] + EXTRA_COLUMNS.get(table, [])
        report[table] = {
            "missing": [c for c in expected if c not in header],
            "extra": [c for c in header if c not in expected],
        }
    return report


def head_rows(path: Path, n: int = 5) -> pd.DataFrame:
    """First ``n`` rows, all as strings."""
    return pd.read_csv(path, nrows=n, dtype=str, keep_default_na=False)


def _read_table(table: str, path: Path) -> pd.DataFrame:
    cols = EXPECTED_COLUMNS[table] + EXTRA_COLUMNS.get(table, [])
    df = pd.read_csv(path, usecols=cols, dtype=str, keep_default_na=False, na_values=[""])
    df = df[cols]
    for c in _INT_COLUMNS.get(table, []):
        df[c] = pd.to_numeric(df[c], errors="raise").astype("Int64")
    for c in _DATETIME_COLUMNS.get(table, []):
        fmt = "%Y-%m-%d" if c == "chartdate" else "%Y-%m-%d %H:%M:%S"
        df[c] = pd.to_datetime(df[c], format=fmt, errors="raise")
    for c in _FLOAT_COPIES.get(table, []):
        df[f"{c}_f"] = pd.to_numeric(df[c], errors="coerce")
    return df


def link_labs(labs: pd.DataFrame, admissions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach each lab row to an admission.

    Rows with ``hadm_id`` are linked by id (``linked_by = "hadm_id"``). Rows with
    null ``hadm_id`` are linked when ``charttime`` falls within
    [admittime, dischtime] of exactly one admission of the same subject
    (``linked_by = "time_window"``). Null-hadm rows matching several admissions
    are counted and left unlinked. Returns (linked rows with ``link_hadm_id``
    and ``linked_by``, counts).
    """
    by_id = labs[labs["hadm_id"].notna()].copy()
    known = set(admissions["hadm_id"].tolist())
    unknown_hadm = int((~by_id["hadm_id"].isin(known)).sum())
    by_id = by_id[by_id["hadm_id"].isin(known)]
    by_id["link_hadm_id"] = by_id["hadm_id"]
    by_id["linked_by"] = "hadm_id"

    null = labs[labs["hadm_id"].isna()]
    adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].rename(
        columns={"hadm_id": "cand_hadm_id"}
    )
    m = null.merge(adm, on="subject_id", how="inner")
    m = m[(m["charttime"] >= m["admittime"]) & (m["charttime"] <= m["dischtime"])]
    n_matches = m.groupby("labevent_id")["cand_hadm_id"].nunique()
    ambiguous_ids = set(n_matches[n_matches > 1].index)
    m = m[~m["labevent_id"].isin(ambiguous_ids)]
    by_time = m.drop(columns=["admittime", "dischtime"]).copy()
    by_time["link_hadm_id"] = by_time.pop("cand_hadm_id").astype("Int64")
    by_time["linked_by"] = "time_window"

    linked = pd.concat([by_id, by_time], ignore_index=True)
    linked["link_hadm_id"] = linked["link_hadm_id"].astype("Int64")
    stats = {
        "total_rows": int(len(labs)),
        "rows_with_hadm_id": int(labs["hadm_id"].notna().sum()),
        "linked_by_hadm_id": int(len(by_id)),
        "hadm_id_not_in_admissions": unknown_hadm,
        "rows_null_hadm_id": int(len(null)),
        "linked_by_time_window": int(len(by_time)),
        "null_hadm_ambiguous_multiple_admissions": int(len(ambiguous_ids)),
        "null_hadm_unlinked": int(len(null) - len(by_time) - len(ambiguous_ids)),
    }
    return linked, stats


def load_tables(cfg: dict[str, Any]) -> Tables:
    """Locate, check and load all tables; link labs to admissions.

    Raises SchemaError on any missing file or missing expected column.
    """
    files = find_table_files(data_root(cfg))
    report = check_columns(files)
    missing = {t: r["missing"] for t, r in report.items() if r["missing"]}
    if missing:
        raise SchemaError(f"expected columns missing: {missing}")
    frames = {t: _read_table(t, p) for t, p in files.items()}
    row_counts = {t: int(len(df)) for t, df in frames.items()}
    linked, stats = link_labs(frames["labevents"], frames["admissions"])
    frames["labevents"] = linked
    return Tables(frames=frames, files=files, lab_link_stats=stats, row_counts=row_counts)
