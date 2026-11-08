"""Cohort selection on full MIMIC-IV with DuckDB, reading the ``csv.gz`` files directly.

Everything here may touch credentialed data, so it prints COUNTS ONLY: never rows,
ids or text. The id lists go to ``cohort.out_dir`` (gitignored); only their SHA-256
hashes are written to ``config/cohort/<split>_hadm_ids.sha256`` (committed).

Selection, as implemented (each step's count is printed):
  1. all ``subject_id`` in ``patients``;
  2. a seeded pool: ``round(pool_fraction * n)`` subjects drawn with
     ``random.Random(f"{seed}:pool")`` from the sorted id list;
  3. Demo subjects removed (``demo_root/demo_subject_id.csv``, else the Demo's
     ``patients``);
  4. admissions of the pool subjects;
  5. hospital stay ``min_los_days <= (dischtime - admittime) <= max_los_days`` (days);
  6. at least ``min_prior_admissions`` admissions of the same subject with
     ``dischtime`` before this ``admittime`` (the ``select-admissions`` definition);
  7. at least ``min_lab_results`` ``labevents`` rows with this ``hadm_id`` (rows
     linked only by time window are not counted, so this is a lower bound of what
     ``build-units`` uses);
  8. a discharge note in MIMIC-IV-Note ``note/discharge`` with this ``hadm_id``
     (skippable: the Demo has no notes);
  9. one admission per subject: the latest eligible one (by admittime, then hadm_id),
     as in ``select-admissions``;
 10. eligible subjects shuffled with ``random.Random(f"{seed}:draw")``; the first
     ``n_dev`` give the dev admissions, the next ``n_test`` the test admissions, so
     dev and test never share a patient.

Extraction (``extract-cohort``) copies, for the patients of the selected admissions,
every row of each table ``build-units`` loads (all their admissions, so prior
admissions and time-window lab linkage work as before), plus the three dictionary
tables in full. Rows are filtered inside DuckDB while scanning the csv.gz files, so
nothing large is loaded into memory. Columns are exactly the Demo's columns, in the
Demo's order, all as strings; ``build-units`` then reads the Parquet unchanged.
The discharge notes of the selected admissions go to a separate ``reference/``
directory: they are evaluation references and never evidence.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ehr_rag.config import ConfigError, project_root, resolve_path
from ehr_rag.data.load import EXPECTED_COLUMNS, EXTRA_COLUMNS, SchemaError, find_table_files, read_header

SPLITS: tuple[str, ...] = ("dev", "test")
NOTE_TABLE = "discharge"
NOTE_COLUMNS: tuple[str, ...] = ("note_id", "subject_id", "hadm_id", "note_seq", "text")
DICTIONARY_TABLES: tuple[str, ...] = ("d_labitems", "d_icd_diagnoses", "d_icd_procedures")
# Row order of the extracted files (numeric keys compared as integers).
ORDER_BY: dict[str, tuple[str, ...]] = {
    "patients": ("subject_id",),
    "admissions": ("subject_id", "hadm_id"),
    "labevents": ("labevent_id",),
    "d_labitems": ("itemid",),
    "prescriptions": ("subject_id", "hadm_id", "pharmacy_id"),
    "diagnoses_icd": ("subject_id", "hadm_id", "seq_num"),
    "d_icd_diagnoses": ("icd_code", "icd_version"),
    "procedures_icd": ("subject_id", "hadm_id", "seq_num"),
    "d_icd_procedures": ("icd_code", "icd_version"),
}
_TEXT_KEYS = {"icd_code"}


class CohortError(RuntimeError):
    """The cohort cannot be selected or extracted as configured."""


# --------------------------------------------------------------------------- paths and hashes


def cohort_dir(cfg: dict[str, Any]) -> Path:
    """Gitignored output directory for id lists, extracted Parquet and reference notes."""
    return resolve_path(cfg, cfg["cohort"]["out_dir"])


def hash_dir(cfg: dict[str, Any]) -> Path:
    """Committed directory holding only the SHA-256 hashes of the id lists."""
    return project_root(cfg) / "config" / "cohort"


def evidence_root(cfg: dict[str, Any]) -> Path:
    """Extracted structured tables (``<out_dir>/evidence/hosp/*.parquet``): ``data.root`` of config/full.yaml."""
    return cohort_dir(cfg) / "evidence"


def reference_dir(cfg: dict[str, Any]) -> Path:
    """Discharge notes of the selected admissions (evaluation references only, never evidence)."""
    return cohort_dir(cfg) / "reference"


def ids_path(cfg: dict[str, Any], split: str) -> Path:
    """``<out_dir>/<split>_hadm_ids.txt``."""
    return cohort_dir(cfg) / f"{split}_hadm_ids.txt"


def hash_path(cfg: dict[str, Any], split: str) -> Path:
    """``config/cohort/<split>_hadm_ids.sha256``."""
    return hash_dir(cfg) / f"{split}_hadm_ids.sha256"


def ids_bytes(ids: list[int]) -> bytes:
    """Canonical file content of an id list: sorted ids, one per line, ``\\n`` endings."""
    return "".join(f"{h}\n" for h in sorted(ids)).encode("ascii")


def sha256_hex(data: bytes) -> str:
    """Hex SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def write_split(cfg: dict[str, Any], split: str, ids: list[int]) -> str:
    """Write the id list (gitignored) and its hash (``sha256sum`` format). Returns the hash."""
    data = ids_bytes(ids)
    p = ids_path(cfg, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    digest = sha256_hex(data)
    h = hash_path(cfg, split)
    h.parent.mkdir(parents=True, exist_ok=True)
    h.write_text(f"{digest}  {p.name}\n", encoding="ascii", newline="\n")
    return digest


def load_split_ids(cfg: dict[str, Any], split: str) -> list[int]:
    """Read a split's id list and check it against the committed hash. Raises ConfigError."""
    if split not in SPLITS:
        raise ConfigError(f"unknown split {split!r}; allowed {list(SPLITS)}")
    p, h = ids_path(cfg, split), hash_path(cfg, split)
    if not p.exists():
        raise ConfigError(f"cohort id list not found: {p} (run select-cohort)")
    if not h.exists():
        raise ConfigError(f"cohort hash not found: {h} (run select-cohort, then commit config/cohort/*.sha256)")
    data = p.read_bytes()
    expected = h.read_text(encoding="ascii").split()[0]
    if sha256_hex(data) != expected:
        raise ConfigError(f"{p.name}: SHA-256 differs from {h}; the id list was changed")
    return [int(x) for x in data.decode("ascii").split()]


# --------------------------------------------------------------------------- DuckDB helpers


def connect() -> Any:
    """In-memory DuckDB connection."""
    import duckdb

    return duckdb.connect(":memory:")


def _q(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def read_csv_sql(path: Path) -> str:
    """DuckDB ``read_csv`` over one (gzipped) MIMIC CSV, every column as VARCHAR.

    Empty fields become NULL, as in ``load._read_table``; quoted fields may contain
    commas and newlines (note text).
    """
    return (f"read_csv({_q(path)}, header=true, all_varchar=true, delim=',', quote='\"', escape='\"', "
            "max_line_size=50000000)")


def find_note_file(note_root: Path) -> Path:
    """``note/discharge.csv.gz`` (or ``.csv``) under the MIMIC-IV-Note root."""
    hits = [note_root / "note" / f"{NOTE_TABLE}{ext}" for ext in (".csv.gz", ".csv")]
    hits = [h for h in hits if h.is_file()]
    if len(hits) != 1:
        raise SchemaError(f"{NOTE_TABLE}: expected exactly one of note/{NOTE_TABLE}.csv(.gz) under {note_root}")
    return hits[0]


def demo_subject_ids(demo_root: Path) -> set[int]:
    """Subject ids of the MIMIC-IV Demo."""
    p = demo_root / "demo_subject_id.csv"
    if not p.is_file():
        p = find_table_files(demo_root)["patients"]
    ids = pd.read_csv(p, usecols=["subject_id"], dtype=str)["subject_id"]
    return {int(x) for x in ids}


# --------------------------------------------------------------------------- selection


@dataclass
class CohortSelection:
    """Result of ``select_cohort``. ``counts`` is the ordered (step, count) list that was printed."""

    splits: dict[str, list[int]]
    counts: list[tuple[str, int]] = field(default_factory=list)


def select_cohort(
    mimic_root: Path,
    note_file: Path | None,
    exclude_subjects: set[int] | None,
    params: dict[str, Any],
    log: Callable[[str], None] = print,
) -> CohortSelection:
    """Run the selection described in the module docstring. Logs counts only.

    ``note_file=None`` skips the discharge-note filter; ``exclude_subjects=None``
    skips the Demo exclusion. Raises CohortError if too few subjects are eligible.
    """
    files = find_table_files(mimic_root)
    con = connect()
    counts: list[tuple[str, int]] = []

    def step(name: str, n: int) -> None:
        counts.append((name, n))
        log(f"  {name:<62} {n:>10,}")

    subjects = sorted(int(r[0]) for r in con.execute(
        f"SELECT DISTINCT CAST(subject_id AS BIGINT) FROM {read_csv_sql(files['patients'])}").fetchall())
    step("subjects in patients", len(subjects))
    k = round(params["pool_fraction"] * len(subjects))
    pool = sorted(random.Random(f"{params['seed']}:pool").sample(subjects, k))
    step(f"seeded pool ({params['pool_fraction']:g} of subjects, seed {params['seed']})", len(pool))
    if exclude_subjects is None:
        log("  (Demo exclusion skipped)")
    else:
        pool = [s for s in pool if s not in exclude_subjects]
        step("pool after removing Demo subjects", len(pool))

    con.execute("CREATE TEMP TABLE pool AS SELECT UNNEST(?::BIGINT[]) AS subject_id", [pool])
    con.execute(f"""
        CREATE TEMP TABLE adm AS
        SELECT CAST(subject_id AS BIGINT) AS subject_id, CAST(hadm_id AS BIGINT) AS hadm_id,
               CAST(admittime AS TIMESTAMP) AS admittime, CAST(dischtime AS TIMESTAMP) AS dischtime
        FROM {read_csv_sql(files['admissions'])}
        WHERE CAST(subject_id AS BIGINT) IN (SELECT subject_id FROM pool)""")
    n = con.execute("SELECT COUNT(*) FROM adm").fetchone()[0]
    step("admissions of pool subjects", n)

    con.execute("""
        CREATE TEMP TABLE cand AS SELECT * FROM adm
        WHERE date_diff('second', admittime, dischtime) / 86400.0 BETWEEN ? AND ?""",
                [float(params["min_los_days"]), float(params["max_los_days"])])
    step(f"hospital stay {params['min_los_days']:g}-{params['max_los_days']:g} days",
         con.execute("SELECT COUNT(*) FROM cand").fetchone()[0])

    con.execute("""
        CREATE OR REPLACE TEMP TABLE cand AS
        SELECT c.* FROM cand c
        WHERE (SELECT COUNT(*) FROM adm b WHERE b.subject_id = c.subject_id AND b.hadm_id <> c.hadm_id
               AND b.dischtime < c.admittime) >= ?""", [params["min_prior_admissions"]])
    step(f">= {params['min_prior_admissions']} prior admission(s)",
         con.execute("SELECT COUNT(*) FROM cand").fetchone()[0])

    log("  (scanning labevents; this reads the whole file)")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cand AS
        SELECT c.* FROM cand c JOIN (
            SELECT TRY_CAST(hadm_id AS BIGINT) AS hadm_id, COUNT(*) AS n
            FROM {read_csv_sql(files['labevents'])}
            WHERE TRY_CAST(hadm_id AS BIGINT) IN (SELECT hadm_id FROM cand)
            GROUP BY 1) l ON l.hadm_id = c.hadm_id
        WHERE l.n >= ?""", [params["min_lab_results"]])
    step(f">= {params['min_lab_results']} lab results (labevents rows with this hadm_id)",
         con.execute("SELECT COUNT(*) FROM cand").fetchone()[0])

    if note_file is None:
        log("  (discharge-note filter skipped)")
    else:
        log("  (scanning discharge notes; hadm_id column only)")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE cand AS
            SELECT c.* FROM cand c
            WHERE c.hadm_id IN (SELECT TRY_CAST(hadm_id AS BIGINT) FROM {read_csv_sql(note_file)})""")
        step("has a discharge note (MIMIC-IV-Note)", con.execute("SELECT COUNT(*) FROM cand").fetchone()[0])

    elig = con.execute("SELECT subject_id, hadm_id, admittime FROM cand").fetchall()
    con.close()
    latest: dict[int, tuple[Any, int]] = {}
    for subj, hadm, t in elig:
        if subj not in latest or (t, hadm) > latest[subj]:
            latest[subj] = (t, hadm)
    step("eligible subjects (latest eligible admission each)", len(latest))

    need = params["n_dev"] + params["n_test"]
    if len(latest) < need:
        raise CohortError(f"only {len(latest)} eligible subjects; n_dev + n_test = {need}")
    order = sorted(latest)
    random.Random(f"{params['seed']}:draw").shuffle(order)
    dev = [latest[s][1] for s in order[: params["n_dev"]]]
    test = [latest[s][1] for s in order[params["n_dev"]: need]]
    step("dev admissions", len(dev))
    step("test admissions", len(test))
    return CohortSelection(splits={"dev": dev, "test": test}, counts=counts)


# --------------------------------------------------------------------------- schema check


def schema_report(full_root: Path, demo_root: Path, extracted_root: Path | None,
                  note_file: Path | None) -> dict[str, dict[str, Any]]:
    """Column differences per table between full MIMIC-IV, the Demo and (if given) the extracted Parquet.

    Reads headers only. Per table: ``demo_not_in_full`` (blocks extraction),
    ``full_not_in_demo`` (dropped by extraction), ``builder_not_in_full`` (columns
    ``build-units`` loads that are missing) and, when extracted files exist,
    ``extracted_differs`` (their columns are not exactly the Demo's, in order).
    For the note table: ``required_not_in_full`` and its ``columns``.
    """
    full, demo = find_table_files(full_root), find_table_files(demo_root)
    extracted = find_table_files(extracted_root) if extracted_root is not None else {}
    out: dict[str, dict[str, Any]] = {}
    for t in EXPECTED_COLUMNS:
        f, d = read_header(full[t]), read_header(demo[t])
        builder = EXPECTED_COLUMNS[t] + EXTRA_COLUMNS.get(t, [])
        r: dict[str, Any] = {
            "demo_not_in_full": [c for c in d if c not in f],
            "full_not_in_demo": [c for c in f if c not in d],
            "builder_not_in_full": [c for c in builder if c not in f],
        }
        if t in extracted:
            r["extracted_differs"] = read_header(extracted[t]) != d
        out[t] = r
    if note_file is not None:
        h = read_header(note_file)
        out[NOTE_TABLE] = {"required_not_in_full": [c for c in NOTE_COLUMNS if c not in h], "columns": h}
    return out


def schema_ok(report: dict[str, dict[str, Any]]) -> bool:
    """True if extraction keeps every Demo column and the builder finds every column it loads."""
    return not any(r.get("demo_not_in_full") or r.get("builder_not_in_full") or r.get("required_not_in_full")
                   or r.get("extracted_differs") for r in report.values())


# --------------------------------------------------------------------------- extraction


def _order_sql(table: str) -> str:
    keys = ORDER_BY[table]
    return ", ".join(k if k in _TEXT_KEYS else f"TRY_CAST({k} AS BIGINT), {k}" for k in keys)


def _cols_sql(cols: list[str]) -> str:
    return ", ".join('"' + c.replace('"', '""') + '"' for c in cols)


def _copy_parquet(con: Any, select_sql: str, dest: Path) -> int:
    """``COPY`` a query to Parquet (written to a temp name, then renamed). Returns the row count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    con.execute(f"COPY ({select_sql}) TO {_q(tmp)} (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(dest)
    return int(con.execute(f"SELECT COUNT(*) FROM read_parquet({_q(dest)})").fetchone()[0])


def extract_cohort(
    full_root: Path,
    demo_root: Path,
    hadm_ids: list[int],
    evidence_out: Path,
    note_file: Path | None,
    reference_out: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Copy the cohort's rows into Parquet (see module docstring). Logs counts only.

    ``note_file=None`` skips the reference notes. Raises SchemaError if a Demo column
    is missing from the full data, CohortError if a selected admission is not in the
    full ``admissions`` table.
    """
    full, demo = find_table_files(full_root), find_table_files(demo_root)
    columns = {t: read_header(demo[t]) for t in EXPECTED_COLUMNS}
    missing = {t: [c for c in columns[t] if c not in read_header(full[t])] for t in EXPECTED_COLUMNS}
    missing = {t: m for t, m in missing.items() if m}
    if missing:
        raise SchemaError(f"Demo columns missing from the full data (see cohort-schema-check): {missing}")
    con = connect()
    target = sorted(set(hadm_ids))
    con.execute("CREATE TEMP TABLE target AS SELECT UNNEST(?::BIGINT[]) AS hadm_id", [target])
    con.execute(f"""
        CREATE TEMP TABLE subj AS
        SELECT DISTINCT TRY_CAST(subject_id AS BIGINT) AS subject_id, TRY_CAST(hadm_id AS BIGINT) AS hadm_id
        FROM {read_csv_sql(full['admissions'])}
        WHERE TRY_CAST(hadm_id AS BIGINT) IN (SELECT hadm_id FROM target)""")
    n_found = con.execute("SELECT COUNT(DISTINCT hadm_id) FROM subj").fetchone()[0]
    if n_found != len(target):
        raise CohortError(f"{len(target) - n_found} of {len(target)} selected admissions are not in admissions")
    n_subj = int(con.execute("SELECT COUNT(DISTINCT subject_id) FROM subj").fetchone()[0])
    log(f"  {'selected admissions':<44} {len(target):>12,}")
    log(f"  {'their patients':<44} {n_subj:>12,}")
    counts: dict[str, int] = {}
    for t in EXPECTED_COLUMNS:
        if t in DICTIONARY_TABLES:
            where = ""
        else:
            where = "WHERE TRY_CAST(subject_id AS BIGINT) IN (SELECT subject_id FROM subj)"
        sql = f"SELECT {_cols_sql(columns[t])} FROM {read_csv_sql(full[t])} {where} ORDER BY {_order_sql(t)}"
        counts[t] = _copy_parquet(con, sql, evidence_out / "hosp" / f"{t}.parquet")
        label = t + (" (all rows)" if t in DICTIONARY_TABLES else "")
        log(f"  {label:<44} {counts[t]:>12,} rows")
    notes: dict[str, int] = {}
    if note_file is None:
        log("  (discharge notes skipped)")
    else:
        h = read_header(note_file)
        lacking = [c for c in NOTE_COLUMNS if c not in h]
        if lacking:
            raise SchemaError(f"{note_file.name}: missing column(s) {lacking}")
        dest = reference_out / f"{NOTE_TABLE}.parquet"
        sql = (f"SELECT {_cols_sql(h)} FROM {read_csv_sql(note_file)} "
               "WHERE TRY_CAST(hadm_id AS BIGINT) IN (SELECT hadm_id FROM target) "
               "ORDER BY TRY_CAST(hadm_id AS BIGINT), TRY_CAST(note_seq AS BIGINT), note_id")
        notes["note rows"] = _copy_parquet(con, sql, dest)
        per = [k for (k,) in con.execute(f"""
            SELECT COUNT(n.h) FROM target t
            LEFT JOIN (SELECT TRY_CAST(hadm_id AS BIGINT) AS h FROM read_parquet({_q(dest)})) n ON n.h = t.hadm_id
            GROUP BY t.hadm_id""").fetchall()]
        notes["admissions without a note"] = sum(1 for k in per if k == 0)
        notes["admissions with one note"] = sum(1 for k in per if k == 1)
        notes["admissions with several notes"] = sum(1 for k in per if k > 1)
        notes["references written"] = write_references(con, dest, reference_out / "references.csv")
        for k, v in notes.items():
            log(f"  {'discharge: ' + k:<44} {v:>12,}")
    con.close()
    return {"n_admissions": len(target), "n_subjects": n_subj, "rows": counts, "notes": notes}


def write_references(con: Any, notes_parquet: Path, dest: Path) -> int:
    """``references.csv`` (``hadm_id,reference``: the ``text-metrics`` format) from the extracted notes.

    Several notes of one admission are joined by a blank line in ``note_seq`` order.
    Returns the number of admissions written.
    """
    import csv

    rows = con.execute(f"""
        SELECT TRY_CAST(hadm_id AS BIGINT) AS h,
               string_agg(text, chr(10) || chr(10) ORDER BY TRY_CAST(note_seq AS BIGINT), note_id)
        FROM read_parquet({_q(notes_parquet)})
        WHERE text IS NOT NULL GROUP BY h ORDER BY h""").fetchall()
    tmp = dest.with_name(dest.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hadm_id", "reference"])
        w.writerows(rows)
    tmp.replace(dest)
    return len(rows)
