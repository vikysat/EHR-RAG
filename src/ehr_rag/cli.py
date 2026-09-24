"""Command-line entrypoint: ``python -m ehr_rag.cli <command> [options]``."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import pandas as pd
    from pathlib import Path

import yaml

from ehr_rag.config import ConfigError, load_config, public_config

DEFAULT_CONFIG = "config/default.yaml"


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Load and validate the config, then print the resolved version."""
    cfg = load_config(args.config)
    print(yaml.safe_dump(public_config(cfg), sort_keys=False, allow_unicode=True))
    print(f"OK: {args.config} is valid (project root: {cfg['_project_root']})")
    return 0


def _print_table(df: "pd.DataFrame") -> None:
    import pandas as pd
    from pathlib import Path

    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 250,
                           "display.max_colwidth", 80):
        print(df.to_string(index=False))


def cmd_inspect_data(args: argparse.Namespace) -> int:
    """Phase 1 data inspection report."""
    from ehr_rag.data.load import (
        EXPECTED_COLUMNS, check_columns, data_root, directory_tree, find_table_files, head_rows, load_tables,
    )
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    run = start_run(cfg, "inspect-data", command="inspect-data")
    root = data_root(cfg)
    print("== Data directory tree")
    print("\n".join(directory_tree(root)))
    files = find_table_files(root)
    print("\n== Table files")
    for t, p in files.items():
        print(f"  {t}: {p.relative_to(root)}")
    report = check_columns(files)
    print("\n== Column check (expected vs real header)")
    mismatch = False
    for t, r in report.items():
        status = "OK" if not r["missing"] else f"MISSING {r['missing']}"
        mismatch |= bool(r["missing"])
        print(f"  {t}: {status}; extra columns not used: {r['extra']}")
    for t, p in files.items():
        print(f"\n== {t}: first 5 rows (expected columns only)")
        _print_table(head_rows(p)[[c for c in EXPECTED_COLUMNS[t] if c not in report[t]["missing"]]])
    if mismatch:
        run.finish(status="schema_mismatch", column_report=report)
        print("\nSTOP: column mismatch against SPEC.")
        return 3
    tables = load_tables(cfg)
    print("\n== Row counts")
    for t, n in tables.row_counts.items():
        print(f"  {t}: {n:,}")
    print("\n== Lab linkage")
    for k, v in tables.lab_link_stats.items():
        print(f"  {k}: {v:,}")
    run.finish(column_report=report, row_counts=tables.row_counts, lab_link_stats=tables.lab_link_stats)
    print(f"\nrun: {run.path}")
    return 0


def cmd_select_admissions(args: argparse.Namespace) -> int:
    """Phase 1 admission metrics and proposal."""
    from ehr_rag.config import project_root, resolve_path
    from ehr_rag.data.load import load_tables
    from ehr_rag.data.select_admissions import apply_selection, compute_metrics, write_outputs

    cfg = load_config(args.config)
    from ehr_rag.runs import start_run

    run = start_run(cfg, "select-admissions", command="select-admissions")
    tables = load_tables(cfg)
    sel = apply_selection(compute_metrics(tables), cfg["selection"])
    cand = resolve_path(cfg, cfg["data"]["derived"]) / "admission_candidates.csv"
    yml = project_root(cfg) / "config" / "selected_admissions.yaml"
    items = write_outputs(sel, cand, yml)
    n_elig = int(sel["eligible"].sum())
    n_elig_subj = int(sel.loc[sel["eligible"], "subject_id"].nunique())
    print(f"admissions: {len(sel)}; eligible: {n_elig} (from {n_elig_subj} patients); "
          f"eligible with >=1 prior admission: {int((sel['eligible'] & (sel['n_prior_admissions'] >= 1)).sum())}")
    print(f"proposed: {len(items)} (selection.n_admissions = {cfg['selection']['n_admissions']})\n")
    import pandas as pd
    from pathlib import Path

    _print_table(pd.DataFrame(items))
    print(f"\nwrote {cand}\nwrote {yml}")
    sel.to_csv(run.path / "admission_candidates.csv", index=False)
    run.finish(n_admissions=len(sel), n_eligible=n_elig, proposed=[i["hadm_id"] for i in items])
    print(f"run: {run.path}")
    return 0


def _selected_ids(cfg: dict) -> list[int]:
    from ehr_rag.config import project_root
    from ehr_rag.data.select_admissions import load_selected_hadm_ids

    return load_selected_hadm_ids(project_root(cfg) / "config" / "selected_admissions.yaml")


def _units_dir(cfg: dict) -> "Path":
    from ehr_rag.config import resolve_path
    from ehr_rag.evidence.build_units import units_dir

    return units_dir(resolve_path(cfg, cfg["data"]["derived"]))


def cmd_build_units(args: argparse.Namespace) -> int:
    """Phase 2: build evidence units + headers for the selected admissions."""
    import json
    import random

    import pandas as pd
    from pathlib import Path

    from ehr_rag.data.load import load_tables
    from ehr_rag.evidence.build_units import build_all
    from ehr_rag.evidence.schema import UNIT_TYPES, read_units
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    seed = cfg["eval"]["seed"]
    run = start_run(cfg, "build-units", command="build-units", seed=seed)
    ids = _selected_ids(cfg)
    out = _units_dir(cfg)
    counts, dropped = build_all(load_tables(cfg), ids, cfg["units"], out)
    counts.to_csv(out / "unit_counts.csv", index=False)
    counts.to_csv(run.path / "unit_counts.csv", index=False)
    dropped_df = pd.DataFrame(dropped, columns=["hadm_id", "unit_type", "reason", "detail", "n_rows"])
    dropped_df.to_csv(out / "dropped.csv", index=False)
    dropped_df.to_csv(run.path / "dropped.csv", index=False)

    print("== Unit counts by type per admission")
    _print_table(pd.concat([counts, pd.DataFrame([{"hadm_id": "TOTAL", **counts.drop(columns="hadm_id").sum()}])]))

    print(f"\n== Example units (3 per type, different admissions, seed {seed})")
    rng = random.Random(seed)
    all_units = {h: read_units(out / f"{h}.jsonl") for h in ids}
    for t in UNIT_TYPES:
        having = [h for h in ids if any(u.unit_type == t for u in all_units[h])]
        for h in sorted(rng.sample(having, min(3, len(having)))):
            u = rng.choice([u for u in all_units[h] if u.unit_type == t])
            print(f"\n--- {t} (hadm {h})")
            print(u.to_json())
    print(f"\n== Example header (hadm {ids[0]})")
    print(json.dumps(json.loads((out / f"{ids[0]}_header.json").read_text(encoding="utf-8")), indent=2))

    print("\n== Dropped / lossy handling (summed over selected admissions)")
    if len(dropped_df):
        g = dropped_df.groupby(["unit_type", "reason"]).agg(events=("hadm_id", "size"), rows=("n_rows", "sum"))
        _print_table(g.reset_index())
    else:
        print("  nothing dropped")
    print(f"\nwrote {out}\nrun: {run.path}")
    run.finish(hadm_ids=ids, n_units=int(counts["total"].sum()), n_dropped_events=len(dropped_df))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m ehr_rag.cli",
        description="EHR-aware clinical RAG with citation-faithfulness probes.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    def add(name: str, func: Callable[[argparse.Namespace], int], help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--config", default=DEFAULT_CONFIG, help=f"config file (default: {DEFAULT_CONFIG})")
        p.set_defaults(func=func)
        return p

    add("validate-config", cmd_validate_config, "Validate the config and print it resolved.")
    add("inspect-data", cmd_inspect_data,
        "Phase 1: print data tree, header check, 5 rows per table, row counts, lab linkage.")
    add("select-admissions", cmd_select_admissions,
        "Phase 1: compute admission metrics, write candidates CSV and proposed selected_admissions.yaml.")
    add("build-units", cmd_build_units,
        "Phase 2: build evidence units and headers for admissions in config/selected_admissions.yaml.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    from ehr_rag.data.load import SchemaError

    try:
        return args.func(args)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2
    except SchemaError as e:
        print(f"STOP - SCHEMA MISMATCH: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
