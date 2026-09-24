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


def _load_all_units(cfg: dict, ids: list[int]) -> dict:
    from ehr_rag.evidence.schema import read_units

    out = _units_dir(cfg)
    return {h: read_units(out / f"{h}.jsonl") for h in ids}


def _encoders(cfg: dict) -> "object":
    from ehr_rag.config import project_root
    from ehr_rag.retrieval.retrieve import Encoders

    return Encoders.from_config(cfg, project_root(cfg) / "cache")


def cmd_retrieve(args: argparse.Namespace) -> int:
    """Phase 3: run section retrieval for all selected admissions."""
    from ehr_rag.config import SECTIONS
    from ehr_rag.retrieval.retrieve import retrieve_admission
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    run = start_run(cfg, "retrieve", command="retrieve")
    ids = [int(h) for h in args.hadm_ids] if args.hadm_ids else _selected_ids(cfg)
    units = _load_all_units(cfg, ids)
    enc = _encoders(cfg)
    results = {}
    for h in ids:
        res = retrieve_admission(units[h], enc, cfg["retrieval"])
        run.write_json(f"retrieval/{h}.json", res.to_json())
        results[h] = res
    for h in ids[: args.show]:
        res = results[h]
        by_id = {u.unit_id: u for u in units[h]}
        print(f"\n{'=' * 100}\nhadm_id {h}  |  primary_dx: {res.primary_dx}  |  "
              f"{len(units[h])} units in pool  |  evidence: {len(res.evidence)} units")
        for s in SECTIONS:
            print(f"\n--- {s}")
            for q in res.queries[s]:
                print(f"  query: {q.query!r} -> top: {q.final()[: cfg['retrieval']['top_k_per_query']]}")
            for uid in res.sections[s]:
                u = by_id[uid]
                print(f"  [{uid}] ({u.unit_type}) {u.text}")
        if res.truncated_by_total_cap:
            print(f"\n  dropped by max_units_total: {res.truncated_by_total_cap}")
        print(f"\n  evidence order: {res.evidence}")
    trunc = {h: len(r.truncated_by_total_cap) for h, r in results.items()}
    print(f"\nunits dropped by max_units_total per admission: {trunc}")
    print(f"run: {run.path}")
    revs = enc.revisions()
    print(f"model revisions: {revs}")
    run.finish(hadm_ids=ids, rerank=cfg["retrieval"]["rerank"], model_revisions=revs)
    return 0


def cmd_export_relevance(args: argparse.Namespace) -> int:
    """Phase 3: export pooled top-10 per (admission, query) pair for manual labeling."""
    from ehr_rag.config import SECTIONS
    from ehr_rag.retrieval.relevance import POOL_DEPTH, POOLING_NOTE, sample_pairs, write_export
    from ehr_rag.retrieval.retrieve import AdmissionRetriever, primary_dx, section_queries
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    seed = cfg["eval"]["seed"]
    run = start_run(cfg, "export-relevance", command="export-relevance", seed=seed)
    ids = _selected_ids(cfg)
    units = _load_all_units(cfg, ids)
    all_pairs = []
    for h in ids:
        qs = section_queries(cfg["retrieval"]["queries"], primary_dx(units[h]))
        all_pairs.extend((h, q) for s in SECTIONS for q in qs[s])
    all_pairs = sorted(set(all_pairs))
    pairs = sample_pairs(all_pairs, cfg["retrieval"]["relevance_export_pairs"], seed)
    enc = _encoders(cfg)
    retrievers: dict = {}
    items = []
    for h, q in pairs:
        if h not in retrievers:
            retrievers[h] = AdmissionRetriever(units[h], enc, cfg["retrieval"]["rrf_k"])
        r = retrievers[h].search(q)
        rankings = {m: [d for d, _ in getattr(r, m)[:POOL_DEPTH]] for m in ("bm25", "dense", "fused")}
        if r.reranked is not None:
            rankings["reranked"] = [d for d, _ in r.reranked[:POOL_DEPTH]]
        items.append({"hadm_id": h, "query": q, "rankings": rankings})
    texts = {h: {u.unit_id: u.text for u in units[h]} for h in ids}
    csv_path, side = write_export(run.path, items, texts)
    n_rows = sum(1 for _ in csv_path.open(encoding="utf-8")) - 1
    print(f"sampled {len(pairs)} of {len(all_pairs)} (admission, query) pairs with seed {seed}")
    for h, q in pairs:
        print(f"  {h}: {q}")
    print(f"\nwrote {csv_path} ({n_rows} rows to label; fill 'relevant' with 0/1)\nwrote {side}")
    print(POOLING_NOTE)
    revs = enc.revisions()
    print(f"model revisions: {revs}")
    run.finish(pairs=[list(p) for p in pairs], n_rows=n_rows, model_revisions=revs)
    return 0


def cmd_retrieval_metrics(args: argparse.Namespace) -> int:
    """Phase 3: Recall@5 and MRR per method from a labeled relevance CSV."""
    import json
    from pathlib import Path

    from ehr_rag.retrieval.relevance import compute_metrics, read_labels
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    labels_path = Path(args.labels)
    rankings_path = Path(args.rankings) if args.rankings else labels_path.parent / "relevance_rankings.json"
    rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
    metrics = compute_metrics(read_labels(labels_path), rankings)
    run = start_run(cfg, "retrieval-metrics", command="retrieval-metrics")
    run.write_json("retrieval_metrics.json", {**metrics, "labels": str(labels_path),
                                              "rankings": str(rankings_path)})
    print(f"{'method':<10} {'Recall@5':>9} {'MRR':>7} {'pairs':>6}")
    for m, v in metrics["methods"].items():
        print(f"{m:<10} {v['recall@5']:>9.3f} {v['mrr']:>7.3f} {v['n_pairs']:>6}")
    if metrics["skipped_no_relevant"]:
        print(f"skipped {len(metrics['skipped_no_relevant'])} pair(s) with no relevant unit in the pool")
    print(metrics["note"])
    run.finish()
    print(f"run: {run.path}")
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
    p = add("retrieve", cmd_retrieve, "Phase 3: per-section hybrid retrieval for the selected admissions.")
    p.add_argument("--hadm-ids", nargs="*", help="restrict to these admissions (default: all selected)")
    p.add_argument("--show", type=int, default=2, help="print sections for the first N admissions (default 2)")
    add("export-relevance", cmd_export_relevance,
        "Phase 3: write relevance_to_label.csv (pooled top-10 of BM25/dense/fused) for manual labeling.")
    p = add("retrieval-metrics", cmd_retrieval_metrics,
            "Phase 3: Recall@5 and MRR per method from a labeled relevance CSV (pooled relevance).")
    p.add_argument("--labels", required=True, help="labeled relevance_to_label.csv")
    p.add_argument("--rankings", help="relevance_rankings.json (default: next to --labels)")
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
