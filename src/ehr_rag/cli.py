"""Command-line entrypoint: ``python -m ehr_rag.cli <command> [options]``."""

from __future__ import annotations

import argparse
import os
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
    from ehr_rag.runs import checker_hashes

    h = checker_hashes(cfg) or {}
    print(f"checker: rules {h.get('rules_sha256')}, lexicon {h.get('lexicon_sha256')}, rewrite prompt "
          f"{h.get('rewrite_prompt_sha256')}; combined sha256 {h.get('combined_sha256')} "
          "(pass it as --checker-sha256 to freeze the checker for a run)")
    print(f"OK: {args.config} is valid (project root: {cfg['_project_root']})")
    return 0


def _quiet(cfg: dict) -> bool:
    """With credentialed data (``data.source: full``) commands print counts only, never rows or text."""
    return cfg["data"]["source"] == "full"


QUIET_NOTE = "(data.source is 'full': rows and text are not printed{})"


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


def cmd_select_cohort(args: argparse.Namespace) -> int:
    """Seeded dev/test cohort from full MIMIC-IV (DuckDB); prints counts only."""
    from ehr_rag.config import resolve_path
    from ehr_rag.data.cohort import (
        SPLITS, CohortError, demo_subject_ids, find_note_file, hash_path, select_cohort, write_split,
    )
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    co = dict(cfg["cohort"])
    for key in ("pool_fraction", "n_dev", "n_test"):
        if getattr(args, key) is not None:
            co[key] = getattr(args, key)
    existing = [hash_path(cfg, s) for s in SPLITS if hash_path(cfg, s).exists()]
    if existing and not args.overwrite:
        raise ConfigError(f"cohort hashes already exist ({', '.join(p.name for p in existing)}); "
                          "the cohort is fixed. Pass --overwrite only to replace it on purpose.")
    note_file = None if args.skip_note_filter else find_note_file(resolve_path(cfg, co["note_root"]))
    exclude = None if args.no_demo_exclusion else demo_subject_ids(resolve_path(cfg, co["demo_root"]))
    run = start_run(cfg, "select-cohort", command="select-cohort", seed=co["seed"])
    print("== Cohort selection (counts only)")
    try:
        sel = select_cohort(resolve_path(cfg, co["mimic_root"]), note_file, exclude, co)
    except CohortError as e:
        print(f"STOP - {e}", file=sys.stderr)
        run.finish(status="failed", error=str(e))
        return 4
    hashes = {s: write_split(cfg, s, sel.splits[s]) for s in SPLITS}
    print("\nwrote the id lists (gitignored) and their hashes:")
    for s in SPLITS:
        print(f"  {hash_path(cfg, s)}  sha256 {hashes[s]}")
    run.finish(params=co, counts=sel.counts, sha256=hashes, skip_note_filter=args.skip_note_filter,
               demo_exclusion=not args.no_demo_exclusion)
    print(f"run: {run.path}")
    return 0


def _split_ids(cfg: dict, splits: list[str]) -> list[int]:
    """Admission ids of the given cohort splits, each checked against its committed hash."""
    from ehr_rag.data.cohort import load_split_ids

    ids: list[int] = []
    for s in dict.fromkeys(splits):
        ids.extend(h for h in load_split_ids(cfg, s) if h not in ids)
    return ids


def cmd_extract_cohort(args: argparse.Namespace) -> int:
    """Copy the cohort's rows from full MIMIC-IV into Parquet; prints counts only."""
    from ehr_rag.config import resolve_path
    from ehr_rag.data.cohort import CohortError, evidence_root, extract_cohort, find_note_file, reference_dir
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    co = cfg["cohort"]
    ids = _split_ids(cfg, args.splits)
    note_file = None if args.skip_notes else find_note_file(resolve_path(cfg, co["note_root"]))
    run = start_run(cfg, "extract-cohort", command="extract-cohort")
    print(f"== Cohort extraction, splits {args.splits} (counts only)")
    try:
        res = extract_cohort(resolve_path(cfg, co["mimic_root"]), resolve_path(cfg, co["demo_root"]), ids,
                             evidence_root(cfg), note_file, reference_dir(cfg))
    except CohortError as e:
        print(f"STOP - {e}", file=sys.stderr)
        run.finish(status="failed", error=str(e))
        return 4
    print(f"\nwrote {evidence_root(cfg)} (data.root for build-units)")
    if note_file is not None:
        print(f"wrote {reference_dir(cfg)} (evaluation references only; never evidence)")
    run.finish(splits=args.splits, **res)
    print(f"run: {run.path}")
    return 0


def cmd_cohort_schema_check(args: argparse.Namespace) -> int:
    """Column differences: full MIMIC-IV vs the Demo (and the extracted Parquet, if present)."""
    from ehr_rag.config import resolve_path
    from ehr_rag.data.cohort import evidence_root, find_note_file, schema_ok, schema_report

    cfg = load_config(args.config)
    co = cfg["cohort"]
    ext = evidence_root(cfg)
    ext_root = ext if (ext / "hosp").is_dir() else None
    note_file = None if args.skip_notes else find_note_file(resolve_path(cfg, co["note_root"]))
    rep = schema_report(resolve_path(cfg, co["mimic_root"]), resolve_path(cfg, co["demo_root"]), ext_root, note_file)
    print("== Column check: full MIMIC-IV vs the Demo (headers only)")
    for t, r in rep.items():
        print(f"\n{t}")
        for k, v in r.items():
            if k == "extracted_differs":
                v = "DIFFERENT from the Demo" if v else "same as the Demo"
                k = "extracted Parquet columns"
            print(f"  {k:<26} {v if v != [] else '-'}")
    if ext_root is None:
        print(f"\n(no extracted Parquet under {ext}; run extract-cohort to include it)")
    ok = schema_ok(rep)
    print("\nOK: every Demo column and every builder column exists in the full data" if ok
          else "\nSTOP: schema differences block extraction or build-units (see above)")
    return 0 if ok else 3


def _selected_ids(cfg: dict, args: argparse.Namespace | None = None) -> list[int]:
    """Target admissions: ``--hadm-ids``, else ``--split`` (hash-checked cohort lists), else
    ``config/selected_admissions.yaml``."""
    from ehr_rag.config import project_root
    from ehr_rag.data.select_admissions import load_selected_hadm_ids

    hadm_ids = getattr(args, "hadm_ids", None)
    split = getattr(args, "split", None)
    if hadm_ids and split:
        raise ConfigError("give either --hadm-ids or --split, not both")
    if hadm_ids:
        return [int(h) for h in hadm_ids]
    if split:
        return _split_ids(cfg, split)
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
    ids = _selected_ids(cfg, args)
    out = _units_dir(cfg)
    counts, dropped = build_all(load_tables(cfg), ids, cfg["units"], out)
    counts.to_csv(out / "unit_counts.csv", index=False)
    counts.to_csv(run.path / "unit_counts.csv", index=False)
    dropped_df = pd.DataFrame(dropped, columns=["hadm_id", "unit_type", "reason", "detail", "n_rows"])
    dropped_df.to_csv(out / "dropped.csv", index=False)
    dropped_df.to_csv(run.path / "dropped.csv", index=False)

    print("== Unit counts by type per admission")
    _print_table(pd.concat([counts, pd.DataFrame([{"hadm_id": "TOTAL", **counts.drop(columns="hadm_id").sum()}])]))

    if _quiet(cfg):
        print("\n" + QUIET_NOTE.format(f"; units and headers are in {out}"))
    else:
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
    from ehr_rag.retrieval.retrieve import evidence_record, retrieve_admission
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    run = start_run(cfg, "retrieve", command="retrieve")
    ids = _selected_ids(cfg, args)
    units = _load_all_units(cfg, ids)
    enc = _encoders(cfg)
    results = {}
    for h in ids:
        res = retrieve_admission(units[h], enc, cfg["retrieval"])
        run.write_json(f"retrieval/{h}.json", evidence_record(res, units[h]))
        results[h] = res
    show = args.show
    if _quiet(cfg) and show:
        print(QUIET_NOTE.format(f"; retrieval results are in {run.path / 'retrieval'}"))
        show = 0
    for h in ids[:show]:
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
    ids = _selected_ids(cfg, args)
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


def _target_ids(cfg: dict, args: argparse.Namespace) -> list[int]:
    ids = _selected_ids(cfg, args)
    if args.limit is not None:
        ids = ids[: args.limit]
    if not ids:
        raise ConfigError("no admissions selected")
    return ids


def shard(ids: list[int], shard_size: int | None, shard_index: int | None) -> tuple[list[int], dict]:
    """Contiguous shard ``shard_index`` (0-based) of ``shard_size`` admissions; no sharding if both are None."""
    if shard_size is None and shard_index is None:
        return ids, {}
    if shard_size is None or shard_index is None:
        raise ConfigError("give both --shard-size and --shard-index")
    if shard_size < 1:
        raise ConfigError("--shard-size must be >= 1")
    n_shards = -(-len(ids) // shard_size)
    if not 0 <= shard_index < n_shards:
        raise ConfigError(f"--shard-index {shard_index} out of range: {len(ids)} admissions in shards of "
                          f"{shard_size} give {n_shards} shard(s), indices 0..{n_shards - 1}")
    part = ids[shard_index * shard_size: (shard_index + 1) * shard_size]
    return part, {"shard_index": shard_index, "shard_size": shard_size, "n_shards": n_shards,
                  "n_admissions_all_shards": len(ids)}


def cmd_generate(args: argparse.Namespace) -> int:
    """Phase 4: generate reports for the configured conditions."""
    from ehr_rag.config import GENERATE_CONDITIONS, validate_for_generation
    from ehr_rag.generation.backends import BackendError, gpu_name, make_backend
    from ehr_rag.generation.generate import ConfirmationRequired, GenerationError, run_generation
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    validate_for_generation(cfg)
    gen = cfg["generation"]
    conditions = args.conditions or list(gen["conditions"])
    bad = [c for c in conditions if c not in GENERATE_CONDITIONS]
    if bad:
        raise ConfigError(f"unknown condition(s) {bad}; allowed {list(GENERATE_CONDITIONS)}")
    ids, shard_info = shard(_target_ids(cfg, args), args.shard_size, args.shard_index)
    from ehr_rag.runs import check_frozen_checker

    check_frozen_checker(cfg, args.checker_sha256)
    run = start_run(cfg, "generate", command="generate", run_id=args.run_id)
    run.meta.update(backend=gen["backend"], model=gen["model"], conditions=conditions, hadm_ids=ids,
                    retrieval_run=args.retrieval_run, prompt_version=gen["prompt_version"],
                    gpu_name=gpu_name(), hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"), split=args.split,
                    **shard_info)
    run.write_json("run_meta.json", run.meta)
    print(f"run: {run.path}")
    if shard_info:
        print(f"shard {shard_info['shard_index']} of {shard_info['n_shards']} (0-based): {len(ids)} of "
              f"{shard_info['n_admissions_all_shards']} admissions")
    try:
        backend = make_backend(gen)
        summary = run_generation(cfg, run, backend, hadm_ids=ids, conditions=conditions,
                                 retrieval_run=args.retrieval_run, yes=args.yes, no_cache=args.no_cache)
    except ConfirmationRequired as e:
        print(f"NOT RUN: {e}")
        run.finish(status="not_confirmed")
        return 5
    except (BackendError, GenerationError, OSError, ValueError) as e:
        print(f"STOP - generation failed: {e}", file=sys.stderr)
        run.finish(status="failed", error=repr(e))
        return 4
    except BaseException as e:
        run.finish(status="failed", error=repr(e))
        raise
    print("\n== Format status per condition")
    print(f"{'condition':<14} {'ok':>4} {'retry_ok':>9} {'format_error':>13} {'skipped':>8}")
    for c, n in summary["counts"].items():
        print(f"{c:<14} {n['ok']:>4} {n['ok_after_retry']:>9} {n['format_error']:>13} {n['skipped']:>8}")
    print(f"\nmodel calls: {summary['n_api_calls']}; cache hits: {summary['n_cache_hits']}; "
          f"reported model id(s): {summary['reported_models']}; revision: {summary['model_revision']}")
    if summary["enforcement"]:
        _print_enforcement(summary["enforcement"])
    run.finish(model_revision=summary["model_revision"], reported_model_ids=summary["reported_models"],
               format_counts=summary["counts"], n_api_calls=summary["n_api_calls"],
               n_cache_hits=summary["n_cache_hits"], skipped_full_context=summary["skipped"],
               enforcement=summary["enforcement"])
    print(f"run: {run.path}")
    return 0


def _print_enforcement(totals: dict) -> None:
    """Counts only: outcomes per enforced condition and how much content was kept."""
    print("\n== Enforcement (lines: kept / repaired / retreated / dropped; numbers kept)")
    for c, t in totals.items():
        share = "n/a" if not t["n_numbers_before"] else f"{t['n_numbers_after'] / t['n_numbers_before']:.3f}"
        print(f"{c:<22} lines {t['n_lines']:>5}  kept {t['n_kept']:>5}  repaired {t['n_repaired']:>4}  "
              f"retreated {t['n_retreated']:>4}  dropped {t['n_dropped']:>4}  "
              f"numbers {t['n_numbers_after']}/{t['n_numbers_before']} ({share})  rewrite calls {t['n_rewrite_calls']}")


def cmd_enforce_filter(args: argparse.Namespace) -> int:
    """Filter-only enforcement of existing reports (CPU, no model): drop lines that fail the checker."""
    from ehr_rag.checker.enforce import filter_generation_runs
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    gen_dirs = [_run_dir(cfg, g, "generation") for g in args.generation_run]
    from ehr_rag.runs import check_frozen_checker

    check_frozen_checker(cfg, args.checker_sha256)
    run = start_run(cfg, "enforce-filter", command="enforce-filter", run_id=args.run_id)
    run.meta.update(generation_runs=args.generation_run, conditions=args.conditions)
    run.write_json("run_meta.json", run.meta)
    print(f"run: {run.path}")
    try:
        totals = filter_generation_runs(cfg, gen_dirs, run.path, args.conditions,
                                        log=(lambda s: None) if _quiet(cfg) else print)
    except (OSError, ValueError) as e:
        print(f"STOP - {e}", file=sys.stderr)
        run.finish(status="failed", error=repr(e))
        return 4
    _print_enforcement(totals)
    run.finish(enforcement=totals)
    print(f"evaluate it with: evaluate --generation-run {' '.join(args.generation_run)} {run.run_id}")
    return 0


def cmd_check_citations(args: argparse.Namespace) -> int:
    """Citation-checker statistics for generation run(s); separate from `evaluate`. Counts only."""
    from ehr_rag.checker.report import run_check
    from ehr_rag.checker.rules import ERROR_TYPES
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    gen_dirs = [_run_dir(cfg, g, "generation") for g in args.generation_run]
    from ehr_rag.runs import check_frozen_checker

    check_frozen_checker(cfg, args.checker_sha256)
    run = start_run(cfg, "check-citations", command="check-citations", seed=cfg["eval"]["seed"], run_id=args.run_id)
    run.meta.update(generation_runs=args.generation_run)
    run.write_json("run_meta.json", run.meta)
    print(f"run: {run.path}")
    try:
        res = run_check(cfg, gen_dirs, run.path, args.review_n, cfg["eval"]["seed"])
    except (OSError, ValueError) as e:
        print(f"STOP - {e}", file=sys.stderr)
        run.finish(status="failed", error=repr(e))
        return 4
    h = run.meta["checker"]
    print(f"checker rules sha256 {h['rules_sha256']}; lexicon sha256 {h['lexicon_sha256']}; "
          f"combined {h['combined_sha256']}")
    print("\n== Status per condition (generated text, before enforcement; lines)")
    print(f"{'condition':<14} {'scope':<27} {'lines':>6} {'pass':>6} {'fail':>6} {'unchecked':>9} {'checkable':>9}")
    for r in res["profile"]:
        if r["scope"] in ("ALL", "History+Findings", "Impression+Recommendations"):
            cs = "n/a" if r["checkable_share"] is None else f"{r['checkable_share']:.3f}"
            print(f"{r['condition']:<14} {r['scope']:<27} {r['n_lines']:>6} {r['n_pass']:>6} {r['n_fail']:>6} "
                  f"{r['n_unchecked']:>9} {cs:>9}")
    print("\n== Error profile (lines with each error type, all sections)")
    print(f"{'condition':<14} " + " ".join(f"{e:>19}" for e in ERROR_TYPES))
    for r in res["profile"]:
        if r["scope"] == "ALL":
            print(f"{r['condition']:<14} " + " ".join(f"{r[e]:>19}" for e in ERROR_TYPES))
    if res["enforcement"]:
        print("\n== Enforcement (repair vs retreat; coverage after enforcement)")
        for r in res["enforcement"]:
            ls = "n/a" if r["lines_kept_share"] is None else f"{r['lines_kept_share']:.3f}"
            ns = "n/a" if r["numbers_kept_share"] is None else f"{r['numbers_kept_share']:.3f}"
            print(f"{r['condition']:<22} lines {r['n_lines']:>5} kept {r['kept']:>5} repaired {r['repaired']:>4} "
                  f"retreated {r['retreated']:>4} dropped {r['dropped']:>4}  lines kept {ls}  numbers kept {ns}")
    print(f"\nwrote checker_summary.md, checker_profile.csv, enforcement_summary.csv, checker_lines.jsonl, "
          f"checker_review.csv ({res['n_review']} rows to check by hand) to {run.path}")
    run.finish(profile=[r for r in res["profile"] if r["scope"] == "ALL"], enforcement=res["enforcement"])
    return 0


def cmd_context_tokens(args: argparse.Namespace) -> int:
    """Phase 4: templated full_context prompt tokens per admission vs max_context_tokens."""
    from ehr_rag.config import validate_for_generation
    from ehr_rag.generation.backends import BackendError, make_backend
    from ehr_rag.generation.generate import context_token_counts

    import pandas as pd

    cfg = load_config(args.config)
    validate_for_generation(cfg)
    ids = _target_ids(cfg, args)
    try:
        rows = context_token_counts(cfg, make_backend(cfg["generation"]), ids)
    except (BackendError, OSError) as e:
        print(f"STOP - could not count tokens: {e}", file=sys.stderr)
        return 4
    _print_table(pd.DataFrame(rows))
    limit = cfg["generation"]["max_context_tokens"]
    how = "exact, model tokenizer" if all(r["tokens_exact"] for r in rows) else "approximate, chars/4"
    print(f"\n{sum(r['fits'] for r in rows)}/{len(rows)} admissions fit max_context_tokens = {limit:,} "
          f"(whole templated prompt; {how})")
    return 0


def _run_dir(cfg: dict, run_id: str, what: str) -> "Path":
    from ehr_rag.config import project_root

    d = project_root(cfg) / "runs" / run_id
    if not d.is_dir():
        raise ConfigError(f"{what} run not found: {d}")
    return d


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Phase 5: citation metrics, manual-review export and a 10-sentence check for generation run(s)."""
    from ehr_rag.eval.citations import GROUPS
    from ehr_rag.eval.evaluate import METRICS, gate_sample, run_evaluation
    from ehr_rag.eval.nli import HFNLI, NLIError
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    gen_dirs = [_run_dir(cfg, g, "generation") for g in args.generation_run]
    seed = cfg["eval"]["seed"]
    run = start_run(cfg, "evaluate", command="evaluate", seed=seed, run_id=args.run_id)
    # one run: its id (as before); several (shards): the list
    run.meta.update(generation_run=args.generation_run[0] if len(gen_dirs) == 1 else args.generation_run)
    run.write_json("run_meta.json", run.meta)
    print(f"run: {run.path}")
    try:
        nli = HFNLI(cfg["eval"]["nli_model"])
    except NLIError as e:
        print(f"STOP - {e}", file=sys.stderr)
        run.finish(status="failed", error=str(e))
        return 4
    info = nli.info()
    print(f"NLI model {info['model']} revision {info['revision']}; id2label {info['id2label']}; "
          f"device {info.get('device')}")
    res = run_evaluation(gen_dirs, run.path, run.run_id, nli, cfg["eval"]["manual_review_n"], seed)

    print(f"\n== Metrics (micro-averaged; abstention lines excluded) for generation run(s) "
          f"{' '.join(args.generation_run)}")
    print(f"{'condition':<13} {'scope':<27} {'sent':>5} {'claim':>5} {'abst':>5} "
          + " ".join(f"{m:>24}" for m in METRICS))
    for r in res["rows"]:
        if r["scope"] in GROUPS:
            vals = " ".join(f"{'n/a' if r[m] is None else format(r[m], '.3f'):>24}" for m in METRICS)
            print(f"{r['condition']:<13} {r['scope']:<27} {r['n_sentences']:>5} {r['n_claims']:>5} "
                  f"{r['n_abstentions']:>5} {vals}")
    print("citation_precision = ALCE definition; citation_precision_plain = share of citations that entail "
          "their claim alone")
    print(f"NLI pairs over {res['nli']['max_length']} tokens, labeled too_long (not entailed): "
          f"{res['nli']['n_too_long_pairs']} ({res['n_too_long_rows']} claim/premise rows in nli_too_long.csv); "
          "truncated: 0")
    print(f"\nformat status: {res['format_counts']}")
    sample = gate_sample(res["sentences"], res["texts"], 10, seed)
    run.write_json("gate_sample.json", sample)
    if _quiet(cfg):
        print("\n" + QUIET_NOTE.format("; the 10-sentence check is in gate_sample.json"))
        sample = []
    else:
        print(f"\n== 10 random sentences (seed {seed}) for checking the harness by hand")
    for i, x in enumerate(sample, 1):
        print(f"\n[{i}] {x['condition']} hadm {x['hadm_id']} {x['section']}")
        print(f"  sentence : {x['sentence']}")
        if x["line"]:
            print(f"  (claim of: {x['line']})")
        print(f"  cited    : {x['cited_ids']}  invalid: {x['invalid_ids']}  irrelevant: {x['irrelevant_ids']}")
        for uid, t in zip(x["cited_ids"], x["cited_texts"]):
            print(f"    [{uid}] {t}")
        print(f"  NLI      : {x['nli_label']}")
    print(f"\nwrote metrics.json, metrics_by_section.csv, metrics_summary.md, sentences.jsonl, nli_too_long.csv, "
          f"manual_review.csv ({res['n_manual_review']} rows) to {run.path}")
    run.finish(nli=res["nli"], format_counts=res["format_counts"], n_manual_review=res["n_manual_review"])
    return 0


def cmd_verifier_agreement(args: argparse.Namespace) -> int:
    """Phase 5: agreement between NLI labels and human labels in a labeled manual_review.csv."""
    import json
    from pathlib import Path

    from ehr_rag.eval.agreement import agreement
    from ehr_rag.eval.manual_review import read_manual_review
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    labels = Path(args.labels)
    metrics_p = labels.parent / "metrics.json"
    ent = "entailment"
    if metrics_p.exists():
        ent = json.loads(metrics_p.read_text(encoding="utf-8"))["nli"]["entailment_label"]
    res = agreement(read_manual_review(labels), ent)
    run = start_run(cfg, "verifier-agreement", command="verifier-agreement")
    run.write_json("verifier_agreement.json", {**res, "labels": str(labels)})
    print(json.dumps(res, indent=2))
    run.finish()
    print(f"run: {run.path}")
    return 0


def cmd_text_metrics(args: argparse.Namespace) -> int:
    """Phase 5: ROUGE-L / BERTScore against a reference file (required; the Demo has none)."""
    from pathlib import Path

    import pandas as pd

    from ehr_rag.eval.evaluate import load_generation_runs
    from ehr_rag.eval.text_metrics import BERTScorer, ReferenceFileError, load_references, summarize, text_metrics
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    try:
        refs = load_references(Path(args.references) if args.references else None)
    except ReferenceFileError as e:
        print(f"STOP - {e}", file=sys.stderr)
        return 2
    gen_dirs = [_run_dir(cfg, g, "generation") for g in args.generation_run]
    reports = {(r.condition, r.hadm_id): r.raw for r in load_generation_runs(gen_dirs)
               if not args.conditions or r.condition in args.conditions}
    embedder, info = None, None
    if args.bertscore:
        try:
            embedder = BERTScorer(cfg["eval"]["bertscore_model"], cfg["eval"]["bertscore_layer"])
        except ReferenceFileError as e:
            print(f"STOP - {e}", file=sys.stderr)
            return 4
        info = embedder.info()
        print(f"BERTScore model {info['model']} revision {info['revision']} layer {info['layer']}; "
              f"windows of {info['window_tokens']} tokens; device {info['device']}")
    rows = text_metrics(reports, refs, embedder)
    summary = summarize(rows)
    run = start_run(cfg, "text-metrics", command="text-metrics", run_id=args.run_id)
    pd.DataFrame(rows).to_csv(run.path / "text_metrics.csv", index=False)
    n_missing = len({h for _, h in reports} - set(refs))
    print(f"reports: {len(reports)}; scored (have a reference): {len(rows)}; "
          f"admissions without a reference: {n_missing}")
    if summary:
        _print_table(pd.DataFrame(summary))
    run.finish(generation_runs=args.generation_run, references=str(args.references), n_reports=len(reports),
               n_scored=len(rows), n_admissions_without_reference=n_missing, summary=summary, bertscore=info)
    print(f"per-report scores: {run.path / 'text_metrics.csv'}\nrun: {run.path}")
    return 0


def cmd_select_targets(args: argparse.Namespace) -> int:
    """Phase 6: perturbation candidates, targets and edits from the ehr_rag reports of a generation run."""
    import pandas as pd

    from ehr_rag.perturb.run_pilot import PilotError, select_and_edit
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    gen_dir = _run_dir(cfg, args.generation_run, "generation")
    run = start_run(cfg, "select-targets", command="select-targets", seed=cfg["perturbation"]["seed"])
    try:
        res = select_and_edit(cfg, gen_dir, run)
    except PilotError as e:
        print(f"STOP - {e}", file=sys.stderr)
        run.finish(status="failed", error=str(e))
        return 4
    n_elig = sum(c.eligible for c in res["candidates"])
    print(f"candidates: {len(res['candidates'])} ({n_elig} eligible); targets: {len(res['targets'])} "
          f"(n_targets {cfg['perturbation']['n_targets']}, max per admission "
          f"{cfg['perturbation']['max_targets_per_admission']}, seed {cfg['perturbation']['seed']})")
    if _quiet(cfg):
        print(QUIET_NOTE.format(f"; targets and edits are in {run.path}"))
    elif res["targets"]:
        _print_table(pd.DataFrame([{k: t.to_dict()[k] for k in ("target_id", "hadm_id", "unit_id", "unit_type",
                                    "label", "value", "fields_matched", "sentence")} for t in res["targets"]]))
    print(f"\nedits applied: {len(res['edits'])}; skipped: {len(res['skipped'])}")
    for e in ([] if _quiet(cfg) else res["edits"]):
        print(f"\n{e['target_id']} {e['edit_type']}: {e['original_value']} -> {e['edited_value']}")
        print(f"  before: {e['original_text']}")
        print(f"  after : {e['edited_text'] or '<unit removed from evidence>'}")
    for sk in res["skipped"]:
        print(f"SKIPPED {sk['target_id']} {sk['edit_type']}: {sk['reason']}")
    run.finish(generation_run=args.generation_run, n_candidates=len(res["candidates"]),
               n_targets=len(res["targets"]), n_edits=len(res["edits"]), n_skipped=len(res["skipped"]))
    print(f"\nrun: {run.path}")
    return 0


def cmd_run_pilot(args: argparse.Namespace) -> int:
    """Phase 6: regenerate with frozen evidence for every edit and control."""
    from ehr_rag.config import validate_for_generation
    from ehr_rag.generation.backends import BackendError, gpu_name, make_backend
    from ehr_rag.generation.generate import ConfirmationRequired, GenerationError
    from ehr_rag.perturb.run_pilot import PilotError, run_pilot
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    validate_for_generation(cfg)
    tdir = _run_dir(cfg, args.targets_run, "select-targets")
    run = start_run(cfg, "run-pilot", command="run-pilot", seed=cfg["perturbation"]["seed"], run_id=args.run_id)
    gen = cfg["generation"]
    run.meta.update(targets_run=args.targets_run, backend=gen["backend"], model=gen["model"], gpu_name=gpu_name(),
                    hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"))
    run.write_json("run_meta.json", run.meta)
    print(f"run: {run.path}")
    try:
        res = run_pilot(cfg, tdir, run, make_backend(gen), yes=args.yes, no_cache=args.no_cache)
    except ConfirmationRequired as e:
        print(f"NOT RUN: {e}")
        run.finish(status="not_confirmed")
        return 5
    except (BackendError, GenerationError, PilotError, OSError, ValueError) as e:
        print(f"STOP - pilot failed: {e}", file=sys.stderr)
        run.finish(status="failed", error=repr(e))
        return 4
    except BaseException as e:
        run.finish(status="failed", error=repr(e))
        raise
    print(f"\ncases: {res['n_cases']}; format status: {res['statuses']}; model calls: {res['n_api_calls']}; "
          f"cache hits: {res['n_cache_hits']}")
    run.finish(model_revision=res["model_revision"], reported_model_ids=res["reported_models"],
               generation_run=res["generation_run"], format_counts=res["statuses"], n_cases=res["n_cases"],
               n_api_calls=res["n_api_calls"], n_cache_hits=res["n_cache_hits"])
    print(f"run: {run.path}")
    return 0


def cmd_analyze_perturbation(args: argparse.Namespace) -> int:
    """Phase 6: provisional outcome labels, summary with raw counts, manual-check CSV."""
    from ehr_rag.config import project_root
    from ehr_rag.perturb.analyze import analyze, summary_markdown
    from ehr_rag.runs import start_run

    cfg = load_config(args.config)
    pdir = _run_dir(cfg, args.pilot_run, "run-pilot")
    run = start_run(cfg, "analyze-perturbation", command="analyze-perturbation")
    res = analyze(pdir, project_root(cfg) / "runs", run.path)
    print(summary_markdown(res["summary"], pdir.name))
    if _quiet(cfg):
        print(QUIET_NOTE.format("; worked examples are in perturbation_manual_check.csv"))
    else:
        print("== Worked examples (up to 3 per edit type)")
    shown: dict[str, int] = {}
    for r in ([] if _quiet(cfg) else res["rows"]):
        if r["kind"] != "edit" or shown.get(r["edit_type"], 0) >= 3:
            continue
        shown[r["edit_type"]] = shown.get(r["edit_type"], 0) + 1
        print(f"\n{r['case_id']} ({r['edit_type']}) -> {r['auto_outcome']}")
        print(f"  original sentence : {r['original_sentence']}")
        print(f"  edit              : {r['original_text']}")
        print(f"                   -> {r['edited_text']}")
        print(f"  regenerated       : {r['regenerated_candidates'] or '<no candidate sentence>'}")
    run.finish(pilot_run=args.pilot_run, summary=res["summary"])
    print(f"\nwrote perturbation_results.csv, perturbation_summary.md, perturbation_manual_check.csv, diffs/ "
          f"to {run.path}")
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
    p = add("select-cohort", cmd_select_cohort,
            "Full MIMIC-IV: seeded dev/test cohort (DuckDB); id lists gitignored, hashes in config/cohort/.")
    p.add_argument("--skip-note-filter", action="store_true",
                   help="do not require a discharge note (the Demo has no notes)")
    p.add_argument("--no-demo-exclusion", action="store_true",
                   help="keep Demo subjects (only for testing on the Demo itself)")
    p.add_argument("--pool-fraction", type=float, help="override cohort.pool_fraction")
    p.add_argument("--n-dev", type=int, help="override cohort.n_dev")
    p.add_argument("--n-test", type=int, help="override cohort.n_test")
    p.add_argument("--overwrite", action="store_true", help="replace existing id lists and hashes")
    p = add("extract-cohort", cmd_extract_cohort,
            "Full MIMIC-IV: copy the cohort's rows (Demo columns) into Parquet, plus reference discharge notes.")
    p.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"],
                   help="cohort splits to extract (default: dev test)")
    p.add_argument("--skip-notes", action="store_true", help="do not extract discharge notes (the Demo has none)")
    p = add("cohort-schema-check", cmd_cohort_schema_check,
            "Column differences between full MIMIC-IV, the Demo and the extracted Parquet (headers only).")
    p.add_argument("--skip-notes", action="store_true", help="do not check the MIMIC-IV-Note discharge table")
    p = add("build-units", cmd_build_units,
            "Phase 2: build evidence units and headers for admissions in config/selected_admissions.yaml "
            "(or --split).")
    p.add_argument("--split", nargs="+", choices=["dev", "test"],
                   help="cohort split(s) from select-cohort (hash-checked) instead of selected_admissions.yaml")
    p = add("retrieve", cmd_retrieve, "Phase 3: per-section hybrid retrieval for the selected admissions.")
    p.add_argument("--hadm-ids", nargs="*", help="restrict to these admissions (default: all selected)")
    p.add_argument("--show", type=int, default=2, help="print sections for the first N admissions (default 2)")
    p.add_argument("--split", nargs="+", choices=["dev", "test"],
                   help="cohort split(s) from select-cohort (hash-checked) instead of selected_admissions.yaml")
    p = add("export-relevance", cmd_export_relevance,
            "Phase 3: write relevance_to_label.csv (pooled top-10 of BM25/dense/fused) for manual labeling.")
    p.add_argument("--split", nargs="+", choices=["dev", "test"],
                   help="cohort split(s) from select-cohort (hash-checked) instead of selected_admissions.yaml")
    p = add("retrieval-metrics", cmd_retrieval_metrics,
            "Phase 3: Recall@5 and MRR per method from a labeled relevance CSV (pooled relevance).")
    p.add_argument("--labels", required=True, help="labeled relevance_to_label.csv")
    p.add_argument("--rankings", help="relevance_rankings.json (default: next to --labels)")
    p = add("generate", cmd_generate, "Phase 4: generate reports per condition for the selected admissions.")
    p.add_argument("--conditions", nargs="+",
                   help="subset of no_retrieval ehr_rag full_context ehr_rag_enforced full_context_enforced "
                        "(default: generation.conditions)")
    p.add_argument("--limit", type=int, help="only the first N selected admissions")
    p.add_argument("--hadm-ids", nargs="*", help="restrict to these admissions (default: all selected)")
    p.add_argument("--split", nargs="+", choices=["dev", "test"],
                   help="cohort split(s) from select-cohort (hash-checked) instead of selected_admissions.yaml")
    p.add_argument("--retrieval-run", help="run id of a saved `retrieve` run (required for ehr_rag)")
    p.add_argument("--yes", action="store_true", help="confirm when new calls exceed generation.confirm_threshold")
    p.add_argument("--no-cache", action="store_true", help="bypass the response cache (fresh calls)")
    p.add_argument("--run-id", help="explicit run id YYYYMMDD-HHMMSS-<name> (used by scripts/generate.sbatch)")
    p.add_argument("--shard-size", type=int,
                   help="admissions per shard (with --shard-index; scripts/generate_array.sbatch sets both)")
    p.add_argument("--shard-index", type=int, help="0-based shard to generate (contiguous blocks of --shard-size)")
    p.add_argument("--checker-sha256", help="refuse to run unless the checker files' combined sha256 equals this")
    p = add("check-citations", cmd_check_citations,
            "Deterministic citation-checker statistics (error profile, coverage, enforcement) for generation runs.")
    p.add_argument("--generation-run", required=True, nargs="+", help="run id(s) of `generate` / `enforce-filter` runs")
    p.add_argument("--review-n", type=int, default=25,
                   help="passing and unchecked lines sampled per condition for checker_review.csv (default 25)")
    p.add_argument("--checker-sha256", help="refuse to run unless the checker files' combined sha256 equals this")
    p.add_argument("--run-id", help="explicit run id YYYYMMDD-HHMMSS-<name>")
    p = add("enforce-filter", cmd_enforce_filter,
            "CPU fallback: copy reports to <condition>_filtered and drop lines failing the citation checker.")
    p.add_argument("--generation-run", required=True, nargs="+", help="run id(s) of `generate` runs (shards)")
    p.add_argument("--conditions", nargs="+", default=["ehr_rag", "full_context"],
                   choices=["ehr_rag", "full_context"], help="base conditions to filter (default: both)")
    p.add_argument("--checker-sha256", help="refuse to run unless the checker files' combined sha256 equals this")
    p.add_argument("--run-id", help="explicit run id YYYYMMDD-HHMMSS-<name>")
    p = add("evaluate", cmd_evaluate,
            "Phase 5: parse reports, NLI citation metrics, manual_review.csv, 10-sentence check.")
    p.add_argument("--generation-run", required=True, nargs="+",
                   help="run id(s) of `generate` runs (several = the shards of one job array)")
    p.add_argument("--run-id", help="explicit run id YYYYMMDD-HHMMSS-<name> (used by scripts/cli.sbatch)")
    p = add("verifier-agreement", cmd_verifier_agreement,
            "Phase 5: raw agreement, Cohen's kappa and confusion matrix, NLI vs your labels.")
    p.add_argument("--labels", required=True, help="labeled manual_review.csv (from an evaluate run)")
    p = add("text-metrics", cmd_text_metrics,
            "ROUGE-L / BERTScore of each report vs the admission's discharge summary; REQUIRES --references "
            "(extract-cohort's reference/references.csv; the Demo has none; never generated).")
    p.add_argument("--generation-run", required=True, nargs="+", help="run id(s) of `generate` runs (shards)")
    p.add_argument("--references", help="CSV with columns hadm_id,reference")
    p.add_argument("--conditions", nargs="+", help="conditions to score (default: all in the run)")
    p.add_argument("--bertscore", action="store_true",
                   help="also compute BERTScore (eval.bertscore_model from the local HF cache)")
    p.add_argument("--run-id", help="explicit run id YYYYMMDD-HHMMSS-<name> (used by scripts/cli.sbatch)")
    p = add("select-targets", cmd_select_targets,
            "Phase 6: candidates, targets (fixed seed) and edits from a generation run's ehr_rag reports.")
    p.add_argument("--generation-run", required=True, help="run id of a `generate` run with ehr_rag reports")
    p = add("run-pilot", cmd_run_pilot,
            "Phase 6: frozen-evidence regenerations for every edit plus unperturbed controls.")
    p.add_argument("--targets-run", required=True, help="run id of a `select-targets` run")
    p.add_argument("--yes", action="store_true", help="confirm when new calls exceed generation.confirm_threshold")
    p.add_argument("--no-cache", action="store_true", help="bypass the response cache (fresh calls)")
    p.add_argument("--run-id", help="explicit run id YYYYMMDD-HHMMSS-<name> (used by scripts/cli.sbatch)")
    p = add("analyze-perturbation", cmd_analyze_perturbation,
            "Phase 6: provisional outcome labels, summary with raw counts, manual-check CSV.")
    p.add_argument("--pilot-run", required=True, help="run id of a `run-pilot` run")
    p = add("context-tokens", cmd_context_tokens,
            "Phase 4: print full_context templated-prompt tokens per admission vs max_context_tokens.")
    p.add_argument("--limit", type=int, help="only the first N selected admissions")
    p.add_argument("--hadm-ids", nargs="*", help="restrict to these admissions (default: all selected)")
    p.add_argument("--split", nargs="+", choices=["dev", "test"],
                   help="cohort split(s) from select-cohort (hash-checked) instead of selected_admissions.yaml")
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
