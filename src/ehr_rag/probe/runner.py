"""``probe-targets`` (CPU) and ``run-probe`` (GPU, resumable).

``probe-targets`` reads the plain ehr_rag reports of one or more generation runs (shards) and writes
``probe_units.csv`` (one row per unit, ``targets.UNIT_COLUMNS``), ``probe_contexts.jsonl`` (the evidence
change of every (unit, context)), ``probe_skips.csv`` and ``probe_meta.json`` (generation run per
admission, split, probe config). The split is the generation runs' recorded ``--split`` (dev or
test; the admissions are checked against the hash-checked cohort list); generation runs without a
split are accepted only with ``data.source: demo`` (split ``demo``).

Sequences (one forward pass each): per unit and context, the original term and its alternative,
forced up to ``term_cut``; if the claim has a number, the original number, the cited-edit value and
the within-flag value, forced up to ``number_cut``. Identical sequences are scored once.

``run-probe`` appends one JSON line per scored sequence to ``probe_scores.jsonl`` after every batch
(flushed and fsynced). Re-running with ``--resume <run id>`` reopens that run, checks that the plan
(every sequence key, the targets run, the model and its revision) is unchanged, and scores only the
sequences not yet in the file (a cut-off last line is discarded).

Threshold guard: on a test-split targets run, ``run-probe`` (and ``analyze-probe``) refuse to run unless
``probe.threshold_file`` exists, its stored sha256 matches its content, and equals ``--threshold-sha256``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from ehr_rag.config import project_root, resolve_path
from ehr_rag.eval.parse import parse_report
from ehr_rag.evidence.build_units import read_header, units_dir
from ehr_rag.evidence.schema import read_units
from ehr_rag.generation.generate import MAX_PROBLEMS_IN_REMINDER, _write_json
from ehr_rag.generation.prompt import build_prompt, load_templates, with_format_reminder
from ehr_rag.perturb.run_pilot import _read_json, _write_csv, cited_claim_counts, ehr_rag_reports, frozen_units
from ehr_rag.probe.score import ProbeError, Scorer
from ehr_rag.probe.targets import UNIT_COLUMNS, apply_spec, probe_report
from ehr_rag.runs import Run

SCORES_FILE = "probe_scores.jsonl"
PLAN_FILE = "probe_run_plan.json"
SEQUENCES_FILE = "probe_sequences.csv"      # (key, probe_id, hadm_id, context, role) of every continuation
SKIP_COLUMNS = ["hadm_id", "section", "sentence_index", "surface", "reason"]


# --------------------------------------------------------------------------- split and threshold guard


def generation_split(cfg: dict[str, Any], gen_dirs: list[Path], hadm_ids: list[int]) -> str:
    """dev / test (checked against the cohort list) or demo (module docstring)."""
    splits = set()
    for g in gen_dirs:
        s = _read_json(g / "run_meta.json").get("split")
        splits.add(tuple(s) if s else None)
    if len(splits) != 1:
        raise ProbeError(f"generation runs of different splits: {sorted(map(str, splits))}")
    s = splits.pop()
    if s is None:
        if cfg["data"]["source"] != "demo":
            raise ProbeError("generation run has no recorded --split; the probe needs dev or test reports")
        return "demo"
    if len(s) != 1:
        raise ProbeError(f"generation run covers several splits {list(s)}; probe one split at a time")
    from ehr_rag.data.cohort import load_split_ids

    allowed = set(load_split_ids(cfg, s[0]))
    if any(h not in allowed for h in hadm_ids):
        raise ProbeError(f"reports include admissions outside the {s[0]} cohort list")
    return s[0]


def threshold_sha(payload: dict[str, Any]) -> str:
    """sha256 of the threshold payload (every key except ``sha256``), canonical JSON."""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def read_threshold(cfg: dict[str, Any]) -> dict[str, Any]:
    """The threshold file, checked against its own stored hash."""
    import yaml

    p = resolve_path(cfg, cfg["probe"]["threshold_file"])
    if not p.exists():
        raise ProbeError(f"threshold file not found: {p} (run probe-threshold on a dev probe run)")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict) or d.get("sha256") != threshold_sha(d):
        raise ProbeError(f"{p}: content does not match its stored sha256; the threshold was edited")
    return d


def check_threshold_for_split(cfg: dict[str, Any], split: str, expected: str | None) -> dict[str, Any] | None:
    """On test: the verified threshold, which must hash to ``expected``. Elsewhere: the threshold if present."""
    if split != "test":
        try:
            return read_threshold(cfg)
        except ProbeError:
            return None
    if not expected:
        raise ProbeError("test split: give --threshold-sha256 (printed by probe-threshold on dev)")
    d = read_threshold(cfg)
    if d["sha256"] != expected:
        raise ProbeError(f"test split: the threshold file's sha256 {d['sha256'][:12]}... differs from "
                         f"--threshold-sha256 {expected[:12]}...; the threshold changed, refusing to run")
    return d


# --------------------------------------------------------------------------- probe-targets


def _rules_words(cfg: dict[str, Any]) -> dict[str, str]:
    import yaml

    from ehr_rag.checker.rules import checker_paths

    d = yaml.safe_load(checker_paths(cfg)["rules"].read_text(encoding="utf-8"))["direction"]
    return {k: str(d[k][0]) for k in ("high", "low", "normal")}


def select_probe(cfg: dict[str, Any], gen_dirs: list[Path], run: Run) -> dict[str, Any]:
    """Units, contexts and skips for every plain ehr_rag report; writes the targets-run files."""
    from ehr_rag.checker.rules import Checker
    from ehr_rag.perturb.run_pilot import original_report

    checker = Checker.from_config(cfg)
    words = _rules_words(cfg)
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    where = ehr_rag_reports(gen_dirs)
    split = generation_split(cfg, sorted({g for g in where.values()}, key=lambda p: p.name), sorted(where))
    units, skips, contexts = [], [], []
    for h in sorted(where):
        raw, shown = original_report(where[h], h, "ehr_rag")
        all_units = read_units(udir / f"{h}.jsonl")
        fr = frozen_units(shown, all_units)
        ctx = checker.admission(fr, all_units)
        parsed = parse_report(raw)
        rp = probe_report(h, raw, parsed, fr, cited_claim_counts(parsed, checker, ctx), checker, ctx, cfg, words)
        first = len(units)
        for u in rp.units:
            u.probe_id = f"P{len(units) + 1:04d}"
            units.append(u)
        for (k, c), spec in rp.specs.items():
            contexts.append({"probe_id": units[first + k].probe_id, "hadm_id": h, "context": c, **spec})
        skips.extend(rp.skips)
    _write_csv(run.path / "probe_units.csv", [u.row() for u in units], UNIT_COLUMNS)
    with (run.path / "probe_contexts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for c in contexts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    _write_csv(run.path / "probe_skips.csv", skips, SKIP_COLUMNS)
    _write_json(run.path / "probe_meta.json", {"generation_runs": {str(h): g.name for h, g in where.items()},
                                               "split": split, "probe": cfg["probe"], "audit": cfg["audit"]})
    reasons: dict[str, int] = {}
    for s in skips:
        key = s["reason"].split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    return {"n_admissions": len(where), "units": units, "n_contexts": len(contexts), "skip_reasons": reasons,
            "split": split}


# --------------------------------------------------------------------------- sequences


def _read_csv(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def seq_key(prompt: str, prefix: str, cont: str) -> str:
    """Identity of one sequence."""
    h = hashlib.sha256()
    for part in (prompt, prefix, cont):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def build_sequences(cfg: dict[str, Any], targets_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """(sequence records, key -> (prompt, prefix, cont) texts kept separately). Checks that the rebuilt
    unperturbed prompt equals the original prompt.txt (prompt_retry.txt if the report used the retry)."""
    root = project_root(cfg)
    meta = _read_json(targets_dir / "probe_meta.json")
    templates = load_templates(root / "prompts", cfg["generation"]["prompt_version"])
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    units = _read_csv(targets_dir / "probe_units.csv")
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    for line in (targets_dir / "probe_contexts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            specs[(c["probe_id"], c["context"])] = c
    seqs: list[dict[str, Any]] = []
    texts: dict[str, Any] = {}
    cache: dict[int, Any] = {}
    for u in units:
        h = int(u["hadm_id"])
        if h not in cache:
            d = root / "runs" / meta["generation_runs"][str(h)] / "ehr_rag" / str(h)
            pj = _read_json(d / "parsed.json")
            raw = (d / pj["raw_file"]).read_text(encoding="utf-8")
            shown = _read_json(d / "evidence_shown.json")
            base = frozen_units(shown, read_units(udir / f"{h}.jsonl"))
            header = read_header(udir, h)
            retry = pj["attempt_used"] == 2
            problems = pj["first_attempt_problems"][:MAX_PROBLEMS_IN_REMINDER]

            def prompt_for(ev: Any, header: Any = header, retry: bool = retry, problems: Any = problems) -> str:
                p = build_prompt(templates, header, ev)
                return with_format_reminder(templates, p, problems) if retry else p

            stored = (d / ("prompt_retry.txt" if retry else "prompt.txt")).read_text(encoding="utf-8")
            if prompt_for(base) != stored:
                raise ProbeError(f"hadm {h}: rebuilt unperturbed prompt differs from the stored prompt "
                                 "(template, header or evidence changed); refusing to run")
            cache = {h: (raw, base, prompt_for)}
        raw, base, prompt_for = cache[h]
        ts, te, tc = int(u["term_start"]), int(u["term_end"]), int(u["term_cut"])
        conts = [("term_orig", tc, raw[tc:te]), ("term_alt", tc, raw[tc:ts] + u["term_alt"])]
        if u["number"]:
            ns, ne, nc = int(u["number_start"]), int(u["number_end"]), int(u["number_cut"])
            conts.append(("num_orig", nc, raw[nc:ne]))
            conts.append(("num_alt_cited", nc, raw[nc:ns] + u["number_alt_cited"]))
            if u["number_alt_within"]:
                conts.append(("num_alt_within", nc, raw[nc:ns] + u["number_alt_within"]))
        for c in u["contexts"].split():
            prompt = prompt_for(apply_spec(base, specs[(u["probe_id"], c)]))
            for role, cut, cont in conts:
                k = seq_key(prompt, raw[:cut], cont)
                texts[k] = (prompt, raw[:cut], cont)
                seqs.append({"key": k, "probe_id": u["probe_id"], "hadm_id": h, "context": c, "role": role})
    return seqs, texts


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    """Scored sequences by key; a cut-off last line (interrupted write) is dropped from the file."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    good: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise ProbeError(f"{path.name}: unreadable line {i + 1}")
        out[r["key"]] = r
        good.append(line)
    if len(good) != sum(1 for x in lines if x.strip()):
        path.write_text("".join(x + "\n" for x in good), encoding="utf-8", newline="\n")
    return out


def run_probe(cfg: dict[str, Any], targets_dir: Path, run: Run, scorer: Scorer, *, resume: bool = False,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """Score every sequence not yet in ``probe_scores.jsonl``, printing the expected forward passes first."""
    seqs, texts = build_sequences(cfg, targets_dir)
    if not seqs:
        raise ProbeError("no probe sequences: the targets run has no units")
    keys = sorted(texts)
    plan = {"targets_run": targets_dir.name, "model": scorer.model, "revision": scorer.revision(),
            "n_sequences": len(keys), "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest()}
    plan_path = run.path / PLAN_FILE
    if resume:
        old = _read_json(plan_path) if plan_path.exists() else None
        if old != plan:
            raise ProbeError("resume: the plan changed (targets, prompts, model or revision); start a new run")
    else:
        _write_json(plan_path, plan)
        _write_csv(run.path / SEQUENCES_FILE, seqs, ["key", "probe_id", "hadm_id", "context", "role"])
    path = run.path / SCORES_FILE
    done = load_scores(path)
    stray = [k for k in done if k not in texts]
    if stray:
        raise ProbeError(f"{len(stray)} scored sequences are not in the plan; refusing to mix runs")
    todo = [k for k in keys if k not in done]
    bs = cfg["probe"]["batch_size"]
    n_units = len({s["probe_id"] for s in seqs})
    log(f"== Probe plan: {n_units} units, {len({(s['probe_id'], s['context']) for s in seqs})} (unit, context) "
        f"pairs, {len(seqs)} scored continuations")
    log(f"  expected forward passes: {len(keys)} distinct sequences ({len(done)} already scored, {len(todo)} to run) "
        f"in {-(-len(todo) // bs)} batches of <= {bs}")
    first = {}
    for s in seqs:
        first.setdefault(s["key"], s)
    todo.sort(key=lambda k: -(len(texts[k][0]) + len(texts[k][1])))      # longest first: OOM shows up early
    n_merge = 0
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for b in range(0, len(todo), bs):
            batch = todo[b: b + bs]
            res = scorer.score([texts[k] for k in batch])
            for k, r in zip(batch, res):
                s = first[k]
                n_merge += bool(r["boundary_merge"])
                f.write(json.dumps({"key": k, "probe_id": s["probe_id"], "hadm_id": s["hadm_id"],
                                    "context": s["context"], "role": s["role"], "tokens": r["tokens"],
                                    "logprobs": r["logprobs"], "boundary_merge": r["boundary_merge"]}) + "\n")
            f.flush()
            os.fsync(f.fileno())
            log(f"  batch {b // bs + 1}/{-(-len(todo) // bs)}: {min(b + bs, len(todo))}/{len(todo)} sequences")
    return {"n_units": n_units, "n_sequences": len(keys), "n_scored_now": len(todo), "n_already": len(done),
            "n_boundary_merge": n_merge, "revision": plan["revision"]}
