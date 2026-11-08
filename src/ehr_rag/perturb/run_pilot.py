"""Target selection run and perturbation pilot run (SPEC Phase 6).

``select_and_edit`` (command ``select-targets``, CPU): reads the ORIGINAL ehr_rag reports
of a generation run, writes ``candidates.csv``, ``targets.csv``, ``edits.jsonl`` (original
and edited fields and text for every applied edit) and ``skipped_edits.csv``.

``run_pilot`` (command ``run-pilot``, GPU): frozen retrieval. For every target the
evidence is exactly the ordered unit list of the original ``evidence_shown.json``;
only the edited unit's text changes (or, for ``delete``, the unit is left out, with no
renumbering). Before anything runs, the unperturbed prompt is rebuilt from
``evidence_shown.json`` and the stored header and must equal the original ``prompt.txt``
byte for byte; otherwise the run stops. Each (target, edit) is regenerated once
(sample_idx 0); each target also gets ``n_controls`` unperturbed regenerations with
sample_idx 1..n_controls so the cache keeps them apart.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

from ehr_rag.config import project_root, resolve_path
from ehr_rag.eval.parse import parse_report
from ehr_rag.evidence.build_units import read_header, units_dir
from ehr_rag.evidence.schema import EvidenceUnit, read_units
from ehr_rag.generation.backends import Backend
from ehr_rag.generation.generate import (
    ConfirmationRequired, Generator, Job, ResponseCache, _write_json,
)
from ehr_rag.generation.prompt import build_prompt, load_templates
from ehr_rag.perturb.edit import EDITS_FOR_TYPE, SkipEdit, apply_edit
from ehr_rag.perturb.select_targets import Candidate, candidates_for_report, select_targets
from ehr_rag.runs import Run


class PilotError(RuntimeError):
    """The pilot cannot run with frozen, verified evidence."""


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    cols = columns or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def original_report(gen_dir: Path, hadm_id: int, condition: str = "ehr_rag") -> tuple[str, dict[str, Any]]:
    """(final raw text, evidence_shown) of one original report."""
    d = gen_dir / condition / str(hadm_id)
    meta = _read_json(d / "parsed.json")
    return (d / meta["raw_file"]).read_text(encoding="utf-8"), _read_json(d / "evidence_shown.json")


def frozen_units(evidence_shown: dict[str, Any], units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """Units in evidence_shown order, checked to be identical (id, type, text) to the current unit files."""
    by_id = {u.unit_id: u for u in units}
    out = []
    for e in evidence_shown["units"]:
        u = by_id.get(e["unit_id"])
        if u is None or u.unit_type != e["unit_type"] or u.text != e["text"]:
            raise PilotError(f"hadm {evidence_shown['hadm_id']} unit {e['unit_id']} differs from the evidence "
                             "the original report was generated with; units were rebuilt?")
        out.append(u)
    return out


def select_and_edit(cfg: dict[str, Any], gen_dir: Path, run: Run) -> dict[str, Any]:
    """Candidates, targets and edits for one generation run; writes the target-run files."""
    pc = cfg["perturbation"]
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    ehr = gen_dir / "ehr_rag"
    if not ehr.is_dir():
        raise PilotError(f"generation run has no ehr_rag reports: {ehr}")
    hadm_ids = sorted(int(p.name) for p in ehr.iterdir() if p.is_dir())
    cands: list[Candidate] = []
    unit_maps: dict[int, dict[str, EvidenceUnit]] = {}
    for h in hadm_ids:
        raw, shown = original_report(gen_dir, h)
        units = read_units(udir / f"{h}.jsonl")
        frozen_units(shown, units)
        unit_maps[h] = {u.unit_id: u for u in units}
        cands.extend(candidates_for_report(h, parse_report(raw), unit_maps[h],
                                           {u["unit_id"] for u in shown["units"]}))
    targets = select_targets(cands, pc["n_targets"], pc["max_targets_per_admission"], pc["seed"])
    edits: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for t in targets:
        unit = unit_maps[t.hadm_id][t.unit_id]
        for et in pc["edit_types"]:
            if et not in EDITS_FOR_TYPE[t.unit_type]:
                continue
            try:
                r = apply_edit(unit, et, t.value, t.fields_matched, seed=f"{pc['seed']}:{t.target_id}:{et}")
            except SkipEdit as e:
                skipped.append({"target_id": t.target_id, "hadm_id": t.hadm_id, "unit_id": t.unit_id,
                                "edit_type": et, "reason": str(e)})
                continue
            edits.append({"target_id": t.target_id, "hadm_id": t.hadm_id, **r.to_dict()})
    _write_csv(run.path / "candidates.csv", [c.to_dict() for c in cands],
               list(Candidate.__dataclass_fields__) if not cands else None)
    _write_csv(run.path / "targets.csv", [t.to_dict() for t in targets],
               list(Candidate.__dataclass_fields__) if not targets else None)
    with (run.path / "edits.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for e in edits:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    _write_csv(run.path / "skipped_edits.csv", skipped,
               ["target_id", "hadm_id", "unit_id", "edit_type", "reason"])
    _write_json(run.path / "targets_meta.json", {"generation_run": gen_dir.name, "perturbation": pc})
    return {"candidates": cands, "targets": targets, "edits": edits, "skipped": skipped}


@dataclasses.dataclass
class Case:
    """One pilot regeneration."""

    case_id: str
    kind: str              # edit | control
    target_id: str
    hadm_id: int
    unit_id: str
    edit_type: str | None
    sample_idx: int
    job: Job


def build_cases(cfg: dict[str, Any], targets_dir: Path, backend: Backend) -> tuple[list[Case], str]:
    """Frozen-evidence jobs for every applied edit and every control. Returns (cases, generation run id)."""
    root = project_root(cfg)
    meta = _read_json(targets_dir / "targets_meta.json")
    gen_dir = root / "runs" / meta["generation_run"]
    templates = load_templates(root / "prompts", cfg["generation"]["prompt_version"])
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    with (targets_dir / "targets.csv").open(encoding="utf-8", newline="") as f:
        targets = list(csv.DictReader(f))
    edits = [json.loads(line) for line in (targets_dir / "edits.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    cases: list[Case] = []
    frozen: dict[int, list[EvidenceUnit]] = {}
    for t in targets:
        h = int(t["hadm_id"])
        if h not in frozen:
            _, shown = original_report(gen_dir, h)
            frozen[h] = frozen_units(shown, read_units(udir / f"{h}.jsonl"))
            rebuilt = build_prompt(templates, read_header(udir, h), frozen[h])
            original = (gen_dir / "ehr_rag" / str(h) / "prompt.txt").read_text(encoding="utf-8")
            if rebuilt != original:
                raise PilotError(f"hadm {h}: rebuilt unperturbed prompt differs from the original prompt.txt "
                                 "(template, header or evidence changed); refusing to run")
        base = frozen[h]
        orig_prompt = build_prompt(templates, read_header(udir, h), base)
        for e in [e for e in edits if e["target_id"] == t["target_id"]]:
            if e["edit_type"] == "delete":
                ev = [u for u in base if u.unit_id != e["unit_id"]]
            else:
                ev = [dataclasses.replace(u, fields=e["edited_fields"], text=e["edited_text"])
                      if u.unit_id == e["unit_id"] else u for u in base]
            prompt = build_prompt(templates, read_header(udir, h), ev)
            n, exact = backend.count_tokens(prompt)
            cases.append(Case(f"{t['target_id']}__{e['edit_type']}", "edit", t["target_id"], h, t["unit_id"],
                              e["edit_type"], 0, Job("ehr_rag", h, prompt, ev, n, exact)))
        n, exact = backend.count_tokens(orig_prompt)
        for k in range(1, cfg["perturbation"]["n_controls"] + 1):
            cases.append(Case(f"{t['target_id']}__control{k}", "control", t["target_id"], h, t["unit_id"], None, k,
                              Job("ehr_rag", h, orig_prompt, base, n, exact)))
    return cases, meta["generation_run"]


def run_pilot(
    cfg: dict[str, Any], targets_dir: Path, run: Run, backend: Backend, *, yes: bool = False,
    no_cache: bool = False, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every case. Raises ConfirmationRequired before any call if needed."""
    gen = cfg["generation"]
    cases, gen_run = build_cases(cfg, targets_dir, backend)
    if not cases:
        raise PilotError("no cases: no targets or every edit was skipped")
    params = {"temperature": gen["temperature"], "max_tokens": gen["max_tokens"], "base_url": gen["base_url"]}
    g = Generator(backend, ResponseCache(project_root(cfg) / "cache" / "generation"), params, no_cache=no_cache)
    g.revision = backend.revision()
    n_cached = 0 if no_cache else sum(g.cache.has(g.key(c.job.prompt, c.sample_idx)) for c in cases)
    n_new = len(cases) - n_cached
    n_edit = sum(1 for c in cases if c.kind == "edit")
    log(f"== Pilot plan: {n_edit} edited + {len(cases) - n_edit} control regenerations = {len(cases)} calls "
        f"({n_cached} cached, {n_new} new); estimated input tokens {sum(c.job.input_tokens for c in cases):,}")
    if n_new > gen["confirm_threshold"] and not yes:
        raise ConfirmationRequired(
            f"{n_new} new calls exceed generation.confirm_threshold ({gen['confirm_threshold']}); re-run with --yes")
    templates = load_templates(project_root(cfg) / "prompts", gen["prompt_version"])
    statuses: dict[str, int] = {}
    rows = []
    for i, c in enumerate(cases, 1):
        d = run.path / "cases" / c.case_id
        status = g.run_job(c.job, templates, d, retrieval_run=None, sample_idx=c.sample_idx)
        _write_json(d / "case.json", {k: v for k, v in dataclasses.asdict(c).items() if k != "job"})
        statuses[status] = statuses.get(status, 0) + 1
        rows.append({"case_id": c.case_id, "kind": c.kind, "target_id": c.target_id, "hadm_id": c.hadm_id,
                     "unit_id": c.unit_id, "edit_type": c.edit_type or "", "sample_idx": c.sample_idx,
                     "status": status})
        log(f"  [{i}/{len(cases)}] {c.case_id}: {status}")
    _write_csv(run.path / "cases.csv", rows)
    _write_json(run.path / "pilot_meta.json", {"targets_run": targets_dir.name, "generation_run": gen_run})
    return {"statuses": statuses, "n_cases": len(cases), "n_api_calls": g.n_calls, "n_cache_hits": g.n_cache_hits,
            "model_revision": g.revision, "reported_models": sorted(g.reported_models), "generation_run": gen_run}


# --------------------------------------------------------------------------- counterfactual audit: targets and edits

AUDIT_TARGET_COLUMNS = ["target_id", "hadm_id", "unit_id", "label", "kind", "status", "n_citing_claims"]


def ehr_rag_reports(gen_dirs: list[Path]) -> dict[int, Path]:
    """hadm_id -> the generation run holding its plain ehr_rag report (shards allowed, no duplicates)."""
    out: dict[int, Path] = {}
    for g in gen_dirs:
        d = g / "ehr_rag"
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_dir():
                h = int(p.name)
                if h in out:
                    raise PilotError(f"ehr_rag hadm {h} is in both {out[h].name} and {g.name}")
                out[h] = g
    if not out:
        raise PilotError("no plain ehr_rag reports in the given generation run(s)")
    return out


def cited_claim_counts(parsed: Any, checker: Any, ctx: Any) -> dict[str, int]:
    """Unit id -> number of claims (list items included) citing it, over every non-abstention line."""
    counts: dict[str, int] = {}
    for s in parsed.sentences():
        if s.is_abstention:
            continue
        for _, ids in checker.items(s, ctx):
            for i in dict.fromkeys(ids):
                if i in ctx.shown:
                    counts[i] = counts.get(i, 0) + 1
    return counts


def select_audit(cfg: dict[str, Any], gen_dirs: list[Path], run: Run) -> dict[str, Any]:
    """``audit-targets``: targets, edits and skips for every plain ehr_rag report; writes the target-run files."""
    from ehr_rag.checker.rules import Checker
    from ehr_rag.perturb.edit import audit_edit
    from ehr_rag.perturb.select_targets import select_audit_targets

    ac = cfg["audit"]
    checker = Checker.from_config(cfg)
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    where = ehr_rag_reports(gen_dirs)

    def edit(u: EvidenceUnit) -> Any:
        return audit_edit(u, checker.lab_group(str(u.fields.get("label") or "")), ac["outside_fraction"],
                          ac["below_range_groups"])

    def editable(u: EvidenceUnit) -> str:
        try:
            edit(u)
        except SkipEdit as e:
            return str(e)
        return ""

    targets, skips, edits = [], [], []
    for h in sorted(where):
        raw, shown = original_report(where[h], h, "ehr_rag")
        units = read_units(udir / f"{h}.jsonl")
        fr = frozen_units(shown, units)
        ctx = checker.admission(fr, units)
        cited = cited_claim_counts(parse_report(raw), checker, ctx)
        t, s = select_audit_targets(h, fr, cited, ac["n_cited_per_admission"], ac["n_null_per_admission"],
                                    ac["seed"], editable)
        targets.extend(t)
        skips.extend(s)
    by_id = {}
    for i, t in enumerate(targets, 1):
        t.target_id = f"A{i:04d}"
        if t.hadm_id not in by_id:
            by_id[t.hadm_id] = {u.unit_id: u for u in read_units(udir / f"{t.hadm_id}.jsonl")}
        r = edit(by_id[t.hadm_id][t.unit_id])
        edits.append({"target_id": t.target_id, "hadm_id": t.hadm_id, "kind": t.kind, **r.to_dict()})
    _write_csv(run.path / "audit_targets.csv", [t.to_dict() for t in targets], AUDIT_TARGET_COLUMNS)
    with (run.path / "audit_edits.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for e in edits:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    _write_csv(run.path / "audit_skips.csv", skips, ["hadm_id", "unit_id", "kind", "reason"])
    _write_json(run.path / "audit_meta.json", {"generation_runs": {str(h): g.name for h, g in where.items()},
                                               "audit": ac})
    reasons: dict[str, int] = {}
    for s in skips:
        reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
    return {"n_admissions": len(where), "targets": targets, "edits": edits, "skips": skips, "skip_reasons": reasons}
