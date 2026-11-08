"""Report generation for the three conditions (SPEC Phase 4).

Per (condition, admission) the run writes to ``runs/<id>/<condition>/<hadm_id>/``:
  prompt.txt            the filled template (the single user message)
  evidence_shown.json   exact ordered unit ids + texts the model saw (source of truth for eval)
  raw.txt               first model output, verbatim
  prompt_retry.txt / raw_retry.txt   only if the first output failed the format check
  prompt_templated.txt (+ _retry)    hf_local only: exact chat-templated model input
  parsed.json           status (ok | ok_after_retry | format_error), problems, lines + trailing citation ids
  call.json             per-attempt metadata (cache key/hit, timing, token counts, reported model)

Evidence per condition:
  no_retrieval  header only; every sentence must end with [NONE]
  ehr_rag       header + the ordered evidence saved by a previous ``retrieve`` run (never re-retrieved)
  full_context  header + all units (E1..En); skipped when the templated prompt exceeds
                ``generation.max_context_tokens`` (counted with the model's tokenizer)

The format check here only decides whether to retry once; full parsing and the
citation metrics belong to Phase 5 (``eval/parse.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ehr_rag.config import SECTIONS, project_root, resolve_path
from ehr_rag.evidence.build_units import check_evidence_sources, read_header, units_dir
from ehr_rag.evidence.schema import EvidenceUnit, read_units
from ehr_rag.generation.backends import Backend, Completion
from ehr_rag.generation.prompt import PromptTemplates, build_prompt, load_templates, with_format_reminder
from ehr_rag.runs import Run

_HEADER_LINES = {f"### {s}": s for s in SECTIONS}
_CITE_GROUP = r"\[(?:NONE|E\d+(?:\s*,\s*E\d+)*)\]"
_TRAILING_RE = re.compile(rf"(?:\s*{_CITE_GROUP})+\s*$")
_CITE_ID_RE = re.compile(r"E\d+|NONE")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
MAX_PROBLEMS_IN_REMINDER = 10


class GenerationError(RuntimeError):
    """Generation cannot proceed (bad inputs, stale retrieval run, ...)."""


class ConfirmationRequired(RuntimeError):
    """More new calls than ``generation.confirm_threshold`` and ``--yes`` not given."""


# --------------------------------------------------------------------------- format check


def check_format(text: str, none_only: bool = False) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Check the report layout; return (sections -> parsed lines, problems).

    Rules: exactly the four ``### <Section>`` header lines in order; one sentence per
    line; no bullets; no blank line between two sentence lines of a section (blank
    lines right before a header are allowed); every line ends with a citation group
    (``[E3]``, ``[E3, E7]``, ``[E3][E7]`` or ``[NONE]``). With ``none_only`` (the
    no_retrieval condition) every citation must be ``[NONE]``. Citation ids are not
    checked against the evidence here (that is a Phase 5 metric).
    """
    sections: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    problems: list[str] = []
    current: str | None = None
    pending_blank = False
    for i, raw in enumerate(text.strip("\n").splitlines(), 1):
        line = raw.rstrip()
        if line in _HEADER_LINES:
            name = _HEADER_LINES[line]
            if name in sections:
                problems.append(f"line {i}: duplicate section header '{line}'")
            order.append(name)
            sections.setdefault(name, [])
            current, pending_blank = name, False
            continue
        if not line.strip():
            pending_blank = current is not None
            continue
        if current is None:
            problems.append(f"line {i}: text before the first section header")
            continue
        if pending_blank:
            problems.append(f"line {i}: blank line inside section {current}")
            pending_blank = False
        if line.lstrip().startswith("#"):
            problems.append(f"line {i}: unexpected heading line '{line.strip()[:40]}' (only the four section headers are allowed)")
        elif _BULLET_RE.match(line):
            problems.append(f"line {i}: bullet or numbered line in {current}")
        m = _TRAILING_RE.search(line)
        if m is None:
            problems.append(f"line {i}: sentence in {current} does not end with a citation")
            cites: list[str] = []
            sentence = line.strip()
        else:
            cites = _CITE_ID_RE.findall(m.group(0))
            sentence = line[: m.start()].strip()
            if not sentence:
                problems.append(f"line {i}: citation without a sentence in {current}")
        if none_only and any(c != "NONE" for c in cites):
            problems.append(f"line {i}: cites evidence ids, but no evidence was provided (use [NONE])")
        sections[current].append({"line": line, "sentence": sentence, "citations": cites})
    missing = [s for s in SECTIONS if s not in sections]
    if missing:
        problems.append(f"missing section header(s): {', '.join('### ' + s for s in missing)}")
    elif len(order) == len(SECTIONS) and order != list(SECTIONS):
        problems.append(f"sections out of order: {order}")
    for s in SECTIONS:
        if s in sections and not sections[s]:
            problems.append(f"section {s} is empty")
    return sections, problems


# --------------------------------------------------------------------------- cache


def cache_key(prompt: str, backend: Backend, revision: str | None, params: dict[str, Any], sample_idx: int) -> str:
    """sha256 over prompt, backend, model, revision, generation params and sample_idx."""
    payload = {
        "prompt": prompt, "backend": backend.name, "model": backend.model, "revision": revision,
        "params": params, "sample_idx": sample_idx,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class ResponseCache:
    """One JSON file per response under ``cache/generation/``."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Completion | None:
        p = self._path(key)
        if not p.exists():
            return None
        return Completion(**json.loads(p.read_text(encoding="utf-8"))["completion"])

    def put(self, key: str, request: dict[str, Any], completion: Completion) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "request": request, "completion": completion.to_dict(),
                                   "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)


# --------------------------------------------------------------------------- inputs


@dataclass
class Job:
    """One (condition, admission) generation."""

    condition: str
    hadm_id: int
    prompt: str
    evidence: list[EvidenceUnit]
    input_tokens: int
    tokens_exact: bool


def load_retrieval_evidence(
    retrieval_run_dir: Path, hadm_ids: list[int], units_by_h: dict[int, list[EvidenceUnit]]
) -> dict[int, list[EvidenceUnit]]:
    """Ordered evidence saved by a ``retrieve`` run, checked against the current unit files.

    Raises GenerationError if a file is missing, predates ``evidence_units``, or any
    unit id / type / text differs from the current units (units rebuilt since).
    """
    out: dict[int, list[EvidenceUnit]] = {}
    for h in hadm_ids:
        p = retrieval_run_dir / "retrieval" / f"{h}.json"
        if not p.exists():
            raise GenerationError(f"retrieval run has no result for hadm {h}: {p}")
        d = json.loads(p.read_text(encoding="utf-8"))
        if "evidence_units" not in d:
            raise GenerationError(f"{p} has no 'evidence_units' (saved by an older retrieve); re-run `retrieve`")
        saved = d["evidence_units"]
        if d["hadm_id"] != h or [e["unit_id"] for e in saved] != d["evidence"]:
            raise GenerationError(f"{p}: inconsistent retrieval record")
        by_id = {u.unit_id: u for u in units_by_h[h]}
        units: list[EvidenceUnit] = []
        for e in saved:
            u = by_id.get(e["unit_id"])
            if u is None or u.unit_type != e["unit_type"] or u.text != e["text"]:
                raise GenerationError(
                    f"hadm {h} unit {e['unit_id']}: differs from the retrieval run {retrieval_run_dir.name} "
                    "(units were rebuilt?); re-run `retrieve`"
                )
            units.append(u)
        out[h] = units
    return out


def build_jobs(
    templates: PromptTemplates,
    backend: Backend,
    hadm_ids: list[int],
    conditions: list[str],
    units_by_h: dict[int, list[EvidenceUnit]],
    headers: dict[int, dict[str, Any]],
    retrieval: dict[int, list[EvidenceUnit]] | None,
    max_context_tokens: int,
) -> tuple[list[Job], list[dict[str, Any]]]:
    """Build prompts for every (condition, admission); return (jobs, skipped full_context admissions)."""
    jobs: list[Job] = []
    skipped: list[dict[str, Any]] = []
    for cond in conditions:
        for h in hadm_ids:
            if cond == "no_retrieval":
                evidence: list[EvidenceUnit] = []
                prompt = build_prompt(templates, headers[h], None)
            elif cond == "ehr_rag":
                if retrieval is None:
                    raise GenerationError("ehr_rag requires the evidence of a retrieve run (--retrieval-run)")
                evidence = retrieval[h]
                prompt = build_prompt(templates, headers[h], evidence)
            elif cond == "full_context":
                evidence = list(units_by_h[h])
                prompt = build_prompt(templates, headers[h], evidence)
            else:
                raise GenerationError(f"unknown condition {cond!r}")
            n, exact = backend.count_tokens(prompt)
            if cond == "full_context" and n > max_context_tokens:
                skipped.append({"condition": cond, "hadm_id": h, "input_tokens": n, "tokens_exact": exact,
                                "max_context_tokens": max_context_tokens, "n_units": len(evidence)})
                continue
            jobs.append(Job(cond, h, prompt, evidence, n, exact))
    return jobs, skipped


# --------------------------------------------------------------------------- execution


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, obj: Any) -> None:
    _write(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


@dataclass
class Generator:
    """Runs jobs against one backend with the response cache."""

    backend: Backend
    cache: ResponseCache
    params: dict[str, Any]
    no_cache: bool = False
    revision: str | None = None
    reported_models: set[str] = field(default_factory=set)
    n_calls: int = 0
    n_cache_hits: int = 0

    def key(self, prompt: str, sample_idx: int = 0) -> str:
        return cache_key(prompt, self.backend, self.revision, self.params, sample_idx=sample_idx)

    def call(self, prompt: str, sample_idx: int = 0) -> tuple[Completion, dict[str, Any]]:
        """One completion (from cache unless --no-cache) plus attempt metadata."""
        k = self.key(prompt, sample_idx)
        t0 = time.perf_counter()
        c = None if self.no_cache else self.cache.get(k)
        cached = c is not None
        if c is None:
            c = self.backend.generate(prompt, self.params["max_tokens"], self.params["temperature"])
            request = {"backend": self.backend.name, "model": self.backend.model, "revision": self.revision,
                       "params": self.params, "sample_idx": sample_idx}
            self.cache.put(k, request, c)
            self.n_calls += 1
        else:
            self.n_cache_hits += 1
        self.reported_models.add(c.model)
        meta = {"cache_key": k, "sample_idx": sample_idx, "cached": cached, "seconds": round(time.perf_counter() - t0, 3),
                **{f: v for f, v in c.to_dict().items() if f not in ("text", "templated_prompt")}}
        return c, meta

    def run_job(
        self, job: Job, templates: PromptTemplates, out_dir: Path, retrieval_run: str | None, sample_idx: int = 0
    ) -> str:
        """Generate one report (with at most one format retry); write all files; return the status."""
        _write(out_dir / "prompt.txt", job.prompt)
        _write_json(out_dir / "evidence_shown.json", {
            "hadm_id": job.hadm_id, "condition": job.condition, "prompt_version": templates.version,
            "retrieval_run": retrieval_run if job.condition == "ehr_rag" else None,
            "units": [{"unit_id": u.unit_id, "unit_type": u.unit_type, "text": u.text} for u in job.evidence],
        })
        none_only = job.condition == "no_retrieval"
        c1, m1 = self.call(job.prompt, sample_idx)
        _write(out_dir / "raw.txt", c1.text)
        if c1.templated_prompt is not None:
            _write(out_dir / "prompt_templated.txt", c1.templated_prompt)
        sections, problems = check_format(c1.text, none_only)
        attempts = [{"attempt": 1, **m1, "problems": problems}]
        first_problems = problems
        status, used = "ok", 1
        if problems:
            retry_prompt = with_format_reminder(templates, job.prompt, problems[:MAX_PROBLEMS_IN_REMINDER])
            _write(out_dir / "prompt_retry.txt", retry_prompt)
            c2, m2 = self.call(retry_prompt, sample_idx)
            _write(out_dir / "raw_retry.txt", c2.text)
            if c2.templated_prompt is not None:
                _write(out_dir / "prompt_retry_templated.txt", c2.templated_prompt)
            sections, problems = check_format(c2.text, none_only)
            attempts.append({"attempt": 2, **m2, "problems": problems})
            status, used = ("ok_after_retry" if not problems else "format_error"), 2
        _write_json(out_dir / "parsed.json", {
            "hadm_id": job.hadm_id, "condition": job.condition, "status": status, "attempt_used": used,
            "raw_file": "raw.txt" if used == 1 else "raw_retry.txt",
            "first_attempt_problems": first_problems, "problems": problems, "sections": sections,
        })
        _write_json(out_dir / "call.json", {
            "hadm_id": job.hadm_id, "condition": job.condition, "status": status,
            "estimated_input_tokens": job.input_tokens, "tokens_exact": job.tokens_exact, "attempts": attempts,
        })
        return status


def run_generation(
    cfg: dict[str, Any],
    run: Run,
    backend: Backend,
    *,
    hadm_ids: list[int],
    conditions: list[str],
    retrieval_run: str | None,
    yes: bool = False,
    no_cache: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Plan, confirm and run generation. Returns a summary (counts per condition etc.).

    Raises ConfirmationRequired before any call if new calls exceed the threshold
    without ``yes``; any backend failure propagates and stops the run.
    """
    gen = cfg["generation"]
    root = project_root(cfg)
    templates = load_templates(root / "prompts", gen["prompt_version"])
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    units_by_h = {h: read_units(udir / f"{h}.jsonl") for h in hadm_ids}
    for units in units_by_h.values():
        check_evidence_sources(units)
    headers = {h: read_header(udir, h) for h in hadm_ids}
    retrieval = None
    if "ehr_rag" in conditions:
        if not retrieval_run:
            raise GenerationError("condition ehr_rag needs --retrieval-run <run_id> (a saved `retrieve` run)")
        rdir = root / "runs" / retrieval_run
        if not rdir.is_dir():
            raise GenerationError(f"retrieval run not found: {rdir}")
        retrieval = load_retrieval_evidence(rdir, hadm_ids, units_by_h)

    jobs, skipped = build_jobs(templates, backend, hadm_ids, conditions, units_by_h, headers, retrieval,
                               gen["max_context_tokens"])
    params = {"temperature": gen["temperature"], "max_tokens": gen["max_tokens"], "base_url": gen["base_url"]}
    g = Generator(backend, ResponseCache(root / "cache" / "generation"), params, no_cache=no_cache)
    g.revision = backend.revision()

    n_cached = 0 if no_cache else sum(g.cache.has(g.key(j.prompt)) for j in jobs)
    n_new = len(jobs) - n_cached
    exact = all(j.tokens_exact for j in jobs)
    log(f"== Generation plan: backend {backend.name}, model {backend.model}, revision {g.revision}")
    for cond in conditions:
        cj = [j for j in jobs if j.condition == cond]
        log(f"  {cond}: {len(cj)} calls, estimated input tokens {sum(j.input_tokens for j in cj):,}")
    for s in skipped:
        log(f"  SKIP full_context hadm {s['hadm_id']}: {s['input_tokens']:,} tokens > max_context_tokens "
            f"{s['max_context_tokens']:,}")
    log(f"  total: {len(jobs)} calls ({n_cached} cached, {n_new} new); estimated input tokens "
        f"{sum(j.input_tokens for j in jobs):,} ({'exact, model tokenizer' if exact else 'approximate, chars/4'}); "
        "format retries (at most one per call) not included")
    run.write_json("skipped.json", skipped)
    if n_new > gen["confirm_threshold"] and not yes:
        raise ConfirmationRequired(
            f"{n_new} new calls exceed generation.confirm_threshold ({gen['confirm_threshold']}); re-run with --yes"
        )

    counts = {c: {"ok": 0, "ok_after_retry": 0, "format_error": 0,
                  "skipped": sum(1 for s in skipped if s["condition"] == c)} for c in conditions}
    for i, job in enumerate(jobs, 1):
        status = g.run_job(job, templates, run.path / job.condition / str(job.hadm_id), retrieval_run)
        counts[job.condition][status] += 1
        log(f"  [{i}/{len(jobs)}] {job.condition} hadm {job.hadm_id}: {status}")
    return {
        "counts": counts,
        "reported_models": sorted(g.reported_models),
        "model_revision": g.revision,
        "n_api_calls": g.n_calls,
        "n_cache_hits": g.n_cache_hits,
        "skipped": skipped,
    }


def context_token_counts(
    cfg: dict[str, Any], backend: Backend, hadm_ids: list[int]
) -> list[dict[str, Any]]:
    """Templated full_context prompt size per admission (for choosing max_context_tokens)."""
    gen = cfg["generation"]
    templates = load_templates(project_root(cfg) / "prompts", gen["prompt_version"])
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    rows = []
    for h in hadm_ids:
        units = read_units(udir / f"{h}.jsonl")
        n, exact = backend.count_tokens(build_prompt(templates, read_header(udir, h), units))
        rows.append({"hadm_id": h, "n_units": len(units), "input_tokens": n, "tokens_exact": exact,
                     "fits": n <= gen["max_context_tokens"]})
    return rows
