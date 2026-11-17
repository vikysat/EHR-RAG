"""Every remaining full-data step, in order, safely re-runnable (``scripts/run_all.sh`` runs this with nohup).

Inputs are the fixed runs in ``INPUTS``. Steps (each ends with a ``STEP n done`` line):

  1  freeze the checker (record the combined sha256 of rules, lexicon and rewrite prompt as they are)
     and ``check-citations`` on the dev generation (statistics only)
  2  ``generate --split test --conditions ehr_rag_enforced`` with the test retrieval run and
     ``--checker-sha256``; the base reports come from the generation cache (the step stops if they
     are not all cached, so the enforced reports always start from the test generation's reports)
  3  ``enforce-filter`` (drop-only baseline) on the test generation
  4  ``evaluate`` the enforced and drop-only outputs
  5  ``check-citations`` on the plain, enforced and drop-only test reports
  6  ``text-metrics --bertscore`` on all test conditions
  7  probe on dev: probe-targets, run-probe, probe-threshold (writes ``probe.threshold_file``; nothing is
     pushed), analyze-probe with the dev evaluation
  8  probe on test with ``--threshold-sha256`` from step 7: probe-targets, run-probe, analyze-probe
  9  ``make-tables`` (evaluations: test + step 4; check run 5; text metrics 6; probe 8; manual labels
     ``runs/<test evaluation>/manual_review_labeled.csv`` if that file exists)

Before steps 2, 7 and 8 the expected GPU calls / forward passes are printed (``--plan-only``; for the
probe this comes after probe-targets, which is what defines the sequences).

Re-running: every run id this creates is appended to the ids file (``key<TAB>value<TAB>time``; the
last line of a key wins) before the command starts, together with the frozen checker hash and the
threshold hash. A sub-step whose recorded run has ``"status": "ok"`` in its run_meta.json is skipped,
after checking that it was made with the same arguments (so a skipped step never silently pairs with
different upstream runs). run-probe continues an unfinished run with ``--resume``; an unfinished
generation is started again under a new run id and takes every finished call from the response cache;
other unfinished steps start again under a new id. A recorded run id whose folder never got a
run_meta.json is reused. The first failing command stops everything (exit 1). Output: step markers,
run ids, hashes and the commands' own count-only output (admissions as adm_001...).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

INPUTS: dict[str, str] = {
    "dev_generation": "20261116-193521-generate",
    "dev_evaluation": "20261117-152134-evaluate",
    "test_retrieval": "20261117-011229-retrieve",
    "test_generation": "20261117-011443-generate",
    "test_evaluation": "20261117-152041-evaluate",
}
DEFAULT_IDS = Path.home() / "scratch" / "logs" / "run_all_ids.txt"
DEFAULT_REFERENCES = "data/cohort/reference/references.csv"
_RUN_FLAGS = ("--run-id", "--resume")

Runner = Callable[[list[str]], "tuple[int, str]"]


class StepError(RuntimeError):
    """A step failed or the recorded state does not allow continuing."""


class Ledger:
    """The ids file: append-only ``key<TAB>value<TAB>time`` lines; the last value of a key wins."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.values: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0]:
                    self.values[parts[0]] = parts[1]

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def record(self, key: str, value: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"{key}\t{value}\t{datetime.now().isoformat(timespec='seconds')}\n")
            f.flush()
            os.fsync(f.fileno())
        self.values[key] = value


def subprocess_runner(argv: list[str]) -> tuple[int, str]:
    """``python -m ehr_rag.cli <argv>``, its output streamed through and also returned."""
    env = {**os.environ, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"), "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen([sys.executable, "-m", "ehr_rag.cli", *argv], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=env)
    lines = []
    assert p.stdout is not None
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    return p.wait(), "".join(lines)


def _strip_run_flags(argv: list[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in _RUN_FLAGS:
            skip = True
        else:
            out.append(a)
    return out


def parse_plan(text: str) -> dict[str, Any]:
    """The JSON of the last ``PLAN {...}`` line of a --plan-only command."""
    for line in reversed(text.splitlines()):
        if line.startswith("PLAN "):
            return json.loads(line[5:])
    raise StepError("the --plan-only command printed no PLAN line")


class Pipeline:
    """The nine steps (module docstring)."""

    def __init__(self, config: str, ids_file: Path, references: str, *, dry_run: bool = False,
                 runner: Runner = subprocess_runner, out: Callable[[str], None] = print) -> None:
        from ehr_rag.config import load_config, project_root

        self.config = config
        self.cfg = load_config(config)
        self.runs = project_root(self.cfg) / "runs"
        self.ledger = Ledger(ids_file)
        self.references = references
        self.dry = dry_run
        self.runner = runner
        self.out = out
        self.n_commands = 0

    # ---------------------------------------------------------------- helpers

    def meta(self, rid: str) -> dict[str, Any] | None:
        p = self.runs / rid / "run_meta.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def status(self, rid: str | None) -> str:
        """ok / failed / interrupted / ... ; 'unfinished' if run_meta.json has no status; 'missing' if none."""
        m = self.meta(rid) if rid else None
        if m is None:
            return "missing"
        return str(m.get("status") or "unfinished")

    def _new_id(self, name: str) -> str:
        from ehr_rag.runs import make_run_id

        base = make_run_id(name)
        rid, i = base, 1
        while (self.runs / rid).exists() or rid in self.ledger.values.values():
            i += 1
            rid = make_run_id(f"{name}-{i}")
        return rid

    def call(self, argv: list[str]) -> str:
        self.n_commands += 1
        rc, text = self.runner(argv)
        if rc != 0:
            raise StepError(f"`{argv[0]}` exited with status {rc}")
        return text

    def cli(self, key: str, name: str, args: list[str], resumable: bool = False) -> str:
        """Run one command as run ``key`` (skip if finished; resume / restart otherwise). Returns its run id."""
        argv0 = [args[0], "--config", self.config, *args[1:]]
        rid = self.ledger.get(key)
        st = self.status(rid)
        if rid and st == "ok":
            recorded = _strip_run_flags(list((self.meta(rid) or {}).get("argv", [])[1:]))
            if recorded and recorded != argv0:
                raise StepError(f"{key}: run {rid} finished with other arguments than now ({' '.join(recorded)}); "
                                "remove its line from the ids file to redo it")
            self.out(f"  {key}: finished earlier as {rid}; skipped")
            return rid
        if rid and resumable and st != "missing":
            how, argv = "resume", argv0 + ["--resume", rid]
        elif rid and st == "missing":
            how, argv = "start (recorded id, never started)", argv0 + ["--run-id", rid]
        else:
            how = "start" if not rid else f"start again (previous run {rid}: {st})"
            if self.dry:
                rid = f"<{key}>"
            else:
                rid = self._new_id(name)
                self.ledger.record(key, rid)
            argv = argv0 + ["--run-id", rid]
        if self.dry:
            self.out(f"  {key}: would {how}: python -m ehr_rag.cli {' '.join(argv)}")
            return rid
        self.out(f"  {key}: {how} as {rid}")
        self.call(argv)
        if self.status(rid) != "ok":
            raise StepError(f"{key}: run {rid} ended with status {self.status(rid)}")
        return rid

    def plan(self, argv: list[str]) -> dict[str, Any] | None:
        if self.dry:
            self.out(f"  plan: would run python -m ehr_rag.cli {argv[0]} --config {self.config} {' '.join(argv[1:])}")
            return None
        return parse_plan(self.call([argv[0], "--config", self.config, *argv[1:]]))

    def done(self, n: int, extra: str = "") -> None:
        self.out(f"STEP {n} done{(' ' + extra) if extra else ''}")

    # ---------------------------------------------------------------- steps

    def check_inputs(self) -> None:
        bad = {k: f"{rid} ({self.status(rid)})" for k, rid in INPUTS.items() if self.status(rid) != "ok"}
        if bad:
            msg = "input runs missing or not finished: " + "; ".join(f"{k} {v}" for k, v in bad.items())
            if self.dry:
                self.out("  WARNING " + msg)
            else:
                raise StepError(msg)

    def step1(self) -> str:
        from ehr_rag.runs import checker_hashes

        h = (checker_hashes(self.cfg) or {}).get("combined_sha256")
        if not h:
            raise StepError("no checker configured")
        prev = self.ledger.get("checker_sha256")
        if prev and prev != h:
            raise StepError(f"the checker changed since it was frozen (recorded {prev[:12]}..., now {h[:12]}...); "
                            "restore config/checker.yaml, the lexicon and the rewrite prompt")
        if not prev and not self.dry:
            self.ledger.record("checker_sha256", h)
        self.out(f"  checker frozen: combined sha256 {h}")
        self.cli("s1_check_dev", "rall-check-dev", ["check-citations", "--generation-run", INPUTS["dev_generation"],
                                                     "--checker-sha256", h])
        self.done(1, f"(checker sha256 {h[:12]}...)")
        return h

    def step2(self, h: str) -> str:
        args = ["generate", "--split", "test", "--retrieval-run", INPUTS["test_retrieval"],
                "--conditions", "ehr_rag_enforced", "--checker-sha256", h]
        if self.status(self.ledger.get("s2_generate_enforced")) != "ok":
            p = self.plan([*args, "--plan-only"])
            if p is not None:
                self.out(f"STEP 2 expected GPU calls: {p['new']} report calls not cached ({p['cached']} of {p['calls']} "
                         f"cached), rewrite calls at most {p['rewrite_max']}")
                if p["base_cached"] < p["base_reports"]:
                    raise StepError(f"only {p['base_cached']} of {p['base_reports']} base test reports are in the "
                                    "generation cache; the enforced reports would not reuse the test generation")
        rid = self.cli("s2_generate_enforced", "rall-generate-enforced", [*args, "--yes"])
        self.done(2)
        return rid

    def step3(self, h: str) -> str:
        rid = self.cli("s3_enforce_filter", "rall-enforce-filter",
                       ["enforce-filter", "--generation-run", INPUTS["test_generation"], "--checker-sha256", h])
        self.done(3)
        return rid

    def step4(self, enf: str, filt: str) -> str:
        rid = self.cli("s4_evaluate", "rall-evaluate-enforced", ["evaluate", "--generation-run", enf, filt])
        self.done(4)
        return rid

    def step5(self, h: str, enf: str, filt: str) -> str:
        rid = self.cli("s5_check_test", "rall-check-test", ["check-citations", "--generation-run",
                                                             INPUTS["test_generation"], enf, filt, "--checker-sha256", h])
        self.done(5)
        return rid

    def step6(self, enf: str, filt: str) -> str:
        rid = self.cli("s6_text_metrics", "rall-text-metrics",
                       ["text-metrics", "--generation-run", INPUTS["test_generation"], enf, filt,
                        "--references", self.references, "--bertscore"])
        self.done(6)
        return rid

    def probe(self, n: int, split: str, gen: str, evaluation: str, thr: list[str]) -> tuple[str, str]:
        t = self.cli(f"s{n}_probe_targets", f"rall-probe-targets-{split}", ["probe-targets", "--generation-run", gen])
        key = f"s{n}_run_probe"
        rp = self.ledger.get(key)
        if self.status(rp) != "ok":
            resume = ["--resume", rp] if rp and self.status(rp) != "missing" else []
            p = self.plan(["run-probe", "--targets-run", t, *thr, "--plan-only", *resume])
            if p is not None:
                self.out(f"STEP {n} expected forward passes: {p['todo']} to run of {p['sequences']} "
                         f"({p['done']} already scored), {p['batches']} batches")
        rp = self.cli(key, f"rall-run-probe-{split}", ["run-probe", "--targets-run", t, *thr], resumable=True)
        return t, rp

    def threshold(self, dev_probe: str) -> str:
        from ehr_rag.config import resolve_path
        from ehr_rag.probe.runner import read_threshold
        from ehr_rag.probe.score import ProbeError

        path = resolve_path(self.cfg, self.cfg["probe"]["threshold_file"])
        if not path.exists():
            if self.dry:
                self.out(f"  threshold: would run probe-threshold --probe-run {dev_probe} (writes {path.name})")
                return "<threshold sha256>"
            self.out(f"  threshold: probe-threshold from {dev_probe}")
            self.call(["probe-threshold", "--config", self.config, "--probe-run", dev_probe])
        else:
            self.out(f"  threshold: {path.name} written earlier; skipped")
        try:
            d = read_threshold(self.cfg)
        except ProbeError as e:
            raise StepError(str(e)) from e
        if d.get("probe_run") != dev_probe and not self.dry:
            raise StepError(f"{path.name} was derived from probe run {d.get('probe_run')}, not {dev_probe}; "
                            "refusing to overwrite it (move it away to derive it again)")
        prev = self.ledger.get("probe_threshold_sha256")
        if prev and prev != d["sha256"]:
            raise StepError(f"{path.name} changed since it was recorded ({prev[:12]}... -> {d['sha256'][:12]}...)")
        if not prev and not self.dry:
            self.ledger.record("probe_threshold_sha256", d["sha256"])
        return str(d["sha256"])

    def step7(self) -> str:
        _, rp = self.probe(7, "dev", INPUTS["dev_generation"], INPUTS["dev_evaluation"], [])
        sha = self.threshold(rp)
        self.cli("s7_analyze_probe", "rall-analyze-probe-dev",
                 ["analyze-probe", "--probe-run", rp, "--evaluate-run", INPUTS["dev_evaluation"]])
        self.done(7, f"(threshold sha256 {sha[:12]}...)")
        return sha

    def step8(self, sha: str) -> str:
        thr = ["--threshold-sha256", sha]
        _, rp = self.probe(8, "test", INPUTS["test_generation"], INPUTS["test_evaluation"], thr)
        a = self.cli("s8_analyze_probe", "rall-analyze-probe-test",
                     ["analyze-probe", "--probe-run", rp, "--evaluate-run", INPUTS["test_evaluation"], *thr])
        self.done(8)
        return a

    def step9(self, ev: str, check: str, tm: str, probe: str) -> str:
        args = ["make-tables", "--evaluate-run", INPUTS["test_evaluation"], ev, "--check-run", check,
                "--text-metrics-run", tm, "--probe-run", probe]
        manual = self.runs / INPUTS["test_evaluation"] / "manual_review_labeled.csv"
        if manual.exists():
            args += ["--manual-labels", str(Path("runs") / INPUTS["test_evaluation"] / manual.name)]
        rid = self.cli("s9_make_tables", "rall-make-tables", args)
        self.done(9)
        return rid

    def run(self) -> None:
        self.out(f"run_all{' (DRY RUN: nothing is executed)' if self.dry else ''}: config {self.config}; "
                 f"ids file {self.ledger.path}")
        self.out("inputs: " + "; ".join(f"{k} {v}" for k, v in INPUTS.items()))
        self.check_inputs()
        h = self.step1()
        enf = self.step2(h)
        filt = self.step3(h)
        ev = self.step4(enf, filt)
        check = self.step5(h, enf, filt)
        tm = self.step6(enf, filt)
        sha = self.step7()
        probe = self.step8(sha)
        self.step9(ev, check, tm, probe)
        self.out("ALL STEPS DONE" if not self.dry else "DRY RUN complete")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ehr_rag.run_all", description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config/full.yaml")
    ap.add_argument("--ids-file", type=Path, default=DEFAULT_IDS)
    ap.add_argument("--references", default=DEFAULT_REFERENCES, help="text-metrics reference CSV")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (skip / resume / start) without running")
    args = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")
    try:
        Pipeline(args.config, args.ids_file, args.references, dry_run=args.dry_run).run()
    except StepError as e:
        print(f"STOP - {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
