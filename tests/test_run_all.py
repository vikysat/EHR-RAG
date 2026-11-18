"""scripts/run_all.sh / ``ehr_rag.run_all``: order, skip, resume, restart, dry run and guards.

The pipeline runs against a FAKE command runner that writes run folders and statuses the way the real
commands do; no real command, model or data is used. Plus the --plan-only flags of generate and run-probe
and merged evaluations in make-tables (FAKE backend and scorer).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from ehr_rag.run_all import INPUTS, Ledger, Pipeline, StepError

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def project(tmp_path: Path) -> dict[str, Any]:
    """A project with the fixed input runs finished, its own copy of the checker files and prompts."""
    (tmp_path / "config").mkdir()
    shutil.copytree(REPO / "prompts", tmp_path / "prompts")
    for f in ("checker.yaml", "checker_lexicon.yaml"):
        shutil.copy(REPO / "config" / f, tmp_path / "config" / f)
    cfg = yaml.safe_load((REPO / "config" / "default.yaml").read_text(encoding="utf-8"))
    cfg["data"].update(root=str(tmp_path / "nodata"), derived=str(tmp_path / "derived"))
    cfg["checker"].update(rules=str(tmp_path / "config" / "checker.yaml"),
                          lexicon=str(tmp_path / "config" / "checker_lexicon.yaml"))
    cfg["probe"]["threshold_file"] = str(tmp_path / "config" / "probe_threshold.yaml")
    p = tmp_path / "config" / "full_test.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for rid in INPUTS.values():
        (tmp_path / "runs" / rid).mkdir(parents=True)
        (tmp_path / "runs" / rid / "run_meta.json").write_text(json.dumps({"status": "ok", "argv": []}))
    return {"root": tmp_path, "config": str(p), "ids": tmp_path / "logs" / "run_all_ids.txt"}


class FakeCLI:
    """Plays the commands: writes runs/<id>/run_meta.json (argv as sys.argv would be) and the threshold file.
    ``fail``: command -> number of times it fails first (status ``unfinished`` for run-probe, as a killed
    session leaves it, else ``failed``)."""

    def __init__(self, root: Path, fail: dict[str, int] | None = None, base_cached: int = 40) -> None:
        self.root, self.fail, self.base_cached = root, dict(fail or {}), base_cached
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> tuple[int, str]:
        self.calls.append(list(argv))
        cmd = argv[0]
        if "--plan-only" in argv:
            if cmd == "generate":
                plan = {"calls": 40, "cached": self.base_cached, "new": 40 - self.base_cached, "base_reports": 40,
                        "base_cached": self.base_cached, "rewrite_max": 9}
            else:
                done = 8 if "--resume" in argv else 0
                plan = {"units": 5, "sequences": 30, "done": done, "todo": 30 - done, "batches": 8}
            return 0, "== plan\nPLAN " + json.dumps(plan) + "\n"
        if cmd == "probe-threshold":
            from ehr_rag.probe.analyze import write_threshold

            write_threshold(self.root / "config" / "probe_threshold.yaml",
                            {"threshold": 0.4, "quantile": 0.95, "n_null": 3, "n_within_flag": 5,
                             "probe_run": argv[argv.index("--probe-run") + 1]})
            return 0, "sha256 ...\n"
        flag = "--resume" if "--resume" in argv else "--run-id"
        rid = argv[argv.index(flag) + 1]
        d = self.root / "runs" / rid
        d.mkdir(parents=True, exist_ok=True)
        mp = d / "run_meta.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {"argv": ["cli.py", *argv]}
        if self.fail.get(cmd, 0) > 0:
            self.fail[cmd] -= 1
            meta.pop("status", None)
            if cmd != "run-probe":
                meta["status"] = "failed"
            mp.write_text(json.dumps(meta))
            return 1, "STOP - fake failure\n"
        meta["status"] = "ok"
        mp.write_text(json.dumps(meta))
        return 0, f"run: {rid}\n"


def _run(project: dict[str, Any], cli: FakeCLI, dry: bool = False) -> list[str]:
    out: list[str] = []
    Pipeline(project["config"], project["ids"], "refs.csv", dry_run=dry, runner=cli, out=out.append).run()
    return out


def _cmds(cli: FakeCLI) -> list[str]:
    return [c[0] + (" plan" if "--plan-only" in c else "") for c in cli.calls]


ORDER = ["check-citations", "generate plan", "generate", "enforce-filter", "evaluate", "check-citations",
         "text-metrics", "probe-targets", "run-probe plan", "run-probe", "probe-threshold", "analyze-probe",
         "probe-targets", "run-probe plan", "run-probe", "analyze-probe", "make-tables"]


def test_runs_every_step_in_order_and_a_rerun_skips_everything(project: dict[str, Any]) -> None:
    cli = FakeCLI(project["root"])
    out = _run(project, cli)
    assert _cmds(cli) == ORDER
    assert [x.split()[1] for x in out if x.startswith("STEP") and " done" in x] == [str(i) for i in range(1, 10)]
    for n in (2, 7, 8):                                   # expected calls / passes before the GPU part
        i_plan = next(i for i, x in enumerate(out) if x.startswith(f"STEP {n} expected"))
        i_done = next(i for i, x in enumerate(out) if x.startswith(f"STEP {n} done"))
        assert i_plan < i_done
    assert "STEP 2 expected GPU calls: 0 report calls not cached (40 of 40 cached), rewrite calls at most 9" in out
    led = Ledger(project["ids"])
    assert set(led.values) >= {"checker_sha256", "s1_check_dev", "s2_generate_enforced", "s3_enforce_filter",
                               "s4_evaluate", "s5_check_test", "s6_text_metrics", "s7_probe_targets", "s7_run_probe",
                               "probe_threshold_sha256", "s7_analyze_probe", "s8_probe_targets", "s8_run_probe",
                               "s8_analyze_probe", "s9_make_tables"}
    gen = next(c for c in cli.calls if c[0] == "generate" and "--plan-only" not in c)
    assert gen[gen.index("--checker-sha256") + 1] == led.get("checker_sha256") and "--yes" in gen
    assert gen[gen.index("--retrieval-run") + 1] == INPUTS["test_retrieval"]
    ev = next(c for c in cli.calls if c[0] == "evaluate")
    assert led.get("s2_generate_enforced") in ev and led.get("s3_enforce_filter") in ev
    test_probe = [c for c in cli.calls if c[0] in ("run-probe", "analyze-probe")][-3:]
    assert all(c[c.index("--threshold-sha256") + 1] == led.get("probe_threshold_sha256") for c in test_probe)
    mt = cli.calls[-1]
    assert mt[mt.index("--evaluate-run") + 1: mt.index("--evaluate-run") + 3] == [INPUTS["test_evaluation"],
                                                                                led.get("s4_evaluate")]

    again = FakeCLI(project["root"])
    out2 = _run(project, again)
    assert again.calls == []                                      # nothing runs, not even a plan
    assert sum("skipped" in x for x in out2) == 14 and out2[-1] == "ALL STEPS DONE"


def test_interrupted_run_probe_resumes_with_the_same_run(project: dict[str, Any]) -> None:
    cli = FakeCLI(project["root"], fail={"run-probe": 1})
    with pytest.raises(StepError, match="run-probe"):
        _run(project, cli)
    assert _cmds(cli)[-1] == "run-probe" and "probe-threshold" not in _cmds(cli)     # stopped at the first error
    rp = Ledger(project["ids"]).get("s7_run_probe")
    cli2 = FakeCLI(project["root"])
    out = _run(project, cli2)
    assert _cmds(cli2)[:3] == ["run-probe plan", "run-probe", "probe-threshold"]       # steps 1-6 and targets skipped
    assert cli2.calls[0][-2:] == ["--resume", rp] and cli2.calls[1][-2:] == ["--resume", rp]
    assert "STEP 7 expected forward passes: 22 to run of 30 (8 already scored), 8 batches" in out
    assert Ledger(project["ids"]).get("s7_run_probe") == rp


def test_failed_generation_starts_again_under_a_new_id(project: dict[str, Any]) -> None:
    cli = FakeCLI(project["root"], fail={"generate": 1})
    with pytest.raises(StepError):
        _run(project, cli)
    first = Ledger(project["ids"]).get("s2_generate_enforced")
    cli2 = FakeCLI(project["root"])
    out = _run(project, cli2)
    second = Ledger(project["ids"]).get("s2_generate_enforced")
    assert second != first and any(f"previous run {first}: failed" in x for x in out)
    lines = [ln.split("\t") for ln in project["ids"].read_text(encoding="utf-8").splitlines()]
    assert [v for k, v, *_ in lines if k == "s2_generate_enforced"] == [first, second]   # every id recorded
    assert second in next(c for c in cli2.calls if c[0] == "evaluate")


def test_recorded_id_that_never_started_is_reused(project: dict[str, Any]) -> None:
    Ledger(project["ids"]).record("s1_check_dev", "20990101-000000-rall-check-dev")   # crashed before start_run
    cli = FakeCLI(project["root"])
    _run(project, cli)
    assert cli.calls[0][-2:] == ["--run-id", "20990101-000000-rall-check-dev"]


def test_dry_run_runs_nothing(project: dict[str, Any]) -> None:
    cli = FakeCLI(project["root"])
    out = _run(project, cli, dry=True)
    assert cli.calls == [] and not project["ids"].exists()
    assert sum("would start" in x for x in out) == 13 and out[-1] == "DRY RUN complete"
    assert any("would run probe-threshold" in x for x in out)
    cli2 = FakeCLI(project["root"], fail={"run-probe": 1})
    with pytest.raises(StepError):
        _run(project, cli2)
    out2 = _run(project, FakeCLI(project["root"]), dry=True)
    assert any("s7_run_probe: would resume" in x for x in out2) and any("s1_check_dev: finished earlier" in x
                                                                        for x in out2)


def test_guards(project: dict[str, Any]) -> None:
    with pytest.raises(StepError, match="base test reports"):          # enforced would not reuse the base reports
        _run(project, FakeCLI(project["root"], base_cached=39))
    assert not any(k == "s2_generate_enforced" for k in Ledger(project["ids"]).values)
    lex = project["root"] / "config" / "checker_lexicon.yaml"
    lex.write_text(lex.read_text(encoding="utf-8") + "\n# tuned\n", encoding="utf-8")
    with pytest.raises(StepError, match="checker changed"):
        _run(project, FakeCLI(project["root"]))


def test_skipped_step_must_have_the_same_arguments(project: dict[str, Any]) -> None:
    _run(project, FakeCLI(project["root"]))
    rid = Ledger(project["ids"]).get("s4_evaluate")
    mp = project["root"] / "runs" / rid / "run_meta.json"
    meta = json.loads(mp.read_text())
    meta["argv"] = [a if a != Ledger(project["ids"]).get("s3_enforce_filter") else "20990101-000000-other"
                    for a in meta["argv"]]
    mp.write_text(json.dumps(meta))
    with pytest.raises(StepError, match="other arguments"):
        _run(project, FakeCLI(project["root"]))


def test_threshold_from_another_probe_run_is_not_overwritten(project: dict[str, Any]) -> None:
    from ehr_rag.probe.analyze import write_threshold

    write_threshold(project["root"] / "config" / "probe_threshold.yaml",
                    {"threshold": 0.1, "quantile": 0.95, "n_null": 1, "n_within_flag": 1, "probe_run": "someone-else"})
    cli = FakeCLI(project["root"])
    with pytest.raises(StepError, match="refusing to overwrite"):
        _run(project, cli)
    assert "probe-threshold" not in _cmds(cli)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_wrapper_dry_run_logs_to_the_log_dir(project: dict[str, Any], tmp_path: Path) -> None:
    logs = tmp_path / "wrapper_logs"
    env = {**os.environ, "RUN_ALL_LOG_DIR": str(logs),
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}
    r = subprocess.run(["bash", str(REPO / "scripts" / "run_all.sh"), "--dry-run", "--config", project["config"],
                        "--references", "refs.csv"], env=env, capture_output=True, text=True, timeout=300)
    log = (logs / "run_all.log").read_text(encoding="utf-8")
    assert r.returncode == 0, log
    assert "DRY RUN complete" in log and "exit status 0" in log and not (logs / "run_all_ids.txt").exists()


# --------------------------------------------------------------------------- --plan-only flags


def test_generate_plan_only_counts_cached_base_reports_and_the_rewrite_bound(env: dict[str, Any],
                                                                           capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.generation.generate import run_generation
    from ehr_rag.run_all import parse_plan
    from ehr_rag.runs import start_run
    from fakes import FakeBackend, realistic_responder

    gen = start_run(env["cfg"], "generate", command="generate")
    run_generation(env["cfg"], gen, FakeBackend(realistic_responder), hadm_ids=[12, 21], conditions=["ehr_rag"],
                   retrieval_run=env["retrieval_run"], log=lambda s: None)
    n_runs = len(list((env["root"] / "runs").iterdir()))
    backend = FakeBackend(realistic_responder)
    res = run_generation(env["cfg"], None, backend, hadm_ids=[12, 21], conditions=["ehr_rag_enforced"],  # type: ignore[arg-type]
                         retrieval_run=env["retrieval_run"], plan_only=True)
    out = capsys.readouterr().out
    assert res["plan"] == parse_plan(out)
    assert res["plan"]["base_reports"] == res["plan"]["base_cached"] == 2 and res["plan"]["new"] == 0
    assert res["plan"]["rewrite_max"] >= 0 and backend.prompts == []                     # no model call
    assert len(list((env["root"] / "runs").iterdir())) == n_runs                         # no run folder
    cold = run_generation(env["cfg"], None, backend, hadm_ids=[12, 21], conditions=["ehr_rag_enforced"],  # type: ignore[arg-type]
                          retrieval_run=env["retrieval_run"], plan_only=True, no_cache=True)
    assert cold["plan"]["new"] == 2


def test_run_probe_plan_only(probe_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main
    from ehr_rag.probe import score
    from ehr_rag.run_all import parse_plan
    from test_probe import FakeScorer

    cfg = probe_env["cfg_path"]
    capsys.readouterr()
    before = set((probe_env["root"] / "runs").iterdir())
    assert main(["run-probe", "--config", cfg, "--targets-run", "20990101-000000-probe-targets", "--plan-only"]) == 0
    assert parse_plan(capsys.readouterr().out) == {"units": 2, "sequences": 30, "done": 0, "reused": 0, "todo": 30, "batches": 8}
    assert set((probe_env["root"] / "runs").iterdir()) == before
    monkeypatch.setattr(score, "make_scorer", lambda g: FakeScorer(fail_after=2))
    with pytest.raises(KeyboardInterrupt):
        main(["run-probe", "--config", cfg, "--targets-run", "20990101-000000-probe-targets",
              "--run-id", "20990101-000001-run-probe"])
    capsys.readouterr()
    assert main(["run-probe", "--config", cfg, "--targets-run", "20990101-000000-probe-targets", "--plan-only",
                 "--resume", "20990101-000001-run-probe"]) == 0
    assert parse_plan(capsys.readouterr().out)["done"] == 8


from test_probe import probe_env  # noqa: E402,F401  (fixture)


def test_make_tables_merges_evaluate_runs_and_refuses_duplicates(tmp_path: Path) -> None:
    from ehr_rag.tables import _merged_sentences

    def ev(name: str, conds: list[str]) -> Path:
        d = tmp_path / name
        d.mkdir()
        rows = [{"condition": c, "hadm_id": 1, "section": "Findings", "index": 0} for c in conds]
        (d / "sentences.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return d

    a, b = ev("a", ["ehr_rag", "full_context"]), ev("b", ["ehr_rag_enforced", "ehr_rag_filtered"])
    assert {s["condition"] for s in _merged_sentences([a, b])} == {"ehr_rag", "full_context", "ehr_rag_enforced",
                                                                   "ehr_rag_filtered"}
    with pytest.raises(ValueError, match="in both"):
        _merged_sentences([a, ev("c", ["ehr_rag"])])
