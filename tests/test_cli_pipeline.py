"""The whole CLI chain on fake data: generate -> evaluate -> select-targets -> run-pilot -> analyze-perturbation.

Model classes are monkeypatched to the fakes in tests/fakes.py; nothing is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ehr_rag.cli import main
from fakes import FakeBackend, FakeNLI, realistic_responder


def _latest(root: Path, command: str) -> str:
    runs = [p for p in (root / "runs").iterdir()
            if (p / "run_meta.json").exists()
            and json.loads((p / "run_meta.json").read_text(encoding="utf-8"))["command"] == command]
    return sorted(runs)[-1].name


def run_pipeline(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Run every Phase 4-6 command via the CLI; return the run ids."""
    import ehr_rag.eval.nli as nli_mod
    import ehr_rag.generation.backends as backends

    monkeypatch.setattr(backends, "make_backend", lambda gen: FakeBackend(realistic_responder))
    monkeypatch.setattr(nli_mod, "HFNLI", lambda name: FakeNLI())
    root = env["root"]
    cfg = str(root / "config" / "test.yaml")
    ids = ["--hadm-ids", "12", "21"]
    assert main(["generate", "--config", cfg, "--retrieval-run", env["retrieval_run"], "--yes", *ids]) == 0
    gen = _latest(root, "generate")
    assert main(["evaluate", "--config", cfg, "--generation-run", gen]) == 0
    ev = _latest(root, "evaluate")
    assert main(["select-targets", "--config", cfg, "--generation-run", gen]) == 0
    tg = _latest(root, "select-targets")
    assert main(["run-pilot", "--config", cfg, "--targets-run", tg]) == 5  # needs --yes (21 > 20)
    assert main(["run-pilot", "--config", cfg, "--targets-run", tg, "--yes",
                 "--run-id", "20990101-000000-run-pilot-job1"]) == 0
    pilot = "20990101-000000-run-pilot-job1"
    assert main(["analyze-perturbation", "--config", cfg, "--pilot-run", pilot]) == 0
    an = _latest(root, "analyze-perturbation")
    return {"generate": gen, "evaluate": ev, "select-targets": tg, "run-pilot": pilot,
            "analyze-perturbation": an, "retrieve": env["retrieval_run"]}


def test_cli_pipeline(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    ids = run_pipeline(env, monkeypatch)
    out = capsys.readouterr().out
    assert "10 random sentences" in out and "Worked examples" in out and "NOT RUN" in out
    runs = env["root"] / "runs"
    meta = json.loads((runs / ids["generate"] / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ok" and meta["model_revision"] == "fake-revision" and "gpu_name" in meta
    assert meta["format_counts"]["full_context"]["ok"] == 2
    assert (runs / ids["evaluate"] / "manual_review.csv").exists()
    pmeta = json.loads((runs / ids["run-pilot"] / "run_meta.json").read_text(encoding="utf-8"))
    assert pmeta["status"] == "ok" and pmeta["targets_run"] == ids["select-targets"]
    assert (runs / ids["analyze-perturbation"] / "perturbation_summary.md").exists()
