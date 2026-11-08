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


def test_demo_notebook_runs_top_to_bottom(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture) -> None:
    """Execute every code cell of notebooks/demo.ipynb against the fake runs (no model calls)."""
    ids = run_pipeline(env, monkeypatch)
    capsys.readouterr()
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb").read_text(encoding="utf-8"))
    monkeypatch.chdir(env["root"])
    ns: dict[str, Any] = {"__name__": "__demo__"}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "demo.ipynb", "exec"), ns)
    out = capsys.readouterr().out
    assert ns["GEN_RUN"] == ids["generate"] and ns["ANALYSIS_RUN"] == ids["analyze-perturbation"]
    assert ns["RETRIEVE_RUN"] == ids["retrieve"] and ns["EVAL_RUN"] == ids["evaluate"]
    for text in ("Age:", "primary diagnosis:", "### Findings", "SUPPORTED", "report diff", "Perturbation summary",
                 "provisional"):
        assert text in out, text
    assert "not available" not in out


# --------------------------------------------------------------------------- sharded generation


def test_shard_blocks_cover_all_admissions_once() -> None:
    from ehr_rag.cli import shard
    from ehr_rag.config import ConfigError

    ids = list(range(1000, 1100))
    parts = [shard(ids, 25, k) for k in range(4)]
    assert [len(p) for p, _ in parts] == [25, 25, 25, 25]
    assert [h for p, _ in parts for h in p] == ids                       # contiguous, disjoint, complete
    assert parts[3][1] == {"shard_index": 3, "shard_size": 25, "n_shards": 4, "n_admissions_all_shards": 100}
    assert shard(ids[:20], 25, 0)[0] == ids[:20]                          # dev: one shard
    assert [len(shard(ids[:30], 25, k)[0]) for k in (0, 1)] == [25, 5]
    assert shard(ids, None, None) == (ids, {})
    with pytest.raises(ConfigError, match="indices 0..3"):
        shard(ids, 25, 4)
    with pytest.raises(ConfigError, match="both"):
        shard(ids, 25, None)


def test_sharded_generate_then_evaluate_equals_unsharded(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture) -> None:
    import ehr_rag.eval.nli as nli_mod
    import ehr_rag.generation.backends as backends

    monkeypatch.setattr(backends, "make_backend", lambda gen: FakeBackend(realistic_responder))
    monkeypatch.setattr(nli_mod, "HFNLI", lambda name: FakeNLI())
    root, cfg = env["root"], str(env["root"] / "config" / "test.yaml")
    base = ["generate", "--config", cfg, "--retrieval-run", env["retrieval_run"], "--yes", "--hadm-ids", "12", "21"]
    assert main(base + ["--run-id", "20990101-000000-generate-job7-shard0", "--shard-size", "1",
                        "--shard-index", "0"]) == 0
    assert main(base + ["--run-id", "20990101-000000-generate-job7-shard1", "--shard-size", "1",
                        "--shard-index", "1"]) == 0
    assert main(base + ["--shard-size", "1", "--shard-index", "2"]) == 2      # out of range: nothing generated
    assert "out of range" in capsys.readouterr().err
    runs = root / "runs"
    m0 = json.loads((runs / "20990101-000000-generate-job7-shard0" / "run_meta.json").read_text(encoding="utf-8"))
    assert m0["hadm_ids"] == [12] and m0["n_shards"] == 2 and m0["status"] == "ok"
    assert sorted(p.name for p in (runs / "20990101-000000-generate-job7-shard1" / "ehr_rag").iterdir()) == ["21"]
    full = "20990101-000000-generate-unsharded"
    assert main(base + ["--run-id", full]) == 0
    shards = ["20990101-000000-generate-job7-shard0", "20990101-000000-generate-job7-shard1"]
    assert main(["evaluate", "--config", cfg, "--generation-run", *shards, "--run-id",
                 "20990101-000001-evaluate-shards"]) == 0
    assert main(["evaluate", "--config", cfg, "--generation-run", full, "--run-id",
                 "20990101-000002-evaluate-full"]) == 0
    a = json.loads((runs / "20990101-000001-evaluate-shards" / "metrics.json").read_text(encoding="utf-8"))
    b = json.loads((runs / "20990101-000002-evaluate-full" / "metrics.json").read_text(encoding="utf-8"))
    assert a["generation_run"] == shards and b["generation_run"] == full
    assert a["conditions"] == b["conditions"]
    mr = [(runs / r / "manual_review.csv").read_text(encoding="utf-8").replace(r, "RUN")
          for r in ("20990101-000001-evaluate-shards", "20990101-000002-evaluate-full")]
    assert mr[0] == mr[1]


def test_overlapping_shards_are_refused(env: dict[str, Any]) -> None:
    from ehr_rag.eval.evaluate import load_generation_runs
    from ehr_rag.generation.generate import run_generation
    from ehr_rag.runs import start_run

    run = start_run(env["cfg"], "generate", command="generate")
    run_generation(env["cfg"], run, FakeBackend(realistic_responder), hadm_ids=[12], conditions=["ehr_rag"],
                   retrieval_run=env["retrieval_run"], log=lambda s: None)
    with pytest.raises(ValueError, match="is in both"):
        load_generation_runs([run.path, run.path])
