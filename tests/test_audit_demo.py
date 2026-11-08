"""The whole counterfactual audit on REAL MIMIC-IV Demo units (open data) with the FAKE backend.

Two Demo admissions are built into a temporary project; retrieval uses the fake encoder and
generation the fake backend, so nothing is downloaded and no model is loaded. Skipped if the
Demo is not present.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from fakes import FakeBackend, FakeEncoder, realistic_responder
from test_audit import _responder

REPO = Path(__file__).resolve().parents[1]
HADM = [23559586, 29276678]


def _demo_root() -> Path:
    cfg = yaml.safe_load((REPO / "config" / "default.yaml").read_text(encoding="utf-8"))
    return REPO / cfg["data"]["root"]


pytestmark = pytest.mark.skipif(not _demo_root().is_dir(), reason="MIMIC-IV Demo not present")


@pytest.fixture()
def demo_project(tmp_path: Path) -> dict[str, Any]:
    from ehr_rag.config import load_config
    from ehr_rag.data.load import load_tables
    from ehr_rag.evidence.build_units import build_all, units_dir
    from ehr_rag.evidence.schema import read_units
    from ehr_rag.generation.generate import run_generation
    from ehr_rag.retrieval.retrieve import Encoders, evidence_record, retrieve_admission
    from ehr_rag.runs import start_run

    raw = yaml.safe_load((REPO / "config" / "default.yaml").read_text(encoding="utf-8"))
    raw["data"].update(root=str(_demo_root()), derived=str(tmp_path / "derived"))
    raw["checker"].update(rules=str(REPO / "config" / "checker.yaml"), lexicon=str(REPO / "config" / "checker_lexicon.yaml"))
    raw["generation"]["max_context_tokens"] = 10 ** 6
    (tmp_path / "config").mkdir()
    (tmp_path / "prompts").mkdir()
    for p in (REPO / "prompts").glob("*.txt"):
        (tmp_path / "prompts" / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    cfg_path = tmp_path / "config" / "demo.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_path)
    out = units_dir(Path(cfg["data"]["derived"]))
    build_all(load_tables(cfg), HADM, cfg["units"], out)
    enc = Encoders(query=FakeEncoder(), doc=FakeEncoder(), cross=None)  # type: ignore[arg-type]
    rrun = start_run(cfg, "retrieve", command="retrieve")
    for h in HADM:
        units = read_units(out / f"{h}.jsonl")
        rrun.write_json(f"retrieval/{h}.json", evidence_record(retrieve_admission(units, enc, cfg["retrieval"]), units))
    gen = start_run(cfg, "generate", command="generate")
    run_generation(cfg, gen, FakeBackend(realistic_responder), hadm_ids=HADM, conditions=["ehr_rag"],
                   retrieval_run=rrun.run_id, log=lambda s: None)
    return {"cfg_path": cfg_path, "root": tmp_path, "gen": gen.run_id}


def test_audit_on_demo_units(demo_project: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture) -> None:
    import ehr_rag.generation.backends as backends
    from ehr_rag.cli import main

    cfg, root = str(demo_project["cfg_path"]), demo_project["root"]
    assert main(["audit-targets", "--config", cfg, "--generation-run", demo_project["gen"],
                 "--run-id", "20990101-000000-audit-targets"]) == 0
    tdir = root / "runs" / "20990101-000000-audit-targets"
    with (tdir / "audit_targets.csv").open(encoding="utf-8") as f:
        targets = list(csv.DictReader(f))
    edits = [json.loads(x) for x in (tdir / "audit_edits.jsonl").read_text(encoding="utf-8").splitlines()]
    by_h = {h: [t for t in targets if int(t["hadm_id"]) == h] for h in HADM}
    for h in HADM:                                       # up to 3 cited + 1 null per admission
        assert 1 <= sum(t["kind"] == "cited" for t in by_h[h]) <= 3
        assert sum(t["kind"] == "null" for t in by_h[h]) == 1
    flagged_first = [t for t in targets if t["kind"] == "cited"]
    assert any(t["status"] != "normal" for t in flagged_first)
    with (tdir / "audit_skips.csv").open(encoding="utf-8") as f:
        assert any(r["reason"] == "no reference range" for r in csv.DictReader(f))   # real unranged Demo labs
    assert {e["edit_type"] for e in edits} <= {"to_normal", "to_high", "to_low"}

    monkeypatch.setattr(backends, "make_backend", lambda g: FakeBackend(_responder("follow", edits)))
    capsys.readouterr()
    assert main(["run-audit", "--config", cfg, "--targets-run", tdir.name, "--yes",
                 "--run-id", "20990101-000001-run-audit"]) == 0
    plan = capsys.readouterr().out
    n_jobs = len(HADM) + len(targets)
    assert f"report jobs {2 * n_jobs}; distinct prompts {n_jobs}" in plan
    assert main(["analyze-audit", "--config", cfg, "--audit-run", "20990101-000001-run-audit",
                 "--run-id", "20990101-000002-analyze-audit"]) == 0
    arun = root / "runs" / "20990101-000002-analyze-audit"
    with (arun / "audit_pairs.csv").open(encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    plain = [p for p in pairs if p["condition"] == "ehr_rag"]
    assert plain and all(p["auto_label"] in ("follows", "other") for p in plain)   # this fake model follows edits
    assert any(p["auto_label"] == "follows" for p in plain)
    with (arun / "audit_null_churn.csv").open(encoding="utf-8") as f:
        churn = list(csv.DictReader(f))
    assert {int(r["hadm_id"]) for r in churn} == set(HADM)                          # every admission had a null edit
    summary = json.loads((arun / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["conditions"]["ehr_rag"]["faithfulness"]["estimate"] == 1.0
    assert summary["conditions"]["ehr_rag"]["faithfulness"]["n_resamples"] == 1000
    assert main(["make-tables", "--config", cfg, "--audit-run", arun.name, "--run-id",
                 "20990101-000003-make-tables"]) == 0
    assert (root / "runs" / "20990101-000003-make-tables" / "table_2_audit.md").exists()
