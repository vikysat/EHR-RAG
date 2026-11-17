"""Citation-dependence probe: dev-only threshold with its hash, the test-split guard, the headline
and table 6 (OBVIOUSLY FAKE data and scorer; fixtures from test_probe)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest

from test_probe import FakeScorer, probe_env  # noqa: F401  (fixture)


def test_threshold_headline_and_table(probe_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main
    from ehr_rag.probe import score

    cfg = probe_env["cfg_path"]
    monkeypatch.setattr(score, "make_scorer", lambda g: FakeScorer())
    assert main(["run-probe", "--config", cfg, "--targets-run", "20990101-000000-probe-targets",
                 "--run-id", "20990101-000001-run-probe"]) == 0
    assert main(["probe-threshold", "--config", cfg, "--probe-run", "20990101-000001-run-probe"]) == 0
    out = capsys.readouterr().out
    sha = re.search(r"sha256 ([0-9a-f]{64})", out).group(1)                 # type: ignore[union-attr]
    assert "threshold 0.0000" in out and "0 matched-null and 2 within-flag drops" in out
    assert main(["probe-threshold", "--config", cfg, "--probe-run", "20990101-000001-run-probe"]) == 4  # exists
    assert main(["analyze-probe", "--config", cfg, "--probe-run", "20990101-000001-run-probe",
                 "--evaluate-run", probe_env["eval"], "--run-id", "20990101-000002-analyze-probe"]) == 0
    run = probe_env["root"] / "runs" / "20990101-000002-analyze-probe"
    with (run / "probe_units_scored.csv").open(encoding="utf-8") as f:
        rows = {r["hadm_id"]: r for r in csv.DictReader(f)}
    # adm 12 has a [HIGH] point: editing it to normal flips the fake model -> dependent;
    # adm 21 is LOW: "Elevated" was already unlikely, the edit changes nothing -> not dependent (0 < 0 is False)
    assert float(rows["12"]["dependence"]) == pytest.approx(5.8) and rows["12"]["not_dependent"] == "False"
    assert float(rows["21"]["dependence"]) == 0 and float(rows["12"]["num_dependence"]) > 3
    assert rows["12"]["supported"] == "True"
    s = json.loads((run / "probe_summary.json").read_text(encoding="utf-8"))
    assert s["threshold_sha256"] == sha and s["splits"]["all"]["n_units"] == s["n_supported"]
    assert set(s["splits"]) >= {"all", "order=number_first", "order=term_first", "corroborated=no", "section=Findings"}
    assert main(["make-tables", "--config", cfg, "--probe-run", run.name,
                 "--run-id", "20990101-000003-make-tables"]) == 0
    table = (probe_env["root"] / "runs" / "20990101-000003-make-tables" / "table_6_probe.md").read_text(encoding="utf-8")
    assert "Table 6" in table and "number token (positive control)" in table


def test_test_split_requires_the_unchanged_threshold(probe_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.config import load_config
    from ehr_rag.probe.analyze import write_threshold
    from ehr_rag.probe.runner import check_threshold_for_split, read_threshold
    from ehr_rag.probe.score import ProbeError

    cfg = load_config(probe_env["cfg_path"])
    with pytest.raises(ProbeError, match="not found"):
        check_threshold_for_split(cfg, "test", "abc")
    assert check_threshold_for_split(cfg, "dev", None) is None
    path = Path(cfg["probe"]["threshold_file"])
    body = write_threshold(path, {"threshold": 0.5, "quantile": 0.95, "n_null": 3, "n_within_flag": 4})
    with pytest.raises(ProbeError, match="--threshold-sha256"):
        check_threshold_for_split(cfg, "test", None)
    with pytest.raises(ProbeError, match="differs"):
        check_threshold_for_split(cfg, "test", "0" * 64)
    assert check_threshold_for_split(cfg, "test", body["sha256"])["threshold"] == 0.5
    path.write_text(path.read_text(encoding="utf-8").replace("threshold: 0.5", "threshold: 0.9"), encoding="utf-8")
    with pytest.raises(ProbeError, match="edited"):
        read_threshold(cfg)
    # run-probe refuses a test-split targets run without the matching hash
    meta_p = probe_env["root"] / "runs" / "20990101-000000-probe-targets" / "probe_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta_p.write_text(json.dumps({**meta, "split": "test"}), encoding="utf-8")
    from ehr_rag.cli import main

    assert main(["run-probe", "--config", probe_env["cfg_path"], "--targets-run", "20990101-000000-probe-targets",
                 "--threshold-sha256", body["sha256"]]) == 4
    assert "edited" in capsys.readouterr().err
