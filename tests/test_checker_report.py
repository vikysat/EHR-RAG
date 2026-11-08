"""check-citations: error profile, repair vs retreat, coverage (FAKE runs, no model)."""

from __future__ import annotations

import csv
import json
from typing import Any

import pytest
import yaml

from ehr_rag.checker.report import enforcement_rows, profile
from fakes import FakeBackend
from test_enforce import _gen, corrupting_responder


def _line(cond: str, section: str, status: str, errors: list[str], n_numbers: int = 1) -> dict[str, Any]:
    return {"condition": cond, "hadm_id": 1, "section": section, "index": 0, "status": status,
            "error_types": errors, "n_numbers": n_numbers, "n_names": 1, "items": [], "errors": []}


def test_profile_counts_and_coverage() -> None:
    lines = [_line("ehr_rag", "Findings", "fail", ["number_mismatch", "name_missing"]),
             _line("ehr_rag", "Findings", "pass", []),
             _line("ehr_rag", "Impression", "unchecked", [], 0),
             _line("no_retrieval", "History", "fail", ["number_mismatch"])]
    rows = {(r["condition"], r["scope"]): r for r in profile(lines)}
    a = rows[("ehr_rag", "ALL")]
    assert (a["n_lines"], a["n_pass"], a["n_fail"], a["n_unchecked"]) == (3, 1, 1, 1)
    assert a["checkable_share"] == pytest.approx(2 / 3) and a["pass_share_of_checkable"] == pytest.approx(1 / 2)
    assert a["number_mismatch"] == 1 and a["name_missing"] == 1 and a["trend_mismatch"] == 0
    assert rows[("ehr_rag", "Findings")]["n_lines"] == 2 and rows[("ehr_rag", "Recommendations")]["n_lines"] == 0
    assert rows[("ehr_rag", "Recommendations")]["checkable_share"] is None
    assert rows[("no_retrieval", "History+Findings")]["number_mismatch"] == 1


def test_enforcement_rows_shares() -> None:
    t = {"n_lines": 10, "n_pass": 5, "n_unchecked": 2, "n_fail": 3, "n_kept": 7, "n_repaired": 1, "n_retreated": 1,
         "n_dropped": 1, "n_rewrite_calls": 3, "n_lines_after": 9, "n_numbers_before": 8, "n_numbers_after": 6,
         "n_names_before": 10, "n_names_after": 9, "n_sections_emptied": 0}
    (r,) = enforcement_rows({"ehr_rag_enforced": t})
    assert (r["kept"], r["repaired"], r["retreated"], r["dropped"]) == (7, 1, 1, 1)
    assert r["lines_kept_share"] == pytest.approx(0.9) and r["numbers_kept_share"] == pytest.approx(0.75)


def test_check_citations_cli(env: dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main

    gen = _gen(env, ["no_retrieval", "ehr_rag", "ehr_rag_enforced"], FakeBackend(corrupting_responder))
    cfg = str(env["root"] / "config" / "test.yaml")
    assert main(["enforce-filter", "--config", cfg, "--generation-run", gen.name, "--conditions", "ehr_rag",
                 "--run-id", "20990101-000000-enforce-filter"]) == 0
    capsys.readouterr()
    assert main(["check-citations", "--config", cfg, "--generation-run", gen.name, "20990101-000000-enforce-filter",
                 "--run-id", "20990101-000001-check-citations"]) == 0
    out = capsys.readouterr().out
    assert "Error profile" in out and "ehr_rag_enforced" in out and "ehr_rag_filtered" in out
    run = env["root"] / "runs" / "20990101-000001-check-citations"
    with (run / "checker_profile.csv").open(encoding="utf-8") as f:
        prof = {(r["condition"], r["scope"]): r for r in csv.DictReader(f)}
    assert set(c for c, _ in prof) == {"no_retrieval", "ehr_rag"}            # enforced text is never re-graded
    assert int(prof[("ehr_rag", "Findings")]["number_mismatch"]) > 0          # the corrupted lab values
    assert int(prof[("no_retrieval", "ALL")]["n_lines"]) > 0
    with (run / "enforcement_summary.csv").open(encoding="utf-8") as f:
        enf = {r["condition"]: r for r in csv.DictReader(f)}
    e, fl = enf["ehr_rag_enforced"], enf["ehr_rag_filtered"]
    assert int(e["repaired"]) > 0 and int(fl["repaired"]) == 0 and int(fl["rewrite_calls"]) == 0
    assert int(e["n_lines"]) == int(fl["n_lines"])                            # same base reports
    assert int(e["kept"]) + int(e["repaired"]) + int(e["retreated"]) + int(e["dropped"]) == int(e["n_lines"])
    assert float(e["numbers_kept_share"]) >= float(fl["numbers_kept_share"])  # rewriting keeps more numbers
    with (run / "checker_review.csv").open(encoding="utf-8") as f:
        review = list(csv.DictReader(f))
    n_fail = sum(int(prof[(c, "ALL")]["n_fail"]) for c in ("no_retrieval", "ehr_rag"))
    assert sum(1 for r in review if r["status"] == "fail") == n_fail
    assert all(r["human_checker_correct"] == "" for r in review)
    summary = (run / "checker_summary.md").read_text(encoding="utf-8")
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["checker"]["lexicon_sha256"] in summary and "independent of `evaluate`" in summary


def test_check_citations_prints_counts_only_on_full_data(env: dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main

    gen = _gen(env, ["ehr_rag"], FakeBackend(corrupting_responder))
    cfg_path = env["root"] / "config" / "test.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["data"]["source"] = "full"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    capsys.readouterr()
    assert main(["check-citations", "--config", str(cfg_path), "--generation-run", gen.name]) == 0
    out = capsys.readouterr().out
    assert "Fakeium" not in out and "reached" not in out and "Fake" not in out
