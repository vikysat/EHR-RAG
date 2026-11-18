"""Citation-dependence probe, prefix-consistency check: prefix_mentions_lab, the cited_edit_prefix
context (values replaced, direction words flipped in the forced report prefix), score reuse across
run-probe runs (only new sequences are scored), and the prefix splits of analyze-probe / table 6.

OBVIOUSLY FAKE units, reports and scorer only (checker_fakes, the fake tables of conftest).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from checker_fakes import checker, fake_units
from test_probe import REPO, FakeScorer, _cfg, _probe, probe_env  # noqa: F401  (fixture)

TARGETS = "20990101-000000-probe-targets"


def test_number_first_prefix_states_the_value_and_the_edit_replaces_it() -> None:
    from ehr_rag.probe.targets import apply_prefix_edits

    report = "### Findings\nPotassium 5.9, elevated [E1], sodium low at 128 [E2].\n"
    rp, _ = _probe(report)
    k, na = rp.units
    assert (k.order, k.prefix_lab_names, k.prefix_values, k.prefix_flips) == ("number_first", 1, 1, 0)
    assert k.prefix_mentions_lab and k.contexts[:3] == ["original", "cited_edit", "cited_edit_prefix"]
    spec = rp.specs[(0, "cited_edit_prefix")]
    assert spec["replace"] == rp.specs[(0, "cited_edit")]["replace"]
    assert apply_prefix_edits(report[:k.term_cut], spec["prefix_edits"]) == "### Findings\nPotassium 4.2,"
    # sodium: the prefix names sodium but states none of its values; potassium's 5.9 is not sodium's
    assert (na.prefix_lab_names, na.prefix_values, na.prefix_mentions_lab) == (1, 0, True)
    assert rp.specs[(1, "cited_edit_prefix")]["prefix_edits"] == []
    assert k.row()["prefix_mentions_lab"] is True


def test_prefix_without_the_lab_and_values_of_other_labs_are_not_counted() -> None:
    rp, _ = _probe()
    k, na = rp.units
    assert (k.prefix_lab_names, k.prefix_values, k.prefix_mentions_lab) == (1, 0, True)   # "Potassium was"
    assert (na.prefix_lab_names, na.prefix_values, na.prefix_mentions_lab) == (0, 0, False)
    # 4.1 is also a potassium value, but here it is a WBC count citing the WBC unit: not the cited lab
    rp, _ = _probe("### Findings\nWBC 4.1 stable [E6].\nPotassium elevated at 5.9 [E1].\n")
    (k,) = [u for u in rp.units if u.cited_unit == "E1"]
    assert (k.prefix_lab_names, k.prefix_values) == (1, 0) and rp.specs[(rp.units.index(k), "cited_edit_prefix")][
        "prefix_edits"] == []


def test_prefix_edit_flips_direction_words_and_replaces_cited_values() -> None:
    """``prefix_consistency`` directly (a unit is its lab's first direction claim, so the probe itself
    rarely meets a flip; the function must still do it)."""
    from ehr_rag.perturb.edit import audit_edit
    from ehr_rag.probe.runner import _rules_words
    from ehr_rag.probe.targets import _bounds, apply_prefix_edits, prefix_consistency

    cfg = _cfg()
    cfg["checker"]["rules"] = str(REPO / "config" / "checker.yaml")
    cfg["_project_root"] = str(REPO)
    units = fake_units()
    ck = checker()
    ctx = ck.admission(units, units)
    e1 = units[0]
    ce = audit_edit(e1, ck.lab_group("Potassium"), cfg["audit"]["outside_fraction"], cfg["audit"]["below_range_groups"],
                    _bounds(ck, e1, cfg["probe"]["physiologic_bounds"]))
    raw = "### Findings\nHigh potassium at 5.9 [E1]; sodium 130 [E2].\nPotassium remains elevated [E1].\n"
    cut = raw.index(" elevated")
    pi = prefix_consistency(ck, ctx, _rules_words(cfg), raw, cut, e1, ce)
    assert (pi.lab_names, pi.values, pi.flips) == (2, 1, 1)
    assert apply_prefix_edits(raw[:cut], pi.edits) == (
        "### Findings\nNormal potassium at 4.2 [E1]; sodium 130 [E2].\nPotassium remains")
    assert apply_prefix_edits(raw[:10], pi.edits) == raw[:10]      # edits past a shorter cut are not applied


def _set_prefix_edit(root: Path) -> None:
    """Give unit P0001's cited_edit_prefix context a (FAKE) prefix edit."""
    p = root / "runs" / TARGETS / "probe_contexts.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r["probe_id"] == "P0001" and r["context"] == "cited_edit_prefix":
            r["prefix_edits"] = [[0, 3, "XYZ"]]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class CountingScorer(FakeScorer):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.seen: list[tuple[str, str, str]] = []

    def score(self, batch: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
        self.seen += batch
        return super().score(batch)


def test_sequences_reuse_and_prefix_analysis(probe_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main
    from ehr_rag.config import load_config
    from ehr_rag.probe import score
    from ehr_rag.probe.runner import build_sequences
    from ehr_rag.run_all import parse_plan

    cfg, root = probe_env["cfg_path"], probe_env["root"]
    tdir = root / "runs" / TARGETS
    seqs, texts = build_sequences(load_config(cfg), tdir)
    # no prefix edit in these fake reports: cited_edit_prefix repeats cited_edit's term sequences (no new pass)
    pre = [s for s in seqs if s["context"] == "cited_edit_prefix"]
    assert {s["role"] for s in pre} == {"term_orig", "term_alt"} and len(texts) == 30
    ce = {(s["probe_id"], s["role"]): s["key"] for s in seqs if s["context"] == "cited_edit"}
    assert all(ce[(s["probe_id"], s["role"])] == s["key"] for s in pre)

    first = CountingScorer()
    monkeypatch.setattr(score, "make_scorer", lambda g: first)
    assert main(["run-probe", "--config", cfg, "--targets-run", TARGETS, "--run-id", "20990101-000001-run-probe"]) == 0
    assert main(["probe-threshold", "--config", cfg, "--probe-run", "20990101-000001-run-probe"]) == 0
    thr_file = Path(load_config(cfg)["probe"]["threshold_file"])
    frozen = thr_file.read_text(encoding="utf-8")

    _set_prefix_edit(root)                                   # now 2 new sequences (P0001's prefix-edited terms)
    seqs, texts = build_sequences(load_config(cfg), tdir)
    new = {k for k, (_, prefix, _) in texts.items() if prefix.startswith("XYZ")}
    assert len(texts) == 32 and len(new) == 2
    capsys.readouterr()
    assert main(["run-probe", "--config", cfg, "--targets-run", TARGETS, "--plan-only",
                 "--reuse-scores", "20990101-000001-run-probe"]) == 0
    assert parse_plan(capsys.readouterr().out) == {"units": 2, "sequences": 32, "done": 0, "reused": 30, "todo": 2,
                                                   "batches": 1}
    monkeypatch.setattr(score, "make_scorer", lambda g: FakeScorer(model="other"))
    assert main(["run-probe", "--config", cfg, "--targets-run", TARGETS, "--run-id", "20990101-000004-run-probe",
                 "--reuse-scores", "20990101-000001-run-probe"]) == 4                 # other model: refused
    assert "refusing to mix scores" in capsys.readouterr().err
    second = CountingScorer()
    monkeypatch.setattr(score, "make_scorer", lambda g: second)
    assert main(["run-probe", "--config", cfg, "--targets-run", TARGETS, "--run-id", "20990101-000005-run-probe",
                 "--reuse-scores", "20990101-000001-run-probe"]) == 0
    assert len(second.seen) == 2 and all(p.startswith("XYZ") for _, p, _ in second.seen)   # only what is new
    run = root / "runs" / "20990101-000005-run-probe"
    rows = [json.loads(x) for x in (run / "probe_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 32 and sum(1 for r in rows if r.get("reused_from") == "20990101-000001-run-probe") == 30
    assert json.loads((run / "run_meta.json").read_text(encoding="utf-8"))["reuse_scores"] == [
        "20990101-000001-run-probe"]

    assert main(["analyze-probe", "--config", cfg, "--probe-run", run.name, "--evaluate-run", probe_env["eval"],
                 "--run-id", "20990101-000006-analyze-probe"]) == 0
    assert thr_file.read_text(encoding="utf-8") == frozen                      # the frozen threshold is reused
    adir = root / "runs" / "20990101-000006-analyze-probe"
    s = json.loads((adir / "probe_summary.json").read_text(encoding="utf-8"))
    pf = s["prefix"]
    assert set(pf["by_prefix_mentions_lab"]) >= {"prefix_mentions_lab=yes", "prefix_mentions_lab=no",
                                                 "section=Findings, prefix_mentions_lab=no"}
    assert pf["dependence_prefix"]["all"]["n_units"] == s["splits"]["all"]["n_units"]
    yes, no = pf["by_prefix_mentions_lab"]["prefix_mentions_lab=yes"], pf["by_prefix_mentions_lab"]["prefix_mentions_lab=no"]
    assert yes["n_units"] + no["n_units"] == s["n_supported"]
    import csv

    with (adir / "probe_units_scored.csv").open(encoding="utf-8") as f:
        scored = {r["hadm_id"]: r for r in csv.DictReader(f)}
    # the FAKE scorer ignores the prefix, so the prefix-edited dependence equals the plain one
    assert float(scored["12"]["dependence_prefix"]) == pytest.approx(float(scored["12"]["dependence"]))
    assert scored["12"]["not_dependent_prefix"] == "False"
    assert main(["make-tables", "--config", cfg, "--probe-run", adir.name,
                 "--run-id", "20990101-000007-make-tables"]) == 0
    table = (root / "runs" / "20990101-000007-make-tables" / "table_6_probe.md").read_text(encoding="utf-8")
    assert "dependence_prefix: all" in table and "prefix check: section=Findings, prefix_mentions_lab=" in table
    assert "Fakeium" not in table


def test_summary_leaves_out_units_of_targets_runs_without_the_prefix_check() -> None:
    from ehr_rag.probe.analyze import summarize, summary_markdown

    rows = [{"probe_id": f"P{i}", "hadm_id": i, "section": "Findings", "order": "term_first", "corroborated": False,
             "dependence": 1.0 * i, "dependence_prefix": None, "prefix_mentions_lab": None, "supported": True,
             "num_dependence": None} for i in range(3)]
    s = summarize(rows, {"threshold": 1.5, "sha256": "x" * 64}, 50, 1)
    assert s["splits"]["all"]["n_units"] == 3 and s["prefix"]["n_supported_with_prefix_flag"] == 0
    assert s["prefix"]["dependence_prefix"]["all"]["n_units"] == 0
    assert "dependence_prefix" in summary_markdown(s, ["r"])
