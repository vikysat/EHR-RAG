"""make-tables from a full FAKE chain: evaluate, check-citations, text-metrics, audit (no model)."""

from __future__ import annotations

import csv
import json
from typing import Any

import pytest

from ehr_rag.cli import main
from fakes import FakeBackend, FakeNLI, realistic_responder
from test_audit import _base_run, _responder, _targets
from test_enforce import corrupting_responder


def _latest(root: Any, command: str) -> str:
    runs = [p for p in (root / "runs").iterdir() if (p / "run_meta.json").exists()
            and json.loads((p / "run_meta.json").read_text(encoding="utf-8"))["command"] == command]
    return sorted(runs)[-1].name


def test_make_tables_from_all_sources(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture) -> None:
    import ehr_rag.eval.nli as nli_mod
    import ehr_rag.generation.backends as backends

    root = env["root"]
    cfg = str(root / "config" / "test.yaml")
    gen = _base_run(env, corrupting_responder, ["no_retrieval", "ehr_rag", "ehr_rag_enforced"])
    monkeypatch.setattr(nli_mod, "HFNLI", lambda name: FakeNLI())
    assert main(["evaluate", "--config", cfg, "--generation-run", gen, "--run-id", "20990101-000001-evaluate"]) == 0
    ev = root / "runs" / "20990101-000001-evaluate"
    with (ev / "manual_review.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):                               # FAKE hand labels: every other one unsupported
        r["human_support"] = "unsupported" if i % 2 else "supported"
    with (ev / "manual_review_labeled.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert main(["check-citations", "--config", cfg, "--generation-run", gen, "--run-id",
                 "20990101-000002-check-citations"]) == 0
    refs = root / "fake_refs.csv"
    with refs.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["hadm_id", "reference"], [12, "FAKE summary: Fakeium reached 5.9."],
                                 [21, "FAKE summary: Fakeium was 2.9."]])
    assert main(["text-metrics", "--config", cfg, "--generation-run", gen, "--references", str(refs),
                 "--run-id", "20990101-000003-text-metrics"]) == 0
    tdir = _targets(env, gen)
    edits = [json.loads(x) for x in (tdir / "audit_edits.jsonl").read_text(encoding="utf-8").splitlines()]
    monkeypatch.setattr(backends, "make_backend", lambda g: FakeBackend(_responder("ignore", edits)))
    assert main(["run-audit", "--config", cfg, "--targets-run", tdir.name, "--yes",
                 "--run-id", "20990101-000004-run-audit"]) == 0
    assert main(["analyze-audit", "--config", cfg, "--audit-run", "20990101-000004-run-audit",
                 "--run-id", "20990101-000005-analyze-audit"]) == 0
    capsys.readouterr()
    assert main(["make-tables", "--config", cfg, "--evaluate-run", ev.name, "--manual-labels",
                 str(ev / "manual_review_labeled.csv"), "--check-run", "20990101-000002-check-citations",
                 "--text-metrics-run", "20990101-000003-text-metrics", "--audit-run", "20990101-000005-analyze-audit",
                 "--run-id", "20990101-000006-make-tables"]) == 0
    out = capsys.readouterr().out
    metrics = json.loads((ev / "metrics.json").read_text(encoding="utf-8"))
    n_groups = sum(1 for c in metrics["conditions"].values() for g in ("History+Findings",
                   "Impression+Recommendations", "ALL") if c[g]["n_sentences"])          # empty groups are left out
    assert f"table 1_support: {n_groups} rows" in out and "table 2_audit: 3 rows" in out
    assert "Fakeium" not in out and "FAKE" not in out                    # counts only
    run = root / "runs" / "20990101-000006-make-tables"
    doc = (run / "tables.md").read_text(encoding="utf-8")
    for k in ("1_support", "2_audit", "3_enforcement", "4_checker", "5_reference"):
        assert (run / f"table_{k}.md").exists(), k
    # table 1 recall (ALL) equals the evaluation's own pooled recall
    recall = metrics["conditions"]["ehr_rag"]["ALL"]["citation_recall"]
    t1 = [ln for ln in (run / "table_1_support.md").read_text(encoding="utf-8").splitlines()
          if ln.startswith("| ehr_rag | ALL")][0].split(" | ")
    assert t1[7].startswith(f"{recall:.3f}") and "[" in t1[7]                 # estimate [95% CI]
    assert "(n=" in t1[8]                                                      # manual unsupported present
    # table 2 equals the audit summary; the paired row is there
    summ = json.loads((root / "runs" / "20990101-000005-analyze-audit" / "audit_summary.json").read_text(encoding="utf-8"))
    t2 = (run / "table_2_audit.md").read_text(encoding="utf-8")
    plain = summ["conditions"]["ehr_rag"]
    assert f"| ehr_rag | {plain['n_pairs']} | {plain['follows']} | {plain['stale']} |" in t2
    assert "enforced - plain (paired)" in t2
    # tables 3 and 4 equal the check-citations totals
    with (root / "runs" / "20990101-000002-check-citations" / "enforcement_summary.csv").open(encoding="utf-8") as f:
        enf = {r["condition"]: r for r in csv.DictReader(f)}["ehr_rag_enforced"]
    t3 = (run / "table_3_enforcement.md").read_text(encoding="utf-8")
    assert f"| ehr_rag_enforced | {enf['n_fail']} | {enf['repaired']} | {enf['retreated']} | {enf['dropped']} |" in t3
    with (root / "runs" / "20990101-000002-check-citations" / "checker_profile.csv").open(encoding="utf-8") as f:
        prof = {(r["condition"], r["scope"]): r for r in csv.DictReader(f)}
    t4 = (run / "table_4_checker.md").read_text(encoding="utf-8")
    nm = prof[("ehr_rag", "ALL")]["number_mismatch"]
    assert f"| ehr_rag | {prof[('ehr_rag', 'ALL')]['n_lines']} | " in t4 and f"| {nm} (" in t4
    # table 5: ROUGE-L per condition, BERTScore n/a without --bertscore
    t5 = (run / "table_5_reference.md").read_text(encoding="utf-8")
    assert "| ehr_rag | 2 | 0." in t5 and "n/a" in t5
    assert "Sources:" in doc


def test_make_tables_needs_a_source(env: dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    assert main(["make-tables", "--config", str(env["root"] / "config" / "test.yaml")]) == 2
    assert "at least one" in capsys.readouterr().err
