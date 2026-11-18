"""The whole citation-dependence probe on REAL MIMIC-IV Demo units (open data) with the FAKE backend
and the FAKE scorer: real lab labels, units and reference ranges go through unit selection, the
edits, the physiologic bounds, scoring, the threshold and the headline. Skipped if the Demo is absent.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from test_audit_demo import HADM, _demo_root, build_demo_project
from test_probe import FakeScorer

pytestmark = pytest.mark.skipif(not _demo_root().is_dir(), reason="MIMIC-IV Demo not present")

BAND_POINTS = ("first", "max", "min", "last")


def _responder(prompt: str, n: int) -> str:
    """Restates the evidence (``realistic_report``) with a direction word on every lab line; every other
    lab line is first restated plainly (no direction word), so that unit's prefix names its lab and value."""
    from fakes import realistic_report

    rep = realistic_report(prompt)
    out = []
    k = 0
    for line in rep.splitlines():
        if " reached " in line:
            if k % 2 == 0:
                out.append(line)
            out.append(f"Elevated {line[0].lower()}{line[1:]}")
            k += 1
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def _side(f: dict[str, Any], p: str) -> str:
    v, lo, hi = Decimal(f[p]["value"]), Decimal(f["ref_low"]), Decimal(f["ref_high"])
    return "high" if v > hi else "low" if v < lo else "normal"


def test_probe_on_demo_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.checker.rules import Checker
    from ehr_rag.cli import main
    from ehr_rag.eval.evaluate import run_evaluation
    from ehr_rag.perturb.edit import physiologic_bounds
    from ehr_rag.probe import score
    from ehr_rag.runs import start_run
    from fakes import FakeNLI

    proj = build_demo_project(tmp_path, _responder)
    cfg, root, ccfg = str(proj["cfg_path"]), proj["root"], proj["cfg"]
    assert main(["probe-targets", "--config", cfg, "--generation-run", proj["gen"],
                 "--run-id", "20990101-000000-probe-targets"]) == 0
    out = capsys.readouterr().out
    assert "split: demo" in out and "expected forward passes" in out
    tdir = root / "runs" / "20990101-000000-probe-targets"
    with (tdir / "probe_units.csv").open(encoding="utf-8") as f:
        units = list(csv.DictReader(f))
    assert units and {int(u["hadm_id"]) for u in units} <= set(HADM)
    for u in units:                                       # offsets point at the term in the stored report
        d = root / "runs" / proj["gen"] / "ehr_rag" / u["hadm_id"]
        raw = (d / json.loads((d / "parsed.json").read_text(encoding="utf-8"))["raw_file"]).read_text(encoding="utf-8")
        assert raw[int(u["term_start"]):int(u["term_end"])] == u["surface"] == "Elevated"
        if u["number"]:
            assert raw[int(u["number_start"]):int(u["number_end"])] == u["number"]
    ck = Checker.from_config(ccfg)
    table = ccfg["probe"]["physiologic_bounds"]
    ctxs = [json.loads(x) for x in (tdir / "probe_contexts.jsonl").read_text(encoding="utf-8").splitlines()]
    for c in ctxs:
        for e in c.get("replace", []):
            f0, f1 = e["original_fields"], e["edited_fields"]
            lo, hi = physiologic_bounds(table, ck.lab_group(str(f1["label"])), f1.get("uom"))
            for p in BAND_POINTS:
                v = Decimal(f1[p]["value"])
                assert v >= lo and (hi is None or v <= hi)                    # clamped, never negative
                if c["context"] == "within_flag":
                    assert _side(f1, p) == _side(f0, p) and v != Decimal(f0[p]["value"])
    with (tdir / "probe_skips.csv").open(encoding="utf-8") as f:
        assert all(r["reason"] for r in csv.DictReader(f))
    # prefix check: every unit has the context; prefix edits replace a stated value of the cited unit (by an
    # edited value of that unit) or a direction word, and lie wholly before the term
    by_pid = {u["probe_id"]: u for u in units}
    pre = {c["probe_id"]: c for c in ctxs if c["context"] == "cited_edit_prefix"}
    assert set(pre) == set(by_pid)
    words = {w.lower() for ws in ck.dir_words for w in ccfg_words(ccfg, ws)}
    for pid, c in pre.items():
        u = by_pid[pid]
        assert c["replace"] == next(x for x in ctxs if x["probe_id"] == pid and x["context"] == "cited_edit")["replace"]
        n_val = int(u["prefix_values"])
        assert u["prefix_mentions_lab"] == str(bool(int(u["prefix_lab_names"]) or n_val))
        assert len(c["prefix_edits"]) == n_val + int(u["prefix_flips"])
        d = root / "runs" / proj["gen"] / "ehr_rag" / u["hadm_id"]
        raw = (d / json.loads((d / "parsed.json").read_text(encoding="utf-8"))["raw_file"]).read_text(encoding="utf-8")
        orig = {Decimal(f["value"]) for f in (c["replace"][0]["original_fields"][p] for p in BAND_POINTS)}
        new = {Decimal(f["value"]) for f in (c["replace"][0]["edited_fields"][p] for p in BAND_POINTS)}
        for a, b, rep in c["prefix_edits"]:
            assert b <= int(u["term_cut"])
            old = raw[a:b]
            if old[:1].isdigit():
                assert ck.number_matches(Decimal(old), list(orig)) and ck.number_matches(Decimal(rep), list(new))
            else:
                assert old.lower() in words

    monkeypatch.setattr(score, "make_scorer", lambda g: FakeScorer())
    assert main(["run-probe", "--config", cfg, "--targets-run", tdir.name, "--run-id", "20990101-000001-run-probe"]) == 0
    erun = start_run(ccfg, "evaluate", command="evaluate")
    run_evaluation([root / "runs" / proj["gen"]], erun.path, erun.run_id, FakeNLI(), 5, 1)
    assert main(["probe-threshold", "--config", cfg, "--probe-run", "20990101-000001-run-probe"]) == 0
    assert main(["analyze-probe", "--config", cfg, "--probe-run", "20990101-000001-run-probe",
                 "--evaluate-run", erun.run_id, "--run-id", "20990101-000002-analyze-probe"]) == 0
    s = json.loads((root / "runs" / "20990101-000002-analyze-probe" / "probe_summary.json").read_text(encoding="utf-8"))
    assert s["n_scored"] == len(units) and s["splits"]["all"]["share"]["n_resamples"] in (0, 1000)
    n_yes = sum(u["prefix_mentions_lab"] == "True" for u in units)
    assert 0 < n_yes < len(units) and sum(len(c["prefix_edits"]) for c in pre.values()) > 0    # not vacuous
    pf = s["prefix"]
    assert pf["n_supported_with_prefix_flag"] == s["n_supported"]
    assert sum(pf["by_prefix_mentions_lab"][f"prefix_mentions_lab={v}"]["n_units"] for v in ("yes", "no")) == \
        s["n_supported"]
    assert pf["dependence_prefix"]["all"]["n_units"] == s["n_supported"]


def ccfg_words(cfg: dict[str, Any], direction: str) -> list[str]:
    """Direction words of the checker rules (high / low / normal)."""
    import yaml

    from ehr_rag.checker.rules import checker_paths

    return [str(w) for w in yaml.safe_load(checker_paths(cfg)["rules"].read_text(encoding="utf-8"))["direction"][direction]]
