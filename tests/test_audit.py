"""Counterfactual citation audit: targets, edits, regeneration, labels, summaries (FAKE backend, Demo units)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from checker_fakes import T1, T2, T3, lab, med
from ehr_rag.perturb.select_targets import lab_status, range_skip_reason, select_audit_targets

REPO = Path(__file__).resolve().parents[1]


def audit_units() -> list[Any]:
    return [
        lab("E1", "Potassium", [(T1, "4.1"), (T2, "5.9"), (T3, "4.3")]),                  # high
        lab("E2", "Sodium", [(T1, "130"), (T3, "128")], lo="133", hi="145"),             # low
        lab("E3", "Chloride", [(T1, "101"), (T3, "103")], lo="96", hi="108"),            # normal
        lab("E4", "Glucose", [(T1, "90"), (T3, "95")], lo="70", hi="100"),               # normal
        lab("E5", "Troponin T", [(T1, "0.02")], lo=None, hi=None),                       # no range
        lab("E6", "Lipase", [(T1, "40")], lo=None, hi="60"),                             # one-sided
        lab("E7", "Magnesium", [(T1, "2.0"), (T3, "1.9")], lo="1.6", hi="2.6"),          # normal, uncited
        med("E8", "Furosemide"),
    ]


# --------------------------------------------------------------------------- task 1: target selection


def test_lab_status_and_range_skips() -> None:
    u = {x.unit_id: x for x in audit_units()}
    assert [lab_status(u[i].fields) for i in ("E1", "E2", "E3")] == ["high", "low", "normal"]
    assert range_skip_reason(u["E5"]) == "no reference range"
    assert range_skip_reason(u["E6"]) == "one-sided reference range"
    assert range_skip_reason(u["E8"]) == "not a numeric lab" and range_skip_reason(u["E1"]) == ""


def test_select_prefers_flagged_cited_units_and_adds_a_null_target() -> None:
    units = audit_units()
    cited = {"E1": 1, "E2": 2, "E3": 1, "E4": 1, "E5": 1, "E8": 1}
    targets, skips = select_audit_targets(12, units, cited, n_cited=3, n_null=1, seed=13)
    kinds = [(t.unit_id, t.kind) for t in targets]
    cited_ids = [u for u, k in kinds if k == "cited"]
    assert len(cited_ids) == 3 and set(cited_ids[:2]) == {"E1", "E2"}      # both flagged units come first
    assert cited_ids[2] in {"E3", "E4"}
    assert [u for u, k in kinds if k == "null"] == ["E7"]                   # the only eligible uncited lab
    assert {(s["unit_id"], s["reason"]) for s in skips} == {("E5", "no reference range"),
                                                             ("E6", "one-sided reference range")}
    assert next(t for t in targets if t.unit_id == "E2").n_citing_claims == 2


def test_select_is_seeded_and_counts_edit_rejections() -> None:
    units = audit_units()
    cited = {"E3": 1, "E4": 1}
    a, _ = select_audit_targets(12, units, cited, 1, 0, seed=13)
    b, _ = select_audit_targets(12, units, cited, 1, 0, seed=13)
    assert [t.unit_id for t in a] == [t.unit_id for t in b]
    picks = {select_audit_targets(12, units, cited, 1, 0, seed=s)[0][0].unit_id for s in range(20)}
    assert picks == {"E3", "E4"}                                            # the seed decides among equals
    t, skips = select_audit_targets(12, units, cited, 2, 0, seed=13, editable=lambda u: "nope" if u.unit_id == "E3" else "")
    assert [x.unit_id for x in t] == ["E4"] and {"unit_id": "E3", "reason": "nope"} in [
        {k: s[k] for k in ("unit_id", "reason")} for s in skips]


# --------------------------------------------------------------------------- Demo units

CFG_PATH = REPO / "config" / "default.yaml"


def _demo_present() -> bool:
    from ehr_rag.config import load_config

    cfg = load_config(CFG_PATH)
    return (Path(cfg["_project_root"]) / cfg["data"]["root"]).is_dir()


demo_only = pytest.mark.skipif(not _demo_present(), reason="MIMIC-IV Demo not present")


@pytest.fixture(scope="module")
def demo_units() -> list[Any]:
    from ehr_rag.config import load_config
    from ehr_rag.data.load import load_tables
    from ehr_rag.evidence.build_units import build_units_for_admission

    cfg = load_config(CFG_PATH)
    return build_units_for_admission(load_tables(cfg), 23559586, cfg["units"])[0]


@demo_only
def test_select_on_demo_units(demo_units: list[Any]) -> None:
    labs = [u for u in demo_units if u.unit_type == "lab"][:40]
    cited = {u.unit_id: 1 for u in labs[::2]}
    targets, skips = select_audit_targets(23559586, labs, cited, 3, 1, seed=13)
    assert len([t for t in targets if t.kind == "cited"]) == 3 and len([t for t in targets if t.kind == "null"]) == 1
    flagged_available = [u for u in labs if u.unit_id in cited and not range_skip_reason(u)
                         and lab_status(u.fields) != "normal"]
    if len(flagged_available) >= 3:
        assert all(t.status != "normal" for t in targets if t.kind == "cited")
    assert all(not range_skip_reason(next(u for u in labs if u.unit_id == t.unit_id)) for t in targets)
    assert all(t.unit_id not in cited for t in targets if t.kind == "null")
    assert all(s["reason"] for s in skips) and skips                      # the Demo has unranged labs


# --------------------------------------------------------------------------- task 2: edits


def _edit(u: Any, group: str | None = None) -> Any:
    from ehr_rag.perturb.edit import audit_edit

    return audit_edit(u, group, 0.5, ["hemoglobin", "hematocrit", "platelet", "albumin"])


def test_flagged_unit_moves_to_the_midpoint_and_loses_its_flag() -> None:
    from ehr_rag.evidence.render import render

    k = audit_units()[0]                                       # potassium 4.1 / 5.9 [HIGH] / 4.3, ref 3.3-5.1
    r = _edit(k)
    assert r.edit_type == "to_normal" and r.edited_value == "4.2" and r.original_value == "4.1 5.9 4.3"
    assert all(r.edited_fields[p]["value"] == "4.2" and r.edited_fields[p]["flag"] is None
               for p in ("first", "max", "min", "last"))
    assert "[HIGH]" not in r.edited_text and "[LOW]" not in r.edited_text and r.edited_fields["n_abnormal"] == 0
    assert r.edited_text == render("lab", r.edited_fields) and r.original_text == k.text
    assert k.fields["max"]["value"] == "5.9"                     # the original unit is untouched


def test_normal_unit_moves_outside_the_range_with_its_flag() -> None:
    cl = audit_units()[2]                                      # chloride 101 / 103, ref 96-108
    r = _edit(cl)
    assert (r.edit_type, r.edited_value) == ("to_high", "114")  # 108 + 0.5 * 12, integer precision kept
    assert r.edited_text.count("[HIGH]") == 4 and r.edited_fields["n_abnormal"] == r.edited_fields["n_measurements"]
    assert all(r.edited_fields[p]["flag"] == "abnormal" for p in ("first", "max", "min", "last"))
    hgb = lab("E9", "Hemoglobin", [(T1, "14.2"), (T3, "15.0")], lo="13.7", hi="17.5")
    r = _edit(hgb, "hemoglobin")                               # configured to go below the range
    assert (r.edit_type, r.edited_value) == ("to_low", "11.8") and r.edited_text.count("[LOW]") == 4


def test_edit_skips() -> None:
    from ehr_rag.perturb.edit import SkipEdit

    with pytest.raises(SkipEdit, match="no reference range"):
        _edit(audit_units()[4])
    with pytest.raises(SkipEdit, match="not positive"):
        _edit(lab("E9", "Platelet Count", [(T1, "20"), (T3, "30")], lo="10", hi="60"), "platelet")
    # integer values, range 0.4-0.6: the midpoint 0.5 rounds to "1" at 0 decimals, still [HIGH]
    with pytest.raises(SkipEdit, match="original precision"):
        _edit(lab("E9", "Fakeone", [(T1, "2"), (T3, "2")], lo="0.4", hi="0.6"))


@demo_only
def test_edits_on_demo_units(demo_units: list[Any]) -> None:
    from ehr_rag.checker.rules import Checker
    from ehr_rag.config import load_config
    from ehr_rag.evidence.render import lab_marker
    from ehr_rag.perturb.edit import SkipEdit

    ck = Checker.from_config(load_config(CFG_PATH))
    n_ok = 0
    for u in demo_units:
        if range_skip_reason(u):
            continue
        try:
            r = _edit(u, ck.lab_group(u.fields["label"]))
        except SkipEdit:
            continue
        n_ok += 1
        f = r.edited_fields
        marks = {lab_marker(f[p]["value"], f[p]["flag"], f["ref_low"], f["ref_high"]) for p in ("first", "max", "min", "last")}
        if lab_status(u.fields) != "normal":
            assert marks == {""} and r.edit_type == "to_normal"
        else:
            assert marks in ({"[HIGH]"}, {"[LOW]"}) and r.edit_type in ("to_high", "to_low")
        assert len({f[p]["value"] for p in ("first", "max", "min", "last")}) == 1
    assert n_ok > 20


def test_audit_targets_cli(env: dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    import json

    from ehr_rag.cli import main
    from ehr_rag.generation.generate import run_generation
    from ehr_rag.runs import start_run
    from fakes import FakeBackend, realistic_responder

    gen = start_run(env["cfg"], "generate", command="generate")
    run_generation(env["cfg"], gen, FakeBackend(realistic_responder), hadm_ids=[12, 21], conditions=["ehr_rag"],
                   retrieval_run=env["retrieval_run"], log=lambda s: None)
    cfg = str(env["root"] / "config" / "test.yaml")
    assert main(["audit-targets", "--config", cfg, "--generation-run", gen.path.name,
                 "--run-id", "20990101-000000-audit-targets"]) == 0
    out = capsys.readouterr().out
    assert "targets: 2 (2 cited, 0 null-edit)" in out and "to_normal" in out
    run = env["root"] / "runs" / "20990101-000000-audit-targets"
    edits = [json.loads(x) for x in (run / "audit_edits.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["edit_type"] for e in edits] == ["to_normal", "to_normal"]        # Fakeium is HIGH / LOW
    assert json.loads((run / "audit_meta.json").read_text(encoding="utf-8"))["generation_runs"] == {
        "12": gen.path.name, "21": gen.path.name}


# --------------------------------------------------------------------------- task 3: regeneration


def _base_run(env: dict[str, Any], responder: Any, conditions: list[str] | None = None) -> str:
    from ehr_rag.generation.generate import run_generation
    from ehr_rag.runs import start_run
    from fakes import FakeBackend

    gen = start_run(env["cfg"], "generate", command="generate")
    run_generation(env["cfg"], gen, FakeBackend(responder), hadm_ids=[12, 21], conditions=conditions or ["ehr_rag"],
                   retrieval_run=env["retrieval_run"], log=lambda s: None)
    return gen.path.name


def _targets(env: dict[str, Any], gen: str, run_id: str = "20990101-000000-audit-targets") -> Path:
    from ehr_rag.cli import main

    assert main(["audit-targets", "--config", str(env["root"] / "config" / "test.yaml"), "--generation-run", gen,
                 "--run-id", run_id]) == 0
    return env["root"] / "runs" / run_id


def test_run_audit_plain_and_enforced(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture) -> None:
    import json

    import ehr_rag.generation.backends as backends
    from ehr_rag.cli import main
    from fakes import FakeBackend, realistic_responder

    gen = _base_run(env, realistic_responder)
    tdir = _targets(env, gen)
    backend = FakeBackend(realistic_responder)
    monkeypatch.setattr(backends, "make_backend", lambda g: backend)
    cfg = str(env["root"] / "config" / "test.yaml")
    capsys.readouterr()
    assert main(["run-audit", "--config", cfg, "--targets-run", tdir.name, "--run-id", "20990101-000001-run-audit"]) == 0
    out = capsys.readouterr().out
    # 2 admissions x (original + 1 target) = 4 distinct prompts; the 2 originals are cache hits
    assert "report jobs 8; distinct prompts 4 = expected report calls (2 cached, 2 new" in out
    assert "at most one rewrite call per failing line in the 4 ehr_rag_enforced reports" in out
    reports = [p for p in backend.prompts if not p.startswith("One sentence")]
    assert len(reports) == 2                                    # only the 2 edited prompts were new calls
    run = env["root"] / "runs" / "20990101-000001-run-audit"
    edits = {json.loads(x)["target_id"]: json.loads(x) for x in
             (tdir / "audit_edits.jsonl").read_text(encoding="utf-8").splitlines()}
    for tid, e in edits.items():
        d = run / "ehr_rag" / tid
        orig = json.loads((run / "ehr_rag" / f"{e['hadm_id']}__original" / "evidence_shown.json").read_text(encoding="utf-8"))
        shown = json.loads((d / "evidence_shown.json").read_text(encoding="utf-8"))
        assert [u["unit_id"] for u in shown["units"]] == [u["unit_id"] for u in orig["units"]]   # retrieval held fixed
        changed = [u["unit_id"] for u, o in zip(shown["units"], orig["units"]) if u["text"] != o["text"]]
        assert changed == [e["unit_id"]] and e["edited_text"] in (d / "prompt.txt").read_text(encoding="utf-8")
        assert (run / "ehr_rag_enforced" / tid / "enforcement.json").exists()
        assert (run / "ehr_rag" / tid / "prompt.txt").read_text(encoding="utf-8") == \
            (run / "ehr_rag_enforced" / tid / "prompt.txt").read_text(encoding="utf-8")       # same prompt
    original = run / "ehr_rag" / "12__original" / "prompt.txt"
    gen_prompt = env["root"] / "runs" / gen / "ehr_rag" / "12" / "prompt.txt"
    assert original.read_text(encoding="utf-8") == gen_prompt.read_text(encoding="utf-8")
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ok" and meta["checker"]["combined_sha256"]


def test_run_audit_shards_and_guards(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture) -> None:
    import json

    import ehr_rag.generation.backends as backends
    from ehr_rag.cli import main
    from fakes import FakeBackend, realistic_responder

    gen = _base_run(env, realistic_responder)
    tdir = _targets(env, gen)
    monkeypatch.setattr(backends, "make_backend", lambda g: FakeBackend(realistic_responder))
    cfg = str(env["root"] / "config" / "test.yaml")
    base = ["run-audit", "--config", cfg, "--targets-run", tdir.name, "--conditions", "ehr_rag"]
    for k in (0, 1):
        assert main(base + ["--shard-size", "1", "--shard-index", str(k), "--run-id",
                            f"20990101-00000{k}-run-audit-job9-shard{k}"]) == 0
    m = [json.loads((env["root"] / "runs" / f"20990101-00000{k}-run-audit-job9-shard{k}" / "run_meta.json")
                    .read_text(encoding="utf-8")) for k in (0, 1)]
    assert [x["hadm_ids"] for x in m] == [[12], [21]] and m[0]["n_shards"] == 2
    assert main(base + ["--shard-size", "1", "--shard-index", "2"]) == 2
    assert main(base + ["--checker-sha256", "0" * 64]) == 2
    assert "checker files changed" in capsys.readouterr().err


def test_run_audit_refuses_changed_prompt(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture) -> None:
    import ehr_rag.generation.backends as backends
    from ehr_rag.cli import main
    from fakes import FakeBackend, realistic_responder

    gen = _base_run(env, realistic_responder)
    tdir = _targets(env, gen)
    p = env["root"] / "runs" / gen / "ehr_rag" / "12" / "prompt.txt"
    p.write_text(p.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    monkeypatch.setattr(backends, "make_backend", lambda g: FakeBackend(realistic_responder))
    assert main(["run-audit", "--config", str(env["root"] / "config" / "test.yaml"), "--targets-run", tdir.name]) == 4
    assert "differs from the original prompt.txt" in capsys.readouterr().err


# --------------------------------------------------------------------------- task 4: labels


def _ctx_and_edit(u: Any, group: str | None = None) -> tuple[Any, Any, Any]:
    import dataclasses

    from checker_fakes import checker

    ck = checker()
    r = _edit(u, group)
    edited = dataclasses.replace(u, fields=r.edited_fields, text=r.edited_text)
    return ck, ck.admission([u], [u]), edited


@pytest.mark.parametrize("claim,label", [
    ("Potassium peaked at 5.9", "stale"),            # an original value
    ("Potassium was 4.2", "follows"),                # the edited value (midpoint)
    ("Hyperkalemia was present", "stale"),           # lab-derived term holds only for the original
    ("Potassium was normal", "other"),               # the original LAST value (4.3) was normal already
    ("There was no hyperkalemia", "follows"),        # negated term
    ("Potassium was elevated", "stale"),
    ("Potassium was checked", "other"),              # cites, states neither
    ("Potassium rose from 5.9 to 4.2", "other"),     # states both
])
def test_labels_for_a_flagged_unit_edited_to_normal(claim: str, label: str) -> None:
    from ehr_rag.perturb.analyze import audit_label, audit_signals

    k = audit_units()[0]                               # potassium 4.1 / 5.9 [HIGH] / 4.3 -> all 4.2
    ck, ctx, edited = _ctx_and_edit(k)
    assert audit_label(audit_signals(ck, ctx, claim, k, edited)) == label


def test_normal_claim_follows_when_the_last_value_was_abnormal() -> None:
    from ehr_rag.perturb.analyze import audit_label, audit_signals

    na = audit_units()[1]                              # sodium 130 / 128, both [LOW] -> 139
    ck, ctx, edited = _ctx_and_edit(na)
    assert audit_label(audit_signals(ck, ctx, "Sodium was normal", na, edited)) == "follows"
    assert audit_label(audit_signals(ck, ctx, "Hyponatremia to 128", na, edited)) == "stale"


@pytest.mark.parametrize("claim,label", [
    ("Chloride was elevated", "follows"),            # 101 / 103 -> all 114 [HIGH]
    ("Chloride was 114", "follows"),
    ("Chloride was normal at 101", "stale"),
    ("Chloride 103", "stale"),
])
def test_labels_for_a_normal_unit_edited_out_of_range(claim: str, label: str) -> None:
    from ehr_rag.perturb.analyze import audit_label, audit_signals

    cl = audit_units()[2]
    ck, ctx, edited = _ctx_and_edit(cl)
    assert audit_label(audit_signals(ck, ctx, claim, cl, edited)) == label


def test_dropped_and_best_match() -> None:
    from ehr_rag.perturb.analyze import audit_label, best_match

    assert audit_label(None) == "dropped"
    orig = {"section": "Findings", "text": "Potassium peaked at 5.9"}
    cands = [{"section": "Impression", "text": "Potassium peaked at 5.9", "line": "a"},
             {"section": "Findings", "text": "Potassium was 4.2 on admission", "line": "b"},
             {"section": "Findings", "text": "Potassium peaked at 4.2", "line": "c"}]
    assert best_match(orig, cands)["line"] == "c"          # same section first, then the most similar
    assert best_match(orig, []) is None


def test_terms_and_churn_helpers() -> None:
    from checker_fakes import checker
    from ehr_rag.eval.parse import parse_report
    from ehr_rag.perturb.analyze import _ir_terms, _term_names, section_churn

    ck = checker()
    k, na = audit_units()[0], audit_units()[1]
    assert _term_names(ck, k, "high") == {"hyperkalemia"} and _term_names(ck, na, "low") == {"hyponatremia"}
    ctx = ck.admission([k], [k])
    rep = parse_report("### Impression\nHyperkalemia was treated. [E1]\n### Recommendations\nRecheck. [E1]\n")
    assert _ir_terms(rep, ctx) == {"hyperkalemia"}
    a = parse_report("### History\nA. [E1]\nB. [E1]\n### Findings\nC. [E1]\nInsufficient evidence. [NONE]\n")
    b = parse_report("### History\nA. [E1]\nB changed. [E1]\n### Findings\nC. [E1]\n")
    churn = {s: (n, c) for s, n, c in section_churn(a, b)}
    assert churn["History"] == (2, 1) and churn["Findings"] == (1, 0) and churn["Impression"] == (0, 0)


# --------------------------------------------------------------------------- task 5: statistics and outputs


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    from ehr_rag.stats import bootstrap, ratio_stat

    groups = {h: (h % 3, 2) for h in range(30)}          # num, den per admission
    stat = ratio_stat(lambda x: x[0], lambda x: x[1])
    a, b = bootstrap(groups, stat, 1000, 13), bootstrap(groups, stat, 1000, 13)
    assert a == b and a["n_resamples"] == 1000 and a["n_groups"] == 30
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"] and a["ci_low"] < a["ci_high"]
    assert bootstrap({}, stat, 10, 1)["estimate"] is None
    assert bootstrap({1: (0, 0)}, stat, 10, 1)["estimate"] is None        # zero denominator


def _pair(cond: str, h: int, label: str, section: str = "Findings") -> dict[str, Any]:
    return {"condition": cond, "hadm_id": h, "auto_label": label, "section": section, "pair_id": f"{cond}{h}{label}"}


def test_summary_faithfulness_paired_and_manual_sample() -> None:
    from ehr_rag.perturb.analyze import audit_manual_sample, summarize_audit

    pairs = ([_pair("ehr_rag", h, "follows" if h % 2 else "stale") for h in range(20)]
             + [_pair("ehr_rag", 3, "dropped"), _pair("ehr_rag", 5, "other", "Impression")]
             + [_pair("ehr_rag_enforced", h, "follows") for h in range(20)])
    churn = [{"condition": "ehr_rag", "section": "Findings", "n_lines": 4, "n_changed": 1, "target_id": "A1",
              "null_unit_cited_after": False}]
    s = summarize_audit({"pairs": pairs, "terms": [], "churn": churn}, 1000, 13)
    p, e = s["conditions"]["ehr_rag"], s["conditions"]["ehr_rag_enforced"]
    assert (p["n_pairs"], p["follows"], p["stale"], p["dropped"], p["other"]) == (22, 10, 10, 1, 1)
    assert p["faithfulness"]["estimate"] == pytest.approx(0.5) and e["faithfulness"]["estimate"] == 1.0
    d = s["paired_difference"]
    assert d["estimate"] == pytest.approx(0.5) and d["n_groups"] == 20 and d["ci_low"] > 0
    assert s["null_churn"] == [{"condition": "ehr_rag", "section": "Findings", "n_lines": 4, "n_changed": 1,
                                "share_changed": 0.25}]
    sample = audit_manual_sample(pairs, 50, 13)
    assert len(sample) == 42 and sample == audit_manual_sample(pairs, 50, 13)   # capped by availability
    few = audit_manual_sample(pairs, 5, 13)
    assert {(x["condition"], x["auto_label"]) for x in few} == {
        ("ehr_rag", "dropped"), ("ehr_rag", "follows"), ("ehr_rag", "other"), ("ehr_rag", "stale"),
        ("ehr_rag_enforced", "follows")}                                         # one per stratum first


# --------------------------------------------------------------------------- end to end with fake "models"


def _responder(kind: str, edits: list[dict[str, Any]]) -> Any:
    """FAKE models for edited prompts: 'follow' restates the edited unit; 'ignore' restates the original
    values (stale); 'drop' leaves the lab out. Rewrites copy the cited unit's max value back."""
    import re

    from fakes import realistic_report

    def respond(prompt: str, n: int) -> str:
        if prompt.startswith("One sentence of a clinical report failed"):
            m = re.search(r"^\[(E\d+)\] \(lab\) (.*?)(?:,|\s\()", prompt, re.MULTILINE)
            v = re.search(r"(?:max|Value) (-?\d+(?:\.\d+)?)", prompt)
            return f"{m.group(2)} reached {v.group(1)}. [{m.group(1)}]" if m and v else "Insufficient evidence. [NONE]"
        edit = next((e for e in edits if e["edited_text"] in prompt), None)
        if edit is None or kind == "follow":
            return realistic_report(prompt)
        if kind == "ignore":
            return realistic_report(prompt.replace(edit["edited_text"], edit["original_text"]))
        rep = realistic_report(prompt)
        return re.sub(r"### Findings\n(?:.*\n)*?(?=### )", "### Findings\nInsufficient evidence. [NONE]\n", rep)
    return respond


@pytest.mark.parametrize("kind,plain,enforced", [
    ("follow", "follows", "follows"),
    ("ignore", "stale", "follows"),      # enforcement rewrites the stale value from the edited unit
    ("drop", "dropped", "dropped"),
])
def test_audit_end_to_end(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
                          kind: str, plain: str, enforced: str) -> None:
    import csv as csv_mod
    import json

    import ehr_rag.generation.backends as backends
    from ehr_rag.cli import main
    from fakes import FakeBackend, realistic_responder

    gen = _base_run(env, realistic_responder)
    tdir = _targets(env, gen)
    edits = [json.loads(x) for x in (tdir / "audit_edits.jsonl").read_text(encoding="utf-8").splitlines()]
    monkeypatch.setattr(backends, "make_backend", lambda g: FakeBackend(_responder(kind, edits)))
    cfg = str(env["root"] / "config" / "test.yaml")
    assert main(["run-audit", "--config", cfg, "--targets-run", tdir.name, "--yes",
                 "--run-id", "20990101-000001-run-audit"]) == 0
    capsys.readouterr()
    assert main(["analyze-audit", "--config", cfg, "--audit-run", "20990101-000001-run-audit",
                 "--run-id", "20990101-000002-analyze-audit"]) == 0
    out = capsys.readouterr().out
    assert "faithfulness [95% CI]" in out and "Fakeium" not in out
    run = env["root"] / "runs" / "20990101-000002-analyze-audit"
    with (run / "audit_pairs.csv").open(encoding="utf-8") as f:
        pairs = list(csv_mod.DictReader(f))
    labels = {(p["condition"], p["auto_label"]) for p in pairs}
    assert labels == {("ehr_rag", plain), ("ehr_rag_enforced", enforced)}
    assert all(p["original_states_value_or_direction"] == "True" for p in pairs if p["condition"] == "ehr_rag")
    summary = json.loads((run / "audit_summary.json").read_text(encoding="utf-8"))
    faith = summary["conditions"]["ehr_rag"]["faithfulness"]["estimate"]
    assert faith == {"follows": 1.0, "stale": 0.0, "dropped": None}[plain]
    for f in ("audit_by_section.csv", "audit_terms.csv", "audit_null_churn.csv", "audit_summary.md",
              "audit_manual_check.csv"):
        assert (run / f).exists(), f
    with (run / "audit_manual_check.csv").open(encoding="utf-8") as f:
        manual = list(csv_mod.DictReader(f))
    assert manual and all(r["human_label"] == "" for r in manual) and manual[0]["edited_unit_text"]
