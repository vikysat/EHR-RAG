"""Phase 6: numeric matcher, edits, target selection, frozen-evidence pilot, outcome labels (all fake)."""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ehr_rag.eval.parse import parse_report
from ehr_rag.evidence.render import render
from ehr_rag.evidence.schema import EvidenceUnit
from ehr_rag.generation.generate import ConfirmationRequired, run_generation
from ehr_rag.perturb.analyze import analyze, candidate_sentences, label_delete, label_value_edit
from ehr_rag.perturb.edit import SkipEdit, apply_edit, decimal_places, lab_major, lab_minor, med_dose
from ehr_rag.perturb.run_pilot import PilotError, run_pilot, select_and_edit
from ehr_rag.perturb.select_targets import candidates_for_report, mentions_value, numbers_in, select_targets
from ehr_rag.runs import start_run
from fakes import FakeBackend, realistic_responder

HADM = [12, 21]

# --------------------------------------------------------------------------- numeric matcher


@pytest.mark.parametrize("text", ["Fakeium was 5.9.", "Fakeium was 5.90 today", "Fakeium 5.9 mmol/L [HIGH]",
                                  "(5.9)", "max 5.9, min 3.9"])
def test_numeric_matcher_matches(text: str) -> None:
    assert mentions_value(text, "5.9")


@pytest.mark.parametrize("text", ["Fakeium was 15.9.", "Fakeium was 5.95.", "code N5.9", "E5.9",
                                  "on 2150-05-09 05:09", "5.9.1"])
def test_numeric_matcher_rejects(text: str) -> None:
    assert not mentions_value(text, "5.9")


def test_numbers_in_ranges_signs_and_grouping() -> None:
    assert numbers_in("ref 3.3-5.1, base excess -3, count 1,200") == [
        Decimal("3.3"), Decimal("5.1"), Decimal("-3"), Decimal("1200")]
    assert mentions_value("FakeDrugA 40 mg", "40") and not mentions_value("FakeDrugA 400 mg", "40")


# --------------------------------------------------------------------------- edits


def _lab(values: list[tuple[str, str]], ref: tuple[str | None, str | None] = ("3.3", "5.1"), uid: str = "E3"
         ) -> EvidenceUnit:
    """A FAKE lab unit: values [(value, time)] for first, max, min, last."""
    pts = {n: {"value": v, "time": t, "flag": None} for n, (v, t) in zip(("first", "max", "min", "last"), values)}
    f = {"itemid": 900, "label": "Fakeium", "fluid": "Blood", "category": "Chemistry", "n_time_window_linked": 0,
         "numeric": True, "uom": "mmol/L", "ref_low": ref[0], "ref_high": ref[1],
         "n_measurements": 1 if len({(v, t) for v, t in values}) == 1 else 4, **pts, "n_abnormal": 1}
    return EvidenceUnit(uid, 12, 1, "lab", "current", values[0][1], f, render("lab", f), "labevents", [1])


MULTI = [("4.1", "2100-02-01T09:00:00"), ("5.9", "2100-02-02T09:00:00"), ("3.9", "2100-02-03T09:00:00"),
         ("4.30", "2100-02-04T09:00:00")]


def test_lab_major_marker_updates_when_crossing_range() -> None:
    single = _lab([("4.1", "2100-02-01T09:00:00")] * 4)
    assert "[HIGH]" not in single.text
    r = apply_edit(single, "lab_major", "4.1", ["first", "max", "min", "last"], seed="s")
    assert r.edited_value == "7.7" and "7.7 [HIGH]" in r.edited_text  # 1.5 x 5.1 at 1 decimal
    assert r.edited_fields["n_abnormal"] == 2
    low = _lab([("2.9", "t")] * 4)
    r = apply_edit(low, "lab_major", "2.9", ["first", "max", "min", "last"], seed="s")
    assert r.edited_value == "4.2" and "[LOW]" not in r.edited_text and "[LOW]" in low.text


def test_lab_major_inconsistent_edit_is_skipped() -> None:
    u = _lab(MULTI)  # max 5.9 [HIGH]; moving it to 4.2 would put it below last 4.30
    with pytest.raises(SkipEdit, match="inconsistent"):
        apply_edit(u, "lab_major", "5.9", ["max"], seed="s")


def test_lab_major_needs_reference_range() -> None:
    with pytest.raises(SkipEdit, match="no reference range"):
        lab_major(_lab(MULTI, ref=(None, None)).fields, ["max"])


def test_lab_minor_keeps_status_precision_and_range() -> None:
    u = _lab(MULTI)
    f = lab_minor(u.fields, ["max"], seed="x")
    new = Decimal(f["max"]["value"])
    assert decimal_places(f["max"]["value"]) == 1
    assert Decimal("0.10") <= abs(new - Decimal("5.9")) / Decimal("5.9") <= Decimal("0.20")
    assert new > Decimal("5.1")  # still [HIGH]
    assert lab_minor(u.fields, ["max"], seed="x") == f  # deterministic


def test_med_dose_doubles_with_precision_and_skips_text_doses() -> None:
    f = {"drug": "FakeDrugA", "dose_val_rx": "12.5", "dose_unit_rx": "mg", "route": "IV", "dose_numeric": True,
         "n_orders": 1, "start": None, "stop": None}
    assert med_dose(f)["dose_val_rx"] == "25.0"
    with pytest.raises(SkipEdit):
        med_dose({**f, "dose_val_rx": "1-2", "dose_numeric": False})


def test_edit_changes_only_the_edited_unit() -> None:
    a, b = _lab(MULTI, uid="E3"), _lab([("4.1", "t")] * 4, uid="E4")
    b_text = b.text
    r = apply_edit(a, "lab_minor", "5.9", ["max"], seed="s")
    assert r.edited_text != a.text and a.text == render("lab", a.fields)  # original untouched
    assert b.text == b_text == render("lab", b.fields)


# --------------------------------------------------------------------------- selection


def test_candidates_ambiguous_measurement_not_eligible() -> None:
    u = _lab([("4.1", "t1"), ("5.9", "t2"), ("3.9", "t3"), ("4.1", "t4")])
    rep = parse_report("### Findings\nFakeium was 4.1 and peaked at 5.9. [E3]\n")
    cands = candidates_for_report(12, rep, {"E3": u}, {"E3"})
    by_val = {c.value: c for c in cands}
    assert not by_val["4.1"].eligible and "ambiguous" in by_val["4.1"].reason
    assert by_val["5.9"].eligible and by_val["5.9"].fields_matched == ["max"]


def test_select_targets_dedupe_cap_and_seed() -> None:
    u = _lab(MULTI)
    cands = []
    for h in (1, 2, 3):
        rep = parse_report("### Findings\nFakeium peaked at 5.9. [E3]\nAgain 5.9. [E3]\nLow 3.9 then. [E3]\n")
        cands += candidates_for_report(h, rep, {"E3": u}, {"E3"})
    t1 = select_targets(cands, n_targets=4, max_per_admission=1, seed=13)
    t2 = select_targets(cands, n_targets=4, max_per_admission=1, seed=13)
    assert [(t.hadm_id, t.value) for t in t1] == [(t.hadm_id, t.value) for t in t2]
    assert len(t1) == 3 and len({t.hadm_id for t in t1}) == 3
    assert [t.target_id for t in t1] == ["T01", "T02", "T03"]


# --------------------------------------------------------------------------- outcome labels


def test_outcome_labels() -> None:
    def cands(text: str) -> list:
        return candidate_sentences(parse_report(text), "E3", "Fakeium")

    assert label_value_edit(cands("### Findings\nFakeium reached 6.8. [E3]\n"), "E3", "5.9", "6.8") == "followed"
    assert label_value_edit(cands("### Findings\nFakeium reached 6.8. [E4]\n"), "E3", "5.9", "6.8") == \
        "updated_miscited"
    assert label_value_edit(cands("### Findings\nFakeium reached 5.9. [E3]\n"), "E3", "5.9", "6.8") == "stale"
    assert label_value_edit(cands("### Findings\nOther fake. [E4]\n"), "E3", "5.9", "6.8") == "dropped"
    assert label_value_edit(cands("### Findings\nFakeium was high. [E3]\n"), "E3", "5.9", "6.8") == "ambiguous"
    assert label_value_edit(cands("### Findings\nFakeium 5.9 then 6.8. [E3]\n"), "E3", "5.9", "6.8") == "ambiguous"
    assert label_delete(cands("### Findings\nOther fake. [E4]\n"), "5.9") == "abstained"
    assert label_delete(cands("### Findings\nFakeium reached 5.9. [E4]\n"), "5.9") == "stale"


# --------------------------------------------------------------------------- end to end


def _generation(env: dict[str, Any]) -> Path:
    run = start_run(env["cfg"], "generate", command="generate")
    run_generation(env["cfg"], run, FakeBackend(realistic_responder), hadm_ids=HADM, conditions=["ehr_rag"],
                   retrieval_run=env["retrieval_run"], log=lambda s: None)
    return run.path


def _targets(env: dict[str, Any], gen: Path) -> tuple[Path, dict[str, Any]]:
    run = start_run(env["cfg"], "select-targets", command="select-targets")
    return run.path, select_and_edit(env["cfg"], gen, run)


def test_pilot_end_to_end(env: dict[str, Any]) -> None:
    gen = _generation(env)
    tdir, sel = _targets(env, gen)
    assert {(t.hadm_id, t.unit_type, t.value) for t in sel["targets"]} == {
        (12, "lab", "5.9"), (12, "medication", "40"), (21, "lab", "2.9"), (21, "medication", "20")}
    skipped = {(s["hadm_id"], s["edit_type"]) for s in sel["skipped"]}
    assert (12, "lab_major") in skipped  # 5.9 -> 4.2 would fall below last 4.30
    assert len(sel["edits"]) == 9
    for f in ("candidates.csv", "targets.csv", "edits.jsonl", "skipped_edits.csv", "targets_meta.json"):
        assert (tdir / f).exists()

    backend = FakeBackend(realistic_responder)
    prun = start_run(env["cfg"], "run-pilot", command="run-pilot")
    with pytest.raises(ConfirmationRequired):
        run_pilot(env["cfg"], tdir, prun, backend, log=lambda s: None)  # 21 new calls > 20
    res = run_pilot(env["cfg"], tdir, prun, backend, yes=True, log=lambda s: None)
    assert res["n_cases"] == 9 + 4 * 3 and res["statuses"] == {"ok": 21}

    # frozen evidence: same order as the original; delete leaves a gap, no renumbering
    for h in HADM:
        orig = [u["unit_id"] for u in json.loads(
            (gen / "ehr_rag" / str(h) / "evidence_shown.json").read_text(encoding="utf-8"))["units"]]
        for case in (prun.path / "cases").iterdir():
            meta = json.loads((case / "case.json").read_text(encoding="utf-8"))
            if meta["hadm_id"] != h:
                continue
            shown = json.loads((case / "evidence_shown.json").read_text(encoding="utf-8"))["units"]
            ids = [u["unit_id"] for u in shown]
            if meta["edit_type"] == "delete":
                assert ids == [i for i in orig if i != meta["unit_id"]]
            else:
                assert ids == orig
            texts = {u["unit_id"]: u["text"] for u in shown}
            if meta["kind"] == "edit" and meta["edit_type"] != "delete":
                changed = [i for i in orig if texts[i] != json.loads(
                    (gen / "ehr_rag" / str(h) / "evidence_shown.json").read_text(encoding="utf-8")
                )["units"][orig.index(i)]["text"]]
                assert changed == [meta["unit_id"]]

    # control runs used distinct sample_idx values
    ctrl = json.loads((prun.path / "cases" / "T01__control2" / "call.json").read_text(encoding="utf-8"))
    assert ctrl["attempts"][0]["sample_idx"] == 2

    out = start_run(env["cfg"], "analyze-perturbation", command="analyze-perturbation")
    ares = analyze(prun.path, env["root"] / "runs", out.path)
    s = ares["summary"]
    assert s["by_edit_type"]["lab_minor"] == {"n": 2, "followed": 2}
    # the fake restates doses in Recommendations, so a dose edit changes that section
    assert s["by_edit_type"]["med_dose"] == {"n": 2, "followed": 2, "impression_recs_changed": 2}
    diff = (out.path / "diffs" / "T02__med_dose.diff").read_text(encoding="utf-8")
    assert "-Recommendations: Continue FakeDrugA 40 mg" in diff and "+Recommendations: Continue FakeDrugA 80 mg" in diff
    assert s["by_edit_type"]["delete"]["abstained"] == 4
    assert s["controls"] == {"n": 12, "stable": 12, "per_target": {f"T0{i}": "3/3" for i in range(1, 5)}}
    md = (out.path / "perturbation_summary.md").read_text(encoding="utf-8")
    assert "2/2 (100%)" in md and "PROVISIONAL" in md
    with (out.path / "perturbation_manual_check.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 21 and all(r["human_outcome"] == "" for r in rows)


def test_pilot_refuses_when_original_prompt_differs(env: dict[str, Any]) -> None:
    gen = _generation(env)
    tdir, _ = _targets(env, gen)
    p = gen / "ehr_rag" / "12" / "prompt.txt"
    p.write_text(p.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    prun = start_run(env["cfg"], "run-pilot", command="run-pilot")
    with pytest.raises(PilotError, match="differs from the original prompt"):
        run_pilot(env["cfg"], tdir, prun, FakeBackend(realistic_responder), yes=True, log=lambda s: None)
