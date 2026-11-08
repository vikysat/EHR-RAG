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
