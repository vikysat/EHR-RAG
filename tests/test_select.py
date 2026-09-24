"""Phase 1: lab linkage and admission metrics on fake fixture data."""

from __future__ import annotations

from pathlib import Path

from ehr_rag.config import load_config
from ehr_rag.data.load import load_tables
from ehr_rag.data.select_admissions import apply_selection, compute_metrics


def test_lab_linkage_counts(fake_config: Path) -> None:
    t = load_tables(load_config(fake_config))
    s = t.lab_link_stats
    assert s["linked_by_hadm_id"] == 7
    assert s["linked_by_time_window"] == 1
    tw = t["labevents"][t["labevents"]["linked_by"] == "time_window"]
    assert tw["labevent_id"].tolist() == [4] and tw["link_hadm_id"].tolist() == [12]


def test_admission_metrics(fake_config: Path) -> None:
    m = compute_metrics(load_tables(load_config(fake_config))).set_index("hadm_id")
    r = m.loc[12]
    assert r["n_distinct_labs"] == 2
    assert r["n_abnormal_labs"] == 2  # 5.9 > 5.1 by range; text result by flag
    assert r["n_distinct_drugs"] == 2
    assert r["n_diagnoses"] == 2 and r["n_procedures"] == 1
    assert r["n_prior_admissions"] == 1
    assert m.loc[11, "n_prior_admissions"] == 0


def test_selection_one_per_subject(fake_config: Path) -> None:
    cfg = load_config(fake_config)
    sel_cfg = dict(cfg["selection"], min_distinct_labs=1, min_abnormal_labs=0, min_distinct_drugs=0,
                   min_diagnoses=1)
    s = apply_selection(compute_metrics(load_tables(cfg)), sel_cfg)
    proposed = s[s["proposed"]].sort_values("candidate_rank")
    assert proposed["subject_id"].is_unique
    # subject 1: latest eligible (12) chosen and ranked first because it has a prior admission
    assert proposed["hadm_id"].tolist() == [12, 21]
