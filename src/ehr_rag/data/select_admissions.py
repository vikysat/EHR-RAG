"""Per-admission metrics, eligibility and target-admission selection.

Definitions (as implemented):

- ``n_distinct_labs``: distinct ``itemid`` among lab rows linked to the admission
  (by hadm_id or by time window, see ``load.link_labs``).
- ``n_abnormal_labs``: number of linked lab result rows that are abnormal. A row
  is abnormal if it has a numeric value and at least one reference bound (a
  0–0 range counts as no range), and
  the value is below the lower or above the upper bound; if the row has no
  reference bound (or no numeric value), it is abnormal iff ``flag == "abnormal"``.
- ``n_distinct_drugs``: distinct non-null ``drug`` names (case-insensitive,
  whitespace-stripped) among the admission's prescriptions.
- ``n_diagnoses`` / ``n_procedures``: number of diagnoses_icd / procedures_icd rows.
- ``los_days``: (dischtime - admittime) in days.
- ``n_prior_admissions``: admissions of the same subject whose dischtime is
  before this admission's admittime.
- ``richness``: n_distinct_labs + n_abnormal_labs + n_distinct_drugs +
  n_diagnoses + n_procedures.

Selection: keep eligible admissions; per subject keep only the latest eligible
one (by admittime); rank by (has prior admission desc, richness desc, hadm_id
asc); propose the top ``selection.n_admissions``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ehr_rag.data.load import Tables


def lab_abnormal_mask(labs: pd.DataFrame) -> pd.Series:
    """Boolean mask of abnormal lab rows (see module docstring)."""
    v = labs["valuenum_f"]
    zero_zero = (labs["ref_range_lower_f"] == 0) & (labs["ref_range_upper_f"] == 0)
    lo = labs["ref_range_lower_f"].mask(zero_zero)
    hi = labs["ref_range_upper_f"].mask(zero_zero)
    has_ref = lo.notna() | hi.notna()
    numeric_cmp = v.notna() & has_ref
    out_of_range = (lo.notna() & (v < lo)) | (hi.notna() & (v > hi))
    by_flag = labs["flag"].fillna("").str.strip().str.lower() == "abnormal"
    return pd.Series(np.where(numeric_cmp, out_of_range.fillna(False), by_flag), index=labs.index).astype(bool)


def compute_metrics(tables: Tables) -> pd.DataFrame:
    """One row per admission with all selection metrics."""
    adm = tables["admissions"].copy()
    labs = tables["labevents"].copy()
    labs["abnormal"] = lab_abnormal_mask(labs)
    lab_g = labs.groupby("link_hadm_id")
    lab_m = pd.DataFrame({
        "n_distinct_labs": lab_g["itemid"].nunique(),
        "n_abnormal_labs": lab_g["abnormal"].sum(),
        "n_lab_rows": lab_g.size(),
        "n_lab_rows_time_window": lab_g["linked_by"].apply(lambda s: int((s == "time_window").sum())),
    })

    rx = tables["prescriptions"]
    rx = rx[rx["drug"].notna()].copy()
    rx["drug_key"] = rx["drug"].str.strip().str.lower()
    drug_m = rx.groupby("hadm_id")["drug_key"].nunique().rename("n_distinct_drugs")
    dx_m = tables["diagnoses_icd"].groupby("hadm_id").size().rename("n_diagnoses")
    px_m = tables["procedures_icd"].groupby("hadm_id").size().rename("n_procedures")

    out = adm[["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"]].set_index("hadm_id")
    out = out.join(lab_m).join(drug_m).join(dx_m).join(px_m)
    count_cols = ["n_distinct_labs", "n_abnormal_labs", "n_lab_rows", "n_lab_rows_time_window",
                  "n_distinct_drugs", "n_diagnoses", "n_procedures"]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out["los_days"] = ((out["dischtime"] - out["admittime"]).dt.total_seconds() / 86400).round(2)

    # Prior admissions: same subject, dischtime strictly before this admittime.
    pairs = adm[["subject_id", "hadm_id", "admittime"]].merge(
        adm[["subject_id", "hadm_id", "dischtime"]].rename(columns={"hadm_id": "other", "dischtime": "o_disch"}),
        on="subject_id",
    )
    prior = pairs[(pairs["other"] != pairs["hadm_id"]) & (pairs["o_disch"] < pairs["admittime"])]
    out = out.join(prior.groupby("hadm_id").size().rename("n_prior_admissions"))
    out["n_prior_admissions"] = out["n_prior_admissions"].fillna(0).astype(int)
    out["richness"] = (out["n_distinct_labs"] + out["n_abnormal_labs"] + out["n_distinct_drugs"]
                       + out["n_diagnoses"] + out["n_procedures"])
    return out.reset_index()


def apply_selection(metrics: pd.DataFrame, sel_cfg: dict[str, Any]) -> pd.DataFrame:
    """Add ``eligible``, ``latest_eligible_for_subject`` and ``proposed_rank`` columns."""
    m = metrics.copy()
    m["eligible"] = (
        (m["n_distinct_labs"] >= sel_cfg["min_distinct_labs"])
        & (m["n_abnormal_labs"] >= sel_cfg["min_abnormal_labs"])
        & (m["n_distinct_drugs"] >= sel_cfg["min_distinct_drugs"])
        & (m["n_diagnoses"] >= sel_cfg["min_diagnoses"])
    )
    elig = m[m["eligible"]].sort_values(["subject_id", "admittime", "hadm_id"])
    latest = elig.groupby("subject_id").tail(1)
    m["latest_eligible_for_subject"] = m["hadm_id"].isin(set(latest["hadm_id"]))
    ranked = latest.assign(has_prior=(latest["n_prior_admissions"] >= 1).astype(int)).sort_values(
        ["has_prior", "richness", "hadm_id"], ascending=[False, False, True]
    )
    rank = {h: i + 1 for i, h in enumerate(ranked["hadm_id"])}
    m["candidate_rank"] = m["hadm_id"].map(rank).astype("Int64")
    m["proposed"] = m["candidate_rank"].notna() & (m["candidate_rank"] <= sel_cfg["n_admissions"])
    return m.sort_values(["candidate_rank", "hadm_id"], na_position="last").reset_index(drop=True)


PROPOSAL_FIELDS: tuple[str, ...] = (
    "hadm_id", "subject_id", "n_prior_admissions", "n_distinct_labs", "n_abnormal_labs",
    "n_distinct_drugs", "n_diagnoses", "n_procedures", "los_days", "richness",
)


def write_outputs(selected: pd.DataFrame, candidates_csv: Path, selected_yaml: Path) -> list[dict[str, Any]]:
    """Write the candidates CSV and the proposed-admissions YAML. Returns the proposal list."""
    candidates_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(candidates_csv, index=False)
    prop = selected[selected["proposed"]].sort_values("candidate_rank")
    items: list[dict[str, Any]] = []
    for _, r in prop.iterrows():
        items.append({k: (float(r[k]) if k == "los_days" else int(r[k])) for k in PROPOSAL_FIELDS})
    header = (
        "# Target admissions proposed by `select-admissions` (Phase 1).\n"
        "# Edit this list by hand to approve/change targets. Only `hadm_id` is used downstream;\n"
        "# the other fields are informational metrics at proposal time.\n"
    )
    selected_yaml.parent.mkdir(parents=True, exist_ok=True)
    selected_yaml.write_text(
        header + yaml.safe_dump({"admissions": items}, sort_keys=False), encoding="utf-8"
    )
    return items


def load_selected_hadm_ids(path: Path) -> list[int]:
    """Read the approved target admission ids from ``selected_admissions.yaml``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("admissions"), list):
        raise ValueError(f"{path}: expected a mapping with an 'admissions' list")
    ids = [int(a["hadm_id"]) for a in data["admissions"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate hadm_id entries")
    return ids
