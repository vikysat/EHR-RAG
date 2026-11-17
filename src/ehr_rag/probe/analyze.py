"""Probe scores, the dev-only threshold, and the headline (``probe-threshold``, ``analyze-probe``; CPU).

Term score of a context: s(c) = log P(original term) - log P(alternative), each the SUM of the
continuation's token log-probabilities under the same forced prefix. (``*_first`` columns use only
the first token where the two continuations differ.) Per unit:

  dependence        s(original) - s(cited_edit)
  within_flag_drop  s(original) - s(within_flag)
  null_drop         s(original) - s(matched_null)
  ablation_drop     s(original) - s(ablation)

Number (positive control): s_num(c) = log P(original number) - log P(cited-edit value);
num_dependence = s_num(original) - s_num(cited_edit). With the within-flag value as the
alternative, num_within_drop = s_wf(original) - s_wf(within_flag).

Threshold (``probe-threshold``, DEV only): the ``probe.threshold_quantile`` quantile (linear
interpolation) of the pooled null_drop and within_flag_drop values. Written to
``probe.threshold_file`` with a sha256 of its content.

Headline (``analyze-probe``): among units whose claim the NLI verifier judged supported by its cited
units (``supported`` in the evaluate run's sentences.jsonl, plain ehr_rag; the claim containing the
term), the share with dependence < threshold ("not dependent"), overall and by order
(number_first / term_first / no_number), corroborated vs not, and section; 95% CIs from
``probe.bootstrap_resamples`` bootstrap resamples of admissions.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ehr_rag.config import SECTIONS
from ehr_rag.probe.runner import SEQUENCES_FILE, SCORES_FILE, _read_csv, load_scores, threshold_sha
from ehr_rag.probe.score import ProbeError, common_prefix
from ehr_rag.stats import bootstrap, fmt_ci, percentile, ratio_stat

SCORED_COLUMNS = ["probe_id", "hadm_id", "section", "claim_kind", "direction", "groups", "order", "corroborated",
                  "contexts", "s_original", "s_cited_edit", "s_within_flag", "s_matched_null", "s_ablation",
                  "dependence", "within_flag_drop", "null_drop", "ablation_drop", "dependence_first",
                  "num_dependence", "num_within_drop", "supported", "not_dependent"]


def _diff(a: dict[str, Any] | None, b: dict[str, Any] | None, first: bool = False) -> float | None:
    if a is None or b is None:
        return None
    if first:
        i = common_prefix(a["tokens"], b["tokens"])
        if i < len(a["tokens"]) and i < len(b["tokens"]):
            return a["logprobs"][i] - b["logprobs"][i]
    return sum(a["logprobs"]) - sum(b["logprobs"])


def _minus(x: float | None, y: float | None) -> float | None:
    return None if x is None or y is None else x - y


def unit_scores(targets_dir: Path, probe_dirs: list[Path]) -> list[dict[str, Any]]:
    """One row per unit with the scores above (None where a context or sequence is missing)."""
    units = _read_csv(targets_dir / "probe_units.csv")
    by: dict[tuple[str, str, str], dict[str, Any]] = {}
    for d in probe_dirs:
        scores = load_scores(d / SCORES_FILE)
        for s in _read_csv(d / SEQUENCES_FILE):
            r = scores.get(s["key"])
            if r is not None:
                by[(s["probe_id"], s["context"], s["role"])] = r
    out = []
    for u in units:
        pid = u["probe_id"]

        def g(c: str, role: str) -> dict[str, Any] | None:
            return by.get((pid, c, role))

        s = {c: _diff(g(c, "term_orig"), g(c, "term_alt")) for c in
             ("original", "cited_edit", "within_flag", "matched_null", "ablation")}
        sf = {c: _diff(g(c, "term_orig"), g(c, "term_alt"), first=True) for c in ("original", "cited_edit")}
        num = {c: _diff(g(c, "num_orig"), g(c, "num_alt_cited")) for c in ("original", "cited_edit")}
        nwf = {c: _diff(g(c, "num_orig"), g(c, "num_alt_within")) for c in ("original", "within_flag")}
        out.append({
            "probe_id": pid, "hadm_id": int(u["hadm_id"]), "section": u["section"], "claim_kind": u["claim_kind"],
            "direction": u["direction"], "groups": u["groups"], "order": u["order"],
            "corroborated": u["corroborated"] == "True", "contexts": u["contexts"],
            **{f"s_{c}": v for c, v in s.items()},
            "dependence": _minus(s["original"], s["cited_edit"]),
            "within_flag_drop": _minus(s["original"], s["within_flag"]),
            "null_drop": _minus(s["original"], s["matched_null"]),
            "ablation_drop": _minus(s["original"], s["ablation"]),
            "dependence_first": _minus(sf["original"], sf["cited_edit"]),
            "num_dependence": _minus(num["original"], num["cited_edit"]),
            "num_within_drop": _minus(nwf["original"], nwf["within_flag"]),
        })
    return out


# --------------------------------------------------------------------------- threshold


def derive_threshold(rows: list[dict[str, Any]], quantile: float) -> dict[str, Any]:
    """Threshold from the pooled null and within-flag drops (module docstring)."""
    nulls = [r["null_drop"] for r in rows if r["null_drop"] is not None]
    wfs = [r["within_flag_drop"] for r in rows if r["within_flag_drop"] is not None]
    pooled = sorted(nulls + wfs)
    if not pooled:
        raise ProbeError("no matched-null or within-flag drops to derive a threshold from")
    return {"threshold": percentile(pooled, quantile * 100), "quantile": quantile, "n_null": len(nulls),
            "n_within_flag": len(wfs)}


def write_threshold(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Write the threshold file with its sha256; returns the payload including ``sha256``."""
    import yaml

    body = {k: v for k, v in payload.items() if k != "sha256"}
    body["derived_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body["sha256"] = threshold_sha(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ("# Written by `probe-threshold` from DEV probe runs. Do not edit: run-probe and analyze-probe on the\n"
            "# test split refuse to run unless this file's content matches sha256 and --threshold-sha256.\n")
    path.write_text(head + yaml.safe_dump(body, sort_keys=True), encoding="utf-8", newline="\n")
    return body


# --------------------------------------------------------------------------- headline


def nli_supported(eval_dir: Path, rows: list[dict[str, Any]], targets_dir: Path) -> None:
    """Set ``supported`` (True/False/None) on each row from the evaluate run's plain ehr_rag claims."""
    units = {u["probe_id"]: u for u in _read_csv(targets_dir / "probe_units.csv")}
    claims: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    with (eval_dir / "sentences.jsonl").open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["condition"] == "ehr_rag":
                claims.setdefault((int(s["hadm_id"]), s["section"], int(s["index"])), []).append(s)
    for r in rows:
        u = units[r["probe_id"]]
        cs = claims.get((r["hadm_id"], u["section"], int(u["sentence_index"])), [])
        hit = [c for c in cs if u["surface"].lower() in c["sentence"].lower()] or (cs if len(cs) == 1 else [])
        r["supported"] = bool(hit[0]["supported"]) if hit else None


def _share_stat(items: list[list[dict[str, Any]]]) -> float | None:
    return ratio_stat(lambda xs: sum(1 for x in xs if x["not_dependent"]), lambda xs: len(xs))(items)


def summarize(rows: list[dict[str, Any]], threshold: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    """Headline shares with admission-bootstrap CIs (module docstring)."""
    thr = threshold["threshold"]
    for r in rows:
        r["not_dependent"] = None if r["dependence"] is None else r["dependence"] < thr
    scored = [r for r in rows if r["dependence"] is not None]
    pool = [r for r in scored if r.get("supported")]

    def share(sel: list[dict[str, Any]], tag: str) -> dict[str, Any]:
        groups: dict[int, list[dict[str, Any]]] = {}
        for r in sel:
            groups.setdefault(r["hadm_id"], []).append(r)
        res = bootstrap(groups, _share_stat, n_boot, f"{seed}:probe:{tag}")
        return {"n_units": len(sel), "n_not_dependent": sum(1 for r in sel if r["not_dependent"]), "share": res}

    splits: dict[str, dict[str, Any]] = {"all": share(pool, "all")}
    for o in ("number_first", "term_first", "no_number"):
        splits[f"order={o}"] = share([r for r in pool if r["order"] == o], o)
    for c in (True, False):
        splits[f"corroborated={'yes' if c else 'no'}"] = share([r for r in pool if r["corroborated"] == c], str(c))
    for sec in SECTIONS:
        splits[f"section={sec}"] = share([r for r in pool if r["section"] == sec], sec)
    num = [r for r in pool if r["num_dependence"] is not None]
    by_h: dict[int, list[dict[str, Any]]] = {}
    for r in num:
        by_h.setdefault(r["hadm_id"], []).append(r)
    mean = ratio_stat(lambda xs: sum(x["num_dependence"] for x in xs), lambda xs: len(xs))
    num_below = ratio_stat(lambda xs: sum(1 for x in xs if x["num_dependence"] < thr), lambda xs: len(xs))
    return {
        "threshold": thr, "threshold_sha256": threshold.get("sha256"), "n_units": len(rows),
        "n_scored": len(scored), "n_supported": len(pool),
        "n_supported_unknown": sum(1 for r in scored if r.get("supported") is None),
        "splits": splits,
        "number_control": {"n_units": len(num),
                           "mean_num_dependence": bootstrap(by_h, mean, n_boot, f"{seed}:probe:num_mean"),
                           "share_below_threshold": bootstrap(by_h, num_below, n_boot, f"{seed}:probe:num_share")},
    }


def summary_markdown(s: dict[str, Any], runs: list[str]) -> str:
    lines = ["# Citation-dependence probe (plain ehr_rag)", "",
             f"Probe run(s): {', '.join(f'`{r}`' for r in runs)}. Threshold {s['threshold']:.4f} "
             f"(sha256 {str(s['threshold_sha256'])[:12]}...). Units {s['n_units']}, scored {s['n_scored']}, "
             f"NLI-supported {s['n_supported']} (no NLI match: {s['n_supported_unknown']}).", "",
             "Share of NLI-supported claims whose term is NOT dependent on the cited unit (dependence < threshold); "
             "95% CI from bootstrap resamples of admissions.", "",
             "| split | units | not dependent | share [95% CI] |", "|---|---:|---:|---|"]
    for k, d in s["splits"].items():
        lines.append(f"| {k} | {d['n_units']} | {d['n_not_dependent']} | {fmt_ci(d['share'])} |")
    nc = s["number_control"]
    lines += ["", f"Positive control (number token, {nc['n_units']} supported units with a number): mean dependence "
              f"{fmt_ci(nc['mean_num_dependence'])}; share below the term threshold "
              f"{fmt_ci(nc['share_below_threshold'])}.", ""]
    return "\n".join(lines)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORED_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
