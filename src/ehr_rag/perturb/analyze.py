"""Perturbation outcome labeling (SPEC Phase 6). Labels are PROVISIONAL until checked by hand.

As implemented, for each regenerated report (final raw output, parsed with eval/parse.py):
  candidate sentences = sentences whose trailing citations include the target unit id,
                        plus sentences containing the unit's lab label / drug name
                        (case-insensitive substring). Abstention lines are ignored.
  value edits (lab_minor, lab_major, med_dose), using the numeric matcher:
    ambiguous         a candidate states the edited value AND a candidate states the original
    followed          a candidate states the edited value and cites the target unit
    updated_miscited  a candidate states the edited value, none of those cites the target
    stale             a candidate still states the original value
    dropped           there are no candidate sentences at all
    ambiguous         otherwise (the fact is mentioned without either value)
  delete:
    stale             a candidate still states the original value
    abstained         there are no candidate sentences
    ambiguous         otherwise
    and separately ``cites_deleted_id``: any sentence cites the deleted unit id.
Impression/Recommendations change: the lines of those two sections are compared with the
original report; a unified diff is stored; every lab_major case is flagged for manual
reading. Control stability: a control run is stable iff the original target sentence line
appears unchanged in the same section of the control report.
"""

from __future__ import annotations

import csv
import dataclasses
import difflib
import json
import random
from pathlib import Path
from typing import Any

from ehr_rag.eval.parse import ParsedReport, Sentence, parse_report
from ehr_rag.perturb.run_pilot import original_report
from ehr_rag.perturb.select_targets import mentions_value

VALUE_OUTCOMES: tuple[str, ...] = ("followed", "updated_miscited", "stale", "dropped", "ambiguous")
DELETE_OUTCOMES: tuple[str, ...] = ("abstained", "stale", "ambiguous")
MANUAL_COLUMNS: tuple[str, ...] = (
    "case_id", "kind", "target_id", "hadm_id", "unit_id", "edit_type", "original_sentence", "original_text",
    "edited_text", "original_value", "edited_value", "regenerated_candidates", "auto_outcome",
    "cites_deleted_id", "control_stable", "impression_recs_changed", "needs_manual_read", "format_status",
    "human_outcome", "notes",
)


def candidate_sentences(parsed: ParsedReport, unit_id: str, label: str) -> list[Sentence]:
    """Sentences citing the unit or mentioning its label / drug name."""
    lab = label.lower()
    return [s for s in parsed.sentences()
            if not s.is_abstention and (unit_id in s.citations or lab in s.text.lower())]


def label_value_edit(cands: list[Sentence], unit_id: str, original: str, edited: str) -> str:
    """Outcome for lab_minor / lab_major / med_dose."""
    with_new = [s for s in cands if mentions_value(s.text, edited)]
    with_old = [s for s in cands if mentions_value(s.text, original)]
    if with_new and with_old:
        return "ambiguous"
    if any(unit_id in s.citations for s in with_new):
        return "followed"
    if with_new:
        return "updated_miscited"
    if with_old:
        return "stale"
    if not cands:
        return "dropped"
    return "ambiguous"


def label_delete(cands: list[Sentence], original: str) -> str:
    """Outcome for delete."""
    if any(mentions_value(s.text, original) for s in cands):
        return "stale"
    if not cands:
        return "abstained"
    return "ambiguous"


def _section_lines(parsed: ParsedReport, sections: tuple[str, ...]) -> list[str]:
    return [f"{s.section}: {s.raw.strip()}" for s in parsed.sentences() if s.section in sections]


def _final_raw(case_dir: Path) -> tuple[str, str]:
    meta = json.loads((case_dir / "parsed.json").read_text(encoding="utf-8"))
    return (case_dir / meta["raw_file"]).read_text(encoding="utf-8"), meta["status"]


def analyze(pilot_dir: Path, runs_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Label every case of a pilot run; write results, summary and the manual-check CSV."""
    pmeta = json.loads((pilot_dir / "pilot_meta.json").read_text(encoding="utf-8"))
    tdir = runs_dir / pmeta["targets_run"]
    gdir = runs_dir / pmeta["generation_run"]
    with (tdir / "targets.csv").open(encoding="utf-8", newline="") as f:
        targets = {t["target_id"]: t for t in csv.DictReader(f)}
    edits = {(e["target_id"], e["edit_type"]): e for e in
             (json.loads(x) for x in (tdir / "edits.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    with (pilot_dir / "cases.csv").open(encoding="utf-8", newline="") as f:
        cases = list(csv.DictReader(f))
    originals: dict[int, ParsedReport] = {}
    (out_dir / "diffs").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for c in cases:
        t = targets[c["target_id"]]
        h = int(c["hadm_id"])
        if h not in originals:
            originals[h] = parse_report(original_report(gdir, h)[0])
        raw, status = _final_raw(pilot_dir / "cases" / c["case_id"])
        parsed = parse_report(raw)
        cands = candidate_sentences(parsed, t["unit_id"], t["label"])
        row: dict[str, Any] = {
            "case_id": c["case_id"], "kind": c["kind"], "target_id": t["target_id"], "hadm_id": h,
            "unit_id": t["unit_id"], "unit_type": t["unit_type"], "label": t["label"],
            "edit_type": c["edit_type"], "original_sentence": t["sentence_line"], "original_value": t["value"],
            "edited_value": "", "original_text": "", "edited_text": "", "auto_outcome": "",
            "cites_deleted_id": "", "control_stable": "", "impression_recs_changed": "",
            "needs_manual_read": False, "format_status": status, "n_candidates": len(cands),
            "regenerated_candidates": "\n".join(f"{s.section}: {s.raw.strip()}" for s in cands),
        }
        if c["kind"] == "control":
            same = [s.raw.strip() for s in parsed.sections.get(t["section"], [])]
            row["control_stable"] = t["sentence_line"] in same
        else:
            e = edits[(t["target_id"], c["edit_type"])]
            row["original_text"] = e["original_text"]
            row["edited_text"] = e["edited_text"] or "<unit removed from evidence>"
            if c["edit_type"] == "delete":
                row["auto_outcome"] = label_delete(cands, t["value"])
                row["cites_deleted_id"] = any(t["unit_id"] in s.citations for s in parsed.sentences())
            else:
                row["edited_value"] = e["edited_value"]
                row["auto_outcome"] = label_value_edit(cands, t["unit_id"], t["value"], e["edited_value"])
            before = _section_lines(originals[h], ("Impression", "Recommendations"))
            after = _section_lines(parsed, ("Impression", "Recommendations"))
            row["impression_recs_changed"] = before != after
            if before != after:
                diff = "\n".join(difflib.unified_diff(before, after, "original", c["case_id"], lineterm=""))
                (out_dir / "diffs" / f"{c['case_id']}.diff").write_text(diff + "\n", encoding="utf-8",
                                                                        newline="\n")
            row["needs_manual_read"] = c["edit_type"] == "lab_major" or row["auto_outcome"] == "ambiguous"
        rows.append(row)

    cols = list(rows[0]) if rows else list(MANUAL_COLUMNS)
    with (out_dir / "perturbation_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "perturbation_manual_check.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANUAL_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r.get(k, "") for k in MANUAL_COLUMNS}, "auto_outcome": r["auto_outcome"],
                        "human_outcome": "", "notes": ""})
    summary = summarize(rows)
    (out_dir / "perturbation_summary.md").write_text(summary_markdown(summary, pilot_dir.name), encoding="utf-8",
                                                     newline="\n")
    (out_dir / "perturbation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8",
                                                       newline="\n")
    return {"rows": rows, "summary": summary}


def _frac(k: int, n: int) -> str:
    return f"{k}/{n}" + (f" ({100 * k / n:.0f}%)" if n else "")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per edit type and outcome, control stability, flags."""
    by_type: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["kind"] != "edit":
            continue
        d = by_type.setdefault(r["edit_type"], {"n": 0})
        d["n"] += 1
        d[r["auto_outcome"]] = d.get(r["auto_outcome"], 0) + 1
        if r["edit_type"] == "delete" and r["cites_deleted_id"]:
            d["cites_deleted_id"] = d.get("cites_deleted_id", 0) + 1
        if r["impression_recs_changed"]:
            d["impression_recs_changed"] = d.get("impression_recs_changed", 0) + 1
    controls = [r for r in rows if r["kind"] == "control"]
    per_target: dict[str, list[bool]] = {}
    for r in controls:
        per_target.setdefault(r["target_id"], []).append(bool(r["control_stable"]))
    return {
        "by_edit_type": by_type,
        "controls": {"n": len(controls), "stable": sum(1 for r in controls if r["control_stable"]),
                     "per_target": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_target.items())}},
        "n_needs_manual_read": sum(1 for r in rows if r["needs_manual_read"]),
    }


def summary_markdown(s: dict[str, Any], pilot_run: str) -> str:
    """``perturbation_summary.md`` (rates always with raw counts)."""
    lines = [f"# Perturbation summary: pilot run `{pilot_run}`", "",
             "Automated labels are PROVISIONAL until checked by hand (perturbation_manual_check.csv).", ""]
    for et, d in sorted(s["by_edit_type"].items()):
        outcomes = DELETE_OUTCOMES if et == "delete" else VALUE_OUTCOMES
        lines += [f"## {et} (n = {d['n']})", "", "| outcome | count |", "|---|---|"]
        lines += [f"| {o} | {_frac(d.get(o, 0), d['n'])} |" for o in outcomes]
        if et == "delete":
            lines.append(f"| still cites deleted id | {_frac(d.get('cites_deleted_id', 0), d['n'])} |")
        lines.append(f"| Impression/Recommendations changed | {_frac(d.get('impression_recs_changed', 0), d['n'])} |")
        lines.append("")
    c = s["controls"]
    lines += ["## Control stability", "",
              f"Target sentence unchanged in {_frac(c['stable'], c['n'])} control runs.", ""]
    lines += [f"- {k}: {v}" for k, v in c["per_target"].items()]
    lines += ["", f"Cases flagged for manual reading (all lab_major + ambiguous): {s['n_needs_manual_read']}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- counterfactual audit labels
#
# For each condition (ehr_rag, ehr_rag_enforced) and cited target U, a PAIR is (an ORIGINAL claim
# citing U, the edited U). The original is that condition's unedited report from the audit run.
# Claims are the checker's (list items keep their opening words). The pair is matched to the
# regenerated claims citing the same unit id: the best match is the candidate in the same section
# with the highest difflib ratio to the original claim (then any section). Signals use the
# checker's extraction: clinical numbers (equal, or rounded to the stated precision) and direction
# claims (direction words near the lab name, and lab-derived terms) about U's lab:
#   follows-signal  a number matching an edited value but no original value, or a direction claim
#                   that holds for the edited unit and not for the original one
#   stale-signal    the same with original and edited swapped
# Labels: follows (follows-signal and no stale-signal), stale (the reverse), dropped (no
# regenerated claim cites U), other (cites U but neither, or both, signals).
# Lab-derived terms: for to_normal edits, the lexicon terms of U's lab and original direction;
# "present" if one is named in the original Impression + Recommendations, "survives" if it is
# still named there after the edit. Null-edit churn: per section, the share of original lines that
# do not appear unchanged in the same section of the null-edited regeneration.

AUDIT_LABELS: tuple[str, ...] = ("follows", "stale", "dropped", "other")
AUDIT_MANUAL_COLUMNS: tuple[str, ...] = (
    "pair_id", "condition", "hadm_id", "target_id", "unit_id", "lab", "edit_type", "section", "original_sentence",
    "original_unit_text", "edited_unit_text", "matched_regenerated_sentence", "auto_label", "follows_signals",
    "stale_signals", "human_label", "notes",
)


def _final_case_text(d: Path) -> str:
    meta = json.loads((d / "parsed.json").read_text(encoding="utf-8"))
    return (d / meta["raw_file"]).read_text(encoding="utf-8")


def _claims(parsed: ParsedReport, checker: Any, ctx: Any) -> list[dict[str, Any]]:
    out = []
    for s in parsed.sentences():
        if s.is_abstention:
            continue
        for k, (text, ids) in enumerate(checker.items(s, ctx)):
            out.append({"section": s.section, "line": s.raw.strip(), "text": text, "ids": ids, "item": k})
    return out


def _point_values(u: Any) -> list[Any]:
    from ehr_rag.perturb.select_targets import LAB_POINTS, to_decimal

    return [d for p in LAB_POINTS if u.fields.get(p) for d in [to_decimal(u.fields[p]["value"])] if d is not None]


def audit_signals(checker: Any, ctx: Any, text: str, original: Any, edited: Any) -> dict[str, list[str]]:
    """Follows / stale evidence in one claim about the edited lab unit (see the section comment)."""
    stated = checker.clinical_numbers(text)
    v0, v1 = _point_values(original), _point_values(edited)
    fol = [f"number {n}" for n in stated if checker.number_matches(n, v1) and not checker.number_matches(n, v0)]
    sta = [f"number {n}" for n in stated if checker.number_matches(n, v0) and not checker.number_matches(n, v1)]
    concept = checker.lab_concept(original)
    for _, m, direction, negated, word in checker.direction_claims(text, ctx.mentions(text)):
        if m.kind == "lab" and m.key != concept:
            continue
        if m.kind == "term" and (concept not in m.term["labs"]
                                 or (m.term.get("fluid") and original.fields.get("fluid") != m.term["fluid"])):
            continue
        if m.kind not in ("lab", "term"):
            continue
        h0 = checker.direction_holds([original], direction, stated)
        h1 = checker.direction_holds([edited], direction, stated)
        if h0 is None or h1 is None:
            continue
        h0, h1 = h0 != negated, h1 != negated
        claim = f"{'not ' if negated else ''}{direction} ('{word}')"
        if h1 and not h0:
            fol.append(claim)
        elif h0 and not h1:
            sta.append(claim)
    return {"follows": fol, "stale": sta}


def audit_label(signals: dict[str, list[str]] | None) -> str:
    """follows / stale / other from the signals; None (no regenerated claim cites the unit) is dropped."""
    if signals is None:
        return "dropped"
    f, s = bool(signals["follows"]), bool(signals["stale"])
    return "follows" if f and not s else "stale" if s and not f else "other"


def best_match(orig: dict[str, Any], cands: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Same section first, then highest difflib similarity to the original claim; None if no candidate."""
    if not cands:
        return None
    return max(cands, key=lambda c: (c["section"] == orig["section"],
                                     difflib.SequenceMatcher(None, orig["text"], c["text"]).ratio()))


def _term_names(checker: Any, original: Any, status: str) -> set[str]:
    concept = checker.lab_concept(original)
    dirs = {"high": {"high"}, "low": {"low"}, "mixed": {"high", "low"}}.get(status, set())
    return {k for k, t in checker.terms.items() if concept in t["labs"] and t["direction"] in dirs
            and (not t.get("fluid") or original.fields.get("fluid") == t["fluid"])}


def _ir_terms(parsed: ParsedReport, ctx: Any) -> set[str]:
    return {m.key for s in parsed.sentences() if s.section in ("Impression", "Recommendations")
            for m in ctx.mentions(s.text) if m.kind == "term"}


def section_churn(original: ParsedReport, regenerated: ParsedReport) -> list[tuple[str, int, int]]:
    """Per section: (section, original non-abstention lines, how many do not appear unchanged after)."""
    from ehr_rag.config import SECTIONS

    out = []
    for sec in SECTIONS:
        before = [s.raw.strip() for s in original.sections.get(sec, []) if not s.is_abstention]
        after = {s.raw.strip() for s in regenerated.sections.get(sec, [])}
        out.append((sec, len(before), sum(1 for x in before if x not in after)))
    return out


def label_audit(cfg: dict[str, Any], audit_dirs: list[Path], runs_dir: Path) -> dict[str, Any]:
    """Pairs, term survival and null-edit churn for one or more run-audit runs (shards)."""
    from ehr_rag.checker.rules import Checker
    from ehr_rag.config import resolve_path
    from ehr_rag.evidence.build_units import units_dir
    from ehr_rag.evidence.schema import read_units
    from ehr_rag.perturb.run_pilot import frozen_units

    metas = [json.loads((d / "audit_run_meta.json").read_text(encoding="utf-8")) for d in audit_dirs]
    if len({m["targets_run"] for m in metas}) != 1:
        raise ValueError("audit runs come from different audit-targets runs")
    tdir = runs_dir / metas[0]["targets_run"]
    with (tdir / "audit_targets.csv").open(encoding="utf-8", newline="") as f:
        targets = list(csv.DictReader(f))
    edits = {e["target_id"]: e for e in (json.loads(x) for x in
             (tdir / "audit_edits.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    checker = Checker.from_config(cfg)
    udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
    case_dirs: dict[tuple[str, str], Path] = {}
    for d in audit_dirs:
        with (d / "audit_cases.csv").open(encoding="utf-8", newline="") as f:
            for c in csv.DictReader(f):
                key = (c["condition"], c["case_id"])
                if key in case_dirs:
                    raise ValueError(f"case {key} appears in more than one audit run")
                case_dirs[key] = d / c["condition"] / c["case_id"]
    pairs, terms, churn = [], [], []
    for cond in [c for c in ("ehr_rag", "ehr_rag_enforced") if any(k[0] == c for k in case_dirs)]:
        for h in sorted({int(t["hadm_id"]) for t in targets}):
            od = case_dirs.get((cond, f"{h}__original"))
            if od is None:
                continue
            units = read_units(udir / f"{h}.jsonl")
            shown = json.loads((od / "evidence_shown.json").read_text(encoding="utf-8"))
            ctx = checker.admission(frozen_units(shown, units), units)
            o_parsed = parse_report(_final_case_text(od))
            o_claims = _claims(o_parsed, checker, ctx)
            by_id = {u.unit_id: u for u in units}
            for t in [t for t in targets if int(t["hadm_id"]) == h]:
                rd = case_dirs.get((cond, t["target_id"]))
                if rd is None:
                    continue
                e = edits[t["target_id"]]
                u0 = by_id[t["unit_id"]]
                u1 = dataclasses.replace(u0, fields=e["edited_fields"], text=e["edited_text"])
                r_parsed = parse_report(_final_case_text(rd))
                r_claims = _claims(r_parsed, checker, ctx)
                if t["kind"] == "null":
                    cited_after = any(t["unit_id"] in c["ids"] for c in r_claims)
                    for sec, n_lines, n_changed in section_churn(o_parsed, r_parsed):
                        churn.append({"condition": cond, "hadm_id": h, "target_id": t["target_id"], "section": sec,
                                      "n_lines": n_lines, "n_changed": n_changed, "null_unit_cited_after": cited_after})
                    continue
                if e["edit_type"] == "to_normal":
                    names = _term_names(checker, u0, t["status"])
                    if names:
                        present = bool(names & _ir_terms(o_parsed, ctx))
                        terms.append({"condition": cond, "hadm_id": h, "target_id": t["target_id"],
                                      "unit_id": t["unit_id"], "lab": t["label"], "terms": " ".join(sorted(names)),
                                      "present_in_original": present,
                                      "survives_edit": present and bool(names & _ir_terms(r_parsed, ctx))})
                cands = [c for c in r_claims if t["unit_id"] in c["ids"]]
                for k, oc in enumerate(c for c in o_claims if t["unit_id"] in c["ids"]):
                    m = best_match(oc, cands)
                    sig = audit_signals(checker, ctx, m["text"], u0, u1) if m else None
                    osig = audit_signals(checker, ctx, oc["text"], u1, u0)   # does the original state its own value?
                    pairs.append({
                        "pair_id": f"{cond}|{t['target_id']}|{k}", "condition": cond, "hadm_id": h,
                        "target_id": t["target_id"], "unit_id": t["unit_id"], "lab": t["label"],
                        "edit_type": e["edit_type"], "original_value": e["original_value"],
                        "edited_value": e["edited_value"], "section": oc["section"], "original_sentence": oc["line"],
                        "original_claim": oc["text"], "original_states_value_or_direction": bool(osig["follows"]),
                        "n_regenerated_citing": len(cands), "matched_section": m["section"] if m else "",
                        "matched_regenerated_sentence": m["line"] if m else "",
                        "matched_claim": m["text"] if m else "", "auto_label": audit_label(sig),
                        "follows_signals": "; ".join(sig["follows"]) if sig else "",
                        "stale_signals": "; ".join(sig["stale"]) if sig else "",
                        "original_unit_text": e["original_text"], "edited_unit_text": e["edited_text"],
                    })
    return {"pairs": pairs, "terms": terms, "churn": churn, "targets_run": tdir.name}


# --------------------------------------------------------------------------- counterfactual audit summaries


def _faith_stat(items: list[list[dict[str, Any]]]) -> float | None:
    f = sum(1 for ps in items for p in ps if p["auto_label"] == "follows")
    s = sum(1 for ps in items for p in ps if p["auto_label"] == "stale")
    return None if f + s == 0 else f / (f + s)


def _by_admission(pairs: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for p in pairs:
        out.setdefault(int(p["hadm_id"]), []).append(p)
    return out


def _counts(pairs: list[dict[str, Any]]) -> dict[str, int]:
    return {"n_pairs": len(pairs), **{lab: sum(1 for p in pairs if p["auto_label"] == lab) for lab in AUDIT_LABELS}}


def summarize_audit(res: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    """Counts and faithfulness = follows / (follows + stale) with 95% bootstrap CIs over admissions."""
    from ehr_rag.config import SECTIONS
    from ehr_rag.stats import bootstrap

    pairs = res["pairs"]
    conds = [c for c in ("ehr_rag", "ehr_rag_enforced") if any(p["condition"] == c for p in pairs)]
    out: dict[str, Any] = {"conditions": {}, "by_section": [], "n_resamples": n_boot, "seed": seed}
    for c in conds:
        ps = [p for p in pairs if p["condition"] == c]
        out["conditions"][c] = {**_counts(ps), "n_admissions": len(_by_admission(ps)),
                                "faithfulness": bootstrap(_by_admission(ps), _faith_stat, n_boot, f"{seed}:{c}")}
        for sec in SECTIONS:
            ss = [p for p in ps if p["section"] == sec]
            out["by_section"].append({"condition": c, "section": sec, **_counts(ss),
                                      "faithfulness": bootstrap(_by_admission(ss), _faith_stat, n_boot,
                                                                f"{seed}:{c}:{sec}")})
    if len(conds) == 2:
        a, b = _by_admission([p for p in pairs if p["condition"] == conds[0]]), \
            _by_admission([p for p in pairs if p["condition"] == conds[1]])
        common = {h: (a[h], b[h]) for h in set(a) & set(b)}

        def diff(items: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]) -> float | None:
            fa, fb = _faith_stat([x for x, _ in items]), _faith_stat([y for _, y in items])
            return None if fa is None or fb is None else fb - fa

        out["paired_difference"] = {"definition": f"faithfulness({conds[1]}) - faithfulness({conds[0]}), same "
                                                  "resampled admissions", **bootstrap(common, diff, n_boot,
                                                                                      f"{seed}:paired")}
    churn: dict[tuple[str, str], list[int]] = {}
    for r in res["churn"]:
        acc = churn.setdefault((r["condition"], r["section"]), [0, 0, 0])
        acc[0] += r["n_lines"]
        acc[1] += r["n_changed"]
    out["null_churn"] = [{"condition": c, "section": s, "n_lines": v[0], "n_changed": v[1],
                          "share_changed": None if not v[0] else v[1] / v[0]} for (c, s), v in sorted(churn.items())]
    out["null_targets"] = {c: {"n": len({r["target_id"] for r in res["churn"] if r["condition"] == c}),
                               "cited_after_edit": len({r["target_id"] for r in res["churn"]
                                                        if r["condition"] == c and r["null_unit_cited_after"]})}
                           for c in conds}
    out["terms"] = {c: {"targets_with_terms": sum(1 for t in res["terms"] if t["condition"] == c),
                        "present_in_original": sum(1 for t in res["terms"] if t["condition"] == c and t["present_in_original"]),
                        "survives_edit": sum(1 for t in res["terms"] if t["condition"] == c and t["survives_edit"])}
                    for c in conds}
    return out


def audit_manual_sample(pairs: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` pairs, equal allocation over (condition, label) strata, round-robin, seeded per stratum."""
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in pairs:
        strata.setdefault((p["condition"], p["auto_label"]), []).append(p)
    for k in strata:
        random.Random(f"{seed}:{k[0]}:{k[1]}").shuffle(strata[k])
    out: list[dict[str, Any]] = []
    taken = {k: 0 for k in strata}
    while len(out) < n and any(taken[k] < len(strata[k]) for k in strata):
        for k in sorted(strata):
            if len(out) < n and taken[k] < len(strata[k]):
                out.append(strata[k][taken[k]])
                taken[k] += 1
    return out


def _fmt_share(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def audit_summary_markdown(s: dict[str, Any], runs: list[str]) -> str:
    """``audit_summary.md``: counts always next to rates."""
    from ehr_rag.stats import fmt_ci

    lines = [f"# Counterfactual citation audit: {' + '.join(runs)}", "",
             "Automatic labels are provisional until checked by hand (audit_manual_check.csv). "
             f"Faithfulness = Follows / (Follows + Stale); 95% CI: {s['n_resamples']} bootstrap resamples of "
             "admissions.", "",
             "| condition | admissions | pairs | Follows | Stale | Dropped | Other | faithfulness [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for c, d in s["conditions"].items():
        lines.append(f"| {c} | {d['n_admissions']} | {d['n_pairs']} | {d['follows']} | {d['stale']} | {d['dropped']} | "
                     f"{d['other']} | {fmt_ci(d['faithfulness'])} |")
    if "paired_difference" in s:
        pdiff = s["paired_difference"]
        lines += ["", f"Paired difference ({pdiff['definition']}; {pdiff['n_groups']} admissions): {fmt_ci(pdiff)}"]
    lines += ["", "## By section of the original sentence", "",
              "| condition | section | pairs | Follows | Stale | Dropped | Other | faithfulness [95% CI] |",
              "|---|---|---|---|---|---|---|---|"]
    for r in s["by_section"]:
        lines.append(f"| {r['condition']} | {r['section']} | {r['n_pairs']} | {r['follows']} | {r['stale']} | "
                     f"{r['dropped']} | {r['other']} | {fmt_ci(r['faithfulness'])} |")
    lines += ["", "## Lab-derived terms in Impression + Recommendations after an edit into the normal range", "",
              "| condition | targets with a term for their lab | term in original | term survives |", "|---|---|---|---|"]
    for c, d in s["terms"].items():
        lines.append(f"| {c} | {d['targets_with_terms']} | {d['present_in_original']} | {d['survives_edit']} |")
    lines += ["", "## Null-edit churn (an uncited unit edited): lines changed per section", "",
              "| condition | section | lines | changed | share |", "|---|---|---|---|---|"]
    for r in s["null_churn"]:
        lines.append(f"| {r['condition']} | {r['section']} | {r['n_lines']} | {r['n_changed']} | "
                     f"{_fmt_share(r['share_changed'])} |")
    for c, d in s["null_targets"].items():
        lines.append(f"\n{c}: {d['n']} null-edit targets; the edited uncited unit is cited after the edit in "
                     f"{d['cited_after_edit']}.")
    return "\n".join(lines) + "\n"


def analyze_audit(cfg: dict[str, Any], audit_dirs: list[Path], runs_dir: Path, out_dir: Path) -> dict[str, Any]:
    """``analyze-audit``: label, summarize and write every output to ``out_dir``."""
    ac = cfg["audit"]
    res = label_audit(cfg, audit_dirs, runs_dir)
    summary = summarize_audit(res, ac["bootstrap_resamples"], ac["seed"])
    for name, rows in (("audit_pairs.csv", res["pairs"]), ("audit_terms.csv", res["terms"]),
                       ("audit_null_churn.csv", res["churn"])):
        with (out_dir / name).open("w", encoding="utf-8", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    by_sec = [{k: v for k, v in r.items() if k != "faithfulness"} | {
        "faithfulness": r["faithfulness"]["estimate"], "ci_low": r["faithfulness"]["ci_low"],
        "ci_high": r["faithfulness"]["ci_high"]} for r in summary["by_section"]]
    with (out_dir / "audit_by_section.csv").open("w", encoding="utf-8", newline="") as f:
        if by_sec:
            w = csv.DictWriter(f, fieldnames=list(by_sec[0]))
            w.writeheader()
            w.writerows(by_sec)
    sample = audit_manual_sample(res["pairs"], ac["manual_check_n"], ac["seed"])
    with (out_dir / "audit_manual_check.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_MANUAL_COLUMNS)
        w.writeheader()
        for p in sample:
            w.writerow({**{k: p.get(k, "") for k in AUDIT_MANUAL_COLUMNS}, "human_label": "", "notes": ""})
    (out_dir / "audit_summary.json").write_text(json.dumps({"targets_run": res["targets_run"], **summary}, indent=2),
                                                encoding="utf-8", newline="\n")
    (out_dir / "audit_summary.md").write_text(audit_summary_markdown(summary, [d.name for d in audit_dirs]),
                                              encoding="utf-8", newline="\n")
    return {"summary": summary, "n_pairs": len(res["pairs"]), "n_manual": len(sample)}
