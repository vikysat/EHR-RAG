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
import difflib
import json
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
