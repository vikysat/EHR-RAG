"""``make-tables``: markdown versions of the paper's tables from finished runs.

Every rate is a pooled ratio over the condition's sentences / citations / pairs with a 95% CI
from ``audit.bootstrap_resamples`` bootstrap resamples of ADMISSIONS (``stats.bootstrap``,
seeded with ``audit.seed``). Sources (each table is written only if its source is given):

  1 support      evaluate run: sentences.jsonl (+ optional labeled manual_review.csv for
                 "manual unsupported" = human_support unsupported / labeled, not_checkable excluded)
  2 audit        analyze-audit run: audit_pairs.csv / audit_summary.json
  3 enforcement  check-citations run: enforcement_per_report.csv
  4 checker      check-citations run: checker_lines.jsonl (generated text, before enforcement)
  5 reference    text-metrics run: text_metrics.csv (mean F1 per condition)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from ehr_rag.config import ALL_CONDITIONS, SECTIONS
from ehr_rag.stats import bootstrap, fmt_ci, ratio_stat

GROUPS: dict[str, tuple[str, ...]] = {
    "History+Findings": ("History", "Findings"),
    "Impression+Recommendations": ("Impression", "Recommendations"),
    "ALL": SECTIONS,
}
ERROR_TYPES = ("invalid_id", "number_mismatch", "name_missing", "direction_mismatch", "trend_mismatch",
               "too_many_citations")


def _read_csv(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _conds(rows: list[dict[str, Any]]) -> list[str]:
    present = {r["condition"] for r in rows}
    return [c for c in ALL_CONDITIONS if c in present]


def _by_hadm(rows: list[Any], key: Callable[[Any], Any] = lambda r: r["hadm_id"]) -> dict[int, list[Any]]:
    out: dict[int, list[Any]] = {}
    for r in rows:
        out.setdefault(int(key(r)), []).append(r)
    return out


def _ci(rows: list[Any], num: Callable[[Any], float], den: Callable[[Any], float], n: int, seed: str) -> dict[str, Any]:
    stat = ratio_stat(lambda xs: sum(num(x) for x in xs), lambda xs: sum(den(x) for x in xs))
    return bootstrap(_by_hadm(rows), stat, n, seed)


def _md(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


# --------------------------------------------------------------------------- table 1: support


def table_support(eval_dir: Path, manual: Path | None, n: int, seed: int) -> tuple[list[str], int]:
    sents = [s for s in _read_jsonl(eval_dir / "sentences.jsonl") if not s["is_abstention"]]
    labels = []
    if manual is not None:
        labels = [r for r in _read_csv(manual) if r.get("human_support") and r["human_support"] != "not_checkable"]
    header = ["condition", "sections", "sentences", "cited", "invalid IDs", "precision (ALCE)", "precision (plain)",
              "recall", "manual unsupported"]
    rows = []
    for c in _conds(sents):
        for g, secs in GROUPS.items():
            ss = [s for s in sents if s["condition"] == c and s["section"] in secs]
            if not ss:
                continue
            k = f"{seed}:{c}:{g}"
            n_lines = len({(s["hadm_id"], s["section"], s["index"]) for s in ss})
            def cit(s: Any) -> int:
                return len(s["cited_ids"]) + len(s["invalid_ids"])
            ml = [r for r in labels if r["condition"] == c and r["section"] in secs]
            man = (_ci(ml, lambda r: r["human_support"] == "unsupported", lambda r: 1, n, k + ":m") if ml
                   else {"estimate": None})
            rows.append([c, g, f"{n_lines} ({len(ss)} claims)",
                         fmt_ci(_ci(ss, lambda s: bool(s["cited_ids"]), lambda s: 1, n, k + ":c")),
                         fmt_ci(_ci(ss, lambda s: len(s["invalid_ids"]), cit, n, k + ":i")),
                         fmt_ci(_ci(ss, lambda s: len(s["cited_ids"]) - len(s["irrelevant_ids"]), cit, n, k + ":pa")),
                         fmt_ci(_ci(ss, lambda s: len(s["supporting_ids"]), cit, n, k + ":pp")),
                         fmt_ci(_ci(ss, lambda s: bool(s["supported"]), lambda s: 1, n, k + ":r")),
                         (fmt_ci(man) + f" (n={len(ml)})") if ml else "n/a"])
    lines = ["## Table 1. Citation support by condition and section group", "",
             "Rates over claims (precision over citations); 95% CI from bootstrap resamples of admissions. "
             "Manual unsupported: share of hand-labeled sentences marked unsupported (not_checkable excluded).", ""]
    return lines + _md(header, rows), len(rows)


# --------------------------------------------------------------------------- table 2: audit


def table_audit(audit_dir: Path) -> tuple[list[str], int]:
    s = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    header = ["condition", "edited pairs", "Follows", "Stale", "Dropped", "Other", "faithfulness"]
    rows = [[c, str(d["n_pairs"]), str(d["follows"]), str(d["stale"]), str(d["dropped"]), str(d["other"]),
             fmt_ci(d["faithfulness"])] for c, d in s["conditions"].items()]
    if "paired_difference" in s:
        rows.append(["enforced - plain (paired)", "", "", "", "", "", fmt_ci(s["paired_difference"])])
    lines = ["## Table 2. Counterfactual citation audit", "",
             "Faithfulness = Follows / (Follows + Stale); 95% CI from bootstrap resamples of admissions; the paired "
             "row uses the same resampled admissions for both conditions.", ""]
    return lines + _md(header, rows), len(rows)


# --------------------------------------------------------------------------- table 3: enforcement


def table_enforcement(check_dir: Path, n: int, seed: int) -> tuple[list[str], int]:
    p = check_dir / "enforcement_per_report.csv"
    reps = [{k: (v if k == "condition" else int(v)) for k, v in r.items()} for r in _read_csv(p)] \
        if p.exists() and p.stat().st_size else []
    header = ["condition", "failing", "repaired", "retreated", "dropped", "sentences kept", "numbers kept"]
    rows = []
    for c in _conds(reps):
        rs = [r for r in reps if r["condition"] == c]
        k = f"{seed}:{c}"
        rows.append([c, str(sum(r["n_fail"] for r in rs)), str(sum(r["n_repaired"] for r in rs)),
                     str(sum(r["n_retreated"] for r in rs)), str(sum(r["n_dropped"] for r in rs)),
                     fmt_ci(_ci(rs, lambda r: r["n_lines_after"], lambda r: r["n_lines"], n, k + ":s")),
                     fmt_ci(_ci(rs, lambda r: r["n_numbers_after"], lambda r: r["n_numbers_before"], n, k + ":n"))])
    lines = ["## Table 3. Enforcement", "",
             "Failing = lines failing the checker before enforcement; sentences / numbers kept = share after "
             "enforcement (95% CI, bootstrap over admissions).", ""]
    return lines + _md(header, rows), len(rows)


# --------------------------------------------------------------------------- table 4: checker error profile


def table_checker(check_dir: Path, n: int, seed: int) -> tuple[list[str], int]:
    lines_ = _read_jsonl(check_dir / "checker_lines.jsonl")
    header = ["condition", "lines", *ERROR_TYPES, "unchecked"]
    rows = []
    for c in _conds(lines_):
        ls = [x for x in lines_ if x["condition"] == c]
        k = f"{seed}:{c}"
        cells = [c, str(len(ls))]
        for et in ERROR_TYPES:
            cnt = sum(1 for x in ls if et in x["error_types"])
            cells.append(f"{cnt} ({fmt_ci(_ci(ls, lambda x, e=et: e in x['error_types'], lambda x: 1, n, k + et))})")
        un = sum(1 for x in ls if x["status"] == "unchecked")
        cells.append(f"{un} ({fmt_ci(_ci(ls, lambda x: x['status'] == 'unchecked', lambda x: 1, n, k + ':u'))})")
        rows.append(cells)
    lines = ["## Table 4. Checker error profile (generated text, before enforcement)", "",
             "Lines with each error type: count (share of lines [95% CI, bootstrap over admissions]); unchecked = "
             "no checkable content.", ""]
    return lines + _md(header, rows), len(rows)


# --------------------------------------------------------------------------- table 5: reference metrics


def table_reference(tm_dir: Path, n: int, seed: int) -> tuple[list[str], int]:
    rows_ = _read_csv(tm_dir / "text_metrics.csv")
    has_bs = bool(rows_) and "bertscore_f1" in rows_[0]
    header = ["condition", "reports", "ROUGE-L F1", "BERTScore F1"]
    rows = []
    for c in _conds(rows_):
        rs = [r for r in rows_ if r["condition"] == c]
        k = f"{seed}:{c}"
        rl = _ci(rs, lambda r: float(r["rougeL_f1"]), lambda r: 1, n, k + ":r")
        bs = _ci(rs, lambda r: float(r["bertscore_f1"]), lambda r: 1, n, k + ":b") if has_bs else {"estimate": None}
        rows.append([c, str(len(rs)), fmt_ci(rl), fmt_ci(bs)])
    lines = ["## Table 5. Reference metrics against the discharge summary", "",
             "Mean F1 per report; 95% CI from bootstrap resamples of admissions.", ""]
    return lines + _md(header, rows), len(rows)


def make_tables(out_dir: Path, *, eval_dir: Path | None, manual: Path | None, check_dir: Path | None,
                tm_dir: Path | None, audit_dir: Path | None, n: int, seed: int) -> dict[str, int]:
    """Write ``table_<k>_<name>.md`` per available source and ``tables.md``; returns rows per table."""
    parts: list[tuple[str, tuple[list[str], int]]] = []
    if eval_dir:
        parts.append(("1_support", table_support(eval_dir, manual, n, seed)))
    if audit_dir:
        parts.append(("2_audit", table_audit(audit_dir)))
    if check_dir:
        parts.append(("3_enforcement", table_enforcement(check_dir, n, seed)))
        parts.append(("4_checker", table_checker(check_dir, n, seed)))
    if tm_dir:
        parts.append(("5_reference", table_reference(tm_dir, n, seed)))
    sources = {"evaluate": eval_dir, "manual labels": manual, "check-citations": check_dir,
               "text-metrics": tm_dir, "analyze-audit": audit_dir}
    doc = ["# Tables", "", "Sources: " + "; ".join(f"{k} `{v.name}`" for k, v in sources.items() if v), ""]
    counts = {}
    for name, (lines, n_rows) in parts:
        (out_dir / f"table_{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        doc += lines + [""]
        counts[name] = n_rows
    (out_dir / "tables.md").write_text("\n".join(doc), encoding="utf-8", newline="\n")
    return counts
