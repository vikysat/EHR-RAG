"""Hand-labeling sheets, the terminal labeler and the analysis of the labels.

Sheets (``labeling-sheets``, written into a gitignored run folder):

  manual_review_sheet.csv   ``n`` sentences (default 50) from ``n`` different test admissions, one each.
                            Admissions are drawn with ``random.Random(f"{seed}:labeling")`` from those that
                            have reports of every condition (default ehr_rag and ehr_rag_enforced); each is
                            given a (condition, section) cell in turn (conditions alternate, sections in a
                            shuffled order per condition), so conditions and the four sections are mixed as
                            evenly as possible (counts differ by at most one). Within the cell one claim
                            with >= 1 valid citation (abstentions excluded) is drawn; if the cell has none, the
                            least-used cell that has one is taken. The items are then shuffled and numbered
                            m01, m02... The condition is NOT in the sheet.
  relabel_sheet.csv         ``n_relabel`` (default 15) of the same items, a new random order, numbered r01...
  labeling_key.csv          per item: condition, the original item (for r..), sentence key, NLI result.

  Sheet columns: item_id, admission (anonymised), section, sentence, cited_units, label, notes.
  Labels: supported / partially supported / unsupported / contradicted.

Retrieval relevance sheets are ``export-relevance``'s ``relevance_to_label.csv`` (column ``relevant``, 0/1).

``label --sheet <path>`` shows one item at a time and saves the sheet after every answer; the position is
kept in ``<sheet>.progress.json`` so a new start resumes where the last one stopped.

Analysis (``make-tables``, table 7): unsupported rate = (unsupported + contradicted) / labeled per condition
and section group; NLI agreement = Cohen's kappa between human "supported" (only ``supported``; partially
supported counts as not supported) and the NLI verdict (the claim's cited units entail it); intra-annotator
agreement = kappa between the first label and the re-label of the same items (four classes, and binary).
CIs: bootstrap resamples of admissions (``stats.bootstrap``).
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from ehr_rag.config import SECTIONS

LABELS: tuple[str, ...] = ("supported", "partially supported", "unsupported", "contradicted")
NOT_SUPPORTED = ("unsupported", "contradicted")
SHEET_COLUMNS: tuple[str, ...] = ("item_id", "admission", "section", "sentence", "cited_units", "label", "notes")
KEY_COLUMNS: tuple[str, ...] = ("item_id", "sheet", "original_item", "condition", "admission", "section",
                                "sentence_key", "evaluate_run", "nli_label", "nli_supported")
MANUAL_SHEET = "manual_review_sheet.csv"
RELABEL_SHEET = "relabel_sheet.csv"
KEY_FILE = "labeling_key.csv"
RELEVANCE_SHEET = "relevance_to_label.csv"
DEFAULT_CONDITIONS = ("ehr_rag", "ehr_rag_enforced")


class LabelingError(ValueError):
    """A sheet cannot be made, read or labeled."""


# --------------------------------------------------------------------------- csv helpers


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return list(r.fieldnames or []), rows


def write_csv_atomic(path: Path, columns: Iterable[str], rows: list[dict[str, Any]]) -> None:
    """Write to a temporary file next to ``path``, then replace it (a crash never leaves half a sheet)."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- making the sheets


def _sentence_key(s: dict[str, Any], alias: Callable[[int], Any] = lambda h: h) -> str:
    """The claim within its evaluation run: condition|admission|section|line index[.claim] (admission as
    given by ``alias``, so the key file never holds a real id with credentialed data)."""
    key = f"{s['condition']}|{alias(int(s['hadm_id']))}|{s['section']}|{s['index']}"
    return f"{key}.{s.get('claim_index', 0)}" if int(s.get("n_claims", 1)) > 1 else key


def build_sheets(sentences: list[dict[str, Any]], texts: Callable[[str, int], dict[str, str]],
                 alias: Callable[[int], Any], *, n: int, n_relabel: int, seed: int,
                 conditions: tuple[str, ...] = DEFAULT_CONDITIONS) -> dict[str, Any]:
    """The three tables (manual, relabel, key) from evaluated sentences.

    ``sentences``: rows of ``sentences.jsonl`` (plus ``evaluate_run``); ``texts(condition, hadm_id)`` gives the
    unit texts the model saw; ``alias(hadm_id)`` the anonymised admission label.
    """
    if n_relabel > n:
        raise LabelingError(f"--n-relabel {n_relabel} is larger than --n {n}")
    eligible: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    present: dict[int, set[str]] = {}
    for s in sentences:
        if s["condition"] not in conditions:
            continue
        h = int(s["hadm_id"])
        present.setdefault(h, set()).add(s["condition"])
        if s["is_abstention"] or not s["cited_ids"]:
            continue
        eligible.setdefault((h, s["condition"], s["section"]), []).append(s)
    adms = sorted(h for h, cs in present.items() if set(conditions) <= cs
                  and any((h, c, sec) in eligible for c in conditions for sec in SECTIONS))
    if len(adms) < n:
        raise LabelingError(f"only {len(adms)} admissions have reports of every condition {list(conditions)} with a "
                            f"cited claim; {n} needed")
    rng = random.Random(f"{seed}:labeling")
    chosen = rng.sample(adms, n)
    # cells in turn: conditions alternate (each section order shuffled per condition), so the conditions differ
    # by at most one item and the sections by at most one per condition
    per_cond = []
    for c in conditions:
        secs = list(SECTIONS)
        rng.shuffle(secs)
        per_cond.append([(c, sec) for sec in secs])
    rng.shuffle(per_cond)
    cells = [cell for group in zip(*per_cond) for cell in group]
    used = {c: 0 for c in cells}
    picked: list[dict[str, Any]] = []
    for i, h in enumerate(chosen):
        want = cells[i % len(cells)]
        options = [want] + sorted((c for c in cells if c != want), key=lambda c: (used[c], cells.index(c)))
        cell = next(c for c in options if (h, c[0], c[1]) in eligible)
        used[cell] += 1
        pool = sorted(eligible[(h, cell[0], cell[1])], key=_sentence_key)
        picked.append(rng.choice(pool))
    rng.shuffle(picked)
    manual, key = [], []
    for i, s in enumerate(picked, 1):
        h = int(s["hadm_id"])
        t = texts(s["condition"], h)
        item = f"m{i:02d}"
        manual.append({"item_id": item, "admission": alias(h), "section": s["section"], "sentence": s["sentence"],
                       "cited_units": "\n".join(f"[{u}] {t.get(u, '<not in evidence>')}" for u in s["cited_ids"]),
                       "label": "", "notes": ""})
        key.append({"item_id": item, "sheet": MANUAL_SHEET, "original_item": "", "condition": s["condition"],
                    "admission": alias(h), "section": s["section"], "sentence_key": _sentence_key(s, alias),
                    "evaluate_run": s.get("evaluate_run", ""), "nli_label": s.get("nli_label", ""),
                    "nli_supported": int(bool(s.get("supported")))})
    again = rng.sample(manual, n_relabel)
    rng.shuffle(again)
    relabel = []
    by_item = {k["item_id"]: k for k in key}
    for i, m in enumerate(again, 1):
        item = f"r{i:02d}"
        relabel.append({**m, "item_id": item, "label": "", "notes": ""})
        key.append({**by_item[m["item_id"]], "item_id": item, "sheet": RELABEL_SHEET, "original_item": m["item_id"]})
    cell_counts = {f"{c}|{sec}": k for (c, sec), k in sorted(used.items())}
    return {"manual": manual, "relabel": relabel, "key": key, "cells": cell_counts}


def write_sheets(out_dir: Path, built: dict[str, Any]) -> dict[str, Path]:
    paths = {"manual": out_dir / MANUAL_SHEET, "relabel": out_dir / RELABEL_SHEET, "key": out_dir / KEY_FILE}
    write_csv_atomic(paths["manual"], SHEET_COLUMNS, built["manual"])
    write_csv_atomic(paths["relabel"], SHEET_COLUMNS, built["relabel"])
    write_csv_atomic(paths["key"], KEY_COLUMNS, built["key"])
    return paths


# --------------------------------------------------------------------------- sheet kinds and status


def sheet_kind(columns: list[str]) -> str:
    """``manual`` (label column, 1-4) or ``relevance`` (relevant column, y/n)."""
    if "relevant" in columns and "unit_text" in columns:
        return "relevance"
    if "label" in columns and "sentence" in columns:
        return "manual"
    raise LabelingError(f"not a labeling sheet (columns {columns})")


def _field(kind: str) -> str:
    return "relevant" if kind == "relevance" else "label"


def _valid(kind: str) -> tuple[str, ...]:
    return ("0", "1") if kind == "relevance" else LABELS


def is_labeled(kind: str, row: dict[str, str]) -> bool:
    return (row.get(_field(kind)) or "").strip() in _valid(kind)


def sheet_status(path: Path) -> tuple[int, int]:
    """(labeled items, items); (0, 0) if the file does not exist."""
    if not path.exists():
        return 0, 0
    cols, rows = read_csv(path)
    kind = sheet_kind(cols)
    return sum(1 for r in rows if is_labeled(kind, r)), len(rows)


def sheet_complete(path: Path) -> bool:
    done, total = sheet_status(path)
    return total > 0 and done == total


# --------------------------------------------------------------------------- the terminal labeler


def read_key() -> str:
    """One key press without Enter on a terminal (POSIX termios, Windows msvcrt); else one line's first
    character. '' at end of input."""
    if sys.stdin.isatty():
        try:
            import termios
            import tty
        except ImportError:                         # Windows
            try:
                import msvcrt

                return msvcrt.getwch()              # type: ignore[attr-defined]
            except ImportError:
                pass
        else:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                return sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    line = sys.stdin.readline()
    if not line:
        return ""
    return line.strip()[:1] or "\n"


def progress_path(sheet: Path) -> Path:
    return sheet.with_name(sheet.name + ".progress.json")


def _show(kind: str, row: dict[str, str], i: int, n: int, n_done: int, out: Callable[[str], None]) -> None:
    out("\n" + "=" * 78)
    out(f"item {i + 1} of {n}   ({n_done} labeled)")
    if kind == "manual":
        out(f"id {row.get('item_id', '')}   admission {row.get('admission', '')}   section {row.get('section', '')}")
        out(f"\nSENTENCE\n  {row.get('sentence', '')}")
        out("\nCITED UNITS")
        for line in (row.get("cited_units") or "<none>").splitlines():
            out(f"  {line}")
        cur = (row.get("label") or "").strip()
        out(f"\ncurrent label: {cur or '-'}")
        out("  1 supported   2 partially supported   3 unsupported   4 contradicted   s skip   b back   q quit")
    else:
        out(f"\nQUERY\n  {row.get('query', '')}")
        out(f"\nUNIT {row.get('unit_id', '')}\n  {row.get('unit_text', '')}")
        cur = (row.get("relevant") or "").strip()
        out(f"\ncurrent: {'relevant' if cur == '1' else 'not relevant' if cur == '0' else '-'}")
        out("  y relevant   n not relevant   s skip   b back   q quit")


def label_sheet(sheet: Path, key: Callable[[], str] = read_key, out: Callable[[str], None] = print) -> dict[str, int]:
    """Label ``sheet`` interactively. Returns {labeled, items, position}."""
    if not sheet.exists():
        raise LabelingError(f"sheet not found: {sheet}")
    cols, rows = read_csv(sheet)
    kind = sheet_kind(cols)
    if not rows:
        raise LabelingError(f"{sheet} has no items")
    field = _field(kind)
    keymap = ({"y": "1", "n": "0"} if kind == "relevance" else {str(i + 1): lab for i, lab in enumerate(LABELS)})
    prog = progress_path(sheet)
    pos = None
    if prog.exists():
        try:
            pos = int(json.loads(prog.read_text(encoding="utf-8"))["position"])
        except (ValueError, KeyError, TypeError):
            pos = None
    if pos is None or not 0 <= pos < len(rows):
        pos = next((i for i, r in enumerate(rows) if not is_labeled(kind, r)), 0)

    def save_progress() -> None:
        tmp = prog.with_name(prog.name + ".tmp")
        tmp.write_text(json.dumps({"position": pos}), encoding="utf-8")
        os.replace(tmp, prog)

    def n_done() -> int:
        return sum(1 for r in rows if is_labeled(kind, r))

    out(f"labeling {sheet} ({kind}): {n_done()} of {len(rows)} labeled; starting at item {pos + 1}")
    revisit = False                       # after the end of the sheet: only the skipped (unlabeled) items
    while True:
        if pos >= len(rows) or (revisit and is_labeled(kind, rows[pos])):
            left = [i for i, r in enumerate(rows) if not is_labeled(kind, r)]
            if not left:
                pos = min(pos, len(rows) - 1)
                save_progress()
                out(f"\nall {len(rows)} items labeled. Saved {sheet}")
                break
            nxt = next((i for i in left if i >= pos), left[0])
            if not revisit:
                out(f"\nend of the sheet; {len(left)} skipped item(s) still unlabeled, going back to item {nxt + 1}")
            revisit, pos = True, nxt
        _show(kind, rows[pos], pos, len(rows), n_done(), out)
        k = key()
        if k == "":
            out("\nend of input; progress saved")
            save_progress()
            break
        k = k.lower()
        if k in keymap:
            rows[pos][field] = keymap[k]
            write_csv_atomic(sheet, cols, rows)
            if n_done() == len(rows):
                save_progress()
                out(f"\nall {len(rows)} items labeled. Saved {sheet}")
                break
            pos += 1
        elif k == "s":
            pos += 1
        elif k == "b":
            revisit, pos = False, max(0, pos - 1)
        elif k == "q":
            save_progress()
            out(f"\nquit; {n_done()} of {len(rows)} labeled. Saved {sheet}; run the same command to continue")
            break
        else:
            out(f"  key {k!r} not used")
            continue
        save_progress()
    return {"labeled": n_done(), "items": len(rows), "position": pos}


# --------------------------------------------------------------------------- analysis


def joined_labels(sheet: Path, key: Path) -> list[dict[str, Any]]:
    """Labeled sheet rows with their key (condition, NLI); unlabeled rows are left out."""
    cols, rows = read_csv(sheet)
    if sheet_kind(cols) != "manual":
        raise LabelingError(f"{sheet} is not a manual-review sheet")
    _, keys = read_csv(key)
    by_item = {k["item_id"]: k for k in keys}
    out = []
    for r in rows:
        lab = (r.get("label") or "").strip()
        if not lab:
            continue
        if lab not in LABELS:
            raise LabelingError(f"{sheet}: item {r['item_id']} has label {lab!r}; allowed {list(LABELS)}")
        k = by_item.get(r["item_id"])
        if k is None:
            raise LabelingError(f"{sheet}: item {r['item_id']} is not in the key file {key}")
        out.append({**k, "label": lab})
    return out


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    from ehr_rag.eval.agreement import cohen_kappa as k

    return k(a, b)


def _kappa_stat(pairs_key: Callable[[Any], tuple[str, str]]) -> Callable[[list[Any]], float | None]:
    def stat(groups: list[Any]) -> float | None:
        items = [x for g in groups for x in g]
        if not items:
            return None
        a, b = zip(*(pairs_key(x) for x in items))
        return cohen_kappa(list(a), list(b))
    return stat


def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r["admission"]), []).append(r)
    return out


def _human_bin(lab: str) -> str:
    return "supported" if lab == "supported" else "not_supported"


def analyze_labels(sheet: Path, key: Path, relabel: Path | None, n: int, seed: int,
                   groups: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Unsupported rate, label distribution, NLI agreement and intra-annotator agreement with CIs."""
    from ehr_rag.stats import bootstrap, ratio_stat

    rows = joined_labels(sheet, key)
    conds = [c for c in DEFAULT_CONDITIONS if any(r["condition"] == c for r in rows)]
    conds += sorted({r["condition"] for r in rows} - set(conds))
    rate_rows = []
    for c in conds + ["ALL"]:
        for g, secs in groups.items():
            rs = [r for r in rows if (c == "ALL" or r["condition"] == c) and r["section"] in secs]
            if not rs:
                continue
            k = f"{seed}:labels:{c}:{g}"
            rate_rows.append({
                "condition": c, "sections": g, "n": len(rs),
                "counts": {lab: sum(1 for r in rs if r["label"] == lab) for lab in LABELS},
                "unsupported_rate": bootstrap(_groups(rs), ratio_stat(
                    lambda grp: sum(1 for r in grp if r["label"] in NOT_SUPPORTED), lambda grp: len(grp)), n, k),
            })
    nli_rows = []
    for c in conds + ["ALL"]:
        rs = [r for r in rows if c == "ALL" or r["condition"] == c]
        if not rs:
            continue
        hum = [_human_bin(r["label"]) for r in rs]
        nli = ["supported" if str(r["nli_supported"]) == "1" else "not_supported" for r in rs]
        nli_rows.append({
            "condition": c, "n": len(rs),
            "raw_agreement": sum(x == y for x, y in zip(hum, nli)) / len(rs),
            "kappa": bootstrap(_groups(rs), _kappa_stat(
                lambda r: (_human_bin(r["label"]), "supported" if str(r["nli_supported"]) == "1" else "not_supported")),
                n, f"{seed}:nli:{c}"),
        })
    intra = None
    if relabel is not None:
        second = joined_labels(relabel, key)
        first = {r["item_id"]: r for r in rows}
        pairs = [{"admission": r["admission"], "first": first[r["original_item"]]["label"], "second": r["label"]}
                 for r in second if r["original_item"] in first]
        if pairs:
            intra = {
                "n": len(pairs),
                "raw_agreement": sum(p["first"] == p["second"] for p in pairs) / len(pairs),
                "kappa": bootstrap(_groups(pairs), _kappa_stat(lambda p: (p["first"], p["second"])), n,
                                   f"{seed}:intra"),
                "kappa_binary": bootstrap(_groups(pairs), _kappa_stat(
                    lambda p: (_human_bin(p["first"]), _human_bin(p["second"]))), n, f"{seed}:intra2"),
            }
    return {"n_labeled": len(rows), "rates": rate_rows, "nli_agreement": nli_rows, "intra_annotator": intra}
