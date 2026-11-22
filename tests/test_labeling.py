"""Labeling sheets, the terminal labeler and the label analysis (table 7), on FAKE sentences."""

from __future__ import annotations

import csv
import io
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Any

import pytest

from ehr_rag.config import SECTIONS
from ehr_rag.labeling import (KEY_FILE, LABELS, MANUAL_SHEET, RELABEL_SHEET, LabelingError, analyze_labels,
                              build_sheets, label_sheet, progress_path, sheet_complete, sheet_status, write_sheets)
from ehr_rag.tables import GROUPS, table_manual

CONDS = ("ehr_rag", "ehr_rag_enforced")


def fake_sentences(n_adm: int = 60) -> list[dict[str, Any]]:
    """Per admission and condition: one FAKE cited claim per section, an abstention and an uncited claim."""
    out = []
    for h in range(1000, 1000 + n_adm):
        for c in CONDS:
            for i, sec in enumerate(SECTIONS):
                out.append({"condition": c, "hadm_id": h, "section": sec, "index": 0, "sentence": f"FAKE {c} {h} {sec}",
                            "cited_ids": [f"E{i + 1}"], "is_abstention": False, "supported": i % 2 == 0,
                            "nli_label": "entailment" if i % 2 == 0 else "neutral", "evaluate_run": "ev"})
                out.append({"condition": c, "hadm_id": h, "section": sec, "index": 1, "sentence": "Insufficient",
                            "cited_ids": [], "is_abstention": True, "supported": False})
                out.append({"condition": c, "hadm_id": h, "section": sec, "index": 2, "sentence": "uncited",
                            "cited_ids": [], "is_abstention": False, "supported": False})
    return out


def texts(condition: str, hadm_id: int) -> dict[str, str]:
    return {f"E{i}": f"FAKE unit {i} of {hadm_id}" for i in range(1, 5)}


def alias(h: int) -> str:
    return f"adm_{h - 999:03d}"


def _read(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture()
def sheets(tmp_path: Path) -> dict[str, Path]:
    built = build_sheets(fake_sentences(), texts, alias, n=50, n_relabel=15, seed=13)
    return write_sheets(tmp_path, built)


def test_sheet_has_one_cited_sentence_per_admission_mixed_and_shuffled(sheets: dict[str, Path]) -> None:
    rows, key = _read(sheets["manual"]), {k["item_id"]: k for k in _read(sheets["key"])}
    assert len(rows) == 50 and len({r["admission"] for r in rows}) == 50
    assert list(rows[0]) == ["item_id", "admission", "section", "sentence", "cited_units", "label", "notes"]
    assert [r["item_id"] for r in rows] == [f"m{i:02d}" for i in range(1, 51)]
    assert all(r["sentence"].startswith("FAKE") and r["cited_units"].startswith("[E") for r in rows)
    assert all("condition" not in r and r["label"] == "" for r in rows)      # condition only in the key
    cells = Counter((key[r["item_id"]]["condition"], r["section"]) for r in rows)
    assert len(cells) == 8 and max(cells.values()) - min(cells.values()) <= 1   # 50 over 8 cells: 6 or 7 each
    assert Counter(key[r["item_id"]]["condition"] for r in rows) == {"ehr_rag": 25, "ehr_rag_enforced": 25}
    order = [int(r["admission"][4:]) for r in rows]
    assert order != sorted(order)                                             # shuffled, not by admission
    for r in rows:                                                            # key agrees with the sheet
        k = key[r["item_id"]]
        assert k["admission"] == r["admission"] and k["section"] == r["section"]
        assert r["sentence"] == f"FAKE {k['condition']} {int(r['admission'][4:]) + 999} {r['section']}"


def test_relabel_sheet_repeats_items_in_a_new_order(sheets: dict[str, Path]) -> None:
    manual = {r["item_id"]: r for r in _read(sheets["manual"])}
    again = _read(sheets["relabel"])
    key = {k["item_id"]: k for k in _read(sheets["key"])}
    assert [r["item_id"] for r in again] == [f"r{i:02d}" for i in range(1, 16)]
    originals = [key[r["item_id"]]["original_item"] for r in again]
    assert len(set(originals)) == 15
    for r, o in zip(again, originals):
        assert r["sentence"] == manual[o]["sentence"] and r["cited_units"] == manual[o]["cited_units"]
        assert key[r["item_id"]]["condition"] == key[o]["condition"]
    assert originals != sorted(originals)


def test_sheets_are_deterministic_and_checked(tmp_path: Path) -> None:
    a = build_sheets(fake_sentences(), texts, alias, n=50, n_relabel=15, seed=13)
    assert a == build_sheets(fake_sentences(), texts, alias, n=50, n_relabel=15, seed=13)
    assert a != build_sheets(fake_sentences(), texts, alias, n=50, n_relabel=15, seed=14)
    with pytest.raises(LabelingError, match="only 40 admissions"):
        build_sheets(fake_sentences(40), texts, alias, n=50, n_relabel=15, seed=13)
    only_plain = [s for s in fake_sentences() if s["condition"] == "ehr_rag"]
    with pytest.raises(LabelingError, match="only 0 admissions"):                   # both conditions needed
        build_sheets(only_plain, texts, alias, n=50, n_relabel=15, seed=13)


def test_missing_cell_falls_back_to_another(tmp_path: Path) -> None:
    sents = [s for s in fake_sentences() if s["section"] != "Impression"]
    built = build_sheets(sents, texts, alias, n=50, n_relabel=5, seed=13)
    assert len(built["manual"]) == 50 and all(r["section"] != "Impression" for r in built["manual"])


# --------------------------------------------------------------------------- the labeler


def keys(seq: str) -> Any:
    it = iter(seq)
    return lambda: next(it, "")


def test_labeler_saves_every_answer_and_resumes(sheets: dict[str, Path]) -> None:
    sheet = sheets["manual"]
    out: list[str] = []
    res = label_sheet(sheet, key=keys("12x3q"), out=out.append)
    assert res == {"labeled": 3, "items": 50, "position": 3}
    assert [r["label"] for r in _read(sheet)][:4] == ["supported", "partially supported", "unsupported", ""]
    assert any("key 'x' not used" in x for x in out)
    res = label_sheet(sheet, key=keys("4"), out=lambda s: None)      # resumes at item 4; end of input saves
    assert [r["label"] for r in _read(sheet)][3] == "contradicted" and res["position"] == 4
    label_sheet(sheet, key=keys("bb1q"), out=lambda s: None)          # back twice: relabel item 3
    assert [r["label"] for r in _read(sheet)][2] == "supported"


def test_labeler_skip_wraps_back_and_finishes(sheets: dict[str, Path]) -> None:
    sheet = sheets["relabel"]
    seq = "s" + "1" * 14 + "2"                                         # skip item 1, label 2-15, then item 1
    out: list[str] = []
    res = label_sheet(sheet, key=keys(seq), out=out.append)
    assert res["labeled"] == 15 and sheet_complete(sheet)
    assert _read(sheet)[0]["label"] == "partially supported"
    assert any("1 skipped item(s) still unlabeled" in x for x in out) and "all 15 items labeled" in out[-1]
    assert progress_path(sheet).exists()
    # after the end only the skipped items come back, in order, and labeling stops when none is left
    sheet = sheets["manual"]
    out = []
    res = label_sheet(sheet, key=keys("ss" + "3" * 48 + "s12"), out=out.append)
    # item 1 skipped again, item 2 -> 1, wraps past the labeled items to item 1 -> 2
    assert res["labeled"] == 50
    assert [r["label"] for r in _read(sheet)][:3] == ["partially supported", "supported", "unsupported"]
    assert "all 50 items labeled" in out[-1]


def test_labeler_relevance_sheet_uses_y_and_n(tmp_path: Path) -> None:
    sheet = tmp_path / "relevance_to_label.csv"
    sheet.write_text("hadm_id,query,unit_id,unit_text,relevant\n1,q,E1,Na 130,\n1,q,E2,K 4,\n1,q,E3,Hb 9,\n",
                     encoding="utf-8")
    out: list[str] = []
    label_sheet(sheet, key=keys("y3n" + "y"), out=out.append)
    assert [r["relevant"] for r in _read(sheet)] == ["1", "0", "1"] and sheet_status(sheet) == (3, 3)
    assert not any("1 " in x and "hadm" in x for x in out)
    from ehr_rag.retrieval.relevance import read_labels

    assert read_labels(sheet) == {(1, "q"): {"E1": 1, "E2": 0, "E3": 1}}         # readable by retrieval-metrics


def test_label_cli_reads_keys_from_stdin(sheets: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main

    monkeypatch.setattr("sys.stdin", io.StringIO("1\n4\nq\n"))
    assert main(["label", "--sheet", str(sheets["manual"])]) == 0
    assert [r["label"] for r in _read(sheets["manual"])][:3] == ["supported", "contradicted", ""]
    assert "SENTENCE" in capsys.readouterr().out
    assert main(["label", "--sheet", str(sheets["key"])]) == 4                  # not a sheet


# --------------------------------------------------------------------------- analysis and table 7


def _label_all(path: Path, labels: list[str]) -> None:
    rows = _read(path)
    for r, lab in zip(rows, labels):
        r["label"] = lab
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_analysis_rates_and_kappas(sheets: dict[str, Path]) -> None:
    key = {k["item_id"]: k for k in _read(sheets["key"])}
    manual = _read(sheets["manual"])
    # human agrees with NLI exactly: supported where NLI supported, else unsupported
    _label_all(sheets["manual"], ["supported" if key[r["item_id"]]["nli_supported"] == "1" else "unsupported"
                                  for r in manual])
    first = {r["item_id"]: r["label"] for r in _read(sheets["manual"])}
    _label_all(sheets["relabel"], [first[key[r["item_id"]]["original_item"]] for r in _read(sheets["relabel"])])
    a = analyze_labels(sheets["manual"], sheets["key"], sheets["relabel"], 200, 13, GROUPS)
    assert a["n_labeled"] == 50
    all_row = next(r for r in a["rates"] if r["condition"] == "ALL" and r["sections"] == "ALL")
    n_unsup = sum(1 for v in first.values() if v == "unsupported")
    assert all_row["n"] == 50 and all_row["unsupported_rate"]["estimate"] == pytest.approx(n_unsup / 50)
    assert all_row["counts"]["supported"] == 50 - n_unsup
    nli_all = next(r for r in a["nli_agreement"] if r["condition"] == "ALL")
    assert nli_all["raw_agreement"] == 1.0 and nli_all["kappa"]["estimate"] == pytest.approx(1.0)
    assert a["intra_annotator"]["n"] == 15 and a["intra_annotator"]["raw_agreement"] == 1.0

    # contradicted counts as unsupported; partially supported as not supported for NLI agreement
    _label_all(sheets["manual"], ["contradicted"] * 25 + ["partially supported"] * 25)
    a = analyze_labels(sheets["manual"], sheets["key"], None, 200, 13, GROUPS)
    all_row = next(r for r in a["rates"] if r["condition"] == "ALL" and r["sections"] == "ALL")
    assert all_row["unsupported_rate"]["estimate"] == 0.5 and a["intra_annotator"] is None
    nli_all = next(r for r in a["nli_agreement"] if r["condition"] == "ALL")
    n_nli = sum(1 for k in key.values() if k["sheet"] == MANUAL_SHEET and k["nli_supported"] == "1")
    assert nli_all["raw_agreement"] == pytest.approx((50 - n_nli) / 50)


def test_partly_labeled_sheet_counts_only_labeled_rows_and_rejects_unknown_labels(sheets: dict[str, Path]) -> None:
    _label_all(sheets["manual"], ["supported", "unsupported"])
    assert sheet_status(sheets["manual"]) == (2, 50) and not sheet_complete(sheets["manual"])
    a = analyze_labels(sheets["manual"], sheets["key"], None, 50, 13, GROUPS)
    assert a["n_labeled"] == 2
    _label_all(sheets["manual"], ["maybe"])
    with pytest.raises(LabelingError, match="maybe"):
        analyze_labels(sheets["manual"], sheets["key"], None, 50, 13, GROUPS)


def test_table_7_lists_rates_and_agreement(sheets: dict[str, Path]) -> None:
    labs = cycle(LABELS)
    _label_all(sheets["manual"], [next(labs) for _ in range(50)])
    _label_all(sheets["relabel"], ["supported"] * 15)
    lines, n_rows = table_manual({"sheet": sheets["manual"], "key": sheets["key"], "relabel": sheets["relabel"]},
                                 100, 13)
    text = "\n".join(lines)
    assert "## Table 7." in text and "| ehr_rag | History+Findings |" in text and "| ALL | ALL | 50 |" in text
    assert "NLI vs human: ehr_rag_enforced" in text and "intra-annotator (4 labels) | 15 |" in text
    assert n_rows == 9 + 3 + 2                                       # 3 conditions x 3 groups; 3 NLI; 2 intra


def test_make_tables_cli_with_sheets_and_retrieval_metrics(env: dict[str, Any], tmp_path: Path) -> None:
    import json

    from ehr_rag.cli import main
    from ehr_rag.runs import start_run

    cfg = str(env["root"] / "config" / "test.yaml")
    built = build_sheets(fake_sentences(), texts, alias, n=50, n_relabel=15, seed=13)
    p = write_sheets(tmp_path, built)
    _label_all(p["manual"], ["supported", "unsupported"] * 25)
    rm = start_run(env["cfg"], "retrieval-metrics", command="retrieval-metrics")
    rm.write_json("retrieval_metrics.json", {"methods": {"bm25": {"recall@5": 0.5, "mrr": 0.25, "n_pairs": 3}},
                                             "skipped_no_relevant": [], "note": "NOTE: pooled."})
    rm.finish()
    assert main(["make-tables", "--config", cfg, "--label-sheet", str(p["manual"]), "--label-key", str(p["key"]),
                 "--retrieval-metrics-run", rm.run_id, "--run-id", "20990101-000000-make-tables"]) == 0
    d = env["root"] / "runs" / "20990101-000000-make-tables"
    assert (d / "table_7_manual.md").exists() and "| bm25 | 0.500 | 0.250 | 3 |" in (d / "table_8_retrieval.md").read_text(
        encoding="utf-8")
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["label_sheets"]["sheet"] == str(p["manual"]) and meta["retrieval_metrics_run"] == rm.run_id
    assert main(["make-tables", "--config", cfg, "--label-sheet", str(p["manual"])]) == 2      # key missing
    assert main(["make-tables", "--config", cfg, "--label-sheet", str(tmp_path / "nope.csv"),
                 "--label-key", str(p["key"])]) == 2
    assert (tmp_path / KEY_FILE).exists() and (tmp_path / RELABEL_SHEET).exists()
