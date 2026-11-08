"""Phase 5: parser, citation metrics, manual review, agreement, text metrics, evaluate (fake NLI only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ehr_rag.eval.agreement import agreement, cohen_kappa
from ehr_rag.eval.citations import aggregate, evaluate_report, metrics_table
from ehr_rag.eval.evaluate import load_generation_run, run_evaluation
from ehr_rag.eval.manual_review import COLUMNS, read_manual_review, sample_sentences
from ehr_rag.eval.parse import parse_report
from ehr_rag.eval.text_metrics import NO_REFERENCES_MESSAGE, ReferenceFileError, load_references, rouge_l
from ehr_rag.generation.generate import run_generation
from ehr_rag.runs import start_run
from fakes import MALFORMED_REPORT, FakeBackend, FakeNLI, realistic_responder

HADM = [12, 21]

# --------------------------------------------------------------------------- parser


def test_parse_single_multiple_adjacent_none() -> None:
    r = parse_report(
        "### History\nOne fake fact. [E3]\n### Findings\nTwo fake facts. [E3, E7]\n"
        "Adjacent fake facts. [E3][E7]\n### Impression\nInsufficient evidence. [NONE]\n"
        "### Recommendations\nSpaced fake. [ E4 ,E5 ]\n"
    )
    assert r.missing_sections == [] and not r.out_of_order
    h, f1, f2 = r.sections["History"][0], r.sections["Findings"][0], r.sections["Findings"][1]
    assert (h.text, h.citations) == ("One fake fact.", ["E3"])
    assert f1.citations == ["E3", "E7"] and f2.citations == ["E3", "E7"]
    assert f2.text == "Adjacent fake facts."
    imp = r.sections["Impression"][0]
    assert imp.citations == ["NONE"] and imp.is_abstention and imp.evidence_ids == []
    assert r.sections["Recommendations"][0].citations == ["E4", "E5"]


def test_parse_duplicate_ids_deduped() -> None:
    s = parse_report("### History\nFake. [E3, E3][E3]\n").sections["History"][0]
    assert s.citations == ["E3"]


def test_parse_missing_and_extra_sections_and_preamble() -> None:
    r = parse_report(MALFORMED_REPORT + "## Notes\nFake extra line. [E1]\n")
    assert r.missing_sections == ["Impression", "Recommendations"]
    assert r.preamble == ["Here is the fake report:"]
    assert r.extra_sections == ["## Notes"] and r.extra_lines == ["Fake extra line. [E1]"]
    bullet = r.sections["History"][0]
    assert "bullet" in bullet.flags and "no_trailing_citation" in bullet.flags and bullet.citations == []


def test_parse_mid_sentence_citation_and_multiple_sentences() -> None:
    r = parse_report("### Findings\nFakeium was 5.9 [E3] and FakeDrugA was given. [E4]\n"
                     "First fake sentence. Second fake sentence. [E1]\nValue 5.9 stays one sentence. [E1]\n")
    a, b, c = r.sections["Findings"]
    assert a.citations == ["E4"] and a.mid_citations == ["E3"] and "mid_sentence_citation" in a.flags
    assert a.text == "Fakeium was 5.9 and FakeDrugA was given."
    assert "multiple_sentences" in b.flags
    assert "multiple_sentences" not in c.flags


def test_parse_out_of_order_and_duplicate() -> None:
    r = parse_report("### Findings\nA. [E1]\n### History\nB. [E1]\n### Findings\nC. [E1]\n")
    assert r.out_of_order and r.duplicate_sections == ["Findings"]
    assert [s.text for s in r.sections["Findings"]] == ["A.", "C."]


# --------------------------------------------------------------------------- citation metrics

EVIDENCE = [
    {"unit_id": "E1", "unit_type": "lab", "text": "Fakeium (Blood): max 5.9."},
    {"unit_id": "E2", "unit_type": "medication", "text": "FakeDrugA 40 mg IV."},
]
REPORT = (
    "### History\n"
    "Fakeium was 5.9. [E1]\n"                                   # supported, 1/1 relevant
    "Fakeium was 5.9 again. [E1, E2]\n"                         # supported, E2 irrelevant
    "### Findings\n"
    "Fakeium was 5.9 and FakeDrugA 40 mg was given. [E1, E2]\n" # supported, both relevant
    "Invented fake value 99. [E1]\n"                            # unsupported, E1 irrelevant
    "### Impression\n"
    "Something fake. [E9]\n"                                    # invalid id only
    "No citation fake here.\n"                                  # uncited
    "### Recommendations\n"
    "Insufficient evidence. [NONE]\n"                           # abstention
)


def test_citation_metrics_hand_computed() -> None:
    ev = evaluate_report(parse_report(REPORT), EVIDENCE, "ehr_rag", 12, FakeNLI())
    assert [e.supported for e in ev] == [True, True, True, False, False, False, False]
    assert ev[1].irrelevant_ids == ["E2"] and ev[2].irrelevant_ids == [] and ev[3].irrelevant_ids == ["E1"]
    assert ev[4].invalid_ids == ["E9"] and ev[4].nli_label == "no_valid_citation"
    agg = aggregate(ev)
    assert agg["n_sentences"] == 6 and agg["n_abstentions"] == 1
    assert agg["cited_rate"] == pytest.approx(4 / 6)
    assert agg["n_citations"] == 7 and agg["invalid_citation_rate"] == pytest.approx(1 / 7)
    assert agg["citation_recall"] == pytest.approx(3 / 6)
    assert agg["citation_precision"] == pytest.approx(4 / 7)


def test_metrics_table_groups_and_empty_denominators() -> None:
    ev = evaluate_report(parse_report(REPORT), EVIDENCE, "ehr_rag", 12, FakeNLI())
    rows = {r["scope"]: r for r in metrics_table(ev, ["ehr_rag"])}
    assert rows["History+Findings"]["n_sentences"] == 4
    assert rows["History+Findings"]["citation_recall"] == pytest.approx(3 / 4)
    assert rows["Impression+Recommendations"]["citation_recall"] == 0.0
    assert rows["Recommendations"]["n_sentences"] == 0 and rows["Recommendations"]["cited_rate"] is None


def test_nli_not_called_for_uncited_or_abstentions() -> None:
    nli = FakeNLI()
    evaluate_report(parse_report("### History\nInsufficient evidence. [NONE]\nFake. [E9]\nFake two.\n"),
                    EVIDENCE, "ehr_rag", 12, nli)
    assert nli.calls == []


# --------------------------------------------------------------------------- manual review / agreement


def _many_sentences() -> list:
    rep = "".join(f"### {s}\n" + "".join(f"Fakeium {s} {i} 5.9. [E1]\n" for i in range(5))
                  for s in ("History", "Findings", "Impression", "Recommendations"))
    return [e for c in ("ehr_rag", "full_context")
            for e in evaluate_report(parse_report(rep), EVIDENCE, c, 12, FakeNLI())]


def test_manual_review_sampling_is_stratified_and_deterministic() -> None:
    sents = _many_sentences()
    a = sample_sentences(sents, 16, seed=13)
    b = sample_sentences(sents, 16, seed=13)
    assert [(s.condition, s.section, s.index) for s in a] == [(s.condition, s.section, s.index) for s in b]
    counts: dict = {}
    for s in a:
        counts[(s.condition, s.section)] = counts.get((s.condition, s.section), 0) + 1
    assert len(counts) == 8 and set(counts.values()) == {2}
    assert len(sample_sentences(sents, 1000, seed=13)) == 40  # capped by availability


def test_agreement_kappa_and_exclusions() -> None:
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "a", "b", "b"]) == pytest.approx(0.0)
    rows = [
        {"nli_label": "entailment", "human_support": "supported"},
        {"nli_label": "entailment", "human_support": "partial"},
        {"nli_label": "neutral", "human_support": "unsupported"},
        {"nli_label": "neutral", "human_support": "not_checkable"},
        {"nli_label": "neutral", "human_support": ""},
    ]
    res = agreement(rows)
    assert res["n_labeled"] == 4 and res["n_not_checkable"] == 1 and res["n_used"] == 3
    assert res["raw_agreement"] == pytest.approx(2 / 3)
    assert res["confusion_nli_x_human"]["entailment"]["partial"] == 1
    with pytest.raises(ValueError, match="unknown human_support"):
        agreement([{"nli_label": "neutral", "human_support": "maybe"}])


# --------------------------------------------------------------------------- text metrics


def test_text_metrics_require_references(tmp_path: Path) -> None:
    with pytest.raises(ReferenceFileError) as e:
        load_references(None)
    assert str(e.value) == NO_REFERENCES_MESSAGE and "never generates" in str(e.value)
    with pytest.raises(ReferenceFileError, match="not found"):
        load_references(tmp_path / "missing.csv")


def test_rouge_l() -> None:
    assert rouge_l("fake a b c", "fake a b c")["f1"] == pytest.approx(1.0)
    r = rouge_l("a b c d", "a c")
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(1.0)
    assert rouge_l("", "a")["f1"] == 0.0


def test_text_metrics_cli_exits_without_references(env: dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main

    cfg_path = Path(env["cfg"]["_project_root"]) / "config" / "test.yaml"
    assert main(["text-metrics", "--config", str(cfg_path), "--generation-run", "x"]) == 2
    assert "No reference reports provided" in capsys.readouterr().err


# --------------------------------------------------------------------------- evaluate end to end


def _generate(env: dict[str, Any], responder: Any) -> Path:
    run = start_run(env["cfg"], "generate", command="generate")
    run_generation(env["cfg"], run, FakeBackend(responder), hadm_ids=HADM,
                   conditions=["no_retrieval", "ehr_rag", "full_context"],
                   retrieval_run=env["retrieval_run"], log=lambda s: None)
    return run.path


def test_evaluate_end_to_end(env: dict[str, Any]) -> None:
    gen = _generate(env, realistic_responder)
    reports = load_generation_run(gen)
    assert len(reports) == 6
    out = start_run(env["cfg"], "evaluate", command="evaluate")
    res = run_evaluation(gen, out.path, out.run_id, FakeNLI(), manual_review_n=10, seed=13)
    for f in ("metrics.json", "metrics_by_section.csv", "metrics_summary.md", "sentences.jsonl",
              "manual_review.csv"):
        assert (out.path / f).exists(), f
    m = json.loads((out.path / "metrics.json").read_text(encoding="utf-8"))
    er = m["conditions"]["ehr_rag"]["ALL"]
    assert er["invalid_citation_rate"] == 0.0 and er["cited_rate"] == 1.0
    assert er["citation_recall"] > 0
    nr = m["conditions"]["no_retrieval"]["ALL"]
    assert nr["n_citations"] == 0 and nr["cited_rate"] in (None, 0.0)
    rows = read_manual_review(out.path / "manual_review.csv")
    assert list(rows[0]) == list(COLUMNS) and 0 < len(rows) <= 10
    assert all(r["human_support"] == "" for r in rows)
    assert "Abstention lines" in (out.path / "metrics_summary.md").read_text(encoding="utf-8")
    assert res["format_counts"]["ehr_rag"] == {"ok": 2}


def test_evaluate_includes_format_errors(env: dict[str, Any]) -> None:
    gen = _generate(env, lambda p, n: MALFORMED_REPORT)
    out = start_run(env["cfg"], "evaluate", command="evaluate")
    res = run_evaluation(gen, out.path, out.run_id, FakeNLI(), manual_review_n=5, seed=13)
    assert res["format_counts"]["ehr_rag"] == {"format_error": 2}
    m = json.loads((out.path / "metrics.json").read_text(encoding="utf-8"))
    rep = [r for r in m["per_report"] if r["condition"] == "ehr_rag"][0]
    assert rep["raw_file"] == "raw_retry.txt"
    assert rep["parse_flags"]["missing_sections"] == ["Impression", "Recommendations"]
    with (out.path / "metrics_by_section.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 3 * 7


# --------------------------------------------------------------------------- scorer fixes (claims, premise, too_long, precision)


def test_parse_trailing_period_after_citations() -> None:
    r = parse_report("### History\nFake fact [E3].\nFake fact two. [E3, E4].\nInsufficient evidence [NONE].\n"
                     "Fake [E3] in the middle.\nFake double [E3]..\n")
    a, b, c, d, e = r.sections["History"]
    assert (a.text, a.citations) == ("Fake fact.", ["E3"]) and "period_after_citation" in a.flags
    assert (b.text, b.citations) == ("Fake fact two.", ["E3", "E4"])
    assert c.is_abstention and c.citations == ["NONE"]
    assert d.citations == [] and d.mid_citations == ["E3"]           # not trailing: still uncited
    assert e.citations == [] and "no_trailing_citation" in e.flags   # only ONE period may follow


def test_split_claims_list_sentence() -> None:
    from ehr_rag.eval.parse import split_claims

    s = parse_report("### Findings\nFake labs: Na 130 [E1], K 5.9 [E2], and Cr 2.1 [E3][E4].\n").sections["Findings"][0]
    assert split_claims(s) == [("Fake labs: Na 130", ["E1"]), ("K 5.9", ["E2"]), ("Cr 2.1", ["E3", "E4"])]
    s = parse_report("### Findings\nFakeium was 5.9 [E3] and FakeDrugA was given. [E4]\n").sections["Findings"][0]
    assert split_claims(s) == [("Fakeium was 5.9", ["E3"]), ("FakeDrugA was given.", ["E4"])]
    s = parse_report("### Findings\nFakeium 5.9 [E3] then something uncited.\n").sections["Findings"][0]
    assert split_claims(s) == [("Fakeium 5.9", ["E3"]), ("then something uncited.", [])]
    s = parse_report("### Findings\nPlain fake sentence. [E1, E2]\n").sections["Findings"][0]
    assert split_claims(s) == [("Plain fake sentence.", ["E1", "E2"])]


LIST_EVIDENCE = [
    {"unit_id": "E1", "unit_type": "lab", "text": "Fakeium (Blood): max 5.9."},
    {"unit_id": "E2", "unit_type": "lab", "text": "Fakeone (Blood): last 4.1."},
    {"unit_id": "E3", "unit_type": "medication", "text": "FakeDrugA 40 mg IV."},
]


def test_list_sentence_scored_per_claim_with_own_citations() -> None:
    from ehr_rag.eval.manual_review import sentence_key

    nli = FakeNLI()
    rep = "### Findings\nFake labs were 5.9 [E1], 4.1 [E2], and 99 [E3].\n"
    ev = evaluate_report(parse_report(rep), LIST_EVIDENCE, "ehr_rag", 12, nli)
    assert [(e.sentence, e.cited_ids, e.supported) for e in ev] == [
        ("Fake labs were 5.9", ["E1"], True), ("4.1", ["E2"], True), ("99", ["E3"], False)]
    assert all("split_claim" in e.flags and e.n_claims == 3 for e in ev)
    assert ev[0].line == "Fake labs were 5.9, 4.1, and 99."
    assert [sentence_key(e) for e in ev] == [f"ehr_rag|12|Findings|0.{k}" for k in range(3)]
    agg = aggregate(ev)
    assert agg["n_sentences"] == 1 and agg["n_claims"] == 3 and agg["n_split_sentences"] == 1
    assert agg["citation_recall"] == pytest.approx(2 / 3) and agg["n_mid_sentence_citation"] == 1
    # each claim's NLI premise is only its own cited unit
    assert {p for p, _ in nli.calls} == {u["text"] for u in LIST_EVIDENCE}
    assert ("Fakeone (Blood): last 4.1.", "4.1") in nli.calls
    assert not any(p == "Fakeium (Blood): max 5.9." and h == "4.1" for p, h in nli.calls)


def test_premise_is_only_the_cited_units() -> None:
    nli = FakeNLI()
    ev = evaluate_report(parse_report("### Findings\nFake 5.9 and 40. [E1, E3]\n"), LIST_EVIDENCE,
                         "ehr_rag", 12, nli)
    t = {u["unit_id"]: u["text"] for u in LIST_EVIDENCE}
    assert {p for p, _ in nli.calls} == {t["E1"] + " " + t["E3"], t["E1"], t["E3"]}   # never E2 or all evidence
    assert ev[0].supported


def test_precision_alce_vs_plain() -> None:
    rep = ("### Findings\n"
           "Fake values 5.9 and 4.1. [E1, E2]\n"     # only jointly entailed: ALCE 2/2 relevant, plain 0/2
           "Fake value 5.9. [E1, E3]\n"              # E1 alone entails, E3 irrelevant: ALCE 1/2, plain 1/2
           "Fake made-up 77. [E9]\n")                # invalid id: irrelevant and not supporting
    a, b, c = evaluate_report(parse_report(rep), LIST_EVIDENCE, "ehr_rag", 12, FakeNLI())
    assert a.supported and a.irrelevant_ids == [] and a.supporting_ids == []
    assert b.supported and b.irrelevant_ids == ["E3"] and b.supporting_ids == ["E1"]
    assert c.invalid_ids == ["E9"] and c.supporting_ids == []
    agg = aggregate([a, b, c])
    assert agg["n_citations"] == 5
    assert agg["citation_precision"] == pytest.approx(3 / 5)
    assert agg["citation_precision_plain"] == pytest.approx(1 / 5)


def test_too_long_counts_as_not_entailed() -> None:
    from ehr_rag.eval.nli import TOO_LONG

    long_unit = {"unit_id": "E4", "unit_type": "lab", "text": "Fakeium 5.9 " + "padding " * 40}
    rep = "### Findings\nFakeium 5.9. [E4]\nFakeium 5.9 again. [E1, E4]\nFakeium 5.9 short. [E1]\n"
    a, b, c = evaluate_report(parse_report(rep), LIST_EVIDENCE + [long_unit], "ehr_rag", 12,
                              FakeNLI(max_length=20))
    assert a.nli_label == TOO_LONG and not a.supported and a.supporting_ids == [] and a.too_long == [["E4"]]
    assert a.irrelevant_ids == ["E4"]                       # too_long never entails, so never relevant alone
    assert b.nli_label == TOO_LONG and not b.supported and b.supporting_ids == ["E1"]
    assert sorted(b.too_long) == [["E1", "E4"], ["E4"]]
    assert c.supported and c.too_long == []
    agg = aggregate([a, b, c])
    assert agg["citation_recall"] == pytest.approx(1 / 3) and agg["n_too_long_claims"] == 2


def test_evaluate_writes_too_long_log(env: dict[str, Any]) -> None:
    gen = _generate(env, realistic_responder)
    out = start_run(env["cfg"], "evaluate", command="evaluate")
    res = run_evaluation(gen, out.path, out.run_id, FakeNLI(max_length=8), manual_review_n=5, seed=13)
    with (out.path / "nli_too_long.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and res["n_too_long_rows"] == len(rows)
    assert all(int(r["n_tokens"]) > int(r["max_length"]) == 8 for r in rows)
    assert {r["kind"] for r in rows} <= {"all", "single", "without_one"}
    m = json.loads((out.path / "metrics.json").read_text(encoding="utf-8"))
    assert m["nli"]["n_too_long_pairs"] > 0 and m["nli"]["n_truncated_pairs"] == 0
    assert "citation_precision_plain" in m["conditions"]["ehr_rag"]["ALL"]
    summary = (out.path / "metrics_summary.md").read_text(encoding="utf-8")
    assert "too_long" in summary and "citation_precision (ALCE)" in summary


def test_hfnli_too_long_pairs_never_reach_the_model() -> None:
    from ehr_rag.eval.nli import HFNLI, TOO_LONG

    nli = HFNLI.__new__(HFNLI)                     # no model download: replace the two model-facing steps
    nli.id2label, nli.entailment_label = {0: "entailment", 1: "neutral"}, "entailment"
    nli.batch_size, nli.max_length, nli._memo, nli.pair_tokens = 2, 10, {}, {}
    nli.model_name, nli.revision, nli.device = "fake-hf", "fake", "cpu"
    seen: list[tuple[str, str]] = []
    nli._n_tokens = lambda batch: [len(p.split()) + len(h.split()) for p, h in batch]  # type: ignore[method-assign]
    nli._predict = lambda batch: (seen.extend(batch), ["entailment"] * len(batch))[1]  # type: ignore[method-assign]
    short, long_ = ("fake premise", "fake claim"), ("fake " * 20, "fake claim")
    assert nli.labels([short, long_, short]) == ["entailment", TOO_LONG, "entailment"]
    assert seen == [short] and nli.pair_tokens[long_] == 22
    assert nli.info()["n_too_long_pairs"] == 1 and nli.info()["n_truncated_pairs"] == 0
