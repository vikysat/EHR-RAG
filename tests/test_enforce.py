"""Enforcement (rewrite / filter) with mock rewriters and the FAKE backend; no model is loaded."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from checker_fakes import REPO, checker, fake_units
from ehr_rag.checker.enforce import ABSTENTION_LINE, enforce_report, load_rewrite_template, parse_rewrite
from fakes import FakeBackend, FakeNLI, realistic_report

TEMPLATE = load_rewrite_template(REPO / "prompts", "rewrite_v1")
REPORT = (
    "### History\nThe patient had acute kidney failure. [E8]\n"
    "### Findings\n"
    "Potassium peaked at 5.9. [E1]\n"                 # passes
    "Potassium peaked at 6.9. [E1]\n"                 # fails: repaired by the mock
    "Sodium was 128 and creatinine 2.4. [E2]\n"       # fails (creatinine not cited): mock retreats
    "Furosemide 80 mg was given. [E5]\n"              # fails twice: dropped
    "The patient was monitored. [E1]\n"               # unchecked: kept
    "### Impression\nHyponatremia was present. [E1]\n"  # fails, mock answers nothing usable: dropped
    "### Recommendations\nInsufficient evidence. [NONE]\n"
)
FIXES = {
    "Potassium peaked at 6.9.": "Potassium peaked at 5.9. [E1]",
    "Sodium was 128 and creatinine 2.4.": "Sodium was 128. [E2]",
    "Furosemide 80 mg was given.": "Furosemide 120 mg was given. [E5]",
    "Hyponatremia was present.": "Insufficient evidence. [NONE]",
}


def mock_rewriter(calls: list[str]) -> Any:
    def rewrite(prompt: str) -> tuple[str, dict[str, Any]]:
        calls.append(prompt)
        line = re.search(r"Sentence:\n(.*)\n", prompt).group(1)   # type: ignore[union-attr]
        key = re.sub(r"\s*\[[^\]]*\]\s*$", "", line)
        return FIXES[key], {"mock": True}
    return rewrite


def test_rewrite_outcomes_repaired_retreated_dropped() -> None:
    from ehr_rag.eval.parse import parse_report

    units = fake_units()
    ck = checker()
    calls: list[str] = []
    text, log = enforce_report(REPORT, ck.admission(units, units), ck, "rewrite", mock_rewriter(calls), TEMPLATE)
    out = {(r["section"], r["line"]): r["outcome"] for r in log["lines"]}
    assert out[("Findings", "Potassium peaked at 5.9. [E1]")] == "kept"
    assert out[("Findings", "Potassium peaked at 6.9. [E1]")] == "repaired"
    assert out[("Findings", "Sodium was 128 and creatinine 2.4. [E2]")] == "retreated"
    assert out[("Findings", "Furosemide 80 mg was given. [E5]")] == "dropped"
    assert out[("Findings", "The patient was monitored. [E1]")] == "kept"
    assert out[("Impression", "Hyponatremia was present. [E1]")] == "dropped"
    assert len(calls) == 4                                   # one rewrite per failing line, never more
    s = log["summary"]
    assert (s["n_kept"], s["n_repaired"], s["n_retreated"], s["n_dropped"]) == (3, 1, 1, 2)
    assert s["n_rewrite_calls"] == 4 and s["n_lines"] == 7 and s["n_lines_after"] == 5
    assert log["sections_emptied"] == ["Impression"]
    parsed = parse_report(text)
    assert [x.raw for x in parsed.sections["Findings"]] == [
        "Potassium peaked at 5.9. [E1]", "Potassium peaked at 5.9. [E1]", "Sodium was 128. [E2]",
        "The patient was monitored. [E1]"]
    assert [x.raw for x in parsed.sections["Impression"]] == [ABSTENTION_LINE]
    assert [x.raw for x in parsed.sections["Recommendations"]] == [ABSTENTION_LINE]
    assert parsed.missing_sections == []


def test_rewrite_prompt_has_only_cited_units_and_messages() -> None:
    units = fake_units()
    ck = checker()
    calls: list[str] = []
    enforce_report("### Findings\nPotassium peaked at 6.9. [E1]\n", ck.admission(units, units), ck, "rewrite",
                   mock_rewriter(calls), TEMPLATE)
    (prompt,) = calls
    assert "Potassium peaked at 6.9. [E1]" in prompt and "number_mismatch: 6.9 is not a value in [E1]" in prompt
    assert units[0].text in prompt                                   # the cited unit
    assert all(u.text not in prompt for u in units[1:])              # and no other unit


def test_filter_mode_drops_without_model_calls() -> None:
    units = fake_units()
    ck = checker()
    text, log = enforce_report(REPORT, ck.admission(units, units), ck, "filter")
    s = log["summary"]
    assert (s["n_kept"], s["n_dropped"], s["n_rewrite_calls"]) == (3, 4, 0)
    assert "6.9" not in text and "Potassium peaked at 5.9. [E1]" in text
    assert s["n_numbers_before"] == 5 and s["n_numbers_after"] == 1   # 5.9 6.9 128 2.4 80 -> 5.9


@pytest.mark.parametrize("answer,ok", [
    ("Potassium peaked at 5.9. [E1]", True),
    ("- Potassium peaked at 5.9. [E1]\nextra chatter", True),        # bullet stripped, first line only
    ('"Potassium peaked at 5.9. [E1]"', True),
    ("Potassium peaked at 5.9.", False),                             # no citation
    ("Insufficient evidence. [NONE]", False),
    ("", False),
])
def test_parse_rewrite(answer: str, ok: bool) -> None:
    s, reason = parse_rewrite(answer, "Findings", 0)
    assert (s is not None) is ok and (reason == "") is ok


# --------------------------------------------------------------------------- through `generate` with the fake backend


def corrupting_responder(prompt: str, n: int) -> str:
    """FAKE model: reports with every lab value off by +4; rewrites copy the cited value back."""
    if prompt.startswith("One sentence of a clinical report failed"):
        m = re.search(r"^\[(E\d+)\] \(lab\) (.*?)(?:,|\s\()", prompt, re.MULTILINE)
        value = re.search(r"max (-?\d+(?:\.\d+)?)", prompt)
        if m and value:
            return f"{m.group(2)} reached {value.group(1)}. [{m.group(1)}]"
        return "Insufficient evidence. [NONE]"
    rep = realistic_report(prompt)
    return re.sub(r"reached (\d+(?:\.\d+)?)", lambda x: f"reached {float(x.group(1)) + 4:g}", rep)


def _gen(env: dict[str, Any], conditions: list[str], backend: FakeBackend) -> Path:
    from ehr_rag.generation.generate import run_generation
    from ehr_rag.runs import start_run

    run = start_run(env["cfg"], "generate", command="generate")
    run_generation(env["cfg"], run, backend, hadm_ids=[12, 21], conditions=conditions,
                   retrieval_run=env["retrieval_run"], log=lambda s: None)
    return run.path


def test_generate_enforced_rewrites_and_reuses_the_base_report(env: dict[str, Any]) -> None:
    from ehr_rag.eval.evaluate import run_evaluation
    from ehr_rag.runs import start_run

    backend = FakeBackend(corrupting_responder)
    gen = _gen(env, ["ehr_rag", "ehr_rag_enforced"], backend)
    base_calls = [p for p in backend.prompts if not p.startswith("One sentence")]
    assert len(base_calls) == 2                        # the enforced condition reused the cached base report
    rewrites = [p for p in backend.prompts if p.startswith("One sentence")]
    assert rewrites, "corrupted lab sentences must be rewritten"
    d = gen / "ehr_rag_enforced" / "12"
    parsed = json.loads((d / "parsed.json").read_text(encoding="utf-8"))
    assert parsed["raw_file"] == "enforced.txt" and parsed["pre_enforcement_raw_file"] == "raw.txt"
    assert parsed["enforced_format_problems"] == []
    log = json.loads((d / "enforcement.json").read_text(encoding="utf-8"))
    assert log["mode"] == "rewrite" and log["checker"]["combined_sha256"]
    assert {r["outcome"] for r in log["lines"]} <= {"kept", "repaired", "retreated", "dropped"}
    assert any(r["outcome"] == "repaired" for r in log["lines"])
    assert list((d / "rewrites").glob("*_prompt.txt"))
    assert (gen / "ehr_rag" / "12" / "raw.txt").read_text(encoding="utf-8") == (d / "raw.txt").read_text(encoding="utf-8")
    meta = json.loads((gen / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["checker"]["rules_sha256"] and meta["checker"]["lexicon_sha256"]
    # a second run: every call (base and rewrites) comes from the cache
    backend2 = FakeBackend(corrupting_responder)
    _gen(env, ["ehr_rag_enforced"], backend2)
    assert backend2.prompts == []
    # evaluation grades the enforced text with NLI only
    out = start_run(env["cfg"], "evaluate", command="evaluate")
    res = run_evaluation(gen, out.path, out.run_id, FakeNLI(), manual_review_n=5, seed=13)
    assert {r["condition"] for r in res["rows"]} == {"ehr_rag", "ehr_rag_enforced"}


def test_enforce_filter_cli(env: dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main

    gen = _gen(env, ["no_retrieval", "ehr_rag", "full_context"], FakeBackend(corrupting_responder))
    cfg = str(env["root"] / "config" / "test.yaml")
    assert main(["enforce-filter", "--config", cfg, "--generation-run", gen.name,
                 "--run-id", "20990101-000000-enforce-filter"]) == 0
    out = capsys.readouterr().out
    assert "ehr_rag_filtered" in out and "full_context_filtered" in out and "dropped" in out
    run = env["root"] / "runs" / "20990101-000000-enforce-filter"
    assert sorted(p.name for p in run.iterdir() if p.is_dir()) == ["ehr_rag_filtered", "full_context_filtered"]
    log = json.loads((run / "ehr_rag_filtered" / "12" / "enforcement.json").read_text(encoding="utf-8"))
    assert log["mode"] == "filter" and log["summary"]["n_rewrite_calls"] == 0
    assert log["summary"]["n_dropped"] > 0
    shown = json.loads((run / "ehr_rag_filtered" / "12" / "evidence_shown.json").read_text(encoding="utf-8"))
    assert shown["condition"] == "ehr_rag_filtered"
    assert main(["evaluate", "--config", cfg, "--generation-run", gen.name, run.name]) == 0
