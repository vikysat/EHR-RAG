"""Phase 4: prompts, format check, backends' shared logic and the generation run (fake backend only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ehr_rag.config import SECTIONS
from ehr_rag.evidence.build_units import read_header
from ehr_rag.generation.backends import BackendError, greedy_generation_config, with_retries
from ehr_rag.generation.generate import (
    ConfirmationRequired, GenerationError, check_format, run_generation,
)
from ehr_rag.generation.prompt import build_prompt, evidence_block, load_templates
from ehr_rag.runs import start_run
from fakes import MALFORMED_REPORT, FakeBackend, good_report

REPO = Path(__file__).resolve().parents[1]
HADM = [12, 21]


# --------------------------------------------------------------------------- fixtures


def _run(env: dict[str, Any], backend: FakeBackend, conditions: list[str], **kw: Any) -> tuple[Any, dict]:
    run = start_run(env["cfg"], "generate", command="generate")
    summary = run_generation(env["cfg"], run, backend, hadm_ids=HADM, conditions=conditions,
                             retrieval_run=env["retrieval_run"], log=lambda s: None, **kw)
    return run, summary


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- format check


def test_check_format_accepts_good_report() -> None:
    sections, problems = check_format(good_report(["E1", "E2"]))
    assert problems == []
    assert list(sections) == list(SECTIONS)
    assert sections["Findings"][0]["citations"] == ["E1", "E2"]
    assert sections["Findings"][1]["citations"] == ["E1", "E2"]  # adjacent brackets
    assert sections["Findings"][1]["sentence"] == "Fake finding with adjacent brackets."
    assert sections["Recommendations"][0]["citations"] == ["NONE"]


@pytest.mark.parametrize("text,none_only,expected", [
    (MALFORMED_REPORT, False, "text before the first section header"),
    (MALFORMED_REPORT, False, "bullet"),
    (MALFORMED_REPORT, False, "does not end with a citation"),
    (MALFORMED_REPORT, False, "missing section header(s): ### Impression, ### Recommendations"),
    ("### History\nA. [E1]\n\nB. [E2]\n### Findings\nC. [E1]\n### Impression\nD. [E1]\n"
     "### Recommendations\nE. [E1]\n", False, "blank line inside section History"),
    ("### Findings\nA. [E1]\n### History\nB. [E1]\n### Impression\nC. [E1]\n### Recommendations\nD. [E1]\n",
     False, "out of order"),
    ("### History\n[E1]\n### Findings\nC. [E1]\n### Impression\nD. [E1]\n### Recommendations\nE. [E1]\n",
     False, "citation without a sentence"),
    (good_report(["E1"]), True, "no evidence was provided"),
    ("### History\n### Findings\nC. [E1]\n### Impression\nD. [E1]\n### Recommendations\nE. [E1]\n",
     False, "section History is empty"),
])
def test_check_format_flags_problems(text: str, none_only: bool, expected: str) -> None:
    _, problems = check_format(text, none_only)
    assert any(expected in p for p in problems), problems


def test_check_format_no_retrieval_none_only_ok() -> None:
    from fakes import NONE_REPORT

    assert check_format(NONE_REPORT, none_only=True)[1] == []


# --------------------------------------------------------------------------- prompts


def test_no_retrieval_prompt_contains_no_unit_text(env: dict[str, Any]) -> None:
    t = load_templates(env["root"] / "prompts", "report_v1")
    for h in HADM:
        prompt = build_prompt(t, read_header(env["units_dir"], h), None)
        assert "[E" not in prompt
        for u in env["units"][h]:
            assert u.text not in prompt
        assert "[NONE]" in prompt and "### Recommendations" in prompt


def test_evidence_block_format_and_order(env: dict[str, Any]) -> None:
    units = env["units"][12][::-1]
    lines = evidence_block(units).splitlines()
    assert lines == [f"[{u.unit_id}] ({u.unit_type}) {u.text}" for u in units]


def test_released_template_has_required_instructions() -> None:
    t = load_templates(REPO / "prompts", "report_v1")
    for s in SECTIONS:
        assert f"### {s}" in t.evidence and f"### {s}" in t.no_retrieval
    assert "ONLY the patient header and the evidence" in t.evidence
    assert "Insufficient evidence. [NONE]" in t.evidence
    assert "[E3] or [E3, E7]" in t.evidence


# --------------------------------------------------------------------------- hf_local generation config


def test_greedy_generation_config_preserves_eos_list() -> None:
    from transformers import GenerationConfig

    base = GenerationConfig(bos_token_id=128000, eos_token_id=[128001, 128008, 128009],
                            do_sample=True, temperature=0.6, top_p=0.9)
    g = greedy_generation_config(base, max_new_tokens=77)
    assert g.eos_token_id == [128001, 128008, 128009]
    assert g.pad_token_id == 128001
    assert g.do_sample is False and g.temperature is None and g.top_p is None and g.top_k is None
    assert g.max_new_tokens == 77
    assert base.do_sample is True and base.temperature == 0.6  # original untouched
    g.validate(strict=True)


# --------------------------------------------------------------------------- retries


class _Err(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"fake status {status}")
        self.status_code = status


def test_with_retries_recovers_from_transient() -> None:
    calls = []

    def fn() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise _Err(429)
        return "ok"

    assert with_retries(fn, sleep=lambda s: None, log=lambda s: None) == "ok" and len(calls) == 3


def test_with_retries_gives_up_and_does_not_retry_permanent() -> None:
    calls = []

    def transient() -> None:
        calls.append(1)
        raise _Err(503)

    with pytest.raises(BackendError, match="persistent"):
        with_retries(transient, max_attempts=3, sleep=lambda s: None, log=lambda s: None)
    assert len(calls) == 3

    def permanent() -> None:
        calls.append(1)
        raise _Err(400)

    calls.clear()
    with pytest.raises(_Err):
        with_retries(permanent, sleep=lambda s: None, log=lambda s: None)
    assert len(calls) == 1


# --------------------------------------------------------------------------- generation runs


def test_generation_writes_outputs_per_condition(env: dict[str, Any]) -> None:
    backend = FakeBackend()
    run, summary = _run(env, backend, ["no_retrieval", "ehr_rag", "full_context"])
    for cond in ("no_retrieval", "ehr_rag", "full_context"):
        assert summary["counts"][cond] == {"ok": 2, "ok_after_retry": 0, "format_error": 0, "skipped": 0}
        for h in HADM:
            d = run.path / cond / str(h)
            for f in ("prompt.txt", "raw.txt", "parsed.json", "evidence_shown.json", "call.json"):
                assert (d / f).exists(), (cond, h, f)
            shown = _read_json(d / "evidence_shown.json")
            ids = [u["unit_id"] for u in shown["units"]]
            if cond == "no_retrieval":
                assert ids == []
            elif cond == "ehr_rag":
                assert ids == env["evidence"][h] and shown["retrieval_run"] == env["retrieval_run"]
            else:
                assert ids == [u.unit_id for u in env["units"][h]]
            texts = {u.unit_id: u.text for u in env["units"][h]}
            assert all(u["text"] == texts[u["unit_id"]] for u in shown["units"])
            prompt = (d / "prompt.txt").read_text(encoding="utf-8")
            assert all(f"[{i}] (" in prompt for i in ids)
    assert len(backend.prompts) == 6
    assert summary["reported_models"] == ["fake-model-reported"]
    assert summary["model_revision"] == "fake-revision"


def test_format_retry_then_format_error_keeps_raw(env: dict[str, Any]) -> None:
    backend = FakeBackend(lambda p, n: MALFORMED_REPORT)
    run, summary = _run(env, backend, ["ehr_rag"])
    assert summary["counts"]["ehr_rag"]["format_error"] == 2
    d = run.path / "ehr_rag" / "12"
    assert (d / "raw.txt").read_text(encoding="utf-8") == MALFORMED_REPORT
    assert (d / "raw_retry.txt").read_text(encoding="utf-8") == MALFORMED_REPORT
    retry_prompt = (d / "prompt_retry.txt").read_text(encoding="utf-8")
    assert retry_prompt.startswith((d / "prompt.txt").read_text(encoding="utf-8").rstrip("\n"))
    assert "FORMAT REMINDER" in retry_prompt and "bullet" in retry_prompt
    parsed = _read_json(d / "parsed.json")
    assert parsed["status"] == "format_error" and parsed["attempt_used"] == 2 and parsed["problems"]
    assert len(backend.prompts) == 4  # exactly one retry per admission


def test_format_retry_can_succeed(env: dict[str, Any]) -> None:
    def responder(prompt: str, n: int) -> str:
        return FakeBackend.default_responder(prompt, n) if "FORMAT REMINDER" in prompt else MALFORMED_REPORT

    run, summary = _run(env, FakeBackend(responder), ["ehr_rag"])
    assert summary["counts"]["ehr_rag"]["ok_after_retry"] == 2
    parsed = _read_json(run.path / "ehr_rag" / "21" / "parsed.json")
    assert parsed["status"] == "ok_after_retry" and parsed["raw_file"] == "raw_retry.txt"
    assert parsed["first_attempt_problems"] and parsed["problems"] == []


def test_response_cache_and_no_cache(env: dict[str, Any]) -> None:
    _run(env, FakeBackend(), ["ehr_rag"])
    b2 = FakeBackend()
    run, summary = _run(env, b2, ["ehr_rag"])
    assert b2.prompts == [] and summary["n_cache_hits"] == 2 and summary["n_api_calls"] == 0
    call = _read_json(run.path / "ehr_rag" / "12" / "call.json")
    assert call["attempts"][0]["cached"] is True
    b3 = FakeBackend()
    _, summary = _run(env, b3, ["ehr_rag"], no_cache=True)
    assert len(b3.prompts) == 2 and summary["n_api_calls"] == 2


def test_cache_key_depends_on_model(env: dict[str, Any]) -> None:
    _run(env, FakeBackend(), ["ehr_rag"])
    b = FakeBackend(model="other-fake-model")
    _run(env, b, ["ehr_rag"])
    assert len(b.prompts) == 2


def test_confirm_threshold_requires_yes(env: dict[str, Any]) -> None:
    env["cfg"]["generation"]["confirm_threshold"] = 1
    backend = FakeBackend()
    with pytest.raises(ConfirmationRequired):
        _run(env, backend, ["ehr_rag"])
    assert backend.prompts == []
    _, summary = _run(env, backend, ["ehr_rag"], yes=True)
    assert summary["counts"]["ehr_rag"]["ok"] == 2


def test_full_context_skipped_when_too_long(env: dict[str, Any]) -> None:
    env["cfg"]["generation"]["max_context_tokens"] = 5
    backend = FakeBackend()
    run, summary = _run(env, backend, ["full_context"])
    assert summary["counts"]["full_context"]["skipped"] == 2 and backend.prompts == []
    skipped = _read_json(run.path / "skipped.json")
    assert [s["hadm_id"] for s in skipped] == HADM and all(s["input_tokens"] > 5 for s in skipped)
    assert not (run.path / "full_context").exists()


def test_persistent_failure_stops_run(env: dict[str, Any]) -> None:
    def responder(prompt: str, n: int) -> str:
        if n >= 1:
            raise BackendError("fake persistent failure")
        return FakeBackend.default_responder(prompt, n)

    run = start_run(env["cfg"], "generate", command="generate")
    with pytest.raises(BackendError):
        run_generation(env["cfg"], run, FakeBackend(responder), hadm_ids=HADM, conditions=["ehr_rag"],
                       retrieval_run=env["retrieval_run"], log=lambda s: None)
    assert (run.path / "ehr_rag" / "12" / "parsed.json").exists()
    assert not (run.path / "ehr_rag" / "21" / "parsed.json").exists()


def test_ehr_rag_requires_retrieval_run(env: dict[str, Any]) -> None:
    run = start_run(env["cfg"], "generate", command="generate")
    with pytest.raises(GenerationError, match="retrieval-run"):
        run_generation(env["cfg"], run, FakeBackend(), hadm_ids=HADM, conditions=["ehr_rag"],
                       retrieval_run=None, log=lambda s: None)


def test_stale_retrieval_run_rejected(env: dict[str, Any]) -> None:
    p = env["root"] / "runs" / env["retrieval_run"] / "retrieval" / "12.json"
    d = _read_json(p)
    d["evidence_units"][0]["text"] += " (edited)"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(GenerationError, match="re-run `retrieve`"):
        _run(env, FakeBackend(), ["ehr_rag"])
    del d["evidence_units"]
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(GenerationError, match="evidence_units"):
        _run(env, FakeBackend(), ["ehr_rag"])


def test_run_config_snapshot_records_generation_settings(env: dict[str, Any]) -> None:
    run, _ = _run(env, FakeBackend(), ["no_retrieval"])
    snap = yaml.safe_load((run.path / "config.yaml").read_text(encoding="utf-8"))
    assert snap["generation"]["temperature"] == 0.0


def test_check_format_accepts_one_period_after_citations() -> None:
    text = good_report(["E1", "E2"]).replace("one. [E1]\n", "one [E1].\n")
    sections, problems = check_format(text)
    assert problems == []
    assert sections["History"][0]["citations"] == ["E1"]
    assert sections["History"][0]["sentence"] == "Fake history sentence one"
    _, problems = check_format(good_report(["E1", "E2"]).replace("one. [E1]\n", "one [E1]..\n"))
    assert any("does not end with a citation" in p for p in problems)
