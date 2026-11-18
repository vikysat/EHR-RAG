"""Citation-dependence probe: edits, units, contexts, scoring, resume, threshold guard, headline.

OBVIOUSLY FAKE units and reports only (checker_fakes, the fake MIMIC tables of conftest) and a FAKE
scorer; the HF scorer's batching is checked with a tiny torch module, no model is downloaded.
"""

from __future__ import annotations

import csv
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from checker_fakes import checker, dx, fake_units

REPO = Path(__file__).resolve().parents[1]


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((REPO / "config" / "default.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- units and contexts

REPORT = """### History
Insufficient evidence. [NONE]
### Findings
Potassium was elevated at 5.9. [E1]
Hyperkalemia persisted. [E1]
Hyponatremia with sodium 128. [E2]
### Impression
No leukocytosis. [E6]
### Recommendations
Creatinine elevated, monitor. [E5]
"""


def _probe(report: str = REPORT, units: list[Any] | None = None) -> Any:
    from ehr_rag.eval.parse import parse_report
    from ehr_rag.perturb.run_pilot import cited_claim_counts
    from ehr_rag.probe.runner import _rules_words
    from ehr_rag.probe.targets import probe_report

    units = units or fake_units() + [dx("E11", "Hyperkalemia", "FAKE9")]
    ck = checker()
    ctx = ck.admission(units, units)
    cfg = _cfg()
    cfg["checker"]["rules"] = str(REPO / "config" / "checker.yaml")
    cfg["_project_root"] = str(REPO)
    parsed = parse_report(report)
    words = _rules_words(cfg)
    return probe_report(7, report, parsed, units, cited_claim_counts(parsed, ck, ctx), ck, ctx, cfg, words), units


def test_units_first_mention_alternatives_and_offsets() -> None:
    rp, _ = _probe()
    assert [(u.claim_kind, u.surface, u.cited_unit) for u in rp.units] == [
        ("word", "elevated", "E1"), ("term", "Hyponatremia", "E2")]          # hyperkalemia: not the first mention
    k, na = rp.units
    assert (k.direction, k.status, k.edit_type, k.term_alt) == ("high", "high", "to_normal", "normal")
    assert (na.direction, na.edit_type, na.term_alt) == ("low", "to_normal", "Normal sodium")   # line start
    for u in rp.units:
        assert REPORT[u.term_start:u.term_end] == u.surface
    assert REPORT[k.term_cut:k.term_start] == " " and na.term_cut == na.term_start
    assert (k.number, k.order, k.number_alt_cited) == ("5.9", "term_first", "4.2")       # midpoint of 3.3-5.1
    assert REPORT[k.number_start:k.number_end] == "5.9" and REPORT[k.number_cut:k.number_start] == " "
    assert (na.number, na.order) == ("128", "term_first") and na.number_alt_cited == "139"
    reasons = {(s["surface"], s["reason"]) for s in rp.skips}
    assert ("No leukocytosis".split()[1], "negated claim") in reasons
    assert ("elevated", "claim does not cite a matching lab unit") in reasons        # creatinine cites a drug


def test_contexts_null_and_corroborators() -> None:
    from ehr_rag.evidence.render import render
    from ehr_rag.probe.targets import apply_spec

    rp, units = _probe()
    k, na = rp.units
    assert k.corroborators == ["E11"] and k.corroborated                     # the Hyperkalemia diagnosis
    assert na.corroborators == [] and "ablation" not in na.contexts
    assert k.null_unit in {"E3", "E10"}                                      # uncited, also HIGH
    assert na.null_unit == "E7"                                              # the only uncited LOW lab
    assert k.contexts == ["original", "cited_edit", "cited_edit_prefix", "within_flag", "matched_null", "ablation"]
    ev = apply_spec(units, rp.specs[(0, "cited_edit")])
    e1 = next(u for u in ev if u.unit_id == "E1")
    assert "[HIGH]" not in e1.text and e1.text == render("lab", e1.fields)
    assert [u.unit_id for u in apply_spec(units, rp.specs[(0, "ablation")])] == [u.unit_id for u in units if u.unit_id != "E11"]
    wf = next(u for u in apply_spec(units, rp.specs[(0, "within_flag")]) if u.unit_id == "E1")
    assert "[HIGH]" in wf.text and wf.text != units[0].text
    assert k.number_alt_within and Decimal(k.number_alt_within) != Decimal("5.9")


def test_unit_without_an_in_band_within_flag_value_has_no_within_flag_context() -> None:
    from checker_fakes import lab

    units = [lab("E1", "Potassium", [("2150-03-01T08:00:00", "6")], lo="3.5", hi="5.5")]   # +-10..20 % -> 5 or 7
    units += [lab("E2", "Sodium", [("2150-03-01T08:00:00", "140")], lo="139", hi="141")]  # band holds only 140
    report = "### Findings\nElevated potassium at 6. [E1]\nSodium normal at 140. [E2]\n"
    rp, _ = _probe(report, units)
    k, na = rp.units
    assert "within_flag" in k.contexts and k.number_alt_within == "7"     # 5 would leave the HIGH band
    assert "within_flag" not in na.contexts and na.number_alt_within == ""
    assert ("normal", "within-flag context missing: no within-flag change moves every value inside its band and "
            "keeps the unit consistent") in {(s["surface"], s["reason"]) for s in rp.skips}


def test_number_first_and_list_items_use_their_own_citation() -> None:
    rp, _ = _probe("### Findings\nPotassium 5.9, elevated [E1], sodium low at 128 [E2].\n")
    assert [(u.surface, u.cited_unit, u.order, u.number) for u in rp.units] == [
        ("elevated", "E1", "number_first", "5.9"), ("low", "E2", "term_first", "128")]


# --------------------------------------------------------------------------- HF scorer arithmetic


class _TinyBackend:
    """Character tokenizer + a torch module that predicts 'next id = id + 1' (FAKE)."""

    model = "tiny"

    def __init__(self) -> None:
        import torch

        class M(torch.nn.Module):
            device = torch.device("cpu")

            class generation_config:          # noqa: N801
                pad_token_id = 0
                eos_token_id = 0

            def forward(self, input_ids: Any, attention_mask: Any, position_ids: Any, logits_to_keep: int,
                        use_cache: bool) -> Any:
                v = 128
                logits = torch.nn.functional.one_hot((input_ids + 1) % v, v).float() * 10.0
                logits = logits + attention_mask[..., None] * 0.0
                return type("Out", (), {"logits": logits[:, -logits_to_keep:]})()

        self._m = M()

    def revision(self) -> str:
        return "tiny-rev"

    def template(self, prompt: str) -> str:
        return f"<{prompt}>"

    def _encode(self, s: str) -> list[int]:
        return [ord(c) % 128 for c in s]

    def _tokenizer(self) -> Any:
        return type("Tok", (), {"pad_token_id": 0})()

    def _load_model(self) -> Any:
        return self._m


def test_hf_scorer_left_padding_matches_single_sequences() -> None:
    pytest.importorskip("torch")
    from ehr_rag.probe.score import HFScorer

    sc = HFScorer(_TinyBackend())
    items = [("p", "ab", "cde"), ("a much longer prompt", "xy", " z"), ("q", "", "bcx")]
    batched = sc.score(items)
    single = [sc.score([x])[0] for x in items]
    for b, s in zip(batched, single):
        assert b["tokens"] == s["tokens"] and b["logprobs"] == pytest.approx(s["logprobs"], abs=1e-5)
    hit, miss = batched[0]["logprobs"][0], batched[2]["logprobs"][2]     # 'c' after 'b' is predicted; 'x' is not
    assert hit > -0.2 and miss < -3 and batched[0]["tokens"] == [ord("c"), ord("d"), ord("e")]


# --------------------------------------------------------------------------- end to end (fake data, fake scorer)


class FakeScorer:
    """FAKE dependent model: 'elevated' is likely iff the evidence shows a [HIGH] marker; numbers are
    likely iff they appear in the evidence. Raises after ``fail_after`` batches (a GPU session ending)."""

    name = "fake"

    def __init__(self, model: str = "fake-scorer", fail_after: int | None = None) -> None:
        self.model, self.fail_after, self.batches = model, fail_after, 0

    def revision(self) -> str:
        return "fake-rev"

    def score(self, batch: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
        if self.fail_after is not None and self.batches >= self.fail_after:
            raise KeyboardInterrupt("GPU session ended")
        self.batches += 1
        out = []
        for prompt, _, cont in batch:
            ev = prompt.split("Evidence (one unit per line", 1)[-1]
            word = cont.strip()
            if re.fullmatch(r"-?\d+(?:\.\d+)?", word):
                lp = -0.1 if re.search(rf"(?<![\d.]){re.escape(word)}(?![\d])", ev) else -4.0
            else:
                high = "[HIGH]" in ev
                lp = -0.1 if (word.lower() == "elevated") == high else -3.0
            out.append({"tokens": [len(word)], "logprobs": [lp], "boundary_merge": False})
        return out


def _responder(prompt: str, n: int) -> str:
    from fakes import realistic_report

    return realistic_report(prompt).replace("Fakeium reached", "Elevated Fakeium reached")


@pytest.fixture()
def probe_env(env: dict[str, Any], capsys: pytest.CaptureFixture) -> dict[str, Any]:
    from ehr_rag.cli import main
    from ehr_rag.eval.evaluate import run_evaluation
    from ehr_rag.generation.generate import run_generation
    from ehr_rag.runs import start_run
    from fakes import FakeBackend, FakeNLI

    cfg_path = env["root"] / "config" / "test.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["probe"]["batch_size"] = 4
    raw["probe"]["threshold_file"] = str(env["root"] / "config" / "probe_threshold.yaml")
    raw["generation"].update(backend="hf_local", model="fake/model", temperature=0)
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    gen = start_run(env["cfg"], "generate", command="generate")
    run_generation(env["cfg"], gen, FakeBackend(_responder), hadm_ids=[12, 21], conditions=["ehr_rag"],
                   retrieval_run=env["retrieval_run"], log=lambda s: None)
    erun = start_run(env["cfg"], "evaluate", command="evaluate")
    run_evaluation([gen.path], erun.path, erun.run_id, FakeNLI(), 5, 1)
    assert main(["probe-targets", "--config", str(cfg_path), "--generation-run", gen.path.name,
                 "--run-id", "20990101-000000-probe-targets"]) == 0
    return {**env, "cfg_path": str(cfg_path), "gen": gen.path.name, "eval": erun.run_id,
            "targets_out": capsys.readouterr().out}


def test_probe_targets_cli(probe_env: dict[str, Any]) -> None:
    out = probe_env["targets_out"]
    assert "split: demo" in out and "units: 2" in out and "expected forward passes: 34" in out
    t = probe_env["root"] / "runs" / "20990101-000000-probe-targets"
    with (t / "probe_units.csv").open(encoding="utf-8") as f:
        units = list(csv.DictReader(f))
    assert [(u["hadm_id"], u["surface"], u["term_alt"], u["contexts"]) for u in units] == [
        ("12", "Elevated", "Normal", "original cited_edit cited_edit_prefix within_flag"),
        ("21", "Elevated", "Normal", "original cited_edit cited_edit_prefix within_flag")]
    assert "Fakeium" not in out


def test_run_probe_resumes_after_an_interruption(probe_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main
    from ehr_rag.probe import score

    cfg = probe_env["cfg_path"]
    monkeypatch.setattr(score, "make_scorer", lambda g: FakeScorer(fail_after=2))
    with pytest.raises(KeyboardInterrupt):
        main(["run-probe", "--config", cfg, "--targets-run", "20990101-000000-probe-targets",
              "--run-id", "20990101-000001-run-probe"])
    run = probe_env["root"] / "runs" / "20990101-000001-run-probe"
    assert "expected forward passes: 30 to run of 30 distinct sequences (0 already scored or reused, 0 of them reused now) in 8 batches" in \
        capsys.readouterr().out
    lines = (run / "probe_scores.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8 and json.loads((run / "run_meta.json").read_text(encoding="utf-8"))["status"] == "interrupted"
    with (run / "probe_scores.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"key": "cut-off')                                          # a write cut off mid-line
    monkeypatch.setattr(score, "make_scorer", lambda g: FakeScorer(model="other"))
    assert main(["run-probe", "--config", cfg, "--targets-run", "20990101-000000-probe-targets",
                 "--resume", run.name]) == 4                                 # different model: plan changed
    monkeypatch.setattr(score, "make_scorer", lambda g: FakeScorer())
    assert main(["run-probe", "--config", cfg, "--targets-run", "20990101-000000-probe-targets",
                 "--resume", run.name]) == 0
    out = capsys.readouterr().out
    assert "22 to run of 30 distinct sequences (8 already scored or reused" in out
    rows = [json.loads(x) for x in (run / "probe_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 30 and len({r["key"] for r in rows}) == 30
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ok" and len(meta["resumed_at"]) == 2
