"""Test doubles. Everything here is OBVIOUSLY FAKE and never used by the real pipeline.

- ``FakeEncoder``: deterministic bag-of-hashed-tokens encoder standing in for MedCPT.
- ``FakeBackend``: generation backend returning canned reports (well-formed and malformed).
- ``realistic_report``: a FAKE report that restates values from the evidence block (so
  perturbation outcomes can be exercised end to end).
- ``FakeNLI``: rule-based stand-in for the NLI model.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

import numpy as np

from ehr_rag.generation.backends import Completion
from ehr_rag.retrieval.bm25 import tokenize


class FakeEncoder:
    """Deterministic bag-of-hashed-tokens encoder standing in for MedCPT in tests."""

    dim = 64

    def encode(self, items: list) -> np.ndarray:
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for i, it in enumerate(items):
            text = it if isinstance(it, str) else f"{it[0]} {it[1]}"
            for tok in tokenize(text):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        out += 1e-6
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def good_report(ids: list[str]) -> str:
    """A correctly formatted FAKE report citing the given evidence ids."""
    a = ids[0] if ids else "E1"
    b = ids[1] if len(ids) > 1 else a
    return (
        "### History\n"
        f"Fake history sentence one. [{a}]\n"
        "\n"
        "### Findings\n"
        f"Fake finding with two citations. [{a}, {b}]\n"
        f"Fake finding with adjacent brackets. [{a}][{b}]\n"
        "### Impression\n"
        f"Fake impression. [{b}]\n"
        "### Recommendations\n"
        "Insufficient evidence. [NONE]\n"
    )


NONE_REPORT = (
    "### History\nFake history from the header only. [NONE]\n"
    "### Findings\nInsufficient evidence. [NONE]\n"
    "### Impression\nInsufficient evidence. [NONE]\n"
    "### Recommendations\nInsufficient evidence. [NONE]\n"
)

MALFORMED_REPORT = (
    "Here is the fake report:\n"
    "### History\n"
    "- Fake bullet without citation\n"
    "### Findings\n"
    "Fake finding. [E1]\n"
)

_EVIDENCE_ID_RE = re.compile(r"^\[(E\d+)\] \(", re.MULTILINE)


class FakeBackend:
    """Canned-response backend. ``responder(prompt, n_calls_so_far) -> text``.

    By default it answers with a well-formed report citing the evidence ids found
    in the prompt (or ``NONE_REPORT`` when there is no evidence block).
    """

    name = "fake"

    def __init__(self, responder: Callable[[str, int], str] | None = None, model: str = "fake-model") -> None:
        self.model = model
        self.responder = responder or self.default_responder
        self.prompts: list[str] = []

    @staticmethod
    def default_responder(prompt: str, n: int) -> str:
        ids = _EVIDENCE_ID_RE.findall(prompt)
        return good_report(ids) if ids else NONE_REPORT

    def revision(self) -> str | None:
        return "fake-revision"

    def count_tokens(self, prompt: str) -> tuple[int, bool]:
        return len(prompt.split()), True

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Completion:
        text = self.responder(prompt, len(self.prompts))
        self.prompts.append(prompt)
        return Completion(
            text=text, model=f"{self.model}-reported", revision="fake-revision", input_tokens=len(prompt.split()),
            output_tokens=len(text.split()), finish_reason="stop", temperature_applied=True,
        )


_EVIDENCE_LINE_RE = re.compile(r"^\[(E\d+)\] \((\w+)\) (.*)$", re.MULTILINE)
_LAB_VALUE_RE = re.compile(r"(?:max|Value) (-?\d+(?:\.\d+)?)")


def realistic_report(prompt: str) -> str:
    """A well-formed FAKE report restating lab values and doses from the evidence block."""
    ev = _EVIDENCE_LINE_RE.findall(prompt)
    if not ev:
        return NONE_REPORT
    hist, find, imp, rec = [], [], [], []
    for uid, typ, text in ev:
        if typ in ("diagnosis", "history_diagnosis") and ": " in text:
            title = text.split(": ", 1)[1].split(" (ICD")[0]
            (hist if typ == "history_diagnosis" else imp).append(f"The patient has {title}. [{uid}]")
        elif typ == "lab":
            m = _LAB_VALUE_RE.search(text)
            if m:
                label = re.split(r" \(|,", text, maxsplit=1)[0]
                find.append(f"{label} reached {m.group(1)}. [{uid}]")
        elif typ == "medication":
            rec.append(f"Continue {text.split(', ordered')[0]}. [{uid}]")
    none = ["Insufficient evidence. [NONE]"]
    out = []
    for name, lines in (("History", hist), ("Findings", find), ("Impression", imp), ("Recommendations", rec)):
        out.append(f"### {name}")
        out.extend(lines[:3] or none)
    return "\n".join(out) + "\n"


def realistic_responder(prompt: str, n: int) -> str:
    """FakeBackend responder built on ``realistic_report``."""
    return realistic_report(prompt)


_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


class FakeNLI:
    """Rule-based FAKE verifier: entailment iff the hypothesis has numbers and all appear in the premise,
    or (without numbers) all its words of >= 5 letters appear in the premise. Otherwise neutral."""

    model_name = "fake-nli"
    entailment_label = "entailment"

    def __init__(self, max_length: int = 512) -> None:
        """``max_length`` is in whitespace tokens: longer pairs get ``too_long`` (as HFNLI)."""
        self.calls: list[tuple[str, str]] = []
        self.max_length = max_length
        self.pair_tokens: dict[tuple[str, str], int] = {}

    @staticmethod
    def rule(premise: str, hyp: str) -> str:
        nums = _NUM_RE.findall(hyp)
        if nums:
            ok = all(n in _NUM_RE.findall(premise) for n in nums)
        else:
            words = [w for w in re.findall(r"[a-z]+", hyp.lower()) if len(w) >= 5]
            ok = bool(words) and all(w in premise.lower() for w in words)
        return "entailment" if ok else "neutral"

    def labels(self, pairs: list[tuple[str, str]]) -> list[str]:
        from ehr_rag.eval.nli import TOO_LONG

        self.calls.extend(pairs)
        out = []
        for p, h in pairs:
            self.pair_tokens[(p, h)] = n = len(p.split()) + len(h.split())
            out.append(TOO_LONG if n > self.max_length else self.rule(p, h))
        return out

    def info(self) -> dict[str, Any]:
        n_long = sum(1 for n in self.pair_tokens.values() if n > self.max_length)
        return {"model": self.model_name, "revision": "fake", "id2label": {0: "entailment", 1: "neutral"},
                "entailment_label": "entailment", "max_length": self.max_length, "too_long_label": "too_long",
                "n_too_long_pairs": n_long, "n_truncated_pairs": 0}
