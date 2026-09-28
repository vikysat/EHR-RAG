"""Test doubles. Everything here is OBVIOUSLY FAKE and never used by the real pipeline.

- ``FakeEncoder``: deterministic bag-of-hashed-tokens encoder standing in for MedCPT.
- ``FakeBackend``: generation backend returning canned reports (well-formed and malformed).
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable

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
