"""Teacher-forced continuation log-probabilities (forward passes only, no generation).

A sequence is (prompt, prefix, continuation): the chat-templated prompt, then the ORIGINAL report up
to the cut point (``prefix``), then the scored ``continuation``. The scorer returns the continuation's
token ids and the log-probability of each of them given everything before it.

``HFScorer`` (``generation.backend: hf_local``): the backend's tokenizer, chat template and bf16 CUDA
model. The context is tokenized alone and with the continuation; continuation tokens are those after
the longest common prefix of the two (``boundary_merge`` is True if the context's own last tokens
changed, i.e. a token spans the cut; the cut is placed before the whitespace preceding the term so
this should not happen with Llama-3-style tokenizers). Batches are LEFT-padded with explicit position
ids, and only the last ``max continuation length + 1`` positions' logits are computed
(``logits_to_keep``), so the vocabulary-sized logits are never materialized for the prompt.
"""

from __future__ import annotations

from typing import Any, Protocol


class ProbeError(RuntimeError):
    """The probe cannot run as configured."""


class Scorer(Protocol):
    """Interface shared by the real and fake scorers."""

    name: str
    model: str

    def revision(self) -> str | None:
        """Model revision (commit hash) if known."""

    def score(self, batch: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
        """For each (prompt, prefix, continuation): {tokens, logprobs, boundary_merge}."""


def common_prefix(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class HFScorer:
    """Teacher forcing with an ``HFLocalBackend`` (see module docstring)."""

    name = "hf_local"

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.model = backend.model

    def revision(self) -> str | None:
        return self.backend.revision()

    def encode(self, prompt: str, prefix: str, cont: str) -> tuple[list[int], int, bool]:
        """(full token ids, number of context tokens, boundary_merge)."""
        t = self.backend.template(prompt) + prefix
        ctx = self.backend._encode(t)
        full = self.backend._encode(t + cont)
        n = common_prefix(ctx, full)
        if n >= len(full):
            raise ProbeError("empty continuation after tokenization")
        return full, n, n < len(ctx)

    def score(self, batch: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
        import torch

        model = self.backend._load_model()
        tok = self.backend._tokenizer()
        enc = [self.encode(*x) for x in batch]
        length = max(len(f) for f, _, _ in enc)
        keep = max(len(f) - n for f, n, _ in enc) + 1
        pad = tok.pad_token_id if tok.pad_token_id is not None else model.generation_config.pad_token_id
        if pad is None:
            eos = model.generation_config.eos_token_id
            pad = eos[0] if isinstance(eos, list) else eos
        ids = torch.full((len(enc), length), int(pad), dtype=torch.long)
        mask = torch.zeros((len(enc), length), dtype=torch.long)
        for i, (f, _, _) in enumerate(enc):
            ids[i, length - len(f):] = torch.tensor(f)
            mask[i, length - len(f):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            out = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                        position_ids=pos.to(model.device), logits_to_keep=keep, use_cache=False)
            logp = torch.log_softmax(out.logits.float(), dim=-1)      # [batch, keep, vocab]
        res = []
        for i, (f, n, merged) in enumerate(enc):
            c = len(f) - n
            # token at padded position length-c+j is predicted by the logits at length-c+j-1,
            # which is kept index keep-1-c+j
            idx = torch.arange(keep - 1 - c, keep - 1, device=logp.device)
            tgt = torch.tensor(f[n:], device=logp.device)
            res.append({"tokens": f[n:], "logprobs": logp[i, idx, tgt].tolist(), "boundary_merge": merged})
        return res


def make_scorer(gen: dict[str, Any]) -> Scorer:
    """The scorer for ``generation`` (only ``hf_local`` exposes token log-probabilities)."""
    if gen["backend"] != "hf_local":
        raise ProbeError(f"the probe needs token log-probabilities: generation.backend must be hf_local, "
                         f"got {gen['backend']!r}")
    from ehr_rag.generation.backends import HFLocalBackend

    return HFScorer(HFLocalBackend(gen["model"]))
