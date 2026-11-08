"""NLI verifier (``eval.nli_model``, default ``cross-encoder/nli-deberta-v3-base``).

The label order is read from the model's own ``config.id2label``; the entailment
label is the one whose name is ``entailment`` (case-insensitive). If the model or
that label cannot be found, loading raises ``NLIError`` (the caller stops).

The premise is only the units a claim cites (built in ``citations.py``). A pair whose
tokenized length exceeds the model's maximum (``min(model_max_length, 512)``) is NOT
run and NOT truncated: it gets the label ``too_long`` (``TOO_LONG``), which is never
the entailment label, so it counts as not entailed. Its token count is kept in
``pair_tokens`` and the evaluation logs every such pair (``nli_too_long.csv``).
Nothing is ever truncated, so ``n_truncated_pairs`` is always 0.
Predictions are memoized per (premise, hypothesis).
"""

from __future__ import annotations

from typing import Any, Protocol

TOO_LONG = "too_long"


class NLIError(RuntimeError):
    """The NLI model could not be loaded or used."""


class NLI(Protocol):
    """Minimal verifier interface used by the metrics."""

    model_name: str
    entailment_label: str

    def labels(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Predicted label name for each (premise, hypothesis) pair."""

    def info(self) -> dict[str, Any]:
        """Model id, revision, label mapping, truncation count."""


class HFNLI:
    """Hugging Face sequence-classification NLI model."""

    def __init__(self, model_name: str, batch_size: int = 16, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as e:  # noqa: BLE001 - any load failure is fatal (SPEC: STOP)
            raise NLIError(f"could not load NLI model {model_name}: {e}") from e
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        ent = [k for k, v in self.id2label.items() if v.lower() == "entailment"]
        if len(ent) != 1:
            raise NLIError(f"{model_name}: no unique 'entailment' label in id2label {self.id2label}")
        if TOO_LONG in self.id2label.values():
            raise NLIError(f"{model_name}: id2label already uses the reserved label {TOO_LONG!r}")
        self.entailment_label = self.id2label[ent[0]]
        self.revision = getattr(self.model.config, "_commit_hash", None)
        if not self.revision:
            raise NLIError(f"could not resolve the Hugging Face revision hash for {model_name}")
        self.batch_size = batch_size
        self.max_length = min(int(self.tok.model_max_length), 512)
        self._memo: dict[tuple[str, str], str] = {}
        self.pair_tokens: dict[tuple[str, str], int] = {}

    def _n_tokens(self, batch: list[tuple[str, str]]) -> list[int]:
        """Full tokenized length (special tokens included, no truncation) of each pair."""
        enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation=False)
        return [len(ids) for ids in enc["input_ids"]]

    def _predict(self, batch: list[tuple[str, str]]) -> list[str]:
        """Model labels for pairs that fit ``max_length`` (never truncated)."""
        enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation=False, padding=True,
                       return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            pred = self.model(**enc).logits.argmax(-1).tolist()
        return [self.id2label[int(k)] for k in pred]

    def labels(self, pairs: list[tuple[str, str]]) -> list[str]:
        todo = list(dict.fromkeys(p for p in pairs if p not in self._memo))
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i: i + self.batch_size]
            fits = []
            for pair, n in zip(batch, self._n_tokens(batch)):
                self.pair_tokens[pair] = n
                if n > self.max_length:
                    self._memo[pair] = TOO_LONG
                else:
                    fits.append(pair)
            if fits:
                for pair, label in zip(fits, self._predict(fits)):
                    self._memo[pair] = label
        return [self._memo[p] for p in pairs]

    def info(self) -> dict[str, Any]:
        return {"model": self.model_name, "revision": self.revision, "id2label": self.id2label,
                "entailment_label": self.entailment_label, "device": self.device,
                "max_length": self.max_length, "too_long_label": TOO_LONG,
                "n_too_long_pairs": sum(1 for v in self._memo.values() if v == TOO_LONG),
                "n_truncated_pairs": 0}
