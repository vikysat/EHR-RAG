"""NLI verifier (``eval.nli_model``, default ``cross-encoder/nli-deberta-v3-base``).

The label order is read from the model's own ``config.id2label``; the entailment
label is the one whose name is ``entailment`` (case-insensitive). If the model or
that label cannot be found, loading raises ``NLIError`` (the caller stops).

Inputs longer than the model's maximum length are truncated from the PREMISE only
(``truncation="only_first"``); the number of truncated pairs is recorded.
Predictions are memoized per (premise, hypothesis).
"""

from __future__ import annotations

from typing import Any, Protocol


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
        self.entailment_label = self.id2label[ent[0]]
        self.revision = getattr(self.model.config, "_commit_hash", None)
        if not self.revision:
            raise NLIError(f"could not resolve the Hugging Face revision hash for {model_name}")
        self.batch_size = batch_size
        self.max_length = min(int(self.tok.model_max_length), 512)
        self.n_truncated = 0
        self._memo: dict[tuple[str, str], str] = {}

    def labels(self, pairs: list[tuple[str, str]]) -> list[str]:
        todo = list(dict.fromkeys(p for p in pairs if p not in self._memo))
        torch = self._torch
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i: i + self.batch_size]
            prem = [p for p, _ in batch]
            hyp = [h for _, h in batch]
            full = self.tok(prem, hyp, truncation=False)["input_ids"]
            self.n_truncated += sum(1 for ids in full if len(ids) > self.max_length)
            enc = self.tok(prem, hyp, truncation="only_first", max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                pred = self.model(**enc).logits.argmax(-1).tolist()
            for pair, k in zip(batch, pred):
                self._memo[pair] = self.id2label[int(k)]
        return [self._memo[p] for p in pairs]

    def info(self) -> dict[str, Any]:
        return {"model": self.model_name, "revision": self.revision, "id2label": self.id2label,
                "entailment_label": self.entailment_label, "device": self.device,
                "max_length": self.max_length, "n_truncated_pairs": self.n_truncated}
