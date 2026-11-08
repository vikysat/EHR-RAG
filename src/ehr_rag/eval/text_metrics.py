"""Reference-based text metrics (ROUGE-L, BERTScore). They REQUIRE a reference file.

References are real discharge summaries of the same admission, written by
``extract-cohort`` to ``<cohort.out_dir>/reference/references.csv`` (columns
``hadm_id,reference``). The MIMIC-IV Demo has no notes, and this code never generates
references; without a reference file the command stops.

Generated reports are scored as their claim text only: section headers, citation
brackets and abstention lines are removed (``report_text``).

- ROUGE-L: LCS over lowercased word tokens; precision, recall, F1. Implemented here.
- BERTScore (``BERTScorer``): greedy cosine matching of contextual token embeddings
  (Zhang et al. 2020) without idf weighting or baseline rescaling, as bert-score's
  defaults. Discharge summaries are much longer than the encoder's 512-token limit,
  so each text is tokenized in full and embedded in consecutive non-overlapping
  windows of ``max_length - 2`` tokens (plus the special tokens per window); the token
  embeddings of all windows are concatenated, so every token of both texts is matched
  and nothing is truncated. Model and layer come from ``eval.bertscore_model`` /
  ``eval.bertscore_layer`` (default roberta-large, layer 17), loaded with
  ``local_files_only=True``: the weights must already be in the local HF cache.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ehr_rag.eval.parse import parse_report

NO_REFERENCES_MESSAGE = (
    "No reference reports provided (--references). The MIMIC-IV Clinical Database Demo has no "
    "reference reports, and this project never generates references. ROUGE-L / BERTScore were not computed."
)


class ReferenceFileError(RuntimeError):
    """Missing or invalid reference file."""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def report_text(raw: str) -> str:
    """A generated report as plain text: its sentences without headers, citations or abstention lines."""
    return "\n".join(s.text for s in parse_report(raw).sentences() if not s.is_abstention)


def rouge_l(candidate: str, reference: str) -> dict[str, float]:
    """ROUGE-L precision, recall and F1 from the longest common subsequence of word tokens."""
    c, r = _tokens(candidate), _tokens(reference)
    if not c or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    prev = [0] * (len(r) + 1)
    for ct in c:
        cur = [0] * (len(r) + 1)
        for j, rt in enumerate(r, 1):
            cur[j] = prev[j - 1] + 1 if ct == rt else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    p, rec = lcs / len(c), lcs / len(r)
    return {"precision": p, "recall": rec, "f1": 0.0 if lcs == 0 else 2 * p * rec / (p + rec)}


def load_references(path: Path | None) -> dict[int, str]:
    """Read ``hadm_id,reference`` CSV. Raises ReferenceFileError if absent or malformed."""
    if path is None:
        raise ReferenceFileError(NO_REFERENCES_MESSAGE)
    if not path.exists():
        raise ReferenceFileError(f"reference file not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or set(rows[0]) != {"hadm_id", "reference"}:
        raise ReferenceFileError(f"{path}: expected columns hadm_id,reference")
    refs = {int(r["hadm_id"]): r["reference"] for r in rows}
    if any(not v.strip() for v in refs.values()):
        raise ReferenceFileError(f"{path}: empty reference text")
    return refs


# --------------------------------------------------------------------------- BERTScore


def windows(ids: list[int], size: int) -> list[list[int]]:
    """Consecutive non-overlapping chunks of at most ``size`` ids (at least one, possibly empty)."""
    return [ids[i: i + size] for i in range(0, len(ids), size)] or [[]]


def greedy_match(cand: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """BERTScore P/R/F1 from L2-normalized token embeddings (rows): greedy max cosine each way."""
    if len(cand) == 0 or len(ref) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    sim = cand @ ref.T
    p, r = float(sim.max(axis=1).mean()), float(sim.max(axis=0).mean())
    return {"precision": p, "recall": r, "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}


class TokenEmbedder(Protocol):
    """Contextual token embeddings of a whole text (one row per token, L2-normalized)."""

    def embed(self, text: str) -> np.ndarray:
        """Token embedding matrix for ``text``."""


class BERTScorer:
    """Windowed BERTScore with a local Hugging Face encoder (see module docstring)."""

    def __init__(self, model_name: str, layer: int, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception as e:  # noqa: BLE001 - a missing model is fatal (never download, never substitute)
            raise ReferenceFileError(
                f"could not load BERTScore model {model_name} from the local Hugging Face cache "
                f"(download it on a node with internet first): {e}") from e
        self.model_name = model_name
        n_layers = int(self.model.config.num_hidden_layers)
        if not 0 <= layer <= n_layers:
            raise ReferenceFileError(f"bertscore_layer {layer} outside 0..{n_layers} for {model_name}")
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        limits = [int(self.tok.model_max_length), int(self.model.config.max_position_embeddings)]
        if getattr(self.model.config, "model_type", "") in ("roberta", "xlm-roberta", "camembert"):
            limits[1] -= 2   # RoBERTa reserves two position ids for padding offsets
        self.max_length = min(limits)
        # Windows are built as [CLS] + tokens + [SEP] (BERT: [CLS]/[SEP], RoBERTa: <s>/</s>); check that
        # this is exactly how the tokenizer itself wraps a single text.
        self.cls, self.sep = self.tok.cls_token_id, self.tok.sep_token_id
        probe = "fake probe text 5.9"
        inner = self.tok(probe, add_special_tokens=False)["input_ids"]
        if self.cls is None or self.sep is None or self.tok(probe)["input_ids"] != [self.cls, *inner, self.sep]:
            raise ReferenceFileError(f"{model_name}: tokenizer does not wrap a text as [CLS] text [SEP]; "
                                     "windowed BERTScore supports BERT/RoBERTa-style encoders only")
        self.window = self.max_length - 2
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def embed(self, text: str) -> np.ndarray:
        torch = self._torch
        ids = self.tok(text.strip(), add_special_tokens=False)["input_ids"]
        rows = []
        for chunk in windows(ids, self.window):
            if not chunk:
                continue
            x = torch.tensor([[self.cls, *chunk, self.sep]], device=self.device)
            with torch.no_grad():
                h = self.model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True)
            rows.append(h.hidden_states[self.layer][0, 1: 1 + len(chunk)].float().cpu().numpy())
        if not rows:
            return np.zeros((0, int(self.model.config.hidden_size)), dtype=np.float32)
        emb = np.concatenate(rows)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)

    def info(self) -> dict[str, Any]:
        return {"model": self.model_name, "revision": self.revision, "layer": self.layer,
                "max_length": self.max_length, "window_tokens": self.window, "device": self.device}


def bertscore(embedder: TokenEmbedder, candidate: str, reference: str) -> dict[str, float]:
    """BERTScore of one candidate against one reference."""
    return greedy_match(embedder.embed(candidate), embedder.embed(reference))


def text_metrics(
    reports: dict[tuple[str, int], str], references: dict[int, str], embedder: TokenEmbedder | None
) -> list[dict[str, Any]]:
    """Per (condition, admission) ROUGE-L (and BERTScore if ``embedder``) for admissions with a reference.

    ``reports`` maps (condition, hadm_id) to the raw report; it is scored as ``report_text``.
    """
    rows = []
    ref_emb: dict[int, np.ndarray] = {}
    for (cond, h), raw in sorted(reports.items()):
        if h not in references:
            continue
        cand = report_text(raw)
        row: dict[str, Any] = {"condition": cond, "hadm_id": h, "candidate_tokens": len(_tokens(cand)),
                               "reference_tokens": len(_tokens(references[h]))}
        row.update({f"rougeL_{k}": v for k, v in rouge_l(cand, references[h]).items()})
        if embedder is not None:
            if h not in ref_emb:
                ref_emb[h] = embedder.embed(references[h])
            row.update({f"bertscore_{k}": v for k, v in greedy_match(embedder.embed(cand), ref_emb[h]).items()})
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per condition: number of reports scored and mean / median of each F1 (counts and scores only)."""
    out = []
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        s: dict[str, Any] = {"condition": cond, "n": len(rs)}
        for m in ("rougeL_f1", "bertscore_f1"):
            if m in rs[0]:
                v = np.array([r[m] for r in rs], dtype=float)
                s[f"{m}_mean"], s[f"{m}_median"] = float(v.mean()), float(np.median(v))
        out.append(s)
    return out
