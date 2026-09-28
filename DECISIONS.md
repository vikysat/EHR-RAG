# DECISIONS

Places where SPEC.md was ambiguous or silent, and the choice made. In each case the
more conservative option was taken, meaning the one that makes the pipeline look no
better than it is, never guesses, and keeps the raw data for manual review.

## Phase 4: generation

- **Backends.** `hf_local` (transformers in-process) was added at the user's request.
  `fake` exists only in `tests/fakes.py` and is rejected by the config.
- **`anthropic` temperature.** The pinned SDK and current Claude models do not accept
  `temperature`. It is not sent, and `call.json` records `temperature_applied: false`.
- **`no_retrieval` format.** Every sentence ends with `[NONE]`. The prompt keeps "use
  ONLY the patient header", so any patient-specific content the model still produces
  is the baseline of what it invents.
- **`max_context_tokens`** counts the whole chat-templated prompt with the model's own
  tokenizer (chars/4 only for API backends without a tokenizer).
- **`ehr_rag` evidence** comes only from a saved `retrieve` run. Generation stops if any
  saved unit id, type or text differs from the current unit files.
- **Format retry.** One retry: the original prompt plus a reminder that lists the
  problems. The confirmation threshold counts only new (uncached) calls.

## Phase 5: evaluation

- **Abstention lines** (`Insufficient evidence. [NONE]`) are excluded from every rate
  and counted as `n_abstentions`, so they neither inflate nor deflate the rates.
- **Only the trailing citation group counts.** Citations in the middle of a line are
  flagged (`mid_sentence_citation`) but do not count, so such a sentence may be uncited.
- **Duplicate ids within one sentence** are counted once.
- **Invalid (fabricated) ids** count as citations in `invalid_citation_rate` and as
  irrelevant citations in `citation_precision`. They are never passed to NLI.
- **`citation_precision`** implements the SPEC definition literally (`entail(empty) =
  False`). One consequence, which matches ALCE: every citation of an unsupported
  sentence is irrelevant unless it entails the sentence on its own.
- **Premise.** Cited unit texts are joined by a single space in citation order. If
  the pair exceeds 512 tokens, only the premise is truncated. The number of truncated
  pairs is reported (`n_truncated_pairs`), because truncation can hide supporting
  evidence.
- **Aggregation** is micro-averaged over sentences or citations. Rates with a zero
  denominator are `null` (n/a), not 0.
- **`format_error` reports are included** in the metrics (parsed best-effort from the
  final raw output) and counted per condition. Dropping them would make a condition
  look better than it is.
- **Manual review sample.** Equal allocation over (condition, section) strata,
  round-robin, abstentions excluded, with one seeded generator per stratum. A
  `sentence_key` column was added after the SPEC columns for joining.
- **Verifier agreement.** Binary: NLI entailment vs human `supported` (`partial` and
  `unsupported` both count as not supported). `not_checkable` is excluded and counted.
  The full NLI × human confusion matrix is also reported.
- **BERTScore** uses the optional `bert-score` package, which is not in
  `requirements.txt` because the Demo has no references and the command refuses to run
  without them. It has not been tested. ROUGE-L is implemented and tested in-repo.
- **Reference file format** (only needed if you ever have references): CSV with
  columns `hadm_id,reference`.
