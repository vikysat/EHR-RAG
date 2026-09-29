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

## Phase 6: perturbation pilot

- **Targets come from ehr_rag reports only**, using the final raw output (the retry
  output if there was a format retry), parsed with the Phase 5 parser. Only trailing
  citations link a sentence to a unit.
- **Lab values that can be targeted:** the shown points first/max/min/last. Reference
  bounds and counts cannot. A number equal to points from different measurements
  (different times) is ambiguous: it is recorded as a candidate but is not eligible.
  One measurement shown as several points (for example `n_measurements = 1`) is edited
  in all of them together.
- **Medications:** only `dose_val_rx` values that are a single number.
- **Numeric matcher.** Dates and clock times are removed before matching, and
  comma-grouped integers (`1,200`) count as one number. A `-` counts as a sign only
  when it is not attached to a preceding word or number (in `3.3-5.1` the second
  number is 5.1).
- **De-duplication:** by (admission, unit, value), keeping the first sentence.
  Selection shuffles with `perturbation.seed` and fills up to `n_targets` with at most
  `max_targets_per_admission` per admission.
- **Edits apply only to the matching unit type:** lab targets get
  lab_minor/lab_major/delete; medication targets get med_dose/delete.
- **`lab_minor`** tries ±10…20 % in a seeded order and takes the first change that is
  still 10–20 % after rounding to the original precision, keeps the marker, and keeps
  the unit consistent. For labs without a reference range, the marker comes from the
  flag, which a value edit does not change.
- **`lab_major`** needs a reference range. Normal → 1.5 × upper bound (0.7 × lower
  bound only if there is no upper bound). Abnormal → middle of the range (needs both
  bounds). The marker must actually change. `n_abnormal` (not rendered) is adjusted
  by ±1.
- **`med_dose` always doubles.** Halving can need more decimal places than the
  original (0.5 → 0.25), which the SPEC forbids.
- **Consistency:** after a lab edit, max must still be ≥ and min ≤ every shown point.
  Inconsistent edits are skipped and logged (`skipped_edits.csv`). Hidden measurements
  are not checked, because only the shown points exist in the fields.
- **Frozen evidence check.** Before any regeneration, the unperturbed prompt is rebuilt
  from `evidence_shown.json` plus the stored header and must equal the original
  `prompt.txt` byte for byte; otherwise the pilot stops. Unit files must also still
  match `evidence_shown.json`.
- **Format retry.** Pilot and control regenerations get the same single format retry
  as Phase 4. Their format status is reported and they are analyzed best-effort.
- **Controls** use `sample_idx` 1..`n_controls` (0 is the original). With greedy
  `hf_local` decoding, controls are expected to be identical; that is still a valid
  stability measurement.
- **Candidate sentences for outcome labels** are those citing the target id or
  containing the lab label / drug name (case-insensitive substring of the full label).
  Abstention lines are ignored.
- **Outcome precedence** (value edits): both the edited and the original value stated
  → ambiguous; else followed; else updated_miscited; else stale; no candidate sentence
  → dropped; otherwise ambiguous. For delete: original value stated → stale; no
  candidate → abstained; otherwise ambiguous.
- **Manual reading:** every lab_major case and every ambiguous case is flagged
  (`needs_manual_read`).
- **Control stability** means the original target sentence line (text plus citations)
  appears unchanged in the same section.
- **Pipeline split:** `select-targets` (CPU) → `run-pilot` (GPU) →
  `analyze-perturbation` (CPU). You can check targets and edits before spending GPU
  time.
