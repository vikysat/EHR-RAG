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
- **Claims (revised).** Originally only the trailing citation group counted and
  citations inside a line were ignored. Now a line with inside citations (a list) is
  split into one claim per citation run before scoring, each with only its own ids;
  all rates are over claims. Leading `,;:`/`and`/`or` are removed from each claim; a
  claim is not given the list's lead-in words (`Labs: Na 130 [E1], K 5.9 [E2]` →
  `Labs: Na 130`, `K 5.9`), so later items are short hypotheses. Text after the last
  citation is an uncited claim. The perturbation pilot still uses trailing
  citations only.
- **Trailing period.** `Fact [E3].` counts as cited by E3 in both the generation
  format check and the parser (the period moves to the text, flag
  `period_after_citation`). Only one period is accepted.
- **Duplicate ids within one claim** are counted once.
- **Invalid (fabricated) ids** count as citations in `invalid_citation_rate` and as
  irrelevant / not supporting in both precisions. They are never passed to NLI.
- **`citation_precision`** implements the SPEC definition literally (`entail(empty) =
  False`). One consequence, which matches ALCE: every citation of an unsupported
  claim is irrelevant unless it entails the claim on its own.
  **`citation_precision_plain`** is reported next to it: citations that entail the
  claim alone / all citations. The two differ when units support a claim only jointly.
- **Premise.** Only the claim's valid cited unit texts, joined by a single space in
  citation order. **too_long (revised):** a pair longer than the NLI limit (512
  tokens) is never truncated and never run; it is labeled `too_long`, counts as not
  entailed everywhere, and is logged in `nli_too_long.csv`. Truncating could hide or
  invent support, so not entailed is the conservative label.
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
- **BERTScore (revised)** is implemented in-repo with transformers instead of the
  bert-score package: the package truncates at 512 tokens, which would compare a report
  against only the opening of a discharge summary. Both texts are embedded in
  consecutive non-overlapping 510-token windows ([CLS] + window + [SEP]) and all token
  embeddings are greedily matched. Defaults as bert-score: roberta-large layer 17, no
  idf, no rescaling. Windows lose context at their borders; overlapping windows were
  not used, for simplicity. The model loads with `local_files_only=True` and never
  downloads.
- **What is scored:** the report's sentence text, without headers, citation brackets
  or abstention lines, against the whole discharge summary (all of an admission's
  discharge notes, joined). All conditions are scored by default.
- **Reference file format:** CSV with columns `hadm_id,reference`, written by
  `extract-cohort`.

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

## Phase 7: demo notebook and README

- **Which runs the notebook uses:** by default it follows the chain from the latest
  successful `analyze-perturbation` run (→ pilot → targets → generation → the
  generation's retrieve run), plus the latest `evaluate` run of that generation run.
  Any run id can be overridden in the first code cell. Sections whose runs are missing
  print "not available" instead of failing.
- **Header source:** the notebook reads the patient header from the stored
  `prompt.txt`, which is exactly what the model saw, so it needs only `runs/`, not
  `data/`.
- **Testing:** the notebook is tested by executing its code cells against runs
  produced by the full CLI chain on fake data. Jupyter is not added to
  `requirements.txt`.

## Full MIMIC-IV cohort

- **Selection order** (`select-cohort`, counts printed after each step): all subjects in
  `patients` → seeded pool (`round(pool_fraction × n)` drawn with
  `random.Random(f"{seed}:pool")` from the sorted ids) → Demo subjects removed
  (`demo_subject_id.csv`) → their admissions → stay 2–14 days (inclusive, from
  admittime/dischtime) → ≥ 1 prior admission (the `select-admissions` definition) →
  ≥ 20 `labevents` rows with this `hadm_id` → a discharge note in `note/discharge`.
- **Lab count** uses rows linked by `hadm_id` only. `build-units` also links null-hadm
  rows by time window, so the evidence never has fewer lab rows than the filter saw.
- **One admission per patient**, the latest eligible one (as in `select-admissions`).
  Eligible patients are shuffled with `random.Random(f"{seed}:draw")`; the first
  `n_dev` are dev, the next `n_test` are test, so the splits never share a patient.
- **Id lists** are sorted, one id per line, under `cohort.out_dir` (gitignored). The
  committed `config/cohort/<split>_hadm_ids.sha256` is in `sha256sum` format. Every
  command that reads a split checks the hash first. `select-cohort` refuses to
  replace existing hashes without `--overwrite`.
- **Testing on the Demo** needs `--skip-note-filter` (no notes) and
  `--no-demo-exclusion` (otherwise nothing is left).
- **Extraction scope:** every row of the cohort's PATIENTS (not only the target
  admissions) for patients, admissions, labevents, prescriptions, diagnoses_icd and
  procedures_icd, because history units and time-window lab linkage need the other
  admissions. Dictionary tables are copied in full. Columns are exactly the Demo's,
  in the Demo's order, stored as strings, so `build-units` gives byte-identical units
  from the Parquet and from the CSV (tested on fake data and on the Demo). Columns
  that only the newer full release has are dropped; a Demo column missing from the
  full data stops extraction.
- **Reference notes:** only the target admissions' discharge notes are extracted, to
  `reference/`, never to the evidence root. Several notes of one admission are joined
  by a blank line in `note_seq` order in `references.csv`.
- **Admission ids for full runs** come from `--split` (hash-checked), so
  `config/selected_admissions.yaml` (the Demo's approved list) stays untouched.
- **Reference-leak guard** is enforced twice: no note table may be under `data.root`
  (`load_tables`), and every unit must come from a structured table (`build-units`,
  and again in `generate` for unit files read from disk).
- **Counts only on full data:** with `data.source: full`, every command that used to
  print units, retrieved sections or report sentences writes them to its run folder
  instead (`evaluate` writes `gate_sample.json`). Ids and counts are still printed.

## Sharded generation

- **Shards are contiguous blocks** of the ordered admission list (`--shard-size`,
  0-based `--shard-index`); the array script sets the index from `SLURM_ARRAY_TASK_ID`
  and the size from `SHARD_SIZE` (default 25). An index past the last shard is a config
  error (exit 2) and generates nothing. `--array` must be given at submission.
- **One run folder per shard** (`<date>-generate-job<array id>-shard<k>`), so shards
  never write the same `run_meta.json`; the response cache is shared (one file per key,
  atomic writes).
- **`evaluate` and `text-metrics` take several `--generation-run` ids.** Reports are
  ordered by condition and hadm_id before scoring, so the metrics and the seeded
  manual-review sample are identical to one unsharded run (tested). The same
  (condition, admission) in two runs is an error. `select-targets` still takes one run.
