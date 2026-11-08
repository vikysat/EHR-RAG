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

## Citation checker (deterministic)

Rules are in `config/checker.yaml`, vocabulary in `config/checker_lexicon.yaml`; the
full definitions are in `src/ehr_rag/checker/rules.py`. Choices where the task left room:

- **Valid ids** are those in `evidence_shown.json` (what the model saw), as in evaluation.
- **List items keep the opening words** (the first item's text before its first number or
  name). This differs on purpose from evaluation's claim split, which does not repeat the
  lead-in; evaluation is unchanged.
- **Numbers:** a stated number matches if it equals a number in a cited unit's text, or if
  the unit value rounded half-up to the stated decimals equals it (5.94 → "5.9", 5.9 →
  "6"). More precision than the unit (5.94 for 5.9) fails. Besides ages, dates and
  section/list numbers, durations ("5 days", "hospital day 3") and ICD versions are
  ignored: durations come from the non-citable header and would otherwise always fail.
- **Names** come from the admission's own units (lab labels, drug names, diagnosis and
  procedure titles and simple variants) plus lexicon synonyms. A drug or diagnosis that is
  in no unit of the admission and not in the lexicon is not detected (limits coverage).
  Short abbreviations with capitals (K, Na, Cr, pH) are case-sensitive.
- **Directions** use the shown points' [HIGH]/[LOW] markers. With a stated number, that
  point decides; otherwise high/low hold if any shown point is marked, normal if every
  cited unit's last point is unmarked. Labs without a reference range are not checkable.
  A lab name supported only by a non-lab unit (the drug Potassium Chloride) plus a
  direction word fails.
- **Trends (adapted):** there is one lab unit per lab per admission, so "two cited units
  of the same lab" would almost never exist. A trend word needs >= 2 shown measurements of
  that lab at different times across the cited units (usually first and last of one
  unit); latest minus earliest must have the word's sign.
- **Lab-derived terms** (hyperkalemia, AKI, anemia, ...) are supported by a cited lab of the
  right group, whose direction is then checked, or by a cited diagnosis title containing
  the term (then no direction check).
- **Negation:** a cue (no, not, without, negative for, ...) within 3 tokens before the
  direction word or the name flips the claim.
- **Status:** fail (any error), pass (no error, something checkable), unchecked (nothing
  checkable). Unchecked lines are reported separately as checker coverage.

## Enforcement

- **Conditions:** `ehr_rag_enforced` / `full_context_enforced` (generate: the base prompt,
  so a cached base report is reused, then one rewrite per failing line by the same model)
  and `ehr_rag_filtered` / `full_context_filtered` (`enforce-filter`, CPU, no model: copies an
  existing run's reports and drops failing lines). No `no_retrieval` variant: it cites
  nothing, so every checkable line would be dropped.
- **Rewrite** (`prompts/rewrite_v1.txt`, versioned like the report prompt): the line, the
  checker's messages and ONLY its cited units. The answer's first non-blank line is used
  (a bullet or quotes are stripped); it must end with an evidence citation, otherwise it is
  dropped (an abstention answer is a drop too). The re-check uses the same evidence.
- **Repair vs retreat:** repaired = passes and keeps at least as many checkable numbers
  AND names as the original line; retreated = passes with fewer; dropped = failed twice.
  A rewrite with no checkable content passes as unchecked and therefore counts as retreated.
- **Emptied sections** get `Insufficient evidence. [NONE]` (logged in `sections_emptied`), so
  the enforced report keeps the four-section format; abstentions are excluded from all
  evaluation rates anyway. Preamble and extra sections are not carried over (counted).
- **Files:** `enforced.txt`, `enforcement.json` (every line, its check, the rewrite prompt and
  answer, the outcome) and `rewrites/`; `parsed.json` points `raw_file` at `enforced.txt`
  and keeps `pre_enforcement_raw_file`, so `evaluate` and `text-metrics` read the enforced text.

## Evaluation independent of the checker

- `ehr_rag.eval` and `ehr_rag.runs` never import `ehr_rag.checker` (tested by walking the
  import graph, lazy imports included). `evaluate` grades enforced and filtered reports
  with NLI only; checker statistics exist only in `check-citations` outputs.
- **Frozen checker:** every run's `run_meta.json` has `checker` (SHA-256 of the rules, the
  lexicon, the rewrite prompt, and a combined hash), computed as plain file hashes.
  `validate-config` prints the combined hash; `generate`, `enforce-filter` and
  `check-citations` take `--checker-sha256` and refuse to run if the files changed.

## Counterfactual citation audit

- **Commands:** `audit-targets` (CPU) → `run-audit` (GPU, sharded) → `analyze-audit` (CPU).
  They extend the Phase 6 modules: selection in `perturb/select_targets.py`, edits in
  `perturb/edit.py`, runs in `perturb/run_pilot.py`, labels in `perturb/analyze.py`. The
  original `select-targets` / `run-pilot` / `analyze-perturbation` are unchanged.
- **Targets:** per admission, up to `audit.n_cited_per_admission` (3) numeric lab units of the
  shown evidence that the plain ehr_rag report cites in any claim (list items included),
  units with a [HIGH]/[LOW] point first; plus `audit.n_null_per_admission` (1) shown numeric
  lab cited nowhere. Each group is shuffled with a seed per admission. Units without a
  TWO-sided reference range (a one-sided range has no midpoint and no width) and units the
  edit cannot be applied to are skipped, counted by reason (`audit_skips.csv`), and the next
  unit is tried.
- **Edits:** every shown measurement (first/max/min/last) is moved to ONE value, so the unit
  has a single unambiguous status. Flagged ("mixed" included) → range midpoint, flags
  cleared, `n_abnormal` 0. Normal → `ref_high + 0.5 × width`, or `ref_low − 0.5 × width`
  for hemoglobin, hematocrit, platelets and albumin (configurable; low values are the
  clinically usual direction for these), flag "abnormal", `n_abnormal = n_measurements`.
  The value keeps the unit's precision; edits whose rounded value lands on the wrong side,
  or a non-positive low value, are skipped.
- **Regeneration (`run-audit`):** per admission and condition, the unedited original (a cache
  hit when the generation run used the same model) plus one report per target, with the
  target's fields and text replaced inside the original `evidence_shown.json` order. The
  rebuilt unperturbed prompt must equal the original `prompt.txt` byte for byte. Greedy
  decoding, same template. `ehr_rag_enforced` shares the base call with `ehr_rag` (same
  prompt) and then runs the frozen checker + one rewrite per failing line, with the checker
  vocabulary built from the edited units (`--checker-sha256` guards the files). Running the
  originals inside the audit run makes each run self-contained, and is how the enforced
  originals exist even when the generation run had no `ehr_rag_enforced`.
- **Sharding** is over admissions (`--shard-size` / `--shard-index`, `scripts/audit_array.sbatch`).
  The printed plan counts distinct prompts (= report calls) with how many are cached;
  rewrites and format retries are bounded but not known in advance.
- **Pairs and labels (`analyze-audit`):** a pair is (an original claim citing the edited unit,
  the edit), per condition; the original is that condition's unedited report from the audit run
  (so enforced pairs come from the enforced original). Claims are the checker's, list items
  keeping their opening words. Each pair is matched to the regenerated claim citing the same
  id in the same section with the highest difflib ratio (then any section). Signals use the
  checker's number and direction extraction: a number or direction claim that holds only for
  the edited unit is a follows-signal; only for the original, a stale-signal. Follows = only
  follows-signals; Stale = only stale-signals; Dropped = no regenerated claim cites the unit;
  Other = cites it with neither or both. Consequence of the checker's normal rule (last point
  unmarked): "potassium was normal" about a unit whose last value was already normal is not
  discriminating and is Other. `original_states_value_or_direction` marks pairs whose
  original claim stated a checkable value at all.
- **Faithfulness** = Follows / (Follows + Stale), pooled over pairs; 95% CI from 1,000 bootstrap
  resamples of admissions (`ehr_rag/stats.py`, seeded, percentile interval). Plain vs enforced
  is paired: the difference is computed on the same resampled admissions (those with pairs in
  both conditions).
- **Lab-derived terms:** for to_normal edits, the lexicon terms of the unit's lab group and
  original direction (hyperkalemia for a high potassium); present = named anywhere in the
  original Impression + Recommendations; survives = still named after the edit.
- **Null-edit churn:** per section, original non-abstention lines that do not appear verbatim
  in the same section of the null-edited regeneration, and whether the edited uncited unit is
  cited afterwards.
- **Manual check:** 50 pairs, equal allocation over (condition, label) strata, round-robin,
  seeded per stratum.

## Paper tables (`make-tables`)

- Each table is written only when its source run is given; stdout gets row counts only.
- Every rate is a pooled ratio with a 95% CI from `audit.bootstrap_resamples` (1,000)
  bootstrap resamples of admissions, seeded. Table 1 recomputes the rates from the evaluation's
  `sentences.jsonl` (same definitions as `evaluate`, plus the CIs); "manual unsupported" is
  human_support = unsupported over labeled rows, not_checkable excluded. Table 3 uses the
  per-report enforcement counts that `check-citations` now writes (`enforcement_per_report.csv`).
  Table 5 is the mean F1 per report. Groups with no sentences are left out.
