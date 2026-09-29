# EHR-RAG

EHR-aware clinical RAG with citation-faithfulness probes. See `SPEC.md` for the
full specification. This README is a stub and is completed in Phase 7.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.11+ for the pinned dependencies (pandas 3.x); the code itself
avoids syntax newer than 3.10.

### GPU server (CUDA 12.x)

The server's NVIDIA driver supports CUDA 12.x only, so torch **must** be installed
from the CUDA 12.6 index `https://download.pytorch.org/whl/cu126`, **before** the
rest of the requirements (otherwise pip installs a different torch build from PyPI):

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt   # torch==2.14.0 is already satisfied and is not replaced
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expected: 2.14.0+cu126 12.6 True
```

Do not use the cu128/cu13x indexes on this server. faiss stays `faiss-cpu` on the
server too.

## Data

Download the MIMIC-IV Clinical Database Demo (v2.2, open access) from PhysioNet
and point `data.root` in `config/default.yaml` at the extracted directory.

## Usage

```bash
python -m ehr_rag.cli --help
```

### Phase 4: generation

Backends (`generation.backend`): `hf_local` (default; Hugging Face transformers,
in-process, bf16 on CUDA, the model's chat template, greedy decoding),
`anthropic`, `openai_compatible`. API backends read the key from the environment
variable named in `generation.api_key_env`; `hf_local` needs no key and refuses to
run without CUDA. The `anthropic` backend cannot set a temperature (the pinned SDK
and current Claude models do not accept one); `call.json` records
`temperature_applied: false` for it.

`ehr_rag` reuses the ordered evidence saved by a `retrieve` run (never re-retrieved),
so run retrieval first and pass its run id:

```bash
python -m ehr_rag.cli retrieve                     # -> runs/<retrieve_run_id>/retrieval/<hadm_id>.json
python -m ehr_rag.cli context-tokens               # full_context prompt tokens vs generation.max_context_tokens
python -m ehr_rag.cli generate --retrieval-run <retrieve_run_id> [--conditions ...] [--limit N] [--yes] [--no-cache]
```

`generate` prints the number of calls and estimated input tokens first and requires
`--yes` when new (uncached) calls exceed `generation.confirm_threshold`. Responses are
cached in `cache/generation/`. Output per condition and admission:
`runs/<id>/<condition>/<hadm_id>/{prompt.txt, raw.txt, parsed.json, evidence_shown.json, call.json}`
(plus `*_retry.txt` after a format retry and `prompt_templated.txt` for `hf_local`).

On the SLURM GPU server (submit from the repo root; the model must already be in `HF_HOME`,
the job forces `HF_HUB_OFFLINE=1`):

```bash
mkdir -p runs
sbatch scripts/generate.sbatch --retrieval-run <retrieve_run_id> --conditions ehr_rag --limit 2   # smoke test
sbatch scripts/generate.sbatch --retrieval-run <retrieve_run_id> --yes                            # all conditions
```

The job writes `nvidia-smi.txt`, `slurm_env.txt`, `stdout.log` and `stderr.log` into the
run folder; SLURM's own log goes to `runs/slurm-<jobid>.out`.

### Phase 5: evaluation

```bash
python -m ehr_rag.cli evaluate --generation-run <generate_run_id>        # or: sbatch scripts/cli.sbatch evaluate ...
python -m ehr_rag.cli verifier-agreement --labels runs/<evaluate_run_id>/manual_review.csv   # after you label it
python -m ehr_rag.cli text-metrics --generation-run <id> --references refs.csv  # exits: the Demo has no references
```

### Phase 6: perturbation pilot

```bash
python -m ehr_rag.cli select-targets --generation-run <generate_run_id>          # CPU: targets + edits
sbatch scripts/cli.sbatch run-pilot --targets-run <select_targets_run_id> --yes  # GPU: frozen-evidence regenerations
python -m ehr_rag.cli analyze-perturbation --pilot-run <run_pilot_run_id>        # CPU: provisional labels
```

Outputs: `candidates.csv`, `targets.csv`, `edits.jsonl`, `skipped_edits.csv` (select-targets);
`cases/<target>__<edit|controlK>/...` and `cases.csv` (run-pilot);
`perturbation_results.csv`, `perturbation_summary.md`, `perturbation_manual_check.csv`,
`diffs/` (analyze-perturbation). Automated outcome labels are provisional until checked.
See `RUNBOOK.md` for the full server sequence.

## Metric definitions (as implemented)

Parsing (`eval/parse.py`): sections are split on the exact lines `### History`,
`### Findings`, `### Impression`, `### Recommendations`; every other non-blank line
in a section is one sentence. Only the trailing citation group counts (`[E3]`,
`[E3, E7]`, `[E3][E7]`, `[NONE]`); ids are de-duplicated per sentence. Citations
elsewhere in a line are flagged `mid_sentence_citation` and ignored; lines with a
sentence boundary (`.`/`!`/`?` + space + capital letter) are flagged
`multiple_sentences`. Missing, duplicate, extra and out-of-order sections and text
before the first header are flagged per report.

Metrics (`eval/citations.py`) use every sentence of the four sections except
abstention lines `Insufficient evidence. [NONE]` (reported as `n_abstentions`).
A citation is *valid* if its id is in the report's `evidence_shown.json`.

- `cited_rate`: sentences with at least one valid citation / sentences.
- `invalid_citation_rate`: E-id citations not in `evidence_shown.json` / all E-id
  citations (`[NONE]` is not a citation).
- `citation_recall` (ALCE, Gao et al. 2023): a sentence is supported iff it has at
  least one valid citation and NLI(premise = texts of its valid cited units joined by
  a space in citation order, hypothesis = sentence) is `entailment`. Recall =
  supported / sentences.
- `citation_precision` (ALCE): a valid citation c of a sentence with valid set C is
  irrelevant iff NLI(c alone) is not entailment AND entail(C − {c}) == entail(C),
  with entail(∅) = false. Invalid ids are irrelevant. Precision = relevant citations /
  all E-id citations.
- Aggregates are micro-averages per condition, per section, and for History+Findings
  vs Impression+Recommendations (entailment is less meaningful for the inferential
  sections). A zero denominator gives n/a.
- NLI model `eval.nli_model` (default `cross-encoder/nli-deberta-v3-base`); the label
  order comes from the model's `config.id2label`. Pairs longer than 512 tokens are
  truncated from the premise only and counted (`n_truncated_pairs` in `metrics.json`).
- Reports that ended in `format_error` are included (best-effort parse of the final
  raw output) and counted per condition.

Manual review (`manual_review.csv`): `eval.manual_review_n` sentences, abstentions
excluded, equal allocation over (condition, section) strata, seeded with `eval.seed`.
Fill `human_support` with supported / partial / unsupported / not_checkable.
`verifier-agreement` binarizes (NLI entailment vs human `supported`; partial and
unsupported = not supported; not_checkable excluded) and reports raw agreement,
Cohen's kappa and the full confusion matrix.

ROUGE-L / BERTScore (`text-metrics`) require a `hadm_id,reference` CSV; without one
the command exits. References are never generated. See `DECISIONS.md` for every
judgment call.
