# RUNBOOK: server runs from Phase 4 through Phase 6

Exact commands, in order, for the Linux SLURM server. Each step says which run id it
produces, which later step consumes it, and what to check. Run every command from the
repository root.

Conventions:
- `<...>` is a value you copy from the output of an earlier step.
- Every command creates `runs/<run_id>/` with `run_id = YYYYMMDD-HHMMSS-<command>[-job<SLURM id>]`.
- The GPU steps go through `sbatch`. CPU steps can run on the login node, or with
  `srun --partition=h100-mig --gres=gpu:nvidia_h100_nvl_3g.47gb:1 --pty bash` if your
  cluster forbids work on login nodes.
- Exit codes: `0` ok, `2` config error, `3` data schema mismatch, `4` failed (read
  `stderr.log` or the error printed), `5` not run because `--yes` was missing.

---

## 0. One-time setup (login node, internet available)

```bash
cd <path to EHR-RAG>
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt          # torch 2.14.0+cu126 is already installed; this adds accelerate
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # expect 2.14.0+cu126 12.6
mkdir -p runs
```

Pre-download every model into `HF_HOME`. GPU nodes run with `HF_HUB_OFFLINE=1`.
Llama is already there; the others are needed for retrieval (MedCPT) and
evaluation (NLI):

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["meta-llama/Llama-3.1-8B-Instruct", "ncbi/MedCPT-Query-Encoder",
          "ncbi/MedCPT-Article-Encoder", "cross-encoder/nli-deberta-v3-base"]:
    print(m, snapshot_download(m))
EOF
```

Check that everything loads offline (this prints revision hashes; any error means a
model is missing from `HF_HOME`):

```bash
HF_HUB_OFFLINE=1 python - <<'EOF'
from transformers import AutoConfig
for m in ["meta-llama/Llama-3.1-8B-Instruct", "ncbi/MedCPT-Query-Encoder",
          "ncbi/MedCPT-Article-Encoder", "cross-encoder/nli-deberta-v3-base"]:
    print(m, AutoConfig.from_pretrained(m)._commit_hash)
EOF
```

Useful SLURM commands: `squeue -u $USER` (queued/running), `sacct -j <jobid>` (state
and exit code), `tail -f runs/<run_id>/stdout.log` (progress), `cat runs/slurm-<jobid>.out`
(prints the run folder name).

If a GPU job hits the 2-hour limit, resubmit it with a longer limit. The option on
the command line overrides the script: `sbatch --time=04:00:00 scripts/...`. Every
model response is cached in `cache/generation/`, so a resubmitted job reuses all
finished calls and writes a new, complete run.

---

## 1. Evidence units (CPU; skip if `data/derived/units/` already exists on the server)

The data are under `data.root` in `config/default.yaml`
(`data/mimic-iv-clinical-database-demo-2.2`). `config/selected_admissions.yaml` is in git.

```bash
python -m ehr_rag.cli build-units
```
Check: the unit-count table lists all 15 admissions, and `data/derived/units/<hadm_id>.jsonl`
exists for each.

## 2. Retrieval (CPU is fine) → RETRIEVE_RUN

Must be run with the current code: `generate` refuses retrieve runs made before
`evidence_units` were saved.

```bash
python -m ehr_rag.cli retrieve
```
The last lines print `run: .../runs/<RETRIEVE_RUN>`. Check:
`ls runs/<RETRIEVE_RUN>/retrieval/ | wc -l` = 15, and
`grep -c evidence_units runs/<RETRIEVE_RUN>/retrieval/*.json` shows 1 per file.

## 3. Context size check (CPU; Llama tokenizer only)

```bash
HF_HUB_OFFLINE=1 python -m ehr_rag.cli context-tokens
```
Check the last line, `N/15 admissions fit max_context_tokens = 32,000`. Admissions
that do not fit are skipped for `full_context` (logged in `skipped.json`); that is
expected behavior, not an error.

---

## 4. Phase 4: generation (GPU) → GEN_RUN

### 4a. Smoke test (2 admissions, ehr_rag only)

```bash
sbatch scripts/generate.sbatch --retrieval-run <RETRIEVE_RUN> --conditions ehr_rag --limit 2
```
When it finishes (`sacct -j <jobid>` shows COMPLETED): `cat runs/slurm-<jobid>.out` gives
the run folder `runs/<SMOKE_RUN>`. Check:
- `runs/<SMOKE_RUN>/stderr.log` is empty or only warnings;
- `runs/<SMOKE_RUN>/nvidia-smi.txt` shows the H100 MIG device;
- `runs/<SMOKE_RUN>/run_meta.json` has `"status": "ok"`, a `gpu_name`, and a `model_revision` hash;
- `runs/<SMOKE_RUN>/ehr_rag/<hadm_id>/raw.txt` looks like a four-section report with `[E..]` citations.

### 4b. Full run (all 15 admissions, all 3 conditions = 45 calls, so `--yes` is required)

```bash
sbatch --time=04:00:00 scripts/generate.sbatch --retrieval-run <RETRIEVE_RUN> --yes
```
The two smoke-test reports come from the cache. The run folder is `runs/<GEN_RUN>`.

### GATE 3 checks
- `tail -n 8 runs/<GEN_RUN>/stdout.log` shows the per-condition table
  `ok / retry_ok / format_error / skipped`.
- Two ehr_rag reports: `runs/<GEN_RUN>/ehr_rag/<hadm_id>/raw.txt` with their
  `evidence_shown.json` (and `prompt.txt` for the evidence block exactly as shown).
- One no_retrieval report: `runs/<GEN_RUN>/no_retrieval/<hadm_id>/raw.txt`. Every line
  should end in `[NONE]`; note any patient-specific facts it invents.
- `runs/<GEN_RUN>/skipped.json` lists the full_context admissions that did not fit.
- For any `format_error`: compare `raw.txt` and `raw_retry.txt`, and read `parsed.json` → `problems`.

---

## 5. Phase 5: evaluation (GPU for NLI speed; CPU also works) → EVAL_RUN

```bash
sbatch scripts/cli.sbatch evaluate --generation-run <GEN_RUN>
```
Run folder: `runs/<EVAL_RUN>` (see `runs/slurm-<jobid>.out`).

### GATE 4 checks
- `runs/<EVAL_RUN>/stdout.log` has:
  - the NLI model line, whose `id2label` must include `entailment`;
  - the metrics table per condition for ALL, History+Findings and
    Impression+Recommendations;
  - **10 random sentences**, each with its parsed citation ids, the cited unit texts
    and the NLI label. Check each by hand: are the ids parsed correctly, are the texts
    the ones the model saw, is the NLI label plausible?
- `runs/<EVAL_RUN>/metrics_summary.md`: the same table in markdown, plus the format
  status per condition.
- `runs/<EVAL_RUN>/metrics.json` → `nli.n_too_long_pairs`, and `nli_too_long.csv`:
  pairs over 512 tokens are labeled `too_long` and count as not entailed (nothing is
  truncated). If there are many, recall is understated for claims citing many units.
- Sanity: `no_retrieval` should have `n_citations = 0` and `cited_rate` 0 or n/a;
  `invalid_citation_rate` shows fabricated ids.

### Manual review and verifier agreement (after Gate 4)
1. Copy `runs/<EVAL_RUN>/manual_review.csv`, open it in a spreadsheet, and fill
   `human_support` with `supported`, `partial`, `unsupported` or `not_checkable`
   (plus optional `human_error_type`, `notes`). Keep the other columns unchanged. Save
   it as CSV (UTF-8), for example `runs/<EVAL_RUN>/manual_review_labeled.csv`.
2. Compute agreement:
   ```bash
   python -m ehr_rag.cli verifier-agreement --labels runs/<EVAL_RUN>/manual_review_labeled.csv
   ```
   The file must stay inside `runs/<EVAL_RUN>/` so the NLI entailment label is read
   from its `metrics.json`. The output gives raw agreement, Cohen's kappa, and the
   binary and full confusion matrices.

(ROUGE-L / BERTScore: `text-metrics` exits because the Demo has no reference reports.
That is intended.)

---

## 6. Phase 6: perturbation pilot

### 6a. Select targets and compute edits (CPU) → TARGETS_RUN

```bash
python -m ehr_rag.cli select-targets --generation-run <GEN_RUN>
```
Check the printed output:
- number of candidates, eligible candidates and targets (up to 15; at most 2 per admission);
- the targets table (unit, value, sentence);
- for every applied edit, `before` / `after` text: the value changes, markers
  `[HIGH]`/`[LOW]` update for lab_major, decimals are kept;
- `SKIPPED` lines with reasons (for example "inconsistent"). Skips are logged, not errors.

Files: `runs/<TARGETS_RUN>/targets.csv`, `edits.jsonl`, `skipped_edits.csv`, `candidates.csv`.

### 6b. Frozen-evidence regenerations (GPU) → PILOT_RUN

Number of calls = applied edits + 3 controls per target (roughly 80–90), so `--yes` is required:

```bash
sbatch --time=04:00:00 scripts/cli.sbatch run-pilot --targets-run <TARGETS_RUN> --yes
```
If it stops with "rebuilt unperturbed prompt differs from the original prompt.txt",
the prompt template, header or units changed since GEN_RUN. Do not work around this;
regenerate from step 4 instead.

Check: `runs/<PILOT_RUN>/run_meta.json` has `"status": "ok"`, and
`runs/<PILOT_RUN>/cases.csv` lists every case with its format status.

### 6c. Outcome labels (CPU) → ANALYSIS_RUN

```bash
python -m ehr_rag.cli analyze-perturbation --pilot-run <PILOT_RUN>
```

### GATE 5 checks
- The printed summary (also `runs/<ANALYSIS_RUN>/perturbation_summary.md`): for each
  edit type, outcome rates as `k/n (%)`; control stability; for delete, how often the
  deleted id is still cited.
- The printed worked examples, up to 3 per edit type: original sentence → edit →
  regenerated sentence with citations.
- `runs/<ANALYSIS_RUN>/perturbation_manual_check.csv`: every case. Fill `human_outcome`
  (followed / updated_miscited / stale / dropped / ambiguous, or abstained / stale for
  delete) and `notes`. Read every row with `needs_manual_read = True` (all lab_major
  and all ambiguous cases) together with `diffs/<case_id>.diff`, which shows the
  Impression/Recommendations changes.
- The automated labels are provisional until you have checked them.

---

## 7. Run-id map

| step | command | produces | consumed by |
|---|---|---|---|
| 2 | `retrieve` | RETRIEVE_RUN | 4a, 4b (`--retrieval-run`) |
| 4b | `generate.sbatch` | GEN_RUN | 5 (`--generation-run`), 6a (`--generation-run`) |
| 5 | `cli.sbatch evaluate` | EVAL_RUN | `verifier-agreement --labels runs/<EVAL_RUN>/...` |
| 6a | `select-targets` | TARGETS_RUN | 6b (`--targets-run`) |
| 6b | `cli.sbatch run-pilot` | PILOT_RUN | 6c (`--pilot-run`) |
| 6c | `analyze-perturbation` | ANALYSIS_RUN | your manual check; demo notebook |

## 8. Demo notebook (Phase 7)

After step 6c, open `notebooks/demo.ipynb` (install Jupyter separately: `pip install
jupyterlab`) and run it top to bottom. By default it uses the latest run of each kind.
You can also set the run ids in its first code cell. It makes no model calls. To run
it on another machine, copy `runs/<RETRIEVE_RUN>`, `runs/<GEN_RUN>`, `runs/<EVAL_RUN>`,
`runs/<TARGETS_RUN>`, `runs/<PILOT_RUN>` and `runs/<ANALYSIS_RUN>` into that
checkout's `runs/`, plus `data/derived/units/` if you want the unit views.

## 9. Troubleshooting

| symptom | cause / fix |
|---|---|
| `couldn't connect to 'https://huggingface.co'` in stderr.log | a model is missing from `HF_HOME`; step 0 download on the login node |
| `hf_local requires CUDA` | job not on a GPU node; submit with the sbatch scripts |
| `NOT RUN: ... re-run with --yes` (exit 5) | add `--yes` |
| `... saved by an older retrieve; re-run retrieve` | redo step 2, then use the new RETRIEVE_RUN |
| `differs from the retrieval run` / `units were rebuilt?` | units changed after retrieval or generation; redo from step 2 (or 4) |
| job TIMEOUT in `sacct` | resubmit with `sbatch --time=04:00:00 ...`; cached calls are reused |
| CUDA out of memory | the 3g.47gb MIG slice fits Llama-3.1-8B in bf16; check `nvidia-smi.txt` for other processes and resubmit |
| `cohort hashes already exist` (exit 2) | the cohort is fixed; `--overwrite` only if you really want a new one |
| `SHA-256 differs` | an id list under `data/cohort/` was changed; re-run `select-cohort` (same seed gives the same lists) |
| `clinical notes found under the evidence data root` | a note file is under `data.root`; notes belong only in `data/cohort/reference/` |

---

## 10. Full MIMIC-IV cohort (dev 20 / test 100)

All commands use `--config config/full.yaml`. With `data.source: full` every command
prints counts only; ids lists, Parquet, units, notes and run outputs stay under the
gitignored `data/`, `cache/` and `runs/`. Generation stays local (`hf_local`).

### 10.0 Setup and download check (login node, internet available)

```bash
cd <path to EHR-RAG>
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt            # adds duckdb and pyarrow
ln -s <MIMIC-IV root containing hosp/>      data/mimiciv
ln -s <MIMIC-IV-Note root containing note/> data/mimic-iv-note
ls data/mimiciv/hosp/labevents.csv.gz data/mimic-iv-note/note/discharge.csv.gz \
   data/mimic-iv-clinical-database-demo-2.2/demo_subject_id.csv
```

Pre-download every model into `HF_HOME` (roberta-large is new: BERTScore):

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["meta-llama/Llama-3.1-8B-Instruct", "ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder",
          "cross-encoder/nli-deberta-v3-base", "roberta-large"]:
    print(m, snapshot_download(m))
EOF
```

Download check, offline (every line must print a revision hash; any error = missing):

```bash
HF_HUB_OFFLINE=1 python - <<'EOF'
from transformers import AutoConfig, AutoModel, AutoTokenizer
for m in ["meta-llama/Llama-3.1-8B-Instruct", "ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder",
          "cross-encoder/nli-deberta-v3-base", "roberta-large"]:
    print(m, AutoConfig.from_pretrained(m)._commit_hash)
AutoTokenizer.from_pretrained("roberta-large", local_files_only=True)
AutoModel.from_pretrained("roberta-large", local_files_only=True)
print("roberta-large loads offline")
EOF
python -c "import duckdb, pyarrow; print(duckdb.__version__, pyarrow.__version__)"   # 1.5.6 26.0.1
```

### 10.1 Cohort selection (CPU; scans labevents and discharge notes once)

```bash
python -m ehr_rag.cli select-cohort --config config/full.yaml
```
Check the count after every filter; the last two counts must be `dev admissions 20`
and `test admissions 100`. It writes `data/cohort/{dev,test}_hadm_ids.txt` (gitignored) and
`config/cohort/{dev,test}_hadm_ids.sha256`. Commit only the hashes:
```bash
git add config/cohort/dev_hadm_ids.sha256 config/cohort/test_hadm_ids.sha256
git status --short        # nothing else under config/cohort/, data/, cache/, runs/
git commit -m "Cohort hashes (dev 20, test 100)" && git push origin main
```

### 10.2 Extraction (CPU)

```bash
python -m ehr_rag.cli extract-cohort --config config/full.yaml
```
Check: row counts per table, `discharge: admissions without a note 0`, and
`references written 120`. Output: `data/cohort/evidence/hosp/*.parquet` and
`data/cohort/reference/{discharge.parquet,references.csv}`.

### 10.3 Schema check (headers only)

```bash
python -m ehr_rag.cli cohort-schema-check --config config/full.yaml
```
Expect exit 0 and `OK`. `full_not_in_demo` lists newer-release columns that extraction
drops (fine). Any `demo_not_in_full` / `builder_not_in_full` or `extracted Parquet
columns DIFFERENT` is exit 3: stop and report. (This also works before 10.2.)

### 10.4 Evidence units (CPU)

```bash
python -m ehr_rag.cli build-units --config config/full.yaml --split dev test
```
Check the unit-count table (120 admissions) and the dropped summary. Units go to
`data/derived_full/units/`.

### 10.5 Retrieval on dev (CPU) → RETRIEVE_DEV

```bash
HF_HUB_OFFLINE=1 python -m ehr_rag.cli retrieve --config config/full.yaml --split dev
HF_HUB_OFFLINE=1 python -m ehr_rag.cli context-tokens --config config/full.yaml --split dev   # optional
```
The last lines print `run: .../runs/<RETRIEVE_DEV>`.

### 10.6 Generation on dev (GPU, job array; 20 admissions = 1 shard of 25)

```bash
sbatch --array=0 scripts/generate_array.sbatch --config config/full.yaml --split dev \
    --retrieval-run <RETRIEVE_DEV> --yes
```
20 admissions × 3 conditions = 60 calls (> `confirm_threshold`, hence `--yes`).
`cat runs/slurm-<jobid>_0.out` gives the run folder
`runs/<date>-generate-job<jobid>-shard0`; its `stdout.log` starts with the call plan and
ends with the format-status table. Test later: `--split test`, its own retrieve run, and
`--array=0-3` (100 admissions = 4 shards of 25).

### 10.7 Evaluation (after generation)

```bash
SHARDS=$(ls runs | grep -- "-generate-job<jobid>-shard")
sbatch scripts/cli.sbatch evaluate --config config/full.yaml --generation-run $SHARDS
sbatch scripts/cli.sbatch text-metrics --config config/full.yaml --generation-run $SHARDS \
    --references data/cohort/reference/references.csv --bertscore
```
`evaluate` writes `gate_sample.json` instead of printing sentences;
`text-metrics` prints counts and per-condition mean/median ROUGE-L and BERTScore F1.
