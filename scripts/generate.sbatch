#!/bin/bash
#SBATCH --job-name=ehr-rag-generate
#SBATCH --partition=h100-mig
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%j.out
#
# Phase 4 generation on a GPU node. Submit from the repository root (runs/ must exist);
# every argument is passed through to `python -m ehr_rag.cli generate`:
#
#   sbatch scripts/generate.sbatch --retrieval-run <retrieve_run_id> [--conditions ...] [--limit N] [--yes]
#
# Smoke test (2 admissions, ehr_rag only):
#   sbatch scripts/generate.sbatch --retrieval-run <retrieve_run_id> --conditions ehr_rag --limit 2
#
# The run folder runs/<run_id>/ gets nvidia-smi.txt, slurm_env.txt, stdout.log and
# stderr.log next to the usual run outputs. Models are read from the local HF cache
# (HF_HOME must already be set in the environment; HF_HUB_OFFLINE=1 is forced here).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
if [ ! -f .venv/bin/activate ]; then
    echo "no .venv in $(pwd); submit from the repository root" >&2
    exit 2
fi
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

RUN_ID="$(date +%Y%m%d-%H%M%S)-generate-job${SLURM_JOB_ID:-local}"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

nvidia-smi > "$RUN_DIR/nvidia-smi.txt" 2>&1 || echo "nvidia-smi failed (exit $?)" >> "$RUN_DIR/nvidia-smi.txt"
{
    echo "job_id=${SLURM_JOB_ID:-}"
    echo "node=$(hostname)"
    echo "partition=${SLURM_JOB_PARTITION:-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "hf_home=${HF_HOME:-<unset>}"
    echo "hf_hub_offline=${HF_HUB_OFFLINE}"
    echo "python=$(command -v python)"
    echo "args=$*"
} > "$RUN_DIR/slurm_env.txt"

echo "run folder: $RUN_DIR"
set +e
python -m ehr_rag.cli generate --run-id "$RUN_ID" "$@" > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log"
status=$?
set -e
echo "generate exited with status $status (logs in $RUN_DIR)"
exit "$status"
