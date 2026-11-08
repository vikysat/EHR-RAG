#!/bin/bash
#SBATCH --job-name=ehr-rag-generate-shard
#SBATCH --partition=h100-mig
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%A_%a.out
#
# Sharded generation: one SLURM array task per shard of SHARD_SIZE admissions (default 25).
# Same partition, GPU and resources as scripts/generate.sbatch. Submit from the repository
# root (runs/ must exist). --array is REQUIRED on the command line: indices 0..ceil(N/25)-1
# for N admissions. Every argument is passed through to `python -m ehr_rag.cli generate`,
# plus --shard-size and --shard-index (= SLURM_ARRAY_TASK_ID):
#
#   dev  (20 admissions,  1 shard):  sbatch --array=0   scripts/generate_array.sbatch --config config/full.yaml --split dev  --retrieval-run <id> --yes
#   test (100 admissions, 4 shards): sbatch --array=0-3 scripts/generate_array.sbatch --config config/full.yaml --split test --retrieval-run <id> --yes
#
# Each shard writes its own run folder runs/<date>-generate-job<array job id>-shard<k>/ (with
# nvidia-smi.txt, slurm_env.txt, stdout.log, stderr.log). Evaluate all shards together:
#   sbatch scripts/cli.sbatch evaluate --config config/full.yaml --generation-run $(ls runs | grep -- "-generate-job<array job id>-shard")
# An index beyond the last shard exits with status 2 without generating anything. Models are
# read from the local HF cache (HF_HOME must be set; HF_HUB_OFFLINE=1 is forced here).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "not an array job: submit with sbatch --array=0-<last shard> scripts/generate_array.sbatch ..." >&2
    exit 2
fi
if [ ! -f .venv/bin/activate ]; then
    echo "no .venv in $(pwd); submit from the repository root" >&2
    exit 2
fi
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

SHARD_SIZE="${SHARD_SIZE:-25}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
RUN_ID="$(date +%Y%m%d-%H%M%S)-generate-job${SLURM_ARRAY_JOB_ID}-shard${SHARD_INDEX}"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

nvidia-smi > "$RUN_DIR/nvidia-smi.txt" 2>&1 || echo "nvidia-smi failed (exit $?)" >> "$RUN_DIR/nvidia-smi.txt"
{
    echo "job_id=${SLURM_JOB_ID:-}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "array_task_id=${SHARD_INDEX}"
    echo "shard_size=${SHARD_SIZE}"
    echo "node=$(hostname)"
    echo "partition=${SLURM_JOB_PARTITION:-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "hf_home=${HF_HOME:-<unset>}"
    echo "hf_hub_offline=${HF_HUB_OFFLINE}"
    echo "python=$(command -v python)"
    echo "args=$*"
} > "$RUN_DIR/slurm_env.txt"

echo "run folder: $RUN_DIR (shard ${SHARD_INDEX}, ${SHARD_SIZE} admissions per shard)"
set +e
python -m ehr_rag.cli generate --run-id "$RUN_ID" --shard-size "$SHARD_SIZE" --shard-index "$SHARD_INDEX" "$@" \
    > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log"
status=$?
set -e
echo "generate shard ${SHARD_INDEX} exited with status $status (logs in $RUN_DIR)"
exit "$status"
