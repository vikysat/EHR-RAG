#!/bin/bash
#SBATCH --job-name=ehr-rag
#SBATCH --partition=h100-mig
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%j.out
#
# Run one GPU-using CLI command that accepts --run-id (evaluate, run-pilot, generate).
# Submit from the repository root (runs/ must exist). The first argument is the
# subcommand; everything after it is passed through:
#
#   sbatch scripts/cli.sbatch evaluate  --generation-run <generate_run_id>
#   sbatch scripts/cli.sbatch run-pilot --targets-run <select_targets_run_id> --yes
#
# The run folder runs/<run_id>/ gets nvidia-smi.txt, slurm_env.txt, stdout.log and
# stderr.log next to the usual outputs. HF_HOME must already be set; HF_HUB_OFFLINE=1
# is forced, so every model must already be in the local HF cache.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch scripts/cli.sbatch <subcommand> [args...]" >&2
    exit 2
fi
CMD="$1"
shift
case "$CMD" in
    evaluate|run-pilot|generate) ;;
    *) echo "subcommand must be one of: evaluate, run-pilot, generate (got '$CMD')" >&2; exit 2 ;;
esac

cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
if [ ! -f .venv/bin/activate ]; then
    echo "no .venv in $(pwd); submit from the repository root" >&2
    exit 2
fi
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

RUN_ID="$(date +%Y%m%d-%H%M%S)-${CMD}-job${SLURM_JOB_ID:-local}"
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
    echo "command=${CMD}"
    echo "args=$*"
} > "$RUN_DIR/slurm_env.txt"

echo "run folder: $RUN_DIR"
set +e
python -m ehr_rag.cli "$CMD" --run-id "$RUN_ID" "$@" > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log"
status=$?
set -e
echo "$CMD exited with status $status (logs in $RUN_DIR)"
exit "$status"
