#!/usr/bin/env bash
# Every remaining full-data step in order (src/ehr_rag/run_all.py has the step list and the rules).
# For the GPU machine (plain Jupyter terminal, no Slurm). From the repository root:
#
#   nohup scripts/run_all.sh > /dev/null 2>&1 &       # run (or re-run after a session ended)
#   scripts/run_all.sh --dry-run                       # print the plan: skip / resume / start per step
#   tail -f ~/scratch/logs/run_all.log
#
# Re-running is safe: finished steps are skipped, run-probe resumes, generation reuses its cache; the
# retrieval metrics and the final tables (step 12) run again whenever their inputs (the labeled sheets) change.
# The end of every run lists the labeling sheets still to label and the `label` commands for them.
# Stops at the first error. Log: ~/scratch/logs/run_all.log; run ids: ~/scratch/logs/run_all_ids.txt.
# RUN_ALL_LOG_DIR overrides ~/scratch/logs. Extra arguments go to `python -m ehr_rag.run_all`
# (--config, --references, --dry-run).
set -euo pipefail

cd "$(dirname "$0")/.."
LOG_DIR="${RUN_ALL_LOG_DIR:-$HOME/scratch/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_all.log"

# one instance at a time: a second start (e.g. a re-run while the first is still alive) exits at once
if command -v flock > /dev/null; then
    exec 9>"$LOG_DIR/run_all.lock"
    if ! flock -n 9; then
        echo "run_all is already running (lock $LOG_DIR/run_all.lock)" | tee -a "$LOG"
        exit 3
    fi
else
    echo "WARNING: flock not found; not guarding against a second instance" | tee -a "$LOG"
fi

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

echo "=== run_all start $(date -Iseconds) args: $*" | tee -a "$LOG"
rc=0
python -m ehr_rag.run_all --ids-file "$LOG_DIR/run_all_ids.txt" "$@" 2>&1 | tee -a "$LOG" || rc=$?
echo "=== run_all end $(date -Iseconds) exit status $rc" | tee -a "$LOG"
exit "$rc"
