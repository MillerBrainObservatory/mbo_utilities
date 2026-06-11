#!/bin/bash
# Chain: array over plane shards, then one dependent aggregate job.
# Computes the array size from the plane count and MBO_PLANES_PER_GPU.
#
#   ./submit_all.sh
#   MBO_PLANES_PER_GPU=2 MBO_INPUT=/path/raw MBO_OUTPUT=/path/results ./submit_all.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PROJECT="${MBO_PROJECT:-/lustre/fs8/mbo/scratch/mbo_soft/repos/mbo_distributed}"
source "$PROJECT/.venv/bin/activate"

NTASKS=$(python run_pipeline.py --print-num-tasks)
echo "array size: $NTASKS task(s) (planes / MBO_PLANES_PER_GPU)"

ARRAY_JID=$(sbatch --parsable --array=0-$((NTASKS - 1)) submit_array.sh)
echo "submitted array job $ARRAY_JID"

AGG_JID=$(sbatch --parsable --dependency=afterok:"$ARRAY_JID" submit_aggregate.sh)
echo "submitted aggregate job $AGG_JID (runs after $ARRAY_JID succeeds)"
