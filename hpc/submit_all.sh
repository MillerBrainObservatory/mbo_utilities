#!/bin/bash
# Chain: array over plane shards, then one dependent aggregate. Both target ONE
# shared dated folder on scratch (decided here once) so the cross-node merge can
# see every shard. Each array task still computes on node-local NVMe and transfers
# its shard into that folder; the aggregate reads it directly.
#
# Run from a writable dir with a logs/ subdir (e.g. $MBO_USER), env file sourced:
#   MBO_NAME=mk355 ./submit_all.sh
#   MBO_NAME=mk355 MBO_PLANES_PER_GPU=2 MBO_INPUT=/path/raw ./submit_all.sh
# Result lands in  <MBO_DEST>/YYYY_MM_DD_<MBO_NAME>  (a _2, _3 ... suffix if it exists).
set -euo pipefail
: "${MBO_REPOS:?source your mbo env file first (defines MBO_REPOS/MBO_LBM/MBO_USER)}"
mkdir -p logs

PROJECT="${MBO_PROJECT:-$MBO_REPOS/mbo_utilities}"
source "$PROJECT/.venv/bin/activate"

export MBO_INPUT="${MBO_INPUT:-$MBO_LBM/2025-07-27_mk355/raw}"

NAME="${MBO_NAME:-s2p}"
DEST_ROOT="${MBO_DEST:-${MBO_USER:?set MBO_DEST or source env (MBO_USER)}/results}"
base="$DEST_ROOT/$(date +%Y_%m_%d)_${NAME}"
FINAL="$base"; n=2
while [ -e "$FINAL" ]; do FINAL="${base}_${n}"; n=$((n + 1)); done
mkdir -p "$FINAL"
echo "output folder: $FINAL"

NTASKS=$(python "$PROJECT/hpc/run_pipeline.py" --print-num-tasks)
echo "array size: $NTASKS task(s) (planes / MBO_PLANES_PER_GPU)"

# Optional partition/gres override (e.g. MBO_PARTITION=hpc_l40s MBO_GRES=gpu:l40s:1).
SB=()
[ -n "${MBO_PARTITION:-}" ] && SB+=(--partition="$MBO_PARTITION")
[ -n "${MBO_GRES:-}" ] && SB+=(--gres="$MBO_GRES")

ARRAY_JID=$(sbatch --parsable --job-name="${NAME}-arr" --array=0-$((NTASKS - 1)) \
  ${SB[@]+"${SB[@]}"} --export=ALL,MBO_DEST_DIR="$FINAL" "$PROJECT/hpc/submit_array.sh")
echo "submitted array job $ARRAY_JID"

AGG_JID=$(sbatch --parsable --job-name="${NAME}-agg" --dependency=afterok:"$ARRAY_JID" \
  ${SB[@]+"${SB[@]}"} --export=ALL,MBO_DEST_DIR="$FINAL" "$PROJECT/hpc/submit_aggregate.sh")
echo "submitted aggregate job $AGG_JID (runs after $ARRAY_JID succeeds)"
