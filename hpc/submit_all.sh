#!/bin/bash
# Chain: array over plane shards, then one dependent aggregate. Both target ONE
# dated folder on scratch (decided here once); SLURM writes every task log into it
# and the shards transfer their results into it, so one scp gets logs + results.
# Nothing is written to the CWD or repo.
#   MBO_NAME=mk355 ./submit_all.sh
#   MBO_NAME=mk355 MBO_PARTITION=hpc_l40s MBO_GRES=gpu:l40s:1 MBO_OPS='{"diameter":3}' ./submit_all.sh
set -euo pipefail
: "${MBO_REPOS:?source your mbo env file first (defines MBO_REPOS/MBO_LBM/MBO_USER)}"
PROJECT="${MBO_PROJECT:-$MBO_REPOS/mbo_utilities}"
# Python env to activate; override with MBO_ENV (default: the shared mbo env).
source "${MBO_ENV:-/lustre/fs8/mbo/scratch/mbo_soft/envs/mbo}/bin/activate"

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
# Reserve the whole node so other tenants' CPU jobs can't contend (uniform timing).
[ -n "${MBO_EXCLUSIVE:-}" ] && SB+=(--exclusive)

ARRAY_JID=$(sbatch --parsable --job-name="${NAME}-arr" --array=0-$((NTASKS - 1)) \
  --output="$FINAL/%x_%A_%a.out" --error="$FINAL/%x_%A_%a.err" \
  ${SB[@]+"${SB[@]}"} --export=ALL,MBO_DEST_DIR="$FINAL" "$PROJECT/hpc/submit_array.sh")
echo "submitted array job $ARRAY_JID"

AGG_JID=$(sbatch --parsable --job-name="${NAME}-agg" --dependency=afterok:"$ARRAY_JID" \
  --output="$FINAL/%x_%j.out" --error="$FINAL/%x_%j.err" \
  ${SB[@]+"${SB[@]}"} --export=ALL,MBO_DEST_DIR="$FINAL" "$PROJECT/hpc/submit_aggregate.sh")
echo "submitted aggregate job $AGG_JID (runs after $ARRAY_JID succeeds)"
