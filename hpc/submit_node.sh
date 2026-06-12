#!/bin/bash
# Whole-node job: grab all 4 A100s on one node, shard planes across the GPUs,
# then run the volumetric merge. Computes on node-local NVMe, then moves the
# results + this job's logs to shared scratch and wipes the node-local copy —
# nothing is left on the node.
#
# Override anything from the command line (no file edits):
#   sbatch --job-name=mk355 \
#          --export=ALL,MBO_DEST=$MBO_USER/results,MBO_OPS='{"diameter":3}' \
#          "$MBO_REPOS/mbo_utilities/hpc/submit_node.sh"
# Result lands in  <MBO_DEST>/YYYY_MM_DD_<job-name>  (a _2, _3 ... suffix if it exists).
# Source your mbo env file first and submit from a dir with a logs/ subdir.
#SBATCH --job-name=s2p-node
#SBATCH --partition=hpc_a100_a
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@120

set -euo pipefail
: "${MBO_REPOS:?source your mbo env file first (defines MBO_REPOS/MBO_LBM/MBO_USER)}"
PROJECT="${MBO_PROJECT:-$MBO_REPOS/mbo_utilities}"
source "$PROJECT/.venv/bin/activate"

JOB="${SLURM_JOB_NAME:-s2p-node}"
JID="${SLURM_JOB_ID:-local}"
SUBMIT="${SLURM_SUBMIT_DIR:-$PWD}"

# Compute on node-local NVMe; final home on shared scratch (override with MBO_DEST).
WORK="${TMPDIR:-/tmp}/${JOB}_${JID}"
DEST_ROOT="${MBO_DEST:-${MBO_USER:?set MBO_DEST or source env (MBO_USER)}/results}"
mkdir -p "$WORK"

export MBO_INPUT="${MBO_INPUT:-$MBO_LBM/2025-07-27_mk355/raw}"
export MBO_OUTPUT="$WORK"

# 4 GPUs share 48 CPUs: cap per-worker BLAS threads so concurrent workers don't oversubscribe.
export MBO_THREADS_PER_WORKER="${MBO_THREADS_PER_WORKER:-3}"
export OMP_NUM_THREADS="$MBO_THREADS_PER_WORKER"
export MKL_NUM_THREADS="$MBO_THREADS_PER_WORKER"

# Move results + logs to a dated, non-colliding folder, then wipe node-local /tmp.
# Runs on normal exit, on pipeline error, and (best effort) on timeout/scancel,
# so the node is left clean either way.
_moved=0
finish() {
  [ "$_moved" = 1 ] && return; _moved=1
  local base="$DEST_ROOT/$(date +%Y_%m_%d)_${JOB}" final n=2
  final="$base"
  while [ -e "$final" ]; do final="${base}_${n}"; n=$((n + 1)); done
  mkdir -p "$final"
  cp -a "$WORK"/. "$final"/ 2>/dev/null || true
  cp -a "$SUBMIT/logs/${JOB}_${JID}".{out,err} "$final"/ 2>/dev/null || true
  rm -rf "$WORK"
  echo "moved results + logs to $final"
}
trap finish EXIT TERM

python "$PROJECT/hpc/run_pipeline.py" --gpus 0,1,2,3
