#!/bin/bash
# Single job: all planes on one GPU. Computes on node-local NVMe, writes its log
# AND results into one dated folder on shared scratch, wipes the node-local copy.
# One scp of that folder gets everything. Nothing is written to the CWD or repo.
#
#   sbatch --job-name=mk355 --export=ALL,MBO_DEST=$MBO_USER/results,MBO_OPS='{"diameter":3}' submit.sh
# Result lands in  <MBO_DEST>/YYYY_MM_DD_<job-name>  (a _2, _3 ... suffix if it exists).
#SBATCH --job-name=s2p
#SBATCH --partition=hpc_a100_a
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --signal=B:TERM@120

set -euo pipefail
: "${MBO_REPOS:?source your mbo env file first (defines MBO_REPOS/MBO_LBM/MBO_USER)}"
PROJECT="${MBO_PROJECT:-$MBO_REPOS/mbo_utilities}"
source "$PROJECT/.venv/bin/activate"

JOB="${SLURM_JOB_NAME:-s2p}"
JID="${SLURM_JOB_ID:-local}"

# One dated output folder on shared scratch; the log lives inside it.
DEST_ROOT="${MBO_DEST:-${MBO_USER:?set MBO_DEST or source env (MBO_USER)}/results}"
base="$DEST_ROOT/$(date +%Y_%m_%d)_${JOB}"; FINAL="$base"; n=2
while [ -e "$FINAL" ]; do FINAL="${base}_${n}"; n=$((n + 1)); done
mkdir -p "$FINAL"
exec > "$FINAL/${JOB}_${JID}.log" 2>&1

# Compute on node-local NVMe, copy results into the folder, wipe /tmp.
WORK="${TMPDIR:-/tmp}/${JOB}_${JID}"
mkdir -p "$WORK"
export MBO_INPUT="${MBO_INPUT:-$MBO_LBM/2025-07-27_mk355/raw}"
export MBO_OUTPUT="$WORK"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

_done=0
finish() {
  [ "$_done" = 1 ] && return; _done=1
  local t0=$SECONDS
  cp -a "$WORK"/. "$FINAL"/ 2>/dev/null || true
  rm -rf "$WORK"
  echo "timing: transfer=$((SECONDS - t0))s total_wall=${SECONDS}s -> $FINAL"
}
trap finish EXIT TERM

srun python "$PROJECT/hpc/run_pipeline.py"
