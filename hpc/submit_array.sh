#!/bin/bash
# Array body: each task processes one plane shard (F planes) on its own GPU,
# computing on node-local NVMe and transferring its shard + log into the shared
# folder set by submit_all.sh (MBO_DEST_DIR). The volumetric merge runs once in
# submit_aggregate.sh after all tasks succeed. Run only via submit_all.sh.
#SBATCH --job-name=s2p-arr
#SBATCH --partition=hpc_a100_a
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --signal=B:TERM@120

set -euo pipefail
: "${MBO_REPOS:?source your mbo env file first (defines MBO_REPOS/MBO_LBM)}"
: "${MBO_DEST_DIR:?run via submit_all.sh (it sets the shared output folder)}"
PROJECT="${MBO_PROJECT:-$MBO_REPOS/mbo_utilities}"
source "$PROJECT/.venv/bin/activate"

JOB="${SLURM_JOB_NAME:-s2p-arr}"
JID="${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
SUBMIT="${SLURM_SUBMIT_DIR:-$PWD}"

WORK="${TMPDIR:-/tmp}/${JOB}_${JID}"
mkdir -p "$WORK"

export MBO_INPUT="${MBO_INPUT:-$MBO_LBM/2025-07-27_mk355/raw}"
export MBO_OUTPUT="$WORK"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Transfer this shard + log into the shared folder, then wipe node-local /tmp.
_moved=0
finish() {
  [ "$_moved" = 1 ] && return; _moved=1
  mkdir -p "$MBO_DEST_DIR"
  cp -a "$WORK"/. "$MBO_DEST_DIR"/ 2>/dev/null || true
  cp -a "$SUBMIT/logs/${JOB}_${JID}".{out,err} "$MBO_DEST_DIR"/ 2>/dev/null || true
  rm -rf "$WORK"
  echo "moved shard + log to $MBO_DEST_DIR"
}
trap finish EXIT TERM

srun python "$PROJECT/hpc/run_pipeline.py"
