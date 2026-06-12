#!/bin/bash
# Array body: one plane shard per task on its own GPU, computed on node-local NVMe
# and transferred into the shared folder submit_all.sh set (MBO_DEST_DIR). SLURM
# writes this task's log straight into that folder. Run only via submit_all.sh.
#SBATCH --job-name=s2p-arr
#SBATCH --partition=hpc_a100_a
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --signal=B:TERM@120

set -euo pipefail
: "${MBO_REPOS:?source your mbo env file first (defines MBO_REPOS/MBO_LBM)}"
: "${MBO_DEST_DIR:?run via submit_all.sh (it sets the shared output folder)}"
PROJECT="${MBO_PROJECT:-$MBO_REPOS/mbo_utilities}"
# Python env to activate; override with MBO_ENV (default: the shared mbo env).
source "${MBO_ENV:-/lustre/fs8/mbo/scratch/mbo_soft/envs/mbo}/bin/activate"

JOB="${SLURM_JOB_NAME:-s2p-arr}"
JID="${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

WORK="${TMPDIR:-/tmp}/${JOB}_${JID}"
mkdir -p "$WORK"
export MBO_INPUT="${MBO_INPUT:-$MBO_LBM/2025-07-27_mk355/raw}"
export MBO_OUTPUT="$WORK"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Transfer this shard into the shared folder, then wipe node-local /tmp.
_done=0
finish() {
  [ "$_done" = 1 ] && return; _done=1
  local t0=$SECONDS
  cp -a "$WORK"/. "$MBO_DEST_DIR"/ 2>/dev/null || true
  rm -rf "$WORK"
  echo "timing: transfer=$((SECONDS - t0))s total_wall=${SECONDS}s"
}
trap finish EXIT TERM

srun python "$PROJECT/hpc/run_pipeline.py"
