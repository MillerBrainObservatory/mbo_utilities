#!/bin/bash
# Volumetric merge/stats over the per-plane outputs the array wrote to the shared
# folder (MBO_DEST_DIR from submit_all.sh). Reads the shared folder directly (no
# node-local staging — it needs every shard), and drops its own log in. Reuses
# existing planes (force off), so it is CPU/IO-bound. Run only via submit_all.sh.
#SBATCH --job-name=s2p-agg
#SBATCH --partition=hpc_a100_a
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${MBO_REPOS:?source your mbo env file first (defines MBO_REPOS/MBO_LBM)}"
: "${MBO_DEST_DIR:?run via submit_all.sh (it sets the shared output folder)}"
PROJECT="${MBO_PROJECT:-$MBO_REPOS/mbo_utilities}"
source "$PROJECT/.venv/bin/activate"

export MBO_INPUT="${MBO_INPUT:-$MBO_LBM/2025-07-27_mk355/raw}"
export MBO_OUTPUT="$MBO_DEST_DIR"
export MBO_HPC_MODE=aggregate

LOG="${SLURM_SUBMIT_DIR:-.}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
trap 'cp -a "$LOG".{out,err} "$MBO_OUTPUT"/ 2>/dev/null || true' EXIT

srun python "$PROJECT/hpc/run_pipeline.py"
