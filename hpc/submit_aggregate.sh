#!/bin/bash
# Volumetric merge/stats/plots over already-processed per-plane outputs.
# Reuses existing planes (force_reg/force_detect off), so it is CPU/IO-bound.
#SBATCH --job-name=s2p-agg
#SBATCH --partition=gpu          # confirm: sinfo -s
#SBATCH --gres=gpu:1             # confirm gres name: sinfo -o "%P %G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PROJECT="${MBO_PROJECT:-/lustre/fs8/mbo/scratch/mbo_soft/repos/mbo_distributed}"
source "$PROJECT/.venv/bin/activate"

export MBO_HPC_MODE=aggregate
srun python run_pipeline.py
