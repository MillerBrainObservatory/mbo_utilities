#!/bin/bash
# Array body: each task processes one plane shard (F planes) on its own GPU.
# --gres=gpu:1 per task makes CUDA renumber the visible GPU to 0, so no manual
# pinning is needed. The --array range is set by submit_all.sh from the plane
# count and MBO_PLANES_PER_GPU. skip_volumetric is implied by SLURM_ARRAY_TASK_ID;
# the volumetric merge runs once in submit_aggregate.sh after all tasks succeed.
#SBATCH --job-name=s2p-arr
#SBATCH --partition=gpu          # confirm: sinfo -s
#SBATCH --gres=gpu:1             # confirm gres name: sinfo -o "%P %G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PROJECT="${MBO_PROJECT:-/lustre/fs8/mbo/scratch/mbo_soft/repos/mbo_distributed}"
source "$PROJECT/.venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

srun python run_pipeline.py
