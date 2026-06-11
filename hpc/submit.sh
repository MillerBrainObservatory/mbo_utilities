#!/bin/bash
# Single job: all planes on one GPU, F workers time-sharing it.
# Local-equivalent run on a compute node. For multi-GPU scaling use submit_all.sh.
# Submit from this directory: cd hpc && sbatch submit.sh
#SBATCH --job-name=s2p
#SBATCH --partition=hpc_a100_a   # public A100 node; confirm idle: sinfo -s
#SBATCH --gres=gpu:a100:1        # confirm gres name: sinfo -o "%P %G"
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${MBO_SCRATCH:?set MBO_SCRATCH to your lustre scratch root}"
USER="${USER:-$(id -un)}"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PROJECT="${MBO_PROJECT:-$MBO_SCRATCH/$USER/repos/mbo_utilities}"
source "$PROJECT/.venv/bin/activate"

# Input: read-only on lustre. Output: node-local NVMe — per-node, single-node runs only.
export MBO_INPUT="${MBO_INPUT:-$MBO_SCRATCH/mbo_data/lbm/2025-07-27_mk355/raw}"
export MBO_OUTPUT="${MBO_OUTPUT:-${TMPDIR:-/tmp}/$USER/mk355_results}"
mkdir -p "$MBO_OUTPUT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

srun python run_pipeline.py
