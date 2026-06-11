#!/bin/bash
# Whole-node job: grab all 4 A100s on one node, shard planes across the GPUs,
# then run the volumetric merge. Self-contained (one process per GPU, then the
# aggregate) and all on one node, so node-local NVMe output works.
# Source your mbo env file first, then submit from a writable dir with a logs/
# subdir (e.g. $MBO_USER): sbatch "$MBO_REPOS/mbo_utilities/hpc/submit_node.sh"
#SBATCH --job-name=s2p-node
#SBATCH --partition=hpc_a100_a
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${MBO_REPOS:?source your mbo env file first (defines MBO_REPOS/MBO_LBM)}"
PROJECT="${MBO_PROJECT:-$MBO_REPOS/mbo_utilities}"
source "$PROJECT/.venv/bin/activate"

# Input: read-only on lustre. Output: node-local NVMe.
export MBO_INPUT="${MBO_INPUT:-$MBO_LBM/2025-07-27_mk355/raw}"
export MBO_OUTPUT="${MBO_OUTPUT:-${TMPDIR:-/tmp}/mk355_results}"
mkdir -p "$MBO_OUTPUT"

# 4 GPUs share 48 CPUs: cap per-worker BLAS threads so concurrent workers don't oversubscribe.
export MBO_THREADS_PER_WORKER="${MBO_THREADS_PER_WORKER:-3}"
export OMP_NUM_THREADS="$MBO_THREADS_PER_WORKER"
export MKL_NUM_THREADS="$MBO_THREADS_PER_WORKER"

python "$PROJECT/hpc/run_pipeline.py" --gpus 0,1,2,3
