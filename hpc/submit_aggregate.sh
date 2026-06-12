#!/bin/bash
# Volumetric merge over the shards the array wrote to MBO_DEST_DIR (set by
# submit_all.sh). Reads/writes that shared folder directly; SLURM writes its log
# there too. Run only via submit_all.sh.
#SBATCH --job-name=s2p-agg
#SBATCH --partition=hpc_a100_a
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
: "${MBO_REPOS:?source your mbo env file first (defines MBO_REPOS/MBO_LBM)}"
: "${MBO_DEST_DIR:?run via submit_all.sh (it sets the shared output folder)}"
PROJECT="${MBO_PROJECT:-$MBO_REPOS/mbo_utilities}"
source "$PROJECT/.venv/bin/activate"

export MBO_INPUT="${MBO_INPUT:-$MBO_LBM/2025-07-27_mk355/raw}"
export MBO_OUTPUT="$MBO_DEST_DIR"
export MBO_HPC_MODE=aggregate

t0=$SECONDS
srun python "$PROJECT/hpc/run_pipeline.py"
echo "timing: aggregate=$((SECONDS - t0))s"

# timing report over the full run (per-plane stages + writes timings.json)
python "$PROJECT/hpc/run_pipeline.py" --report-timings "$MBO_OUTPUT" || true
