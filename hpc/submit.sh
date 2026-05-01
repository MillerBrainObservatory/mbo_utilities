#!/bin/bash
#SBATCH --job-name=s2p
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${MBO_ENV:-mbo}"

srun python run_pipeline.py
