#!/bin/bash -l

#SBATCH --job-name=assemble
#SBATCH --output=logs/sr_%x_%j.out
#SBATCH --error=logs/sr_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=256G
#SBATCH --partition=hpc_a10_a

indir="$SCRATCH_USER/data/single_hemi/"
outdir="~/caiman_data/hpc/"

# Enable error handling and robust shell options
set -euo pipefail

echo "----------------"
echo "Job Info"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Current Directory: $(pwd)"
echo "----------------"

echo "Memory Info:"
free -h
echo "----------------"

scontrol show -dd job $SLURM_JOB_ID
printenv

# Run assembly 
# -----------------------------------------------------
echo "----------------"
echo "Running sr on staged files..."
sr "$indir/" --save $indir/zplanes --overwrite --assemble

echo "Job Complete!"
echo "----------------"

