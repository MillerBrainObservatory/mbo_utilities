#!/bin/bash -l

#SBATCH --job-name=mcorr
#SBATCH --output=~/logs/s_r%x_%j.out
#SBATCH --error=~/logs/sr_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=256G
#SBATCH --partition=hpc_a10_a

indir="$USER/data/high_res/"
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

# Cleanup on script exit
# -----------------------------------------------------
#cleanup() {
#        echo "Cleaning up temporary directory $tmpdir"
#        rm -rf "$tmpdir"
#}
#trap cleanup EXIT

# Run assembly 
# -----------------------------------------------------
echo "----------------"
echo "Running sr on staged files..."
sr "$indir/" --save $indir/zplanes --overwrite --assemble
echo "----------------"

# Transfer the results back to local machine
# -----------------------------------------------------
completion_time=$(date +"%Y-%m-%d_%H-%M-%S")
tmp_dest="$outdir/${completion_time}"

echo "----------------"
echo "Transferring results from $tmpdir to remote directory $tmp_dest"
rsync -av -e "ssh" --include 'plane_*' "$indir/zplanes/" rbo:"$tmp_dest"
echo "Transfer complete."
echo "----------------"

echo "Job Complete!"
echo "----------------"

