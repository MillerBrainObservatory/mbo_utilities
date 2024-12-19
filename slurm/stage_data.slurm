#!/bin/bash -l

#SBATCH --job-name=mcorr
#SBATCH --output=stage_data_%j.out
#SBATCH --error=stage_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=256G
#SBATCH --partition=hpc_a10_a

copydir="$MBO_DATA/lbm/high_res/raw/"
outdir="~/caiman_data/hpc/"

# Create a temporary directory to store raw files
tmpdir=$(mktemp -d /tmp/data_XXXXXX)

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
cleanup() {
        echo "Cleaning up temporary directory $tmpdir"
        rm -rf "$tmpdir"
}
trap cleanup EXIT

# Stage raw data into temporary directory
# -----------------------------------------------------
echo "Staging raw data from $copydir to $tmpdir"
rsync -av --include '*/' --include 'plane_*' --exclude '*' "$copydir/" "$tmpdir/"
echo "Data successfully staged to $tmpdir"

echo "----------------"
echo "Listing contents of staged data:"
ls -la "$tmpdir"
echo "----------------"

# Create batch directory
# -----------------------------------------------------
echo "----------------"
echo "Creating batch directory at $tmpdir/batch"
mkdir -p "$tmpdir/batch"
echo "----------------"

# Run lcp on the staged files  
# -----------------------------------------------------
echo "----------------"
echo "Running lcp on staged files..."
lcp --batch_path "$tmpdir/batch/batch.pickle" --run mcorr cnmf \
        --max_shifts 10 10 \
        --K 150 \
        --gSig_filt 0 0 \
        --overlaps 4 4 \
        --strides 14 14 \
        --data_path "$tmpdir"
echo "----------------"

# Transfer the results back to local machine
# -----------------------------------------------------
completion_time=$(date +"%Y-%m-%d_%H-%M-%S")
tmp_dest="~/caiman_data/hpc/${completion_time}"

echo "----------------"
echo "Transferring results from $tmpdir to remote directory $tmp_dest"
rsync -av -e "ssh" --exclude 'plane_*' --include '*' "$tmpdir/" rbo:"$tmp_dest"
echo "Transfer complete."
echo "----------------"

# Final cleanup (only run if everything else succeeds)
# -----------------------------------------------------
echo "----------------"
echo "Cleaning up temporary directory $tmpdir"
rm -rf "$tmpdir"

echo "Job Complete!"
echo "----------------"

