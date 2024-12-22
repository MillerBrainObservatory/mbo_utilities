#!/bin/bash -l

#SBATCH --job-name=mcorr
#SBATCH --output=logs/lcp_%x_%j.out
#SBATCH --time=9:00:00
#SBATCH --error=logs/lcp_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=256G
#SBATCH --partition=hpc_a10_a

copydir="$SCRATCH/$USER/data/bi_hemi/zplanes/test/"
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

ls $tmpdir

# Run lcp on the staged files  
# -----------------------------------------------------
echo "----------------"
echo "Running lcp on staged files..."
lcp --batch_path "$tmpdir/batch.pickle" --run mcorr cnmf \
        --gSig 2 \
        --gSiz 6 6 \
        --rf 100 \
        --stride 30 \
        --K 10000 \
        --data_path "$tmpdir/"

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

