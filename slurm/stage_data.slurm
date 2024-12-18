#!/bin/bash -l

#SBATCH --job-name=mcorr
#SBATCH --output=stage_data_%j.out
#SBATCH --error=stage_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=hpc_a10_a

# Enable error handling and robust shell options
set -euo pipefail

echo "----------------"
echo "Job Info"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Current Directory: $(pwd)"
echo "----------------"

# Print system info for logs
echo "CPU Info:"
lscpu
echo "----------------"

echo "Memory Info:"
free -h
echo "----------------"

echo "GPU Info:"
nvidia-smi || echo "No GPUs found"
echo "----------------"

# Create a temporary working directory
tmpdir=$(mktemp -d /tmp/data_XXXXXX)
echo "Temporary directory created at: $tmpdir"

# Ensure cleanup happens on script exit
cleanup() {
  echo "Cleaning up temporary directory $tmpdir"
  rm -rf "$tmpdir"
}
trap cleanup EXIT

# Create a batch directory inside the temp directory
batchdir="$tmpdir/batch"
mkdir -p "$batchdir"
echo "Batch directory created at: $batchdir"

# Stage data from $MBO_DATA to the temporary directory
echo "Staging data from $MBO_DATA/lbm/high_res/plane* to $tmpdir"
rsync -av --include 'plane_*' --exclude '*' "${MBO_DATA%/}/lbm/high_res/" "$tmpdir/"

echo "Data staged successfully in $tmpdir"
echo "Contents of $tmpdir:"
ls -la "$tmpdir"

# Run lcp on the staged files
echo "Running lcp on transferred files..."
lcp "$batchdir/batch.pickle" --create --run mcorr cnmf \
    --max_shifts 20 20 \
    --gSig_filt 3 3 \
    --overlaps 4 4 \
    --strides 14 14 \
    --data_path "$tmpdir"

echo "lcp processing complete."
echo "Contents of $tmpdir after lcp:"
ls -lR "$tmpdir"

# Create destination directory on $SCRATCH
completion_time=$(date +"%Y-%m-%d_%H-%M-%S")
tmp_dest="$SCRATCH/$USER/${completion_time}"
echo "Creating destination directory: $tmp_dest"
mkdir -p "$tmp_dest"

echo "Copying batch directory from $batchdir to $tmp_dest"
cp -r "$batchdir" "$tmp_dest"

echo "Final contents of destination directory $tmp_dest:"
ls -lR "$tmp_dest"

echo "Job Complete!"

