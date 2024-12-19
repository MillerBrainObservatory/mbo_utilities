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

# cleanup on script exit
# -----------------------------------------------------
cleanup() {
  echo "Cleaning up temporary directory $tmpdir"
  rm -rf "$tmpdir"
}
trap cleanup EXIT

# Stage data
# -----------------------------------------------------
echo "Staging data to $tmpdir"
rsync -av --include 'plane_*' --exclude '*' "$copydir" "$tmpdir/"
echo "Successful staging."

ls -la "$tmpdir"

# Run lcp on the staged files  
# -----------------------------------------------------

echo "Running lcp on transferred files..."
lcp --batch_path "$tmpdir" --run mcorr cnmf \
    --max_shifts 10 10 \
    --K 150 \
    --gSig_filt 0 0 \
    --overlaps 4 4 \
    --strides 14 14 \
    --data_path "$tmpdir"

completion_time=$(date +"%Y-%m-%d_%H-%M-%S")
tmp_dest="~/caiman_data/hpc/${completion_time}"

rsync -av -e --exclude 'plane_*' --include '*' "ssh" "$tmpdir/" rbo:"$tmp_dest"

echo "Job Complete!"

