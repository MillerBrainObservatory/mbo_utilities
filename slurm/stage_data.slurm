#!/bin/bash -l
#SBATCH --job-name=mcorr
#SBATCH --output=stage_data_%j.out
#SBATCH --error=stage_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=hpc_a10_a

hostname
pwd
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "----------------"
echo "CPU Info:"
lscpu
echo "----------------"
echo "Memory Info:"
free -h
echo "----------------"
echo "GPU Info:"
nvidia-smi || echo "No GPUs found"
echo "----------------"

tmpdir=$(mktemp -d /tmp/data_XXXXXX)

batchdir="$tmpdir/batch"
mkdir -p "$batchdir"

echo "----------------"
rsync -av "$MBO_DATA/lbm/high_res/plane*" "$tmpdir/"

echo "Data staged in $tmpdir"
echo "----------------"

ls -la "$tmpdir"

echo "Running lcp on transferred files"

lcp "$batchdir/batch.pickle" --create --run mcorr cnmf --max_shifts 20 20 --gSig_filt 3 3 --overlaps 4 4 --strides 14 14 --data_path "$tmpdir"/*.tif

echo "Contents of $tmpdir after lcp:"
ls -lR "$tmpdir"

tmp_dest="$SCRATCH/$USER/out"
mkdir -p temp_dest

echo "$tmp_dest"
echo "Created!"

ls -lR "$batchdir"

cp -r "$batchdir" "$tmp_dest"

echo "Complete!"
