#SBATCH --job-name=batch
#SBATCH -o test_%j.out
#SBATCH -e test_%j.err
#SBATCH -t 15:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --partition=hpc_a10_a

copydir="$SCRATCH/$USER/data/test"
outdir="~/caiman_data/hpc"

gSig_variants=(4 10)
K_variants=(10 25)

echo "Parameters:"
echo "gSig variants: ${gSig_variants[*]}"
echo "K variants: ${K_variants[*]}"

ls $copydir

# Create a temporary directory to store raw files
tmpdir=$(mktemp -d /tmp/data_XXXXXX)

# Enable error handling and robust shell options
set -euo pipefail

source $SCRATCH/mbo_soft/miniforge3/etc/profile.d/conda.sh

conda activate lcp

start_time=$(date +%s)

echo "----------------"
echo "Job Info"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
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
rsync -av --include '*/' --include 'plane_*' --exclude '*' "$copydir/" "$tmpdir"
echo "Data successfully staged to $tmpdir"

ls $tmpdir

# Run lcp on the staged files
# -----------------------------------------------------
echo "----------------"
echo "Running mcorr"
srun lcp --batch_path "$tmpdir" --run mcorr \
        --data_path "$tmpdir"

export tmpdir

echo "Running cnmf with parameter grid"
parallel --jobs $SLURM_CPUS_PER_TASK \
    "srun lcp --batch_path \"$tmpdir\" --run cnmf --gSig {1} --K {2} --data_path 0" ::: ${gSig_variants[*]} ::: ${K_variants[*]}

echo "----------------"

# Transfer the results back to local machine
# -----------------------------------------------------
tmp_dest="~/caiman_data/hpc/${SLURM_JOB_ID}"

echo "----------------"
echo "Transferring results from $tmpdir to remote directory $tmp_dest"

# copy logs
cp "test_${SLURM_JOB_ID}.out" "$tmpdir/"
cp "test_${SLURM_JOB_ID}.err" "$tmpdir/"

# This IP is not static, when we get a static IP this will no longer work.
rsync -av -e "ssh -i ~/.ssh/id_rsa -o IdentitiesOnly=yes" --exclude 'plane_*' --include '*' "$tmpdir/" rbo@129.85.3.34:"$tmp_dest"
echo "Transfer complete."
echo "----------------"

# Final cleanup (only run if everything else succeeds)
# -----------------------------------------------------
echo "----------------"
echo "Cleaning up temporary directory $tmpdir"
rm -rf "$tmpdir"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Execution Time: $(date -u -d @${elapsed_time} +'%H hours, %M minutes, %S seconds')"

echo "Job Complete!"
echo "----------------"
