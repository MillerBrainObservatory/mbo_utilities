# HPC pipeline runner

Minimal SLURM wrapper around `lbm_suite2p_python.pipeline`. Edit the two
paths at the top of `run_pipeline.py`, then submit.

## Files

- `run_pipeline.py` — edit `input_dir` and `output_dir` at the top.
- `submit.sh` — SLURM wrapper. Activates conda env and runs the script.
- `logs/` — created on first submit.

## One-time cluster setup

```bash
conda create -n mbo python=3.12 -y
conda activate mbo
pip install mbo_utilities lbm_suite2p_python
```

Adjust `submit.sh` if needed: `--partition`, the `conda.sh` path, or set
`MBO_ENV` to override the env name without editing the file.

## Stage data

The reader scans a directory for ScanImage TIFFs sharing a base name. If
chunks are mixed with other recordings, move them to their own subdir:

```bash
mkdir -p <DATA_ROOT>/<DATASET>
mv <DATA_ROOT>/<DATASET>_*.tif <DATA_ROOT>/<DATASET>/
```

## Run

Edit `run_pipeline.py` — set `input_dir` to the staged dataset folder and
`output_dir` to where results should land. Then:

```bash
cd <PATH_TO_HPC_DIR>
sbatch submit.sh
```

Watch:

```bash
squeue -u $USER
tail -f logs/s2p_<JOBID>.out
```

## Reload results

The output directory is a Suite2p volume root — open it in MBO Studio
(`uv run mbo <OUTPUT_DIR>`) to view registered binaries and re-sweep
detection without re-registering.
