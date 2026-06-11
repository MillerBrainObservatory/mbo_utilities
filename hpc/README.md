# HPC pipeline runner

One config-driven wrapper around `lbm_suite2p_python.pipeline`. The same
`run_pipeline.py` runs locally, as a single GPU job, or as a SLURM array —
the mode is detected from the environment.

## Files

- `run_pipeline.py` — the runner. Edit the defaults at the top or override via env.
- `submit.sh` — single GPU job (all planes, F workers time-share one GPU).
- `submit_array.sh` — array body; each task does one plane shard on its own GPU.
- `submit_aggregate.sh` — volumetric merge over already-processed planes.
- `submit_all.sh` — submits the array, then a dependent aggregate job.
- `logs/` — created on first submit.

## One-time cluster setup (uv)

```bash
cd /lustre/fs8/mbo/scratch/mbo_soft/repos/mbo_distributed
uv add mbo_utilities lbm_suite2p_python   # torch-cu126 already present
```

`submit*.sh` activate `$PROJECT/.venv`; set `MBO_PROJECT` to override the path.
Confirm the partition and GPU gres name for your scheduler before submitting:

```bash
sinfo -s                # partition names
sinfo -o "%P %G"        # gres string (gpu:1 vs gpu:a100:1 ...)
```

## Config

Defaults live at the top of `run_pipeline.py`; override any with env vars:

| Env | Meaning | Default |
|---|---|---|
| `MBO_INPUT` | directory of ScanImage TIFFs | staged `mk355/raw` |
| `MBO_OUTPUT` | Suite2p volume root | `mk355/results` |
| `MBO_PLANES_PER_GPU` | pack factor `F` (planes per GPU) | 4 |
| `MBO_THREADS_PER_WORKER` | BLAS/OMP threads per worker | `cpus // workers` |
| `MBO_OPS_JSON` | path to a JSON file of `ops` overrides | unset |

Worker counts derive from `F` and the SLURM CPU allocation
(`sched_getaffinity`, not `os.cpu_count()`), so a job never oversubscribes its
node.

## Pick the pack factor F

`F` is the only tuning knob: the most planes that fit on one GPU before cellpose
OOMs. Measure it once on your data — run a single job and watch the GPU:

```bash
nvidia-smi dmon -s um    # u = util %, m = mem used, per second
```

Sweep `MBO_PLANES_PER_GPU=1,2,4` and take the largest value where memory stays
under the card limit and utilization stops climbing.

## Run

Local or single GPU job (all planes, one GPU):

```bash
sbatch submit.sh                       # on the cluster
python run_pipeline.py                 # locally / interactively
python run_pipeline.py /path/raw /path/results   # positional input/output
```

Local multi-GPU (one pinned process per GPU, then aggregate):

```bash
python run_pipeline.py --gpus 0,1,2
```

SLURM array across GPU nodes (one shard per GPU, then a dependent merge):

```bash
./submit_all.sh
```

Watch:

```bash
squeue -u "$USER"
tail -f logs/s2p-arr_*_*.out
```

## Reload results

The output directory is a Suite2p volume root — open it in MBO Studio
(`uv run mbo <OUTPUT_DIR>`) to view registered binaries and re-sweep detection
without re-registering.
