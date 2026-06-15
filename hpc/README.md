# HPC pipeline runner

Submit the `lbm_suite2p_python` pipeline to SLURM from a TOML config — no shell
scripts to edit. Backed by [submitit](https://github.com/facebookincubator/submitit).

## Install

```bash
uv add "mbo_utilities[hpc]"      # pulls submitit
```

## Run

```bash
mbo hpc info                         # partitions: nodes/CPUs/GPUs/free mem (regex, default 'hpc')
mbo hpc init /path/to/raw            # write hpc.toml (input/output/params filled in)
mbo hpc check hpc.toml               # does the request fit the data + partition?
mbo hpc run hpc.toml --dry-run       # preview the jobs, submit nothing
mbo hpc run hpc.toml                 # single GPU job, all planes
mbo hpc run hpc.toml --mode array    # one array task per plane shard + dependent merge
mbo hpc run hpc.toml --local         # run inline on this machine (no SLURM)
mbo hpc status <output_dir>          # per-stage timings; no arg -> squeue
```

`mbo hpc info` shows whether `--mode array` can spread (NODES > 1) and how much
memory/GPU to request. `mbo hpc check` sizes per-plane memory from the input,
reports the estimated peak vs `mem_gb`, and warns on structural mistakes (array
on a single-node partition, `cpus_per_task` × tasks over the node's CPUs, gres
over the node's GPUs) with suggested TOML deltas.

## Sizing a job (avoid the OOM / packing trap)

One job uses one GPU (`gres`); `planes_per_gpu` (F) planes share it, each holding
a movie in RAM, so **peak RAM ≈ F × per-plane size**. `mem_gb` is a hard per-job
cap — set it too low and the job is OOM-killed at the cgroup limit even when the
node has memory to spare. `--mode array` only helps when tasks spread across
**multiple** nodes; on a single-node partition the tasks pile onto one node and
`cpus_per_task × tasks` must fit its CPUs. Run `mbo hpc info` then
`mbo hpc check hpc.toml --mode <mode>` before submitting.

Edit `hpc.toml`, or override single keys on the command line:

```bash
mbo hpc run hpc.toml --partition hpc_l40s --gres gpu:l40s:1 --planes-per-gpu 3
```

## Config

`mbo hpc init` writes a commented `hpc.toml` with four tables:

| Table | Keys |
|---|---|
| `[io]` | `input`, `output`, `name`, `dated_subfolder` |
| `[slurm]` | `partition`, `gres`, `cpus_per_task`, `mem_gb`, `time`, `exclusive`, `array_parallelism`, `account`, `qos` |
| `[pipeline]` | cluster knobs: `planes_per_gpu` (pack factor F), `threads_per_worker`, `node_local` |
| `[parameters]` | one flat table forwarded to processing — suite2p ops (`diameter`, `anatomical_only`, …) **and** lbm pipeline knobs (`keep_reg`, `keep_raw`, `fix_phase`, `norm_method`, …). Each key is routed by name: known pipeline kwargs → `pipeline()`, `fix_phase`/`use_fft` → phase correction, everything else → suite2p ops. |

The final results land in `<output>/<date>_<name>` (a `_2`, `_3`… suffix if it
exists). Under SLURM, each job computes on node-local NVMe (`node_local = true`)
and copies results back; logs go to `<output_dir>/logs`.

## Pick the pack factor F

`planes_per_gpu` is the only tuning knob: the most planes that fit on one GPU
before cellpose OOMs. Measure once — run a single job and watch the GPU:

```bash
nvidia-smi dmon -s um            # u = util %, m = mem used, per second
```

Take the largest `planes_per_gpu` where memory stays under the card limit and
utilization stops climbing. Workers and threads derive from F and the SLURM CPU
allocation (`sched_getaffinity`, not `os.cpu_count()`), so a job never
oversubscribes its node.

## Python API

```python
from mbo_utilities.hpc import HpcConfig, submit

cfg = HpcConfig.from_toml("hpc.toml")
cfg.pipeline.planes_per_gpu = 3
submit(cfg, mode="array")            # or "single" / "local"; dry_run=True to preview
```

## Reload results

The output directory is a Suite2p volume root — open it in MBO Studio
(`mbo <OUTPUT_DIR>`) to view registered binaries and re-sweep detection.

## Legacy shell scripts

`submit.sh`, `submit_array.sh`, `submit_aggregate.sh`, `submit_all.sh`,
`submit_node.sh`, and `run_pipeline.py` are the previous env-var + `sbatch`
path. They remain as a fallback until the submitit path is validated on the
cluster; prefer `mbo hpc` for new runs.
