(hpc_usage)=

# HPC / SLURM

Run the LBM-Suite2p pipeline on a SLURM cluster (or locally) from a TOML config. Built on [submitit](https://github.com/facebookincubator/submitit).

| Command | Description |
|---------|-------------|
| `mbo hpc info` | Show partitions: nodes, CPUs, GPU usage, free memory |
| `mbo hpc init` | Write a commented `hpc.toml` |
| `mbo hpc check` | Verify the request fits the data and the partition |
| `mbo hpc run` | Submit the run (single / array / local) |
| `mbo hpc status` | Job state, an output dir's timings, or your queue |
| `mbo hpc watch` | Follow a run's `.err`/`.out` logs |

## Typical flow

```bash
mbo hpc info                          # size a job: which partition has free GPUs
mbo hpc init /data/raw                # write /data/raw/hpc.toml (edit it)
mbo hpc check hpc.toml --mode array   # does the request fit the data + partition?
mbo hpc run hpc.toml --mode array     # submit array + dependent aggregate
mbo hpc watch hpc.toml                # follow the newest run's logs
```

## Config

`mbo hpc init` writes a commented `hpc.toml`. The four tables:

```toml
[io]
input  = "/lustre/.../raw"          # directory of ScanImage TIFFs
output = "/lustre/.../results"      # WRITABLE root; prefer scratch
name   = "s2p"                      # label for the dated output subfolder

[slurm]
partition         = "hpc_a100_a"    # see `mbo hpc info`
gres              = "gpu:a100:1"    # GPUs per job
cpus_per_task     = 16              # workers derive from this and F
mem_gb            = 128             # per job; size to the node, not a small default
time              = "24:00:00"
array_parallelism = 0               # max concurrent array tasks (0 = scheduler default)

[pipeline]
planes_per_gpu = 4                  # pack factor F: planes sharing one GPU
node_local     = true              # stage on node-local NVMe, copy results back

[parameters]                        # suite2p ops + pipeline knobs (keep_reg, diameter, ...)
algorithm = "cellpose"
keep_reg  = false
```

One job holds one GPU. `planes_per_gpu` (F) planes share it, each holding a movie in RAM, so peak RAM ≈ F × per-plane size. Set `mem_gb` to the node's capacity — too low OOMs at the cgroup cap. `mbo hpc check` does this math for you.

## Modes

```bash
mbo hpc run hpc.toml                  # single (default)
mbo hpc run hpc.toml --mode array
mbo hpc run hpc.toml --local
mbo hpc run hpc.toml --dry-run        # print the job layout, submit nothing
```

| Mode | What it does | Use when |
|------|--------------|----------|
| `single` (default) | One GPU job over all planes (F packed per GPU), volumetric merge inline | The dataset fits one job's wall-time limit |
| `array` | One array task per F-plane shard, then a dependent aggregate that merges the volume | Spreading shards across **multiple** nodes cuts wall time |
| `local` | Runs the compute inline in this process, off the scheduler | Testing, or a workstation with a GPU |

**`--mode array` only helps across multiple nodes.** On a single-node partition all tasks pile onto one node, where `cpus_per_task × tasks` must fit the node's CPUs and they share its GPUs — the same wall time as `single`. Use `mbo hpc info` to see a partition's `NODES` count, and `mbo hpc check --mode array` to catch the single-node case before submitting.

<details>
<summary><b>run overrides (set config fields on the command line)</b></summary>

| Option | Description |
|--------|-------------|
| `--input` / `--output` / `--name` | Override the `[io]` fields |
| `--partition` / `--gres` / `--time` | Override the `[slurm]` fields |
| `--planes-per-gpu` | Override pack factor F |
| `--local` | Shortcut for `--mode local` |
| `--gpu` | Local-run CUDA device index (nvidia-smi order); `-1` = auto. Ignored under SLURM |

</details>

## Monitor

```bash
mbo hpc status 5162141                # job state, exit code, failure diagnosis
mbo hpc status /data/results/2025_..  # timings.json summary for an output dir
mbo hpc status                        # the last run you launched
mbo hpc watch                         # follow the last run's logs
mbo hpc watch 5162141                 # follow logs by job id (shows state first)
mbo hpc watch hpc.toml -o             # follow a run's .out instead of .err
```

While `watch` follows a terminal: `o`/`e` switch out/err, `n`/`p` switch task logs, `q` quits.

<details>
<summary><b>Diagnostics</b></summary>

| Command | Description |
|---------|-------------|
| `mbo hpc check` | Memory math + structural fixes for a config vs. the partition |

</details>

## Shared environment (`mbo_server_configs`)

Separate from the `mbo hpc` CLI above: the shared software under `/lustre/fs8/mbo/scratch/mbo_soft` (CLI tools, neovim, the `mbo` venv, repos) is exposed through `MBO_*` variables and navigation aliases. Source it once from `~/.bashrc`:

```bash
source /lustre/fs8/mbo/scratch/mbo_soft/repos/mbo_server_configs/config/hpc/mbo.sh
```

Variables (defined in `config/hpc/env.sh`):

| Variable | Value | Points to |
|----------|-------|-----------|
| `MBO_ROOT` | `/lustre/fs8/mbo` | lab root — change this only to move filesystems |
| `MBO_SCRATCH` | `$MBO_ROOT/scratch` | scratch root |
| `MBO_STORE` | `$MBO_ROOT/store` | long-term store |
| `MBO_SOFT` | `$MBO_SCRATCH/mbo_soft` | shared software root |
| `MBO_BIN` | `$MBO_SOFT/bin` | shared bin (on `PATH`) |
| `MBO_REPOS` | `$MBO_SOFT/repos` | shared repos |
| `MBO_NVIM` | `$MBO_SOFT/neovim` | neovim install |
| `MBO_ENVS` | `$MBO_SOFT/envs` | shared venvs dir |
| `MBO_ENV` | `$MBO_ENVS/mbo` | default shared venv |
| `MBO_DATA` | `$MBO_SCRATCH/mbo_data` | data root |
| `MBO_LBM` | `$MBO_DATA/lbm` | LBM data |
| `MBO_LSM` | `$MBO_DATA/lsm` | LSM data |
| `MBO_USER` | `$MBO_SCRATCH/$USER` | your personal scratch (override: set `MBO_USER` first) |

Also sets `UV_LINK_MODE=hardlink`, `UV_CACHE_DIR=$MBO_USER/.uv/cache`, `UV_PYTHON_INSTALL_DIR=$MBO_USER/.uv/python`.

Navigation aliases and helpers (defined in `config/hpc/mbo.sh`):

| Command | Action |
|---------|--------|
| `cdsoft` / `cdrepos` / `cdscratch` / `cdme` | cd to `$MBO_SOFT` / `$MBO_REPOS` / `$MBO_SCRATCH` / `$MBO_USER` |
| `cddata` / `cdlbm` / `cdlsm` | cd to `$MBO_DATA` / `$MBO_LBM` / `$MBO_LSM` |
| `mbo-activate [env]` | source a shared venv (default `mbo`) |
| `mbo-run <cmd> [args]` | run an executable from the shared `mbo` venv |
| `mbo-jobs` / `mbo-gpus` | `squeue --me` / list HPC GPU node availability |
| `mbo-gpu [part] [time] [n]` / `mbo-cpu [part] [time]` | interactive GPU / CPU shell via `srun` |
| `mbo-stage <path> [dest]` / `mbo-pull` / `mbo-push` | rsync data-transfer helpers |
| `mbo-nvim-setup` / `mbo-update` | install nvim tools / pull latest configs |

Verify live values after login with `env | grep '^MBO_'`.
