# Logging system review

Review only — no code changed. Audited `log.py`, `cli.py`, `hpc/`, the IO/array
layer, and the GUI Process Console path. All structural claims below were
verified empirically (running the loggers) and adversarially re-checked.

## Direct answer

Outside the GUI the system cannot easily switch between info / debug / critical.
And even when a level is set, most of the codebase's log calls emit nothing. Two
compounding defects explain why an HPC run looks silent.

## Defect 1 — child loggers are silent no-ops (the main one)

`log.get("sub")` (log.py:16-26) returns `logging.getLogger("mbo.sub")`, forces
`propagate=False` (log.py:22), and attaches only `_extra_handlers` — empty unless
`log.attach()` ran, and `attach()` is called only in the GUI (preview_data.py:259).
The single `StreamHandler` lives on the root `mbo` logger (log.py:36).

Verified behavior (traced through CPython `callHandlers` and run directly):

| Call | INFO | DEBUG | WARNING+ |
|---|---|---|---|
| `log.get()` (root `mbo`) | shows | shows (MBO_DEBUG=1) | shows |
| `log.get("writers")` etc. (child) | silent | silent | leaks bare via Python `lastResort` only |

Almost the entire codebase uses the child form — `log.get("writers")`,
`log.get("reader")`, `log.get("arrays.bin")`, `log.get("arrays.zarr")`, and ~12
others. Only three sites use the working root form (lazy_array.py:99,
numpy.py:247/262). So every `logger.info("Writing volumetric zarr…")`,
`"Detected Suite2p volume…"`, `"computing axial plane shifts…"`,
`"Copying z-plane data…"` is a no-op in any script/CLI/HPC run. `MBO_DEBUG=1`
raises the child *level* but there is still no handler, so DEBUG/INFO still go
nowhere.

Likely original intent was "one handler, no duplicate lines" (the `_worker.py:46`
comment worries about exactly that). But `propagate=False` on the children severs
them from that one handler. You need either propagation or a handler on the
children — currently it is neither.

## Defect 2 — HPC does not use logging at all

`hpc/` has zero logging calls; it is 63 bare `print(..., flush=True)`
(pipeline.py, submit.py, logs.py). These always reach the SLURM `.out` file, but
carry no level, no timestamp, no module, and cannot be silenced or raised. Every
heavy op is bracketed by at most a before/after print with no progress in
between:

- input staging copy ~366 GB -> only the one `input staging: …` line before a
  multi-minute `shutil.copytree`, nothing during (pipeline.py:335 -> 359)
- `imread` of the raw -> `shape=…` printed only after it returns
  (pipeline.py:487-489)
- the 16-minute suite2p compute -> one `single job: N planes…` line before, a
  timing table after; zero periodic progress (pipeline.py:526 -> 532)
- node-local copy-back, aggregate merge, cellpose download -> after-only or
  failure-only

Anything seen mid-run (tqdm / per-plane) comes from suite2p internals, not this
code. The library INFO that would narrate (reader/writers) is muted by Defect 1.

## Level control, precisely

- `MBO_DEBUG` (log.py:29-30): binary DEBUG-vs-INFO, read once at import, no
  WARNING/ERROR/CRITICAL.
- `set_global_level()` (log.py:7, full DEBUG…CRITICAL): the only real switch,
  wired only in GUI widgets (run_gui.py:690, _options_popup.py:195,
  gui_logger.py:134, file_dialog.py:300).
- No `--log-level` / `-v` / `--quiet` on `mbo` or `mbo hpc`. `mbo convert --debug`
  (cli.py:565) only flips the single `mbo.writer` logger and is passed as an
  imwrite kwarg — it does not touch reader/writers/arrays.

## Two logger trees

Five sites bypass `log` via `logging.getLogger("mbo_utilities")` — a different
top-level hierarchy, not a child of `mbo` (metadata/output.py:138,
features/_slicing.py:164, gui/viewers/__init__.py:52,
viewers/pollen_calibration.py:164). `attach()` and `get_package_loggers()` filter
`startswith("mbo.")` while `set_global_level` filters `startswith("mbo")` — so
these never appear in the GUI panel and never get the handler.

## What actually reaches a terminal today

Only tqdm bars at four write sites (`Saving`, `Writing TIFF/H5/Zarr` —
_writers.py:319/945/1177/1614). Silent with no bar and no log: `register_z` axial
shift compute (_registration.py:337), `imread` dispatch/large-file open, the zarr
volume-merge copy loop (zarr.py:662), mp4 encode. `imwrite(debug=True)` suppresses
the tqdm bars (`and not debug`, _writers.py:318) while only un-silencing the
`writer` logger — so debug bin-writes show nothing at all.

## GUI Process Console

Works via two channels: a per-task `.log` file fed by the `mbo.worker`
`RotatingFileHandler` (_worker.py:56-60) plus redirected subprocess stdout/stderr,
and a sidecar JSON progress file. Only `mbo.worker` is wired; `mbo.writers` /
`mbo.reader` / `mbo.arrays.*` are not bridged, so write/read internals are
invisible in the console too. Only `isoview` and `consolidate` are manually
bridged (tasks.py:694, 1039). For `task_save_as` the user sees just a progress bar
and a couple of start/finish lines.

## Recommendations (review only)

1. Fix the child-logger sink (one small change, unblocks everything). In
   `log.get`, either drop `propagate=False` on children so they reach the root
   `mbo` handler, or add the root handler to children. This makes every existing
   `logger.info` across the repo emit, and removes the need for the hand-built
   bridges in tasks.py (694, 1039).
2. Add real level control on CLI + HPC. A `-v/-vv/--log-level {critical..debug}`
   option at the `mbo` and `mbo hpc` group level that calls `set_global_level`;
   read `MBO_DEBUG` at call time, not import; support all five levels.
3. Route HPC through `logging`. Replace the 63 `print` calls with root-`mbo`
   logging (level + timestamp + node/job tag), and add periodic heartbeats to the
   long ops — staging copy %, a compute heartbeat during the 16-min pipeline,
   copy-back progress.
4. Instrument the silent heavy ops — `register_z`, `imread` open, zarr-merge, mp4
   encode, staging/copy-back — at least a root-level start/end line plus a
   progress signal.
5. Collapse the second logger tree. Convert the 5 `logging.getLogger("mbo_utilities")`
   sites to `log.get(...)`, and make `attach`/`get_package_loggers`/
   `set_global_level` use a single consistent prefix.
6. Process Console: once (1) lands, the core loggers flow to the per-task `.log`
   without per-subsystem bridges; today only `isoview` and `consolidate` are
   bridged.

Item 1 is the highest-leverage fix and is small.
