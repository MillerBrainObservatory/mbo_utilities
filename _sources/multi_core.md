# multi-core parallel processing

architecture notes for suite2p parallel plane processing via the gui RUN tab.

## execution modes

the gui provides two modes, toggled by "Run in background" checkbox.

### mode A: background subprocess (default)

- `process_manager.py:spawn()` launches a detached subprocess per channel
- invoked as `python -m mbo_utilities.gui._worker suite2p {args_json}`
- subprocess calls `lbm_suite2p_python.pipeline()` which processes all selected planes **sequentially**
- survives gui closure; monitored via json sidecar files (`progress_{uuid}.json`)
- watchdog thread kills the process if no progress for 120 minutes
- windows: uses `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`
- unix: uses `start_new_session=True`

### mode B: in-gui daemon threads

- builds a list of per-plane config dicts (`settings.py:1906-1952`)
- if parallel enabled + multiple jobs: `ThreadPoolExecutor(max_workers=max_jobs)` at `settings.py:1963`
- each plane submitted as a future, results collected via `as_completed()`
- suite2p internal parallelism disabled (`num_workers=0`) to avoid oversubscription
- runs inside a daemon thread so gui stays responsive
- if parallel disabled or single job: runs sequentially in a single daemon thread

## key files

| file | role |
|------|------|
| `gui/widgets/pipelines/settings.py` | gui controls, job config building, thread dispatch |
| `gui/_worker.py` | subprocess entry point, watchdog, status reporting |
| `gui/tasks.py` | task registry, `task_suite2p()`, `_ChannelView`, `TaskMonitor` |
| `gui/widgets/process_manager.py` | `ProcessManager.spawn()`, `ProcessInfo`, sidecar reads |
| `gui/panels/process_manager.py` | gui display panel for background processes |

## worker function

`_run_plane_worker_thread(config)` at `settings.py:2014` is a pure function taking a snapshot dict.
all gui state is pre-extracted into the config before thread launch to avoid data races.
each worker: writes binary data via `imwrite(.bin)` -> calls `lbm_suite2p_python.run_plane()`.

## data flow

```
run_process()
  |
  +-- background mode? --> ProcessManager.spawn() --> _worker.py --> task_suite2p() --> pipeline()
  |
  +-- daemon thread mode?
        |
        +-- build jobs[] (one config dict per plane/channel/roi combo)
        |
        +-- parallel? --> ThreadPoolExecutor --> _run_plane_worker_thread(config) per future
        |
        +-- sequential? --> _run_plane_worker_thread(config) in loop
```

## open bugs

### [lbm-suite2p-python#80](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/issues/80) - `is_volumetric` UnboundLocalError

`pipeline()` in `run_lsp.py:289` crashes because `is_volumetric` is only assigned inside
`if roi_mode is not None and supports_roi(arr):`. the normal path (`roi_mode=None`) skips
the block entirely, leaving the variable unbound.

error log from a real run (pid=25612, 10 tiff files, planes=[1,3,5]):
```
UnboundLocalError: cannot access local variable 'is_volumetric' where it is not associated with a value
  File "lbm_suite2p_python/run_lsp.py", line 289, in pipeline
```

### [#158](https://github.com/MillerBrainObservatory/mbo_utilities/issues/158) - shared `arr.metadata` mutated across threads

`_run_plane_worker_thread()` at `settings.py:2089-2094` calls `arr.metadata.pop()` on the
original shared array object. all parallel threads share the same `arr`, creating a dict
mutation race. the local `lazy_mdata` copy is correct but the original shouldn't be touched.

### [#159](https://github.com/MillerBrainObservatory/mbo_utilities/issues/159) - parallel checkbox hidden for 5D data

`settings.py:688` only checks `data.ndim == 4` for num_planes detection. 5D TCZYX data
(`ndim == 5`) falls through, `num_planes` stays 1, and the parallel checkbox never appears.

### [#160](https://github.com/MillerBrainObservatory/mbo_utilities/issues/160) - log directory path mismatch

sidecar writers use `~/mbo/logs/`, but `spawn()` creates log files in `~/.mbo/logs/`.
works by accident since they're tracked independently, but fragile.

### [#161](https://github.com/MillerBrainObservatory/mbo_utilities/issues/161) - file handle leak + double `_save()`

`spawn()` opens `f_out` for log redirection but never closes it in the parent process.
also calls `self._save()` twice in a row.

## potential improvements

### no per-plane parallelism in background mode

background subprocess mode (the default) processes planes sequentially via `pipeline()`.
users who enable "parallel" but leave "run in background" checked get no parallelism.
the parallel checkbox only applies to daemon-thread mode. consider either disabling
the checkbox when background mode is on, or passing parallelism settings to the worker.

### no cancellation for in-gui mode

background subprocess has kill support via ProcessManager, but the in-gui ThreadPoolExecutor
has no cancel button. `self._active_executor` is stored but never used for on-demand shutdown.
`executor.shutdown(wait=False, cancel_futures=True)` (python 3.9+) would enable this.

### no progress reporting for in-gui parallel mode

background mode reports granular progress through sidecar files. in-gui parallel mode only
logs completion messages with no progress bar or per-plane status visible during execution.

### ThreadPoolExecutor vs ProcessPoolExecutor

suite2p is cpu-heavy. python's GIL means threads can't achieve true cpu parallelism for
python-bound work. numpy/C extensions release the GIL so threads do help for the heavy
compute, but binary extraction and python orchestration serialize. ProcessPoolExecutor
would give true parallelism but requires picklable arguments (arr and logger are not).

### executor.shutdown(wait=False) in finally block

at `settings.py:1987`, `shutdown(wait=False)` is called after `as_completed()`. in the
happy path all futures are done. but if an exception interrupts the `as_completed()` loop,
some futures may still be running. `wait=True` would be safer since we're already in a
daemon thread.
