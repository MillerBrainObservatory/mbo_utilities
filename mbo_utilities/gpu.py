"""GPU compute monitoring and on/off control.

Reports per-device memory (total/used/free) and per-process VRAM, and
provides a single switch for whether CUDA compute is used.

Two concerns, kept separate:

monitoring
    ``gpu_devices()``   per-device memory + utilization (nvidia-smi).
    ``gpu_processes()`` per-process VRAM. On Windows (WDDM) nvidia-smi
                        cannot report per-process memory, so we read the
                        same GPU performance counters Task Manager uses.
    ``format_gpu_report()`` text block for the CLI and GUI.

control
    A single env var ``MBO_GPU`` drives ``CUDA_VISIBLE_DEVICES``, which
    every CUDA consumer (torch / cupy / cellpose) honours — including
    spawned worker subprocesses, which inherit the environment.
        MBO_GPU=0 / off / cpu   -> CUDA_VISIBLE_DEVICES="" (force CPU)
        MBO_GPU=1 / on / auto    -> leave device visibility unchanged
        MBO_GPU=N / "0,1"        -> CUDA_VISIBLE_DEVICES=N (pin device)

This module has no heavy imports at module load; everything is lazy.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any

_SMI_TIMEOUT = 5


def _no_window_flags() -> int:
    """CREATE_NO_WINDOW on Windows so subprocess calls don't flash a console."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def _run(cmd: list[str], timeout: int = _SMI_TIMEOUT) -> str | None:
    """Run a command, return stdout or None on any failure."""
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_no_window_flags(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def has_nvidia_smi() -> bool:
    """True if nvidia-smi is callable (an NVIDIA driver is present)."""
    return _run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]) is not None


def gpu_devices() -> list[dict[str, Any]]:
    """Per-device memory and utilization.

    Returns one dict per device with keys: index, name, total_mb, used_mb,
    free_mb, util_pct, temp_c. Empty list if no NVIDIA GPU / driver.
    """
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []

    def _num(s: str) -> float | None:
        s = s.strip()
        try:
            return float(s)
        except ValueError:
            return None

    devices = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, util, temp = parts[:7]
        try:
            index = int(idx)
        except ValueError:
            continue
        devices.append({
            "index": index,
            "name": name,
            "total_mb": _num(total),
            "used_mb": _num(used),
            "free_mb": _num(free),
            "util_pct": _num(util),
            "temp_c": _num(temp),
        })
    return devices


def _pid_name(pid: int) -> str:
    """Best-effort process name for a pid."""
    try:
        import psutil
        return psutil.Process(pid).name()
    except Exception:
        return "?"


def _gpu_processes_windows() -> list[dict[str, Any]]:
    """Per-process dedicated VRAM via Windows GPU performance counters.

    nvidia-smi cannot report per-process memory under the Windows display
    driver (WDDM); the '\\GPU Process Memory(*)\\Dedicated Usage' counter
    is what Task Manager reads. Instance names encode the pid, e.g.
    'pid_23584_luid_0x..._phys_0'. A process can have several instances;
    we sum them per pid. Process names come from Get-Process in the same
    call (so this works without psutil installed).
    """
    ps = (
        "$n=@{}; Get-Process | ForEach-Object { $n[$_.Id]=$_.ProcessName }; "
        "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' "
        "-ErrorAction SilentlyContinue).CounterSamples | "
        "Where-Object {$_.CookedValue -gt 0} | ForEach-Object { "
        "if ($_.InstanceName -match 'pid_(\\d+)_') { $p=[int]$Matches[1]; "
        "$nm=$n[$p]; if (-not $nm) {$nm='?'}; "
        "'{0};{1};{2}' -f $p,$nm,[int64]$_.CookedValue } }"
    )
    out = _run([
        "powershell", "-NoProfile", "-NonInteractive", "-Command", ps
    ], timeout=15)
    if not out:
        return []

    by_pid: dict[int, int] = {}
    names: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.strip().split(";")
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            val = int(parts[2])
        except ValueError:
            continue
        by_pid[pid] = by_pid.get(pid, 0) + val
        names[pid] = parts[1]

    procs = [
        {"pid": pid, "name": names.get(pid) or _pid_name(pid),
         "used_mb": b / (1024 * 1024), "device": None}
        for pid, b in by_pid.items()
    ]
    procs.sort(key=lambda p: p["used_mb"], reverse=True)
    return procs


def _gpu_processes_nvsmi() -> list[dict[str, Any]]:
    """Per-process VRAM via nvidia-smi (Linux / datacenter TCC driver)."""
    out = _run([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    procs = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        pid_s, name, mem_s = parts[:3]
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        try:
            used_mb = float(mem_s)
        except ValueError:
            used_mb = None  # '[N/A]' under unsupported drivers
        procs.append({
            "pid": pid,
            "name": name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
            "used_mb": used_mb,
            "device": None,
        })
    procs.sort(key=lambda p: (p["used_mb"] is None, -(p["used_mb"] or 0)))
    return procs


def gpu_processes() -> list[dict[str, Any]]:
    """Per-process VRAM, platform-aware.

    Returns dicts with keys: pid, name, used_mb, device. On Windows uses
    GPU performance counters; elsewhere uses nvidia-smi compute-apps.
    """
    if sys.platform == "win32":
        return _gpu_processes_windows()
    return _gpu_processes_nvsmi()


def _adapter_summary(adapter: Any) -> str:
    """Short label for a wgpu adapter, e.g. 'NVIDIA RTX A4000 [DiscreteGPU] via Vulkan'."""
    summary = getattr(adapter, "summary", None)
    if isinstance(summary, str) and summary:
        return summary
    info = getattr(adapter, "info", {}) or {}
    name = info.get("device", info.get("description", "?"))
    type_ = info.get("adapter_type", info.get("device_type", ""))
    backend = info.get("backend_type", "")
    label = name
    if type_:
        label += f" [{type_}]"
    if backend:
        label += f" via {backend}"
    return label


def render_gpu(live_adapter: Any | None = None) -> dict[str, Any] | None:
    """The wgpu adapter fastplotlib renders with.

    Pass ``live_adapter`` (``fig.renderer.device.adapter``) for ground truth
    — that is the only reliable source inside a running GUI. Without it this
    enumerates adapters and resolves the persisted ``gpu_index`` preference
    (falling back to wgpu's default); only safe to call outside an active
    render frame (e.g. the CLI).
    """
    if live_adapter is not None:
        info = getattr(live_adapter, "info", {}) or {}
        return {
            "summary": _adapter_summary(live_adapter),
            "name": info.get("device", info.get("description", "?")),
            "type": info.get("adapter_type", info.get("device_type", "?")),
            "source": "live",
            "index": None,
        }
    try:
        import fastplotlib as fpl
        adapters = fpl.enumerate_adapters()
    except Exception:
        return None
    if not adapters:
        return None
    try:
        from mbo_utilities.preferences import get_gpu_index
        idx = get_gpu_index()
    except Exception:
        idx = -1

    source = "preference"
    if not (0 <= idx < len(adapters)):
        idx, source = 0, "auto"
        try:
            import wgpu
            default_info = wgpu.gpu.request_adapter_sync().info
            infos = [a.info for a in adapters]
            idx = infos.index(default_info)
        except Exception:
            pass
    a = adapters[idx]
    info = getattr(a, "info", {}) or {}
    return {
        "summary": _adapter_summary(a),
        "name": info.get("device", info.get("description", "?")),
        "type": info.get("adapter_type", info.get("device_type", "?")),
        "source": source,
        "index": idx,
    }


def compute_gpu() -> dict[str, Any]:
    """The device suite2p / cellpose / cupy compute would use.

    Reflects the CUDA_VISIBLE_DEVICES policy: when compute is disabled this
    is CPU; otherwise it is the first visible CUDA device, named from
    nvidia-smi (or torch if already imported). The suite2p torch_device
    dropdown can still override a single run to CPU/MPS.
    """
    if gpu_compute_disabled():
        return {"backend": "cpu", "enabled": False, "name": "CPU (forced via env)",
                "index": None, "torch_index": None}

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = 0
    if cvd:
        first = cvd.split(",")[0].strip()
        if first.isdigit():
            phys = int(first)

    name = None
    for d in gpu_devices():
        if d["index"] == phys:
            name = d["name"]
            break

    torch_index = 0
    try:
        import sys as _sys
        if "torch" in _sys.modules:
            import torch
            if torch.cuda.is_available():
                torch_index = torch.cuda.current_device()
                name = torch.cuda.get_device_name(torch_index)
    except Exception:
        pass

    if name is None:
        return {"backend": "cpu", "enabled": False, "name": "CPU (no CUDA device)",
                "index": None, "torch_index": None}
    return {"backend": "cuda", "enabled": True, "name": name,
            "index": phys, "torch_index": torch_index}


def gpu_compute_disabled() -> bool:
    """True if CUDA compute is turned off via env (CVD empty/-1 or MBO_GPU off)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        return True
    raw = os.environ.get("MBO_GPU")
    if raw is not None and raw.strip().lower() in ("0", "off", "false", "no", "cpu", "none"):
        return True
    return False


def apply_gpu_policy(value: str | int | bool | None = None) -> str | None:
    """Resolve MBO_GPU (or an explicit value) into CUDA_VISIBLE_DEVICES.

    Call once at process startup, before torch/cupy import. Returns the
    CUDA_VISIBLE_DEVICES string that was set, or None if left unchanged.

    value None reads the MBO_GPU env var. Accepted values:
        falsey ("0"/"off"/"false"/"no"/"cpu"/"none") -> force CPU ("")
        "1"/"on"/"true"/"auto"/"gpu" or unset         -> unchanged
        an int or device list like "0,1"              -> pin those devices
    """
    if value is None:
        value = os.environ.get("MBO_GPU")
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in ("", "auto", "1", "on", "true", "yes", "gpu"):
        return None
    if token in ("0", "off", "false", "no", "cpu", "none"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return ""
    # device index or comma list -> pin. Order by PCI bus so the index
    # matches nvidia-smi (CUDA defaults to FASTEST_FIRST otherwise); a
    # user-set CUDA_DEVICE_ORDER still wins.
    if re.fullmatch(r"\d+(,\d+)*", token):
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = token
        return token
    return None


def format_gpu_report(show_processes: bool = False, top: int = 12) -> str:
    """Human-readable report: render GPU, compute GPU, device memory.

    Per-process VRAM is included only when ``show_processes`` is True.
    """
    lines: list[str] = []

    # roles: which GPU renders, which GPU computes
    rg = render_gpu()
    if rg is not None:
        tag = {"live": "", "preference": " (selected)",
               "auto": " (wgpu default)"}.get(rg["source"], "")
        lines.append(f"Render GPU  (fastplotlib): {rg['summary']}{tag}")
    cg = compute_gpu()
    if cg["enabled"]:
        lines.append(
            f"Compute GPU (suite2p/cellpose/cupy): {cg['name']}  (cuda:{cg['torch_index']})"
        )
    else:
        lines.append(f"Compute GPU (suite2p/cellpose/cupy): {cg['name']}")

    devices = gpu_devices()
    if not devices:
        lines.append("")
        if not has_nvidia_smi():
            lines.append("No NVIDIA GPU detected (nvidia-smi not found).")
        else:
            lines.append("nvidia-smi returned no devices.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Device memory:")
    for d in devices:
        total = d["total_mb"] or 0
        used = d["used_mb"] or 0
        free = d["free_mb"] or 0
        pct = (used / total * 100) if total else 0
        extra = []
        if d["util_pct"] is not None:
            extra.append(f"util {d['util_pct']:.0f}%")
        if d["temp_c"] is not None:
            extra.append(f"{d['temp_c']:.0f}C")
        extra_s = (", " + ", ".join(extra)) if extra else ""
        lines.append(
            f"  GPU {d['index']}: {d['name']} - {used:.0f}/{total:.0f} MB "
            f"used ({pct:.0f}%), {free:.0f} MB free{extra_s}"
        )

    if show_processes:
        procs = gpu_processes()
        lines.append("")
        if not procs:
            lines.append("Processes: none reported.")
        else:
            lines.append(f"Processes (top {min(top, len(procs))} by VRAM):")
            lines.append(f"  {'PID':>7}  {'VRAM':>10}  NAME")
            for p in procs[:top]:
                mem = p["used_mb"]
                mem_s = f"{mem:.0f} MB" if mem is not None else "n/a"
                lines.append(f"  {p['pid']:>7}  {mem_s:>10}  {p['name']}")
            if sys.platform == "win32":
                lines.append(
                    "  (per-process VRAM from Windows GPU counters; "
                    "matches Task Manager)"
                )
    return "\n".join(lines)
