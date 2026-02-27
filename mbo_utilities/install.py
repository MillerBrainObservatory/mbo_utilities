"""installation status checker for mbo_utilities.

checks gpu config (pytorch cuda, cupy cuda), optional deps,
qt conflicts, and gui availability. provides structured data
for CLI and GUI display.
"""

from dataclasses import dataclass, field
from enum import Enum
import importlib.util
import subprocess
import sys
import contextlib


def _has_module(name: str) -> bool:
    """check if a module is importable."""
    return importlib.util.find_spec(name) is not None


def _get_version(pkg: str) -> str:
    """get installed package version or empty string."""
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return ""


# HAS_* flags - lightweight, uses find_spec only
HAS_SUITE2P: bool = _has_module("suite2p")
HAS_SUITE3D: bool = _has_module("suite3d")
HAS_CUPY: bool = _has_module("cupy")
HAS_TORCH: bool = _has_module("torch")
HAS_RASTERMAP: bool = _has_module("rastermap")
HAS_IMGUI: bool = _has_module("imgui_bundle")
HAS_FASTPLOTLIB: bool = _has_module("fastplotlib")
HAS_PYQT6: bool = _has_module("PyQt6")
HAS_NAPARI: bool = _has_module("napari")
HAS_GUI: bool = HAS_IMGUI and HAS_FASTPLOTLIB and HAS_PYQT6


class Status(Enum):
    """installation status for a feature."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    MISSING = "missing"


@dataclass
class FeatureStatus:
    """status of a single feature/package."""

    name: str
    status: Status
    version: str = ""
    message: str = ""
    hint: str = ""
    gpu_ok: bool | None = None


@dataclass
class CudaInfo:
    """cuda environment information."""

    nvcc_version: str | None = None
    driver_version: str | None = None
    pytorch_cuda: str | None = None
    cupy_cuda: str | None = None
    device_name: str | None = None
    device_count: int = 0


@dataclass
class InstallStatus:
    """complete installation status."""

    mbo_version: str = ""
    python_version: str = ""
    cuda_info: CudaInfo = field(default_factory=CudaInfo)
    features: list[FeatureStatus] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(
            f.status in (Status.OK, Status.MISSING) for f in self.features
        ) and not self.conflicts


# cuda environment helpers

def _get_nvcc_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    return line.split("release")[-1].strip().split(",")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_nvidia_smi_cuda() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"], check=False, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    return line.split("CUDA Version:")[-1].strip().split()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# individual feature checks - each returns FeatureStatus
# these actually import and test, not just find_spec

def _check_pytorch() -> tuple[FeatureStatus, str | None]:
    """check pytorch and verify cuda works."""
    try:
        import torch
        version = torch.__version__

        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:
                device_name = "unknown GPU"
            return FeatureStatus(
                name="PyTorch",
                status=Status.OK,
                version=version,
                message=f"CUDA {cuda_ver}, {device_name}",
                gpu_ok=True,
            ), cuda_ver

        # cpu-only build or no gpu
        compiled_cuda = getattr(torch.version, "cuda", None)
        if compiled_cuda:
            msg = f"compiled for CUDA {compiled_cuda} but no GPU detected"
        else:
            msg = "CPU-only build"
        return FeatureStatus(
            name="PyTorch",
            status=Status.WARN,
            version=version,
            message=msg,
            hint="uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126",
            gpu_ok=False,
        ), compiled_cuda
    except ImportError:
        return FeatureStatus(
            name="PyTorch",
            status=Status.MISSING,
            message="not installed",
            hint="uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126",
        ), None
    except Exception as e:
        return FeatureStatus(
            name="PyTorch",
            status=Status.ERROR,
            message=f"import failed: {str(e)[:60]}",
        ), None


def _check_cupy() -> tuple[FeatureStatus, str | None]:
    """check cupy with actual gpu test."""
    try:
        import cupy as cp
        version = cp.__version__
    except ImportError:
        return FeatureStatus(
            name="CuPy",
            status=Status.MISSING,
            message="not installed (needed for Suite3D GPU)",
            hint="uv pip install cupy-cuda12x",
        ), None
    except Exception as e:
        return FeatureStatus(
            name="CuPy",
            status=Status.ERROR,
            message=f"import failed: {str(e)[:60]}",
            hint="uv pip install cupy-cuda12x",
        ), None

    # test basic gpu allocation
    try:
        _ = cp.array([1, 2, 3])
        cuda_ver_int = cp.cuda.runtime.runtimeGetVersion()
        cuda_ver = f"{cuda_ver_int // 1000}.{(cuda_ver_int % 1000) // 10}"
    except Exception as e:
        return FeatureStatus(
            name="CuPy",
            status=Status.ERROR,
            version=version,
            message=f"CUDA init failed: {str(e)[:50]}",
            gpu_ok=False,
        ), None

    # test nvrtc kernel compilation (suite3d needs this)
    try:
        kernel = cp.ElementwiseKernel(
            "float32 x", "float32 y", "y = x * 2", "install_check_kernel"
        )
        test_in = cp.array([1.0], dtype="float32")
        kernel(test_in, cp.empty_like(test_in))
    except Exception:
        return FeatureStatus(
            name="CuPy",
            status=Status.ERROR,
            version=version,
            message=f"CUDA {cuda_ver} ok but NVRTC failed (install CUDA toolkit)",
            hint="install full CUDA toolkit from https://developer.nvidia.com/cuda-downloads",
            gpu_ok=False,
        ), cuda_ver

    try:
        device = cp.cuda.Device()
        device_name = device.attributes.get("Name", "GPU")
    except Exception:
        device_name = "GPU"

    return FeatureStatus(
        name="CuPy",
        status=Status.OK,
        version=version,
        message=f"CUDA {cuda_ver}, {device_name}",
        gpu_ok=True,
    ), cuda_ver


def _check_gui() -> FeatureStatus:
    """check gui stack (imgui, fastplotlib, pyqt6, wgpu, rendercanvas)."""
    missing = []
    versions = []

    for mod, pkg, label in [
        ("imgui_bundle", "imgui-bundle", "imgui-bundle"),
        ("fastplotlib", "mbo-fastplotlib", "mbo-fastplotlib"),
        ("PyQt6", "PyQt6", "pyqt6"),
        ("wgpu", "wgpu", "wgpu"),
        ("rendercanvas", "rendercanvas", "rendercanvas"),
        ("pygfx", "pygfx", "pygfx"),
    ]:
        if _has_module(mod):
            ver = _get_version(pkg)
            if ver:
                versions.append(f"{label} {ver}")
        else:
            missing.append(label)

    if missing:
        return FeatureStatus(
            name="GUI",
            status=Status.MISSING,
            message=f"missing: {', '.join(missing)}",
            hint='uv pip install "mbo_utilities[gui]"',
        )

    # verify wgpu can enumerate adapters
    try:
        import wgpu
        enumerate = getattr(wgpu.gpu, "enumerate_adapters_sync", wgpu.gpu.enumerate_adapters)
        adapters = enumerate()
        if not adapters:
            return FeatureStatus(
                name="GUI",
                status=Status.WARN,
                message="no GPU adapters found by wgpu",
                hint="check GPU drivers and vulkan/dx12 support",
            )
    except Exception as e:
        return FeatureStatus(
            name="GUI",
            status=Status.WARN,
            message=f"wgpu adapter check failed: {str(e)[:40]}",
        )

    return FeatureStatus(
        name="GUI",
        status=Status.OK,
        message=f"{len(versions)} packages ok",
    )


def _check_package(name: str, module: str, hint: str = "") -> FeatureStatus:
    """generic check: try importing a package."""
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", _get_version(module))
        return FeatureStatus(name=name, status=Status.OK, version=version)
    except ImportError:
        return FeatureStatus(
            name=name, status=Status.MISSING,
            message="not installed", hint=hint,
        )
    except Exception as e:
        return FeatureStatus(
            name=name, status=Status.ERROR,
            message=f"import error: {str(e)[:50]}",
        )


def _detect_conflicts() -> list[str]:
    """detect known packaging conflicts."""
    conflicts = []

    # pyqt5 + pyqt6 conflict
    if _has_module("PyQt5") and _has_module("PyQt6"):
        conflicts.append(
            "pyqt5 and pyqt6 both installed - this causes conflicts. "
            "run: uv pip uninstall pyqt5 pyqt5-qt5 pyqt5-sip"
        )

    # check torch cuda mismatch with cupy
    if _has_module("torch") and _has_module("cupy"):
        try:
            import torch
            import cupy as cp
            torch_cuda = getattr(torch.version, "cuda", None)
            if torch_cuda and torch.cuda.is_available():
                cupy_ver = cp.cuda.runtime.runtimeGetVersion()
                cupy_major = cupy_ver // 1000
                torch_major = int(torch_cuda.split(".")[0])
                if cupy_major != torch_major:
                    conflicts.append(
                        f"CUDA version mismatch: torch CUDA {torch_cuda}, "
                        f"cupy CUDA {cupy_major}.x - these should match"
                    )
        except Exception:
            pass

    return conflicts


def check_installation(callback=None) -> InstallStatus:
    """run full installation check and return structured status."""
    def _update(p: float, msg: str):
        if callback:
            with contextlib.suppress(Exception):
                callback(p, msg)

    status = InstallStatus()

    # basic info
    _update(0.05, "checking versions...")
    try:
        import mbo_utilities
        status.mbo_version = getattr(mbo_utilities, "__version__", "unknown")
    except ImportError:
        status.mbo_version = "not installed"
    status.python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # cuda environment
    _update(0.1, "checking CUDA...")
    status.cuda_info.nvcc_version = _get_nvcc_version()
    status.cuda_info.driver_version = _get_nvidia_smi_cuda()

    # pytorch
    _update(0.2, "checking PyTorch...")
    pytorch_status, pytorch_cuda = _check_pytorch()
    status.cuda_info.pytorch_cuda = pytorch_cuda
    status.features.append(pytorch_status)

    # get gpu info from pytorch if available
    if pytorch_status.gpu_ok:
        try:
            import torch
            status.cuda_info.device_name = torch.cuda.get_device_name(0)
            status.cuda_info.device_count = torch.cuda.device_count()
        except Exception:
            pass

    # cupy
    _update(0.35, "checking CuPy...")
    cupy_status, cupy_cuda = _check_cupy()
    status.cuda_info.cupy_cuda = cupy_cuda
    status.features.append(cupy_status)

    # get gpu info from cupy if pytorch didn't provide it
    if cupy_status.gpu_ok and not status.cuda_info.device_name:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            status.cuda_info.device_name = device.attributes.get("Name", None)
            status.cuda_info.device_count = cp.cuda.runtime.getDeviceCount()
        except Exception:
            pass

    # gui stack
    _update(0.5, "checking GUI...")
    status.features.append(_check_gui())

    # suite2p
    _update(0.6, "checking Suite2p...")
    s2p = _check_package(
        "Suite2p", "suite2p",
        hint='uv pip install "mbo_utilities[suite2p]"',
    )
    if s2p.status == Status.OK and pytorch_status.gpu_ok is False:
        s2p = FeatureStatus(
            name="Suite2p", status=Status.WARN, version=s2p.version,
            message="installed but torch has no GPU",
            hint="uv pip install torch --index-url https://download.pytorch.org/whl/cu126",
            gpu_ok=False,
        )
    elif s2p.status == Status.OK:
        s2p.gpu_ok = pytorch_status.gpu_ok
    status.features.append(s2p)

    # suite3d
    _update(0.7, "checking Suite3D...")
    s3d = _check_package(
        "Suite3D", "suite3d",
        hint='uv pip install "mbo_utilities[suite3d]"',
    )
    if s3d.status == Status.OK and not HAS_CUPY:
        s3d = FeatureStatus(
            name="Suite3D", status=Status.WARN, version=s3d.version,
            message="installed but cupy missing (no GPU acceleration)",
            hint="uv pip install cupy-cuda12x",
            gpu_ok=False,
        )
    elif s3d.status == Status.OK and cupy_status.gpu_ok is False:
        s3d = FeatureStatus(
            name="Suite3D", status=Status.WARN, version=s3d.version,
            message="installed but cupy GPU not working",
            hint=cupy_status.hint,
            gpu_ok=False,
        )
    elif s3d.status == Status.OK:
        s3d.gpu_ok = cupy_status.gpu_ok
    status.features.append(s3d)

    # lbm_suite2p_python
    _update(0.8, "checking LBM-Suite2p-Python...")
    status.features.append(_check_package(
        "LBM-Suite2p-Python", "lbm_suite2p_python",
        hint="uv pip install lbm_suite2p_python",
    ))

    # rastermap
    _update(0.85, "checking Rastermap...")
    status.features.append(_check_package(
        "Rastermap", "rastermap",
        hint='uv pip install "mbo_utilities[rastermap]"',
    ))

    # napari
    _update(0.9, "checking Napari...")
    status.features.append(_check_package(
        "Napari", "napari",
        hint='uv pip install "mbo_utilities[napari]"',
    ))

    # conflicts
    _update(0.95, "checking conflicts...")
    status.conflicts = _detect_conflicts()

    _update(1.0, "done")
    return status


def print_status_cli(status: InstallStatus):
    """print installation status to CLI with colors."""
    import click

    click.echo(f"\nmbo_utilities v{status.mbo_version} | Python {status.python_version}")
    click.echo("-" * 50)

    # cuda info
    if status.cuda_info.nvcc_version or status.cuda_info.driver_version:
        click.echo("\nCUDA:")
        if status.cuda_info.device_name:
            click.echo(f"  GPU:          {status.cuda_info.device_name}")
        if status.cuda_info.driver_version:
            click.echo(f"  driver CUDA:  {status.cuda_info.driver_version}")
        if status.cuda_info.nvcc_version:
            click.echo(f"  toolkit:      {status.cuda_info.nvcc_version}")
        if status.cuda_info.pytorch_cuda:
            click.echo(f"  torch CUDA:   {status.cuda_info.pytorch_cuda}")
        if status.cuda_info.cupy_cuda:
            click.echo(f"  cupy CUDA:    {status.cuda_info.cupy_cuda}")
    elif not status.cuda_info.device_name:
        click.echo("\nCUDA:")
        click.secho("  no GPU detected (nvidia-smi not found)", fg="yellow")

    # features
    click.echo("\npackages:")
    for f in status.features:
        ver = f" v{f.version}" if f.version and f.version != "installed" else ""

        if f.status == Status.OK:
            icon = click.style("[ok]", fg="green")
            name = click.style(f"{f.name}{ver}", fg="green")
            extra = f"  {f.message}" if f.message and "ok" not in f.message else ""
        elif f.status == Status.WARN:
            icon = click.style("[!!]", fg="yellow")
            name = click.style(f"{f.name}{ver}", fg="yellow")
            extra = f"  {f.message}" if f.message else ""
        elif f.status == Status.ERROR:
            icon = click.style("[xx]", fg="red")
            name = click.style(f"{f.name}{ver}", fg="red")
            extra = f"  {f.message}" if f.message else ""
        else:
            icon = click.style("[ -]", fg="bright_black")
            name = click.style(f.name, fg="bright_black")
            extra = ""

        click.echo(f"  {icon} {name}{extra}")
        if f.hint and f.status in (Status.MISSING, Status.WARN, Status.ERROR):
            click.echo(f"       {click.style(f.hint, fg='cyan')}")

    # conflicts
    if status.conflicts:
        click.echo("")
        click.secho("conflicts:", fg="red", bold=True)
        for c in status.conflicts:
            click.echo(f"  {click.style('!', fg='red')} {c}")

    # summary
    click.echo("")
    if status.all_ok:
        click.secho("installation ok", fg="green", bold=True)
    else:
        click.secho("issues detected", fg="yellow")
