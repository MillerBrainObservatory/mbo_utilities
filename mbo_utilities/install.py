"""installation status checker for mbo_utilities optional dependencies.

checks for proper GPU configuration of suite2p (pytorch), suite3d (cupy),
and rastermap. provides structured data for CLI and GUI display.

also provides HAS_* flags for quick import checks without actually importing.
"""

from dataclasses import dataclass, field
from enum import Enum
import importlib.util
import subprocess
import sys


# quick import availability checks (no actual imports)
def _check_import(module_name: str) -> bool:
    """check if a module can be imported without actually importing it."""
    return importlib.util.find_spec(module_name) is not None


# suite2p pipeline (lbm_suite2p_python is the main entry point)
HAS_SUITE2P: bool = _check_import("lbm_suite2p_python")

# suite3d volumetric registration (requires both suite3d AND cupy for GPU)
HAS_SUITE3D: bool = _check_import("suite3d") and _check_import("cupy")

# cupy for GPU acceleration (used by suite3d and cusignal)
HAS_CUPY: bool = _check_import("cupy")

# pytorch for neural network operations
HAS_TORCH: bool = _check_import("torch")

# rastermap for dimensionality reduction
HAS_RASTERMAP: bool = _check_import("rastermap")

# gui dependencies
# imgui bundle - core GUI framework
HAS_IMGUI: bool = _check_import("imgui_bundle")

# fastplotlib - visualization
HAS_FASTPLOTLIB: bool = _check_import("fastplotlib")

# pyside6 - optional Qt backend (forces Qt over glfw if available)
HAS_PYSIDE6: bool = _check_import("PySide6")


class Status(Enum):
    """installation status for a feature."""
    OK = "ok"           # installed and working
    WARN = "warn"       # installed but degraded (e.g., no GPU)
    ERROR = "error"     # installed but broken
    MISSING = "missing" # not installed


@dataclass
class FeatureStatus:
    """status of a single feature/package."""
    name: str
    status: Status
    version: str = ""
    message: str = ""
    gpu_ok: bool | None = None  # None = n/a, True/False for GPU features


@dataclass
class CudaInfo:
    """cuda environment information."""
    nvcc_version: str | None = None      # cuda toolkit version (nvcc)
    driver_version: str | None = None    # nvidia driver cuda version (nvidia-smi)
    pytorch_cuda: str | None = None      # pytorch compiled cuda version
    cupy_cuda: str | None = None         # cupy cuda runtime version
    device_name: str | None = None       # gpu name
    device_count: int = 0


@dataclass
class InstallStatus:
    """complete installation status."""
    mbo_version: str = ""
    python_version: str = ""
    cuda_info: CudaInfo = field(default_factory=CudaInfo)
    features: list[FeatureStatus] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        """true if all installed features are working properly."""
        return all(f.status in (Status.OK, Status.MISSING) for f in self.features)


def _get_nvcc_version() -> str | None:
    """get cuda toolkit version from nvcc."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # parse "release X.Y" from output
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split("release")[-1].strip().split(",")[0]
                    return parts.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_nvidia_smi_cuda() -> str | None:
    """get driver-supported cuda version from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # also get cuda version
            result2 = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result2.returncode == 0:
                # parse "CUDA Version: X.Y" from output
                for line in result2.stdout.split("\n"):
                    if "CUDA Version" in line:
                        parts = line.split("CUDA Version:")[-1].strip().split()[0]
                        return parts
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _check_pytorch() -> tuple[FeatureStatus, str | None]:
    """check pytorch installation and cuda support."""
    pytorch_cuda = None
    try:
        import torch
        version = torch.__version__

        # check cuda availability
        if torch.cuda.is_available():
            pytorch_cuda = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            return FeatureStatus(
                name="PyTorch",
                status=Status.OK,
                version=version,
                message=f"CUDA {pytorch_cuda}, {device_name}",
                gpu_ok=True
            ), pytorch_cuda
        else:
            # pytorch installed but no cuda
            pytorch_cuda = getattr(torch.version, "cuda", None)
            return FeatureStatus(
                name="PyTorch",
                status=Status.WARN,
                version=version,
                message="CPU only (no CUDA)",
                gpu_ok=False
            ), pytorch_cuda
    except ImportError:
        return FeatureStatus(
            name="PyTorch",
            status=Status.MISSING,
            message="not installed"
        ), None


def _check_cupy() -> tuple[FeatureStatus, str | None]:
    """check cupy installation and cuda support."""
    cupy_cuda = None
    try:
        import cupy as cp
        version = cp.__version__

        # check cuda functionality
        try:
            # simple cuda test
            _ = cp.array([1, 2, 3])
            cuda_ver = cp.cuda.runtime.runtimeGetVersion()
            cuda_major = cuda_ver // 1000
            cuda_minor = (cuda_ver % 1000) // 10
            cupy_cuda = f"{cuda_major}.{cuda_minor}"

            # test nvrtc (required for suite3d)
            try:
                kernel = cp.ElementwiseKernel(
                    'float32 x', 'float32 y', 'y = x * 2', 'test_kernel'
                )
                test_in = cp.array([1.0], dtype='float32')
                test_out = cp.empty_like(test_in)
                kernel(test_in, test_out)

                device = cp.cuda.Device()
                device_name = device.attributes.get("Name", "GPU")
                return FeatureStatus(
                    name="CuPy",
                    status=Status.OK,
                    version=version,
                    message=f"CUDA {cupy_cuda}, {device_name}",
                    gpu_ok=True
                ), cupy_cuda
            except Exception:
                return FeatureStatus(
                    name="CuPy",
                    status=Status.ERROR,
                    version=version,
                    message="NVRTC missing (install CUDA toolkit)",
                    gpu_ok=False
                ), cupy_cuda
        except Exception as e:
            return FeatureStatus(
                name="CuPy",
                status=Status.ERROR,
                version=version,
                message=f"CUDA init failed: {str(e)[:40]}",
                gpu_ok=False
            ), None
    except ImportError:
        return FeatureStatus(
            name="CuPy",
            status=Status.MISSING,
            message="not installed"
        ), None


def _check_suite2p() -> FeatureStatus:
    """check suite2p installation."""
    try:
        import suite2p
        version = getattr(suite2p, "__version__", "installed")
        return FeatureStatus(
            name="Suite2p",
            status=Status.OK,
            version=version,
            message="ready"
        )
    except ImportError:
        return FeatureStatus(
            name="Suite2p",
            status=Status.MISSING,
            message="not installed"
        )


def _check_suite3d() -> FeatureStatus:
    """check suite3d installation."""
    try:
        import suite3d
        version = getattr(suite3d, "__version__", "installed")
        return FeatureStatus(
            name="Suite3D",
            status=Status.OK,
            version=version,
            message="ready"
        )
    except ImportError:
        return FeatureStatus(
            name="Suite3D",
            status=Status.MISSING,
            message="not installed"
        )


def _check_rastermap() -> FeatureStatus:
    """check rastermap installation."""
    try:
        import rastermap
        version = getattr(rastermap, "__version__", "installed")
        return FeatureStatus(
            name="Rastermap",
            status=Status.OK,
            version=version,
            message="ready"
        )
    except ImportError:
        return FeatureStatus(
            name="Rastermap",
            status=Status.MISSING,
            message="not installed"
        )


def check_installation(callback: type[object] | None = None) -> InstallStatus:
    """run full installation check and return structured status.
    
    Args:
        callback: optional callable(progress: float, message: str) for status updates
    """
    def _update(p: float, msg: str):
        if callback:
            try:
                callback(p, msg)
            except Exception:
                pass

    status = InstallStatus()
    _update(0.1, "Checking Python version...")

    # basic info
    try:
        import mbo_utilities
        status.mbo_version = getattr(mbo_utilities, "__version__", "unknown")
    except ImportError:
        status.mbo_version = "not installed"

    status.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # cuda environment
    _update(0.2, "Checking CUDA environment...")
    status.cuda_info.nvcc_version = _get_nvcc_version()
    status.cuda_info.driver_version = _get_nvidia_smi_cuda()

    # check pytorch (needed for suite2p GPU)
    _update(0.3, "Checking PyTorch...")
    pytorch_status, pytorch_cuda = _check_pytorch()
    status.cuda_info.pytorch_cuda = pytorch_cuda
    status.features.append(pytorch_status)

    # check cupy (needed for suite3d)
    _update(0.5, "Checking CuPy...")
    cupy_status, cupy_cuda = _check_cupy()
    status.cuda_info.cupy_cuda = cupy_cuda
    status.features.append(cupy_status)

    # get gpu name from cupy if available
    if cupy_status.gpu_ok:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            status.cuda_info.device_name = device.attributes.get("Name", None)
            status.cuda_info.device_count = cp.cuda.runtime.getDeviceCount()
        except Exception:
            pass

    # check pipelines
    _update(0.7, "Checking Suite2p...")
    suite2p_status = _check_suite2p()
    # if suite2p is installed but pytorch has no GPU, warn
    if suite2p_status.status == Status.OK and pytorch_status.gpu_ok is False:
        suite2p_status = FeatureStatus(
            name="Suite2p",
            status=Status.WARN,
            version=suite2p_status.version,
            message="PyTorch has no GPU support",
            gpu_ok=False
        )
    elif suite2p_status.status == Status.OK:
        suite2p_status.gpu_ok = pytorch_status.gpu_ok
    status.features.append(suite2p_status)

    _update(0.8, "Checking Suite3D...")
    suite3d_status = _check_suite3d()
    # if suite3d is installed but cupy has no GPU, warn
    if suite3d_status.status == Status.OK and cupy_status.gpu_ok is False:
        suite3d_status = FeatureStatus(
            name="Suite3D",
            status=Status.WARN,
            version=suite3d_status.version,
            message="CuPy has no GPU support",
            gpu_ok=False
        )
    elif suite3d_status.status == Status.OK:
        suite3d_status.gpu_ok = cupy_status.gpu_ok
    status.features.append(suite3d_status)

    _update(0.9, "Checking Rastermap...")
    status.features.append(_check_rastermap())
    
    _update(1.0, "Done")
    return status


def print_status_cli(status: InstallStatus):
    """print installation status to CLI with colors."""
    import click

    click.echo(f"\nmbo_utilities v{status.mbo_version} | Python {status.python_version}")
    click.echo("=" * 50)

    # cuda info
    if status.cuda_info.nvcc_version or status.cuda_info.driver_version:
        click.echo("\nCUDA Environment:")
        if status.cuda_info.nvcc_version:
            click.echo(f"  CUDA Toolkit (nvcc): {status.cuda_info.nvcc_version}")
        if status.cuda_info.driver_version:
            click.echo(f"  Driver CUDA:         {status.cuda_info.driver_version}")
        if status.cuda_info.device_name:
            click.echo(f"  GPU:                 {status.cuda_info.device_name}")

    # features table
    click.echo("\nFeatures:")
    for f in status.features:
        # format version string (skip 'v' prefix if version is 'installed')
        if f.version and f.version != "installed":
            ver = f" v{f.version}"
        elif f.version == "installed":
            ver = ""
        else:
            ver = ""

        if f.status == Status.OK:
            icon = click.style("[OK]", fg="green")
            msg = click.style(f"{f.name}{ver}", fg="green")
            extra = f" ({f.message})" if f.message and f.message != "ready" else ""
        elif f.status == Status.WARN:
            icon = click.style("[! ]", fg="yellow")
            msg = click.style(f"{f.name}{ver}", fg="yellow")
            extra = f" - {f.message}" if f.message else ""
        elif f.status == Status.ERROR:
            icon = click.style("[X ]", fg="red")
            msg = click.style(f"{f.name}{ver}", fg="red")
            extra = f" - {f.message}" if f.message else ""
        else:  # MISSING
            icon = click.style("[ -]", fg="bright_black")
            msg = click.style(f"{f.name}", fg="bright_black")
            extra = " (not installed)"

        click.echo(f"  {icon} {msg}{extra}")

    # summary
    click.echo("")
    if status.all_ok:
        click.secho("Installation OK", fg="green", bold=True)
    else:
        click.secho("Issues detected - see warnings above", fg="yellow")
