"""environment-specific cache for fast startup.

caches package availability, versions, and other slow-to-compute info
per Python environment to avoid redundant checks on every run.

cache location: ~/mbo/cache/envs/{env_hash}.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def get_env_hash() -> str:
    """generate short hash from sys.executable path."""
    return hashlib.md5(sys.executable.encode()).hexdigest()[:8]


def get_cache_dir() -> Path:
    """get cache directory for environment caches."""
    # avoid importing file_io (has heavy deps like numpy)
    cache_dir = Path.home() / "mbo" / "cache" / "envs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path() -> Path:
    """get cache file path for current environment."""
    return get_cache_dir() / f"{get_env_hash()}.json"


def load_cache() -> dict | None:
    """load cache for current environment, return None if missing/corrupted."""
    path = get_cache_path()
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cache(data: dict) -> None:
    """save cache for current environment."""
    path = get_cache_path()
    data["env_path"] = sys.executable
    data["env_hash"] = get_env_hash()
    data["last_updated"] = datetime.now().isoformat()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # don't crash if cache write fails


def clear_cache() -> bool:
    """delete cache for current environment. returns True if deleted."""
    path = get_cache_path()
    if path.exists():
        try:
            path.unlink()
            return True
        except Exception:
            pass
    return False


def clear_all_caches() -> int:
    """delete all environment caches. returns count of deleted files."""
    cache_dir = get_cache_dir()
    count = 0
    for f in cache_dir.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count


def is_cache_valid(cache: dict | None, max_age_hours: int = 24) -> bool:
    """check if cache is still valid (not expired, same mbo version)."""
    if not cache:
        return False
    try:
        from mbo_utilities import __version__
        if cache.get("mbo_version") != __version__:
            return False  # invalidate on version change
        updated = datetime.fromisoformat(cache["last_updated"])
        return datetime.now() - updated < timedelta(hours=max_age_hours)
    except Exception:
        return False


def get_cached_packages() -> dict | None:
    """get cached package info, or None if cache invalid."""
    cache = load_cache()
    if is_cache_valid(cache):
        return cache.get("packages")
    return None


def get_cached_install_type() -> str | None:
    """get cached install type, or None if cache invalid."""
    cache = load_cache()
    if is_cache_valid(cache):
        return cache.get("install_type")
    return None


def get_cached_pypi_version(max_age_hours: int = 1) -> str | None:
    """get cached pypi version if checked recently."""
    cache = load_cache()
    if not cache:
        return None
    pypi = cache.get("pypi_check", {})
    try:
        checked = datetime.fromisoformat(pypi.get("checked_at", ""))
        if datetime.now() - checked < timedelta(hours=max_age_hours):
            return pypi.get("latest_version")
    except Exception:
        pass
    return None


def update_cache(key: str, value: Any) -> None:
    """update a single key in the cache."""
    cache = load_cache() or {}
    cache[key] = value
    save_cache(cache)


def update_pypi_cache(latest_version: str) -> None:
    """update pypi version check in cache."""
    cache = load_cache() or {}
    cache["pypi_check"] = {
        "latest_version": latest_version,
        "checked_at": datetime.now().isoformat(),
    }
    save_cache(cache)


def build_full_cache() -> dict:
    """build complete cache data from scratch."""
    from mbo_utilities import __version__

    cache = {
        "mbo_version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "install_type": _detect_install_type(),
        "packages": _check_all_packages(),
    }
    return cache


def _detect_install_type() -> str:
    """detect how mbo_utilities is installed."""
    exe_str = sys.executable.lower()
    if ".local" in exe_str or ("uv" in exe_str and "tools" in exe_str):
        return "uv tool"
    elif "envs" in exe_str or "venv" in exe_str or ".venv" in exe_str:
        return "environment"
    elif "conda" in exe_str or "miniconda" in exe_str or "anaconda" in exe_str:
        return "conda"
    else:
        return "system"


def _check_import(module_name: str) -> bool:
    """check if a module can be imported without actually importing it."""
    import importlib.util
    return importlib.util.find_spec(module_name) is not None


def _get_package_version(module_name: str) -> str | None:
    """get installed version of a package."""
    try:
        from importlib.metadata import version
        return version(module_name)
    except Exception:
        return None


def _check_all_packages() -> dict:
    """check availability and versions of all optional packages."""
    packages = {}

    # package definitions: (cache_key, module_to_check, package_name_for_version)
    checks = [
        ("suite2p", "lbm_suite2p_python", "lbm-suite2p-python"),
        ("suite3d", "suite3d", "suite3d"),
        ("cupy", "cupy", "cupy"),
        ("torch", "torch", "torch"),
        ("rastermap", "rastermap", "rastermap"),
        ("imgui_bundle", "imgui_bundle", "imgui-bundle"),
        ("fastplotlib", "fastplotlib", "fastplotlib"),
        ("pyqt6", "PyQt6", "PyQt6"),
        ("napari", "napari", "napari"),
        ("napari_ome_zarr", "napari_ome_zarr", "napari-ome-zarr"),
        ("napari_animation", "napari_animation", "napari-animation"),
    ]

    for key, module, pkg_name in checks:
        available = _check_import(module)
        info: dict[str, Any] = {"available": available}
        if available:
            ver = _get_package_version(pkg_name)
            if ver:
                info["version"] = ver
        packages[key] = info

    return packages


def ensure_cache() -> dict:
    """ensure cache exists and is valid, building if needed."""
    cache = load_cache()
    if is_cache_valid(cache):
        return cache

    # build fresh cache
    cache = build_full_cache()
    save_cache(cache)
    return cache
