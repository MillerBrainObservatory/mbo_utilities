"""Create a desktop shortcut/launcher for the mbo GUI.

Windows: a .lnk on the Desktop that runs mbo.exe through a hidden-window
VBScript launcher (no console), using the bundled icon.
Linux: a freedesktop .desktop entry on the Desktop.

This mirrors the desktop-shortcut step of scripts/install.ps1 so local
installs (plain pip/uv pip install) can get the same icon without the web
installer. The bundled icon is read from the installed package, so no
download is needed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "Miller Brain Studio"
DESCRIPTION = "MBO Image Viewer"


def _find_mbo_exe() -> Path | None:
    """Locate the mbo launcher for the running install.

    Prefer the launcher next to the running interpreter so a local env
    targets itself rather than a global mbo that happens to be on PATH.
    """
    base = Path(sys.executable).parent
    names = ["mbo.exe", "mbo"] if os.name == "nt" else ["mbo"]
    for d in (base, base / "Scripts", base / "bin"):
        for n in names:
            p = d / n
            if p.exists():
                return p
    exe = shutil.which("mbo")
    return Path(exe) if exe else None


def _bundled_icon(suffix: str) -> Path | None:
    p = Path(__file__).resolve().parent / "assets" / "app_settings" / f"icon{suffix}"
    return p if p.exists() else None


def _copy_icon(suffix: str, base: Path) -> str:
    """Copy the bundled icon next to the launcher so the shortcut keeps its
    icon even if the package is upgraded or moved. Returns the path to use,
    or "" if no icon is available."""
    src = _bundled_icon(suffix)
    if src is None:
        return ""
    dst = base / f"icon{suffix}"
    try:
        shutil.copyfile(src, dst)
        return str(dst)
    except OSError:
        return str(src)


def create_desktop_shortcut(name: str = APP_NAME) -> Path:
    """Create a desktop shortcut that opens the GUI. Returns its path."""
    if sys.platform == "win32":
        return _create_windows_shortcut(name)
    if sys.platform.startswith("linux"):
        return _create_linux_shortcut(name)
    raise RuntimeError(f"Desktop shortcuts are not supported on {sys.platform}.")


def _create_windows_shortcut(name: str) -> Path:
    from mbo_utilities import get_mbo_dirs

    exe = _find_mbo_exe()
    if exe is None:
        raise RuntimeError("Could not find mbo.exe. Is mbo_utilities installed on PATH?")

    base = get_mbo_dirs()["base"]
    icon = _copy_icon(".ico", base)

    # hidden-window launcher: run mbo.exe with no console window
    vbs = base / "mbo_launcher.vbs"
    vbs.write_text(
        'Set WshShell = CreateObject("WScript.Shell")\r\n'
        f'WshShell.Run """{exe}""", 0, False\r\n',
        encoding="ascii",
    )

    # create the .lnk via WScript.Shell COM; values passed as env vars to
    # avoid quoting/escaping issues with paths
    ps = (
        "$ErrorActionPreference='Stop'\n"
        "$desktop=[Environment]::GetFolderPath('Desktop')\n"
        "$lnk=Join-Path $desktop ($env:MBO_SC_NAME + '.lnk')\n"
        "$sc=(New-Object -ComObject WScript.Shell).CreateShortcut($lnk)\n"
        "$sc.TargetPath='wscript.exe'\n"
        "$sc.Arguments='\"' + $env:MBO_SC_VBS + '\"'\n"
        "$sc.WorkingDirectory=[Environment]::GetFolderPath('UserProfile')\n"
        "if ($env:MBO_SC_ICON) { $sc.IconLocation=$env:MBO_SC_ICON }\n"
        "$sc.Description=$env:MBO_SC_DESC\n"
        "$sc.Save()\n"
        "Write-Output $lnk\n"
    )
    env = dict(os.environ)
    env.update(
        MBO_SC_NAME=name,
        MBO_SC_VBS=str(vbs),
        MBO_SC_ICON=icon,
        MBO_SC_DESC=DESCRIPTION,
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to create shortcut:\n"
            + (result.stderr.strip() or result.stdout.strip())
        )
    out = result.stdout.strip()
    return Path(out) if out else base / f"{name}.lnk"


def _create_linux_shortcut(name: str) -> Path:
    from mbo_utilities import get_mbo_dirs

    exe = _find_mbo_exe()
    if exe is None:
        raise RuntimeError(
            "Could not find the mbo launcher. Is mbo_utilities installed on PATH?"
        )

    base = get_mbo_dirs()["base"]
    icon = _copy_icon(".png", base)

    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    entry = desktop / "miller-brain-studio.desktop"
    entry.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={name}\n"
        f"Comment={DESCRIPTION}\n"
        f"Exec={exe}\n"
        + (f"Icon={icon}\n" if icon else "")
        + "Terminal=false\n"
        "Categories=Science;Graphics;\n",
        encoding="utf-8",
    )
    entry.chmod(0o755)
    # GNOME requires the launcher be marked trusted to run from the desktop
    try:
        subprocess.run(
            ["gio", "set", str(entry), "metadata::trusted", "true"],
            check=False, capture_output=True,
        )
    except FileNotFoundError:
        pass
    return entry
