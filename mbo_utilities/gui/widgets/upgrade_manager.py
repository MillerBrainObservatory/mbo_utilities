"""upgrade manager widget for checking and installing mbo_utilities updates."""

import json
import subprocess
import sys
import threading
import urllib.request
from dataclasses import dataclass, field
from enum import Enum

from imgui_bundle import imgui

# minimum version to show in version selector
MIN_VERSION = "2.2.0"


class CheckStatus(Enum):
    """status of the version check."""

    IDLE = "idle"
    CHECKING = "checking"
    DONE = "done"
    ERROR = "error"


class UpgradeStatus(Enum):
    """status of the upgrade process."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


def _parse_version(v: str) -> tuple:
    """parse version string to tuple for comparison."""
    try:
        # handle versions like "2.2.0", "2.10.1", "3.0.0a1"
        # strip any pre-release suffixes for comparison
        base = v.split("a")[0].split("b")[0].split("rc")[0].split(".dev")[0]
        parts = base.split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _version_gte(v1: str, v2: str) -> bool:
    """check if v1 >= v2."""
    return _parse_version(v1) >= _parse_version(v2)


def _get_install_type() -> str:
    """Determine the installation type based on executable path."""
    exe_str = sys.executable.lower()
    if ".local" in exe_str or ("uv" in exe_str and "tools" in exe_str):
        return "uv tool"
    elif "envs" in exe_str or "venv" in exe_str or ".venv" in exe_str:
        return "environment"
    elif "conda" in exe_str or "miniconda" in exe_str or "anaconda" in exe_str:
        return "conda"
    else:
        return "system"


@dataclass
class UpgradeManager:
    """
    manages version checking and upgrading for mbo_utilities.

    uses pypi as the version database.
    """

    enabled: bool = True
    current_version: str = "unknown"
    latest_version: str | None = None
    available_versions: list[str] = field(default_factory=list)
    selected_version_idx: int = 0
    check_status: CheckStatus = CheckStatus.IDLE
    upgrade_status: UpgradeStatus = UpgradeStatus.IDLE
    error_message: str = ""
    upgrade_message: str = ""
    executable_path: str = ""
    install_type: str = ""
    _check_thread: threading.Thread | None = field(default=None, repr=False)
    _upgrade_thread: threading.Thread | None = field(default=None, repr=False)

    def __post_init__(self):
        self._load_current_version()
        self.executable_path = sys.executable
        self.install_type = _get_install_type()

    def _load_current_version(self):
        """Load the current installed version."""
        try:
            import mbo_utilities
            self.current_version = getattr(mbo_utilities, "__version__", "unknown")
        except ImportError:
            self.current_version = "unknown"

    def check_for_upgrade(self):
        """Start async version check against pypi."""
        if self.check_status == CheckStatus.CHECKING:
            return  # already checking

        self.check_status = CheckStatus.CHECKING
        self.error_message = ""

        def _check():
            try:
                url = "https://pypi.org/pypi/mbo-utilities/json"
                with urllib.request.urlopen(url, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    self.latest_version = data["info"]["version"]

                    # get all available versions >= MIN_VERSION
                    all_versions = list(data.get("releases", {}).keys())

                    # filter to stable releases >= MIN_VERSION
                    valid_versions = []
                    for v in all_versions:
                        # skip pre-release versions (alpha, beta, rc, dev)
                        if any(tag in v for tag in ["a", "b", "rc", "dev"]):
                            continue
                        if _version_gte(v, MIN_VERSION):
                            valid_versions.append(v)

                    # sort descending (newest first)
                    valid_versions.sort(key=_parse_version, reverse=True)
                    self.available_versions = valid_versions

                    # set selected to current version if available
                    if self.current_version in valid_versions:
                        self.selected_version_idx = valid_versions.index(self.current_version)
                    else:
                        self.selected_version_idx = 0

                    self.check_status = CheckStatus.DONE
            except Exception as e:
                self.error_message = str(e)
                self.check_status = CheckStatus.ERROR

        self._check_thread = threading.Thread(target=_check, daemon=True)
        self._check_thread.start()

    def start_upgrade(self):
        """Start async upgrade to latest version."""
        if self.latest_version:
            self.install_version(self.latest_version)

    def install_version(self, version: str):
        """Start async install of a specific version."""
        if self.upgrade_status == UpgradeStatus.RUNNING:
            return  # already running

        self.upgrade_status = UpgradeStatus.RUNNING
        self.upgrade_message = ""

        def _install():
            try:
                package_spec = f"mbo-utilities=={version}"

                # try uv first, fall back to pip
                result = subprocess.run(
                    [sys.executable, "-m", "uv", "pip", "install", package_spec],
                    check=False, capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode != 0:
                    # try regular pip
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package_spec],
                        check=False, capture_output=True,
                        text=True,
                        timeout=300,
                    )

                if result.returncode == 0:
                    self.upgrade_status = UpgradeStatus.SUCCESS
                    self.upgrade_message = f"installed v{version}. restart to use."
                    # reload version
                    self._load_current_version()
                else:
                    self.upgrade_status = UpgradeStatus.ERROR
                    self.upgrade_message = result.stderr or "install failed"
            except subprocess.TimeoutExpired:
                self.upgrade_status = UpgradeStatus.ERROR
                self.upgrade_message = "install timed out"
            except Exception as e:
                self.upgrade_status = UpgradeStatus.ERROR
                self.upgrade_message = str(e)

        self._upgrade_thread = threading.Thread(target=_install, daemon=True)
        self._upgrade_thread.start()

    @property
    def upgrade_available(self) -> bool:
        """Check if an upgrade is available."""
        if self.check_status != CheckStatus.DONE:
            return False
        if self.latest_version is None or self.current_version == "unknown":
            return False

        try:
            from packaging.version import parse
            return parse(self.current_version) < parse(self.latest_version)
        except ImportError:
            # fallback: simple string comparison
            return self.current_version != self.latest_version

    @property
    def is_dev_build(self) -> bool:
        """Check if running a dev build (newer than pypi)."""
        if self.check_status != CheckStatus.DONE:
            return False
        if self.latest_version is None or self.current_version == "unknown":
            return False

        try:
            from packaging.version import parse
            return parse(self.current_version) > parse(self.latest_version)
        except ImportError:
            return False


def draw_upgrade_manager(manager: UpgradeManager):
    """
    Draw the upgrade manager ui.

    shows current version, check button, version selector, and install option.
    """
    if not manager.enabled:
        return

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # header
    imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Version Info")
    imgui.spacing()

    # current version and install type
    imgui.text(f"Installed: v{manager.current_version}")
    imgui.same_line()
    imgui.text_disabled(f"({manager.install_type})")

    # executable path (truncated, with tooltip for full path)
    exe_display = manager.executable_path
    if len(exe_display) > 50:
        exe_display = "..." + exe_display[-47:]
    imgui.text_disabled(f"Python: {exe_display}")
    if imgui.is_item_hovered():
        imgui.set_tooltip(manager.executable_path)

    # latest version (if checked)
    if manager.check_status == CheckStatus.DONE and manager.latest_version:
        imgui.text_disabled(f"Latest: v{manager.latest_version}")

    imgui.spacing()

    # status display
    if manager.check_status == CheckStatus.CHECKING:
        imgui.text_disabled("Checking PyPI...")
    elif manager.check_status == CheckStatus.ERROR:
        imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), f"Error: {manager.error_message[:50]}")
    elif manager.check_status == CheckStatus.DONE:
        if manager.upgrade_available:
            imgui.text_colored(
                imgui.ImVec4(0.4, 1.0, 0.4, 1.0),
                f"Update available: v{manager.latest_version}"
            )
        elif manager.is_dev_build:
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
                "Development build"
            )
        else:
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
                "Up to date"
            )

    imgui.spacing()

    # check button
    button_width = 100
    checking = manager.check_status == CheckStatus.CHECKING
    upgrading = manager.upgrade_status == UpgradeStatus.RUNNING

    if checking:
        imgui.begin_disabled()

    if imgui.button("Check", imgui.ImVec2(button_width, 0)):
        manager.check_for_upgrade()

    if checking:
        imgui.end_disabled()

    # version selector (only show if we have versions)
    if manager.check_status == CheckStatus.DONE and manager.available_versions:
        imgui.spacing()
        imgui.text("Switch version:")

        if upgrading:
            imgui.begin_disabled()

        # version dropdown
        imgui.set_next_item_width(120)
        changed, new_idx = imgui.combo(
            "##version_select",
            manager.selected_version_idx,
            manager.available_versions
        )
        if changed:
            manager.selected_version_idx = new_idx

        imgui.same_line()

        # install button
        selected_ver = manager.available_versions[manager.selected_version_idx]
        is_current = selected_ver == manager.current_version

        if is_current:
            imgui.begin_disabled()

        # color button based on action
        if _version_gte(selected_ver, manager.current_version):
            # upgrade (green)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.6, 0.2, 1.0))
            btn_label = "Upgrade" if selected_ver != manager.current_version else "Current"
        else:
            # downgrade (orange)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.7, 0.5, 0.2, 1.0))
            btn_label = "Downgrade"

        if imgui.button(btn_label, imgui.ImVec2(button_width, 0)):
            manager.install_version(selected_ver)

        imgui.pop_style_color()

        if is_current:
            imgui.end_disabled()

        if upgrading:
            imgui.end_disabled()

        # show version count
        imgui.text_disabled(f"{len(manager.available_versions)} versions (>= {MIN_VERSION})")

    # install status
    if manager.upgrade_status == UpgradeStatus.RUNNING:
        imgui.spacing()
        imgui.text_disabled("Installing...")
    elif manager.upgrade_status == UpgradeStatus.SUCCESS:
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), manager.upgrade_message)
    elif manager.upgrade_status == UpgradeStatus.ERROR:
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), f"Error: {manager.upgrade_message[:60]}")

    imgui.spacing()
