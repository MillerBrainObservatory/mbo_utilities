#!/usr/bin/env bash

# MBO Utilities Installation Script for Linux/macOS
# Installs mbo CLI via uv tool, optionally creates a dev environment
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.sh | bash
#
# Environment variables:
#   MBO_ENV_PATH     - custom path for dev environment (default: ~/mbo/envs/mbo_utilities)
#   MBO_SKIP_ENV     - set to "1" to skip dev environment creation
#   MBO_OVERWRITE    - set to "1" to overwrite existing installations

set -euo pipefail

GITHUB_REPO="MillerBrainObservatory/mbo_utilities"
DEFAULT_ENV_PATH="$HOME/mbo/envs/mbo_utilities"

# Installation state
MBO_ENV_PATH="${MBO_ENV_PATH:-}"
SKIP_ENV="${MBO_SKIP_ENV:-0}"
OVERWRITE="${MBO_OVERWRITE:-0}"
PLATFORM=""
SOURCE=""
INSTALL_SPEC=""
BRANCH_REF=""
EXTRAS=()
INSTALL_CLI=true
INSTALL_ENV=true

# GPU detection
GPU_AVAILABLE=false
GPU_NAME=""
CUDA_VERSION=""
TOOLKIT_INSTALLED=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
WHITE='\033[1;37m'
NC='\033[0m'

error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

show_banner() {
    echo ""
    echo -e "${CYAN}  __  __ ____   ___  ${NC}"
    echo -e "${CYAN} |  \\/  | __ ) / _ \\ ${NC}"
    echo -e "${CYAN} | |\\/| |  _ \\| | | |${NC}"
    echo -e "${CYAN} | |  | | |_) | |_| |${NC}"
    echo -e "${CYAN} |_|  |_|____/ \\___/ ${NC}"
    echo ""
    echo "MBO Utilities Installer"
    echo ""
}

check_platform() {
    local platform=$(uname -s)
    if [[ "$platform" == "Linux" ]]; then
        info "Detected Linux"
        PLATFORM="linux"
    elif [[ "$platform" == "Darwin" ]]; then
        info "Detected macOS"
        PLATFORM="macos"
    else
        error "Unsupported platform: $platform"
        exit 1
    fi
}

# =============================================================================
# System Dependencies
# =============================================================================

check_build_tools() {
    # Check for C compiler (required for some Python packages)
    local has_cc=false

    if command -v gcc &> /dev/null; then
        has_cc=true
    elif command -v clang &> /dev/null; then
        has_cc=true
    elif command -v cc &> /dev/null; then
        has_cc=true
    fi

    echo "$has_cc"
}

check_ffmpeg() {
    if command -v ffmpeg &> /dev/null; then
        local version=$(ffmpeg -version 2>&1 | head -1 | sed 's/ffmpeg version \([^ ]*\).*/\1/')
        echo "$version"
    else
        echo ""
    fi
}

show_system_dependencies() {
    echo ""
    echo -e "${WHITE}System Dependencies${NC}"
    echo ""

    local has_cc=$(check_build_tools)
    local ffmpeg_ver=$(check_ffmpeg)
    local has_missing=false

    # C compiler (required)
    if [[ "$has_cc" == "true" ]]; then
        echo -e "  [${GREEN}OK${NC}] C compiler (gcc/clang) ${GRAY}(required)${NC}"
    else
        echo -e "  [${RED}MISSING${NC}] C compiler (gcc/clang) ${GRAY}(required)${NC}"
        has_missing=true
    fi

    # ffmpeg (optional)
    if [[ -n "$ffmpeg_ver" ]]; then
        echo -e "  [${GREEN}OK${NC}] ffmpeg ${GRAY}(optional, for video export)${NC}"
    else
        echo -e "  [  ] ffmpeg ${GRAY}(optional, for video export)${NC}"
    fi

    echo ""

    if [[ "$has_missing" == "true" ]]; then
        echo ""
        error "Required system dependencies are missing."
        echo ""
        if [[ "$PLATFORM" == "linux" ]]; then
            echo -e "  ${YELLOW}Install build tools:${NC}"
            echo -e "    ${CYAN}Ubuntu/Debian:${NC} sudo apt install build-essential"
            echo -e "    ${CYAN}Fedora:${NC}        sudo dnf groupinstall 'Development Tools'"
            echo -e "    ${CYAN}Arch:${NC}          sudo pacman -S base-devel"
        elif [[ "$PLATFORM" == "macos" ]]; then
            echo -e "  ${YELLOW}Install Xcode Command Line Tools:${NC}"
            echo -e "    ${CYAN}xcode-select --install${NC}"
        fi
        echo ""

        echo -e "  [1] Continue anyway (may fail)"
        echo -e "  [2] Exit"
        echo ""

        while true; do
            read -p "Select option (1-2): " choice
            case $choice in
                1) return 0 ;;
                2) exit 0 ;;
                *) warn "Invalid selection." ;;
            esac
        done
    fi

    return 0
}

# =============================================================================
# uv Installation
# =============================================================================

check_uv_installed() {
    if command -v uv &> /dev/null; then
        info "uv is already installed: $(uv --version)"
        return 0
    else
        return 1
    fi
}

install_uv() {
    info "Installing uv..."
    if ! command -v curl &> /dev/null; then
        error "curl is required. Please install curl first."
        exit 1
    fi

    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        success "uv installed successfully"
        export PATH="$HOME/.local/bin:$PATH"

        if ! command -v uv &> /dev/null; then
            warn "uv was installed but not found in PATH for this session"
            warn "You may need to restart your terminal or run:"
            warn "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
    else
        error "Failed to install uv"
        exit 1
    fi
}

get_uv_tool_bin_dir() {
    # `uv tool dir --bin` is the canonical query. appending /bin to
    # `uv tool dir` (without --bin) would point at the tool *install*
    # directory, not the executable directory — different paths.
    local bin_dir=$(uv tool dir --bin 2>/dev/null || echo "")
    if [[ -n "$bin_dir" ]]; then
        echo "$bin_dir"
    else
        echo "$HOME/.local/bin"
    fi
}

# =============================================================================
# Installation Type Selection
# =============================================================================

show_install_type_prompt() {
    echo ""
    echo -e "${WHITE}Installation Type${NC}"
    echo ""
    echo -e "  ${GRAY}CLI             - global 'mbo' command on your PATH, just runs the GUI.${NC}"
    echo -e "  ${GRAY}                  Self-contained; no activation, no imports, no notebooks.${NC}"
    echo -e "  ${GRAY}Local env       - project-local venv you 'cd' into and run 'uv run ...' or${NC}"
    echo -e "  ${GRAY}                  import from. Use this for scripts, notebooks, development.${NC}"
    echo -e "  ${GRAY}Both            - pick this if you want the GUI anywhere AND a local env to code in.${NC}"
    echo ""
    echo -e "  ${CYAN}[1] Environment + CLI${NC} - Create Python venv with mbo_utilities + global CLI (Recommended)"
    echo -e "  ${CYAN}[2] Environment${NC}       - Create Python venv only (for library use)"
    echo -e "  ${CYAN}[3] CLI${NC}               - Install global CLI only (mbo command)"
    echo ""

    while true; do
        read -p "Select installation type (1-3): " choice
        case $choice in
            1)
                INSTALL_CLI=true
                INSTALL_ENV=true
                return
                ;;
            2)
                INSTALL_CLI=false
                INSTALL_ENV=true
                return
                ;;
            3)
                INSTALL_CLI=true
                INSTALL_ENV=false
                return
                ;;
            *)
                warn "Invalid selection."
                ;;
        esac
    done
}

# =============================================================================
# Source Selection
# =============================================================================

get_pypi_version() {
    curl -s "https://pypi.org/pypi/mbo-utilities/json" 2>/dev/null | \
        python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])" 2>/dev/null || echo ""
}

get_github_branches() {
    curl -s "https://api.github.com/repos/$GITHUB_REPO/branches" 2>/dev/null | \
        python3 -c "import sys,json; [print(b['name']) for b in json.load(sys.stdin)]" 2>/dev/null || echo ""
}

show_source_selection() {
    echo ""
    echo -e "${WHITE}Installation Source${NC}"
    echo ""

    local pypi_version=$(get_pypi_version)
    if [[ -n "$pypi_version" ]]; then
        echo -e "  ${CYAN}[1] PyPI (stable)${NC}"
        echo -e "      ${GRAY}Version: $pypi_version${NC}"
    else
        echo -e "  ${CYAN}[1] PyPI (stable)${NC}"
        echo -e "      ${YELLOW}Version: unknown (could not fetch)${NC}"
    fi
    echo ""
    echo -e "  ${CYAN}[2] GitHub (development)${NC}"
    echo -e "      ${GRAY}Install from a specific branch or tag${NC}"
    echo ""

    while true; do
        read -p "Select source (1-2): " choice
        case $choice in
            1)
                SOURCE="pypi"
                INSTALL_SPEC="mbo_utilities"
                BRANCH_REF=""
                return
                ;;
            2)
                break
                ;;
            *)
                warn "Invalid selection. Enter 1 or 2."
                ;;
        esac
    done

    echo ""
    info "Fetching available branches..."

    local branches=$(get_github_branches)
    local branch_array=()
    local idx=1

    echo ""
    echo -e "${WHITE}Available Branches:${NC}"

    # Show main/master first
    local main_branch=""
    if echo "$branches" | grep -q "^master$"; then
        main_branch="master"
    elif echo "$branches" | grep -q "^main$"; then
        main_branch="main"
    fi

    if [[ -n "$main_branch" ]]; then
        echo -e "  ${CYAN}[$idx]${NC} $main_branch ${GRAY}(default)${NC}"
        branch_array+=("$main_branch")
        ((idx++))
    fi

    # Show other branches
    while IFS= read -r branch; do
        if [[ -n "$branch" && "$branch" != "master" && "$branch" != "main" ]]; then
            echo -e "  ${CYAN}[$idx]${NC} $branch"
            branch_array+=("$branch")
            ((idx++))
            if [[ $idx -gt 10 ]]; then break; fi
        fi
    done <<< "$branches"

    echo ""
    echo -e "  ${YELLOW}[c] Custom branch/tag name${NC}"
    echo ""

    while true; do
        read -p "Select branch/tag (1-$((idx-1)) or 'c' for custom): " choice
        if [[ "$choice" == "c" ]]; then
            read -p "Enter branch or tag name: " custom_ref
            SOURCE="github"
            INSTALL_SPEC="mbo_utilities @ git+https://github.com/$GITHUB_REPO@$custom_ref"
            BRANCH_REF="$custom_ref"
            return
        elif [[ "$choice" =~ ^[0-9]+$ ]] && [[ "$choice" -ge 1 ]] && [[ "$choice" -lt "$idx" ]]; then
            local selected="${branch_array[$((choice-1))]}"
            SOURCE="github"
            INSTALL_SPEC="mbo_utilities @ git+https://github.com/$GITHUB_REPO@$selected"
            BRANCH_REF="$selected"
            return
        else
            warn "Invalid selection."
        fi
    done
}

# =============================================================================
# GPU Detection
# =============================================================================

check_nvidia_gpu() {
    GPU_AVAILABLE=false
    GPU_NAME=""
    CUDA_VERSION=""
    TOOLKIT_INSTALLED=false

    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
        if [[ -n "$GPU_NAME" ]]; then
            GPU_AVAILABLE=true

            # Check for CUDA toolkit
            if command -v nvcc &> /dev/null; then
                CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' || echo "")
                if [[ -n "$CUDA_VERSION" ]]; then
                    TOOLKIT_INSTALLED=true
                fi
            elif [[ -n "${CUDA_PATH:-}" ]] && [[ -x "$CUDA_PATH/bin/nvcc" ]]; then
                CUDA_VERSION=$("$CUDA_PATH/bin/nvcc" --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' || echo "")
                if [[ -n "$CUDA_VERSION" ]]; then
                    TOOLKIT_INSTALLED=true
                fi
            fi
        fi
    fi
}

show_optional_dependencies() {
    echo ""
    echo -e "${WHITE}Optional Dependencies${NC}"
    echo ""

    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        echo -e "  ${GREEN}GPU detected:${NC} $GPU_NAME"
        if [[ "$TOOLKIT_INSTALLED" == "true" ]]; then
            echo -e "  ${GREEN}CUDA Toolkit:${NC} $CUDA_VERSION"
        else
            echo -e "  ${YELLOW}CUDA Toolkit: Not installed (GPU packages will use CPU)${NC}"
        fi
    else
        echo -e "  ${YELLOW}No NVIDIA GPU detected (GPU features will be slower)${NC}"
    fi
    echo ""

    echo -e "  ${CYAN}[1] Suite2p${NC}   - 2D cell extraction (PyTorch + CUDA)"
    echo -e "  ${CYAN}[2] Rastermap${NC} - Dimensionality reduction"
    echo -e "  ${CYAN}[3] All${NC}       - Install all processing pipelines"
    echo -e "  ${CYAN}[4] None${NC}      - Base installation only (fastest)"
    echo ""

    while true; do
        read -p "Select option (1-4, or comma-separated like 1,2): " choice
        if [[ "$choice" =~ ^[1-4](,[1-4])*$ ]]; then
            break
        fi
        warn "Invalid selection. Enter 1-4 or comma-separated."
    done

    EXTRAS=()
    IFS=',' read -ra choices <<< "$choice"

    for c in "${choices[@]}"; do
        c=$(echo "$c" | tr -d ' ')
        case $c in
            1) EXTRAS+=("suite2p") ;;
            2) EXTRAS+=("rastermap") ;;
            3) EXTRAS=("all"); break ;;
            4) EXTRAS=(); break ;;
        esac
    done

    # Warn if GPU packages selected without toolkit
    if [[ ${#EXTRAS[@]} -gt 0 ]]; then
        local has_gpu_package=false
        for e in "${EXTRAS[@]}"; do
            if [[ "$e" == "suite2p" || "$e" == "all" ]]; then
                has_gpu_package=true
                break
            fi
        done

        if [[ "$has_gpu_package" == "true" && "$GPU_AVAILABLE" == "true" && "$TOOLKIT_INSTALLED" == "false" ]]; then
            echo ""
            warn "CUDA Toolkit not installed. GPU packages will use CPU (slower)."
            warn "Install CUDA Toolkit for GPU acceleration."
        elif [[ "$has_gpu_package" == "true" && "$GPU_AVAILABLE" == "false" ]]; then
            echo ""
            warn "No NVIDIA GPU detected. These packages will run in CPU-only mode."
            read -p "Continue anyway? (y/n): " continue_choice
            if [[ "$continue_choice" != "y" ]]; then
                EXTRAS=()
            fi
        fi
    fi
}

# =============================================================================
# Environment Location
# =============================================================================

show_env_location_prompt() {
    if [[ -n "$MBO_ENV_PATH" ]]; then
        info "Environment location: $MBO_ENV_PATH (from MBO_ENV_PATH)"
        return
    fi

    echo ""
    echo -e "${WHITE}Environment Location${NC}"
    echo ""
    echo -e "  ${GRAY}Default:${NC} ${CYAN}$DEFAULT_ENV_PATH${NC}"
    echo ""
    read -p "Press Enter for default, or enter custom path: " user_input

    if [[ -z "$user_input" ]]; then
        MBO_ENV_PATH="$DEFAULT_ENV_PATH"
    else
        # Expand ~ to home directory
        user_input="${user_input/#\~/$HOME}"
        # Resolve to absolute path
        if [[ -d "$user_input" ]]; then
            MBO_ENV_PATH="$(cd "$user_input" && pwd)"
        elif [[ -d "$(dirname "$user_input")" ]]; then
            MBO_ENV_PATH="$(cd "$(dirname "$user_input")" && pwd)/$(basename "$user_input")"
        else
            MBO_ENV_PATH="$user_input"
        fi
    fi
}

# =============================================================================
# Version Query
# =============================================================================

get_mbo_version_from_env() {
    local python_path="$1"

    if [[ ! -x "$python_path" ]]; then
        echo ""
        return
    fi

    "$python_path" -c "
import sys
try:
    from importlib.metadata import version, distribution
    v = version('mbo_utilities')
    d = distribution('mbo_utilities')
    direct_url = None
    try:
        import json
        from pathlib import Path
        dist_info = Path(d._path)
        direct_url_file = dist_info / 'direct_url.json'
        if direct_url_file.exists():
            direct_url = json.loads(direct_url_file.read_text())
    except: pass
    if direct_url and 'vcs_info' in direct_url:
        vcs = direct_url['vcs_info']
        commit = vcs.get('commit_id', 'unknown')[:8]
        branch = vcs.get('requested_revision', 'unknown')
        print(f'{v}|git|{branch}|{commit}')
    else:
        print(f'{v}|pypi||')
except Exception as e:
    print(f'error|{e}||')
" 2>/dev/null || echo ""
}

parse_version_info() {
    local version_output="$1"

    if [[ -z "$version_output" || "$version_output" == error* ]]; then
        echo ""
        return
    fi

    echo "$version_output"
}

# =============================================================================
# CLI Tool Installation
# =============================================================================

install_mbo_tool() {
    echo ""
    info "Installing mbo CLI tool via uv tool install..."

    # Build spec with extras
    local full_spec="$INSTALL_SPEC"
    if [[ ${#EXTRAS[@]} -gt 0 ]]; then
        local extras_str=$(IFS=','; echo "${EXTRAS[*]}")
        if [[ "$SOURCE" == "pypi" ]]; then
            full_spec="mbo_utilities[$extras_str]"
        else
            # Git URL format: insert extras after package name
            full_spec="mbo_utilities[$extras_str] @ git+https://github.com/$GITHUB_REPO@$BRANCH_REF"
        fi
    fi

    info "  Spec: $full_spec"

    # Check if already installed
    local existing_tools=$(uv tool list 2>/dev/null || echo "")
    if echo "$existing_tools" | grep -q "mbo[_-]utilities"; then
        if [[ "$OVERWRITE" == "1" ]]; then
            info "Uninstalling existing mbo_utilities..."
            uv tool uninstall mbo_utilities 2>/dev/null || true
        else
            echo ""
            warn "mbo_utilities is already installed as a tool."
            echo ""
            echo -e "  ${CYAN}[1] Upgrade${NC}   - Uninstall and reinstall"
            echo -e "  ${CYAN}[2] Skip${NC}      - Keep existing installation"
            echo -e "  ${CYAN}[3] Cancel${NC}    - Exit"
            echo ""

            while true; do
                read -p "Select option (1-3): " choice
                case $choice in
                    1)
                        info "Uninstalling existing mbo_utilities..."
                        uv tool uninstall mbo_utilities 2>/dev/null || true
                        break
                        ;;
                    2)
                        info "Keeping existing installation."
                        return 0
                        ;;
                    3)
                        info "Installation cancelled."
                        exit 0
                        ;;
                    *)
                        warn "Invalid selection."
                        ;;
                esac
            done
        fi
    fi

    # Install tool
    if uv tool install "$full_spec" --python 3.12; then
        # ensure the tool bin dir is on the user's shell PATH. uv's own
        # install-time PATH wiring doesn't always fire (locked-down
        # configs, unusual shells). idempotent — safe to run every time.
        uv tool update-shell 2>&1 || true

        # make `mbo` reachable in THIS shell too so the user doesn't
        # need to open a new terminal just to try it.
        local bin_dir=$(get_uv_tool_bin_dir)
        if [[ -n "$bin_dir" && ":$PATH:" != *":$bin_dir:"* ]]; then
            export PATH="$bin_dir:$PATH"
        fi

        success "mbo CLI tool installed successfully"
        return 0
    else
        error "Failed to install mbo tool"
        return 1
    fi
}

# =============================================================================
# Dev Environment Installation
# =============================================================================

install_dev_environment() {
    echo ""
    info "Creating development environment at: $MBO_ENV_PATH"

    local python_path="$MBO_ENV_PATH/bin/python"
    local env_exists=false
    local dir_exists=false

    [[ -x "$python_path" ]] && env_exists=true
    [[ -d "$MBO_ENV_PATH" ]] && dir_exists=true

    # Check if directory exists but is not a valid venv (likely a project directory)
    if [[ "$dir_exists" == "true" && "$env_exists" == "false" ]]; then
        # Check if it looks like a project directory
        local is_project_dir=false
        if [[ -f "$MBO_ENV_PATH/pyproject.toml" || -d "$MBO_ENV_PATH/.git" || -f "$MBO_ENV_PATH/setup.py" ]]; then
            is_project_dir=true
        fi

        if [[ "$is_project_dir" == "true" ]]; then
            local venv_path="$MBO_ENV_PATH/.venv"
            local venv_python_path="$venv_path/bin/python"
            local venv_exists=false
            [[ -x "$venv_python_path" ]] && venv_exists=true

            echo ""
            warn "This looks like a project directory: $MBO_ENV_PATH"

            if [[ "$venv_exists" == "true" ]]; then
                # Query existing installation
                local version_output=$(get_mbo_version_from_env "$venv_python_path")

                echo ""
                echo -e "  ${WHITE}Existing .venv found:${NC}"
                if [[ -n "$version_output" && "$version_output" != error* ]]; then
                    IFS='|' read -r ver src branch commit <<< "$version_output"
                    echo -ne "    ${GRAY}mbo_utilities:${NC} "
                    echo -ne "${CYAN}v$ver${NC}"
                    if [[ "$src" == "git" ]]; then
                        echo -e " ${GRAY}(git: $branch@$commit)${NC}"
                    else
                        echo -e " ${GRAY}(PyPI)${NC}"
                    fi
                else
                    echo -e "    ${GRAY}mbo_utilities:${NC} ${YELLOW}not installed or error reading${NC}"
                fi
                echo -e "    ${GRAY}Python: $venv_python_path${NC}"
                echo ""
                echo -e "  ${CYAN}[1] Overwrite${NC} - Delete .venv and recreate"
                echo -e "  ${CYAN}[2] Update${NC}    - Install/upgrade mbo_utilities in existing .venv"
                echo -e "  ${CYAN}[3] Skip${NC}      - Don't modify dev environment"
                echo ""

                while true; do
                    read -p "Select option (1-3): " choice
                    case $choice in
                        1)
                            info "Removing existing .venv..."
                            rm -rf "$venv_path"
                            MBO_ENV_PATH="$venv_path"
                            python_path="$venv_python_path"
                            env_exists=false
                            break
                            ;;
                        2)
                            info "Updating existing .venv..."
                            MBO_ENV_PATH="$venv_path"
                            python_path="$venv_python_path"
                            env_exists=true
                            break
                            ;;
                        3)
                            info "Skipping dev environment."
                            return 0
                            ;;
                        *)
                            warn "Invalid selection."
                            ;;
                    esac
                done
            else
                # No .venv exists, offer to create one
                echo ""
                echo -e "  ${CYAN}[1] Create .venv${NC} - Create environment at $venv_path (Recommended)"
                echo -e "  ${CYAN}[2] Skip${NC}         - Don't create dev environment"
                echo ""

                while true; do
                    read -p "Select option (1-2): " choice
                    case $choice in
                        1)
                            MBO_ENV_PATH="$venv_path"
                            python_path="$venv_python_path"
                            env_exists=false
                            break
                            ;;
                        2)
                            info "Skipping dev environment."
                            return 0
                            ;;
                        *)
                            warn "Invalid selection."
                            ;;
                    esac
                done
            fi
        else
            # Not a project directory, but directory exists without venv
            local venv_path="$MBO_ENV_PATH/.venv"
            echo ""
            warn "Directory exists but is not a virtual environment: $MBO_ENV_PATH"
            echo ""
            echo -e "  ${CYAN}[1] Create .venv inside${NC} - Create environment at $venv_path"
            echo -e "  ${CYAN}[2] Skip${NC}                - Don't create dev environment"
            echo ""

            while true; do
                read -p "Select option (1-2): " choice
                case $choice in
                    1)
                        MBO_ENV_PATH="$venv_path"
                        python_path="$MBO_ENV_PATH/bin/python"
                        env_exists=false
                        break
                        ;;
                    2)
                        info "Skipping dev environment."
                        return 0
                        ;;
                    *)
                        warn "Invalid selection."
                        ;;
                esac
            done
        fi
    elif [[ "$env_exists" == "true" ]]; then
        # EnvPath itself is a valid venv
        if [[ "$OVERWRITE" == "1" ]]; then
            warn "Removing existing environment..."
            rm -rf "$MBO_ENV_PATH"
            env_exists=false
        else
            # Query existing installation
            local version_output=$(get_mbo_version_from_env "$python_path")

            echo ""
            warn "Environment already exists at: $MBO_ENV_PATH"
            echo ""
            if [[ -n "$version_output" && "$version_output" != error* ]]; then
                IFS='|' read -r ver src branch commit <<< "$version_output"
                echo -ne "  ${GRAY}Installed:${NC} "
                echo -ne "${CYAN}mbo_utilities v$ver${NC}"
                if [[ "$src" == "git" ]]; then
                    echo -e " ${GRAY}(git: $branch@$commit)${NC}"
                else
                    echo -e " ${GRAY}(PyPI)${NC}"
                fi
                echo ""
            fi
            echo -e "  ${CYAN}[1] Overwrite${NC} - Delete and recreate"
            echo -e "  ${CYAN}[2] Update${NC}    - Install into existing"
            echo -e "  ${CYAN}[3] Skip${NC}      - Don't modify dev environment"
            echo ""

            while true; do
                read -p "Select option (1-3): " choice
                case $choice in
                    1)
                        info "Removing existing environment..."
                        rm -rf "$MBO_ENV_PATH"
                        env_exists=false
                        break
                        ;;
                    2)
                        info "Updating existing environment..."
                        break
                        ;;
                    3)
                        info "Skipping dev environment."
                        return 0
                        ;;
                    *)
                        warn "Invalid selection."
                        ;;
                esac
            done
        fi
    fi

    # Create virtual environment if it doesn't exist
    if [[ "$env_exists" == "false" ]]; then
        local env_parent=$(dirname "$MBO_ENV_PATH")
        mkdir -p "$env_parent"

        info "Creating virtual environment with Python 3.12..."
        if ! uv venv "$MBO_ENV_PATH" --python 3.12; then
            error "Failed to create virtual environment"
            return 1
        fi
        python_path="$MBO_ENV_PATH/bin/python"
    fi

    # Build spec with extras
    local full_spec="$INSTALL_SPEC"
    if [[ ${#EXTRAS[@]} -gt 0 ]]; then
        local extras_str=$(IFS=','; echo "${EXTRAS[*]}")
        if [[ "$SOURCE" == "pypi" ]]; then
            full_spec="mbo_utilities[$extras_str]"
        else
            full_spec="mbo_utilities[$extras_str] @ git+https://github.com/$GITHUB_REPO@$BRANCH_REF"
        fi
    fi

    info "Installing mbo_utilities into environment..."
    info "  Spec: $full_spec"

    if uv pip install --python "$python_path" "$full_spec"; then
        success "Development environment created successfully"
        return 0
    else
        error "Failed to install into environment"
        return 1
    fi
}

# =============================================================================
# Desktop Entry / App Bundle
# =============================================================================

create_desktop_entry_linux() {
    if [[ "$INSTALL_CLI" != "true" ]]; then
        return
    fi

    info "Creating desktop entry..."

    local app_dir="$HOME/.local/share/applications"
    local icon_dir="$HOME/mbo"
    local desktop_file="$app_dir/mbo-utilities.desktop"
    local icon_path="$icon_dir/icon.png"

    mkdir -p "$app_dir" "$icon_dir"

    # Download icon
    local download_ref="${BRANCH_REF:-master}"
    if curl -LsSf "https://raw.githubusercontent.com/$GITHUB_REPO/$download_ref/mbo_utilities/assets/app_settings/icon.png" -o "$icon_path" 2>/dev/null; then
        info "Icon downloaded"
    elif curl -LsSf "https://raw.githubusercontent.com/$GITHUB_REPO/master/mbo_utilities/assets/app_settings/icon.png" -o "$icon_path" 2>/dev/null; then
        info "Icon downloaded"
    else
        warn "Could not download icon"
        icon_path=""
    fi

    # Find mbo executable
    local mbo_path=$(which mbo 2>/dev/null || echo "$HOME/.local/bin/mbo")

    cat > "$desktop_file" << EOF
[Desktop Entry]
Type=Application
Name=MBO Utilities
Comment=Miller Brain Observatory Image Viewer
Exec=$mbo_path
Icon=$icon_path
Terminal=false
Categories=Science;Education;
EOF

    chmod +x "$desktop_file"

    # Copy to desktop if exists
    if [[ -d "$HOME/Desktop" ]]; then
        cp "$desktop_file" "$HOME/Desktop/"
        chmod +x "$HOME/Desktop/mbo-utilities.desktop"
        success "Desktop shortcut created"
    fi

    success "Application menu entry created"
}

create_desktop_entry_macos() {
    if [[ "$INSTALL_CLI" != "true" ]]; then
        return
    fi

    info "Creating macOS app bundle..."

    local app_dir="$HOME/Applications"
    local app_path="$app_dir/MBO Utilities.app"
    local icon_dir="$HOME/mbo"

    mkdir -p "$app_dir" "$icon_dir"
    mkdir -p "$app_path/Contents/MacOS"
    mkdir -p "$app_path/Contents/Resources"

    local mbo_path=$(which mbo 2>/dev/null || echo "$HOME/.local/bin/mbo")

    # Create launcher script
    cat > "$app_path/Contents/MacOS/MBO Utilities" << EOF
#!/bin/bash
export PATH="$HOME/.local/bin:\$PATH"
"$mbo_path"
EOF
    chmod +x "$app_path/Contents/MacOS/MBO Utilities"

    # Create Info.plist
    cat > "$app_path/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>MBO Utilities</string>
    <key>CFBundleIdentifier</key>
    <string>com.millerbrainobservatory.mbo-utilities</string>
    <key>CFBundleName</key>
    <string>MBO Utilities</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF

    # Download icon
    local download_ref="${BRANCH_REF:-master}"
    curl -LsSf "https://raw.githubusercontent.com/$GITHUB_REPO/$download_ref/mbo_utilities/assets/app_settings/icon.png" \
        -o "$app_path/Contents/Resources/icon.png" 2>/dev/null || true

    success "App bundle created at: $app_path"

    # Create symlink on desktop
    if [[ -d "$HOME/Desktop" ]]; then
        ln -sf "$app_path" "$HOME/Desktop/MBO Utilities.app"
        success "Desktop alias created"
    fi
}

# =============================================================================
# Usage Instructions
# =============================================================================

show_usage_instructions() {
    echo ""
    echo -e "${WHITE}Installation Complete${NC}"
    echo ""

    # two distinct usage modes depending on what was installed. CLI works
    # from any directory; environment is project-local and needs a `cd`
    # (or activation) first. label them so users running both don't
    # conflate the two.
    local show_both=false
    if [[ "$INSTALL_CLI" == "true" && "$INSTALL_ENV" == "true" && -n "$MBO_ENV_PATH" && -d "$MBO_ENV_PATH" ]]; then
        show_both=true
    fi

    local section=0
    if [[ "$INSTALL_CLI" == "true" ]]; then
        section=$((section + 1))
        local bin_dir=$(get_uv_tool_bin_dir)
        local header="CLI - available system-wide"
        if $show_both; then header="(${section}) $header"; fi
        echo -e "  ${GRAY}${header}${NC}"
        echo -e "    ${WHITE}mbo${NC}                    # open GUI"
        echo -e "    ${WHITE}mbo /path/to/data${NC}      # open specific file"
        echo -e "    ${WHITE}mbo --help${NC}             # show all commands"
        echo -e "    ${GRAY}Location: $bin_dir/mbo${NC}"
        echo ""
    fi

    if [[ "$INSTALL_ENV" == "true" && -n "$MBO_ENV_PATH" && -d "$MBO_ENV_PATH" ]]; then
        section=$((section + 1))
        # $MBO_ENV_PATH points at the actual venv dir (ends in /.venv when
        # the user pointed at a project root). `uv run` wants the project
        # dir, not the venv dir — strip the trailing .venv so `cd` lands
        # on the project and all the other `uv ...` commands Just Work.
        local cd_path="$MBO_ENV_PATH"
        if [[ "$cd_path" =~ /\.venv/?$ ]]; then
            cd_path="${cd_path%/.venv}"
            cd_path="${cd_path%/.venv/}"
        fi

        local header="Local environment - use from the env directory"
        if $show_both; then header="(${section}) $header"; fi
        echo -e "  ${GRAY}${header}${NC}"
        echo -e "    ${WHITE}cd $cd_path${NC}"
        echo -e "    ${WHITE}uv run mbo${NC}             # open GUI (uses this env)"
        echo -e "    ${WHITE}uv run mbo --help${NC}      # show all commands"
        echo -e "    ${WHITE}uv run python${NC}          # interactive session with this env"
        echo -e "    ${WHITE}uv pip install <pkg>${NC}   # add a package to this env"
        echo -e "    ${WHITE}uv pip list${NC}            # show installed packages + versions"
        echo ""
        echo -e "  ${GRAY}Use in VSCode:${NC}"
        echo -e "    ${WHITE}Ctrl+Shift+P -> 'Python: Select Interpreter'${NC}"
        echo -e "    ${WHITE}Choose: $MBO_ENV_PATH/bin/python${NC}"
        echo ""
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    show_banner
    check_platform

    # Step 0: Check system dependencies
    show_system_dependencies

    # Check/install uv
    if ! check_uv_installed; then
        install_uv
    fi

    # Step 1: Choose installation type
    show_install_type_prompt

    # Step 2: Choose source
    show_source_selection

    # Step 3: Detect GPU
    check_nvidia_gpu

    # Step 4: Choose extras
    show_optional_dependencies

    # Step 5: Get environment location if needed
    if [[ "$INSTALL_ENV" == "true" && "$SKIP_ENV" != "1" ]]; then
        show_env_location_prompt
    fi

    # Step 6: Install CLI tool if requested
    if [[ "$INSTALL_CLI" == "true" ]]; then
        if ! install_mbo_tool; then
            error "CLI installation failed."
            exit 1
        fi
    fi

    # Step 7: Create dev environment if requested
    if [[ "$INSTALL_ENV" == "true" && "$SKIP_ENV" != "1" && -n "$MBO_ENV_PATH" ]]; then
        install_dev_environment
    fi

    # Step 8: Create desktop entry
    if [[ "$INSTALL_CLI" == "true" ]]; then
        if [[ "$PLATFORM" == "linux" ]]; then
            create_desktop_entry_linux
        elif [[ "$PLATFORM" == "macos" ]]; then
            create_desktop_entry_macos
        fi
    fi

    # Show usage instructions
    show_usage_instructions

    success "Installation completed!"
    echo ""
    if [[ "$INSTALL_CLI" == "true" ]]; then
        echo -e "${YELLOW}You may need to restart your terminal for PATH changes to take effect.${NC}"
    fi
}

main "$@"
