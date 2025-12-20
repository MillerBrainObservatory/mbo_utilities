#!/usr/bin/env bash

# MBO Utilities Installation Script for Linux/macOS
# installs uv if not present, creates environment at ~/mbo/envs/, installs mbo_utilities
#
# usage:
#   curl -LsSf https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.sh | bash

set -euo pipefail

GITHUB_REPO="MillerBrainObservatory/mbo_utilities"
MBO_ENV_PATH="$HOME/mbo/envs/mbo_utilities"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }

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
            warning "uv was installed but not found in PATH for this session"
            warning "You may need to restart your terminal or run:"
            warning "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
    else
        error "Failed to install uv"
        exit 1
    fi
}

get_pypi_version() {
    curl -s "https://pypi.org/pypi/mbo-utilities/json" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])" 2>/dev/null || echo ""
}

get_github_branches() {
    curl -s "https://api.github.com/repos/$GITHUB_REPO/branches" 2>/dev/null | python3 -c "import sys,json; [print(b['name']) for b in json.load(sys.stdin)]" 2>/dev/null || echo ""
}

show_source_selection() {
    echo ""
    echo -e "${CYAN}Installation Source${NC}"
    echo ""

    local pypi_version=$(get_pypi_version)
    if [[ -n "$pypi_version" ]]; then
        echo -e "  [1] PyPI (stable)"
        echo -e "      Version: $pypi_version"
    else
        echo -e "  [1] PyPI (stable)"
        echo -e "      ${YELLOW}Version: unknown (could not fetch)${NC}"
    fi
    echo ""
    echo -e "  [2] GitHub (development)"
    echo -e "      Install from a specific branch or tag"
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
                warning "Invalid selection. Enter 1 or 2."
                ;;
        esac
    done

    echo ""
    info "Fetching available branches..."

    local branches=$(get_github_branches)
    local branch_array=()
    local idx=1

    echo ""
    echo -e "${CYAN}Available Branches:${NC}"

    # show main/master first
    local main_branch=""
    if echo "$branches" | grep -q "^master$"; then
        main_branch="master"
    elif echo "$branches" | grep -q "^main$"; then
        main_branch="main"
    fi

    if [[ -n "$main_branch" ]]; then
        echo -e "  [$idx] $main_branch (default)"
        branch_array+=("$main_branch")
        ((idx++))
    fi

    # show other branches
    while IFS= read -r branch; do
        if [[ -n "$branch" && "$branch" != "master" && "$branch" != "main" ]]; then
            echo -e "  [$idx] $branch"
            branch_array+=("$branch")
            ((idx++))
            if [[ $idx -gt 10 ]]; then break; fi
        fi
    done <<< "$branches"

    echo ""
    echo -e "  [c] Custom branch/tag name"
    echo ""

    while true; do
        read -p "Select branch/tag (1-$((idx-1)) or 'c' for custom): " choice
        if [[ "$choice" == "c" ]]; then
            read -p "Enter branch or tag name: " custom_ref
            SOURCE="github"
            INSTALL_SPEC="git+https://github.com/$GITHUB_REPO@$custom_ref"
            BRANCH_REF="$custom_ref"
            return
        elif [[ "$choice" =~ ^[0-9]+$ ]] && [[ "$choice" -ge 1 ]] && [[ "$choice" -lt "$idx" ]]; then
            local selected="${branch_array[$((choice-1))]}"
            SOURCE="github"
            INSTALL_SPEC="git+https://github.com/$GITHUB_REPO@$selected"
            BRANCH_REF="$selected"
            return
        else
            warning "Invalid selection."
        fi
    done
}

check_nvidia_gpu() {
    GPU_AVAILABLE=false
    GPU_NAME=""
    CUDA_VERSION=""
    TOOLKIT_INSTALLED=false

    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
        if [[ -n "$GPU_NAME" ]]; then
            GPU_AVAILABLE=true

            # check for cuda toolkit
            if command -v nvcc &> /dev/null; then
                CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' || echo "")
                if [[ -n "$CUDA_VERSION" ]]; then
                    TOOLKIT_INSTALLED=true
                fi
            elif [[ -n "$CUDA_PATH" ]] && [[ -x "$CUDA_PATH/bin/nvcc" ]]; then
                CUDA_VERSION=$("$CUDA_PATH/bin/nvcc" --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' || echo "")
                if [[ -n "$CUDA_VERSION" ]]; then
                    TOOLKIT_INSTALLED=true
                fi
            fi
        fi
    fi
}

get_pytorch_index_url() {
    local cuda_ver="$1"
    if [[ -z "$cuda_ver" ]]; then
        echo ""
        return
    fi

    local major=$(echo "$cuda_ver" | cut -d. -f1)
    local minor=$(echo "$cuda_ver" | cut -d. -f2)

    if [[ "$major" == "11" ]]; then
        echo "https://download.pytorch.org/whl/cu118"
    elif [[ "$major" == "12" ]]; then
        if [[ "$minor" -le 1 ]]; then
            echo "https://download.pytorch.org/whl/cu121"
        else
            echo "https://download.pytorch.org/whl/cu124"
        fi
    else
        echo ""
    fi
}

show_optional_dependencies() {
    echo ""
    echo -e "${CYAN}Optional Dependencies${NC}"
    echo ""

    if [[ "$GPU_AVAILABLE" == true ]]; then
        echo -e "  ${GREEN}GPU detected:${NC} $GPU_NAME"
        if [[ "$TOOLKIT_INSTALLED" == true ]]; then
            echo -e "  ${GREEN}CUDA Toolkit:${NC} $CUDA_VERSION"
        else
            echo -e "  ${YELLOW}CUDA Toolkit: Not installed (PyTorch will use CPU)${NC}"
        fi
    else
        echo -e "  ${YELLOW}No NVIDIA GPU detected (GPU features will be slower)${NC}"
    fi
    echo ""

    echo -e "  [1] Suite2p   - 2D cell extraction (PyTorch + CUDA)"
    echo -e "  [2] Suite3D   - 3D volumetric registration (CuPy + CUDA)"
    echo -e "  [3] Rastermap - Dimensionality reduction"
    echo -e "  [4] All       - Install all processing pipelines"
    echo -e "  [5] None      - Base installation only (fastest)"
    echo ""

    while true; do
        read -p "Select option (1-5, or comma-separated like 1,3): " choice
        if [[ "$choice" =~ ^[1-5](,[1-5])*$ ]]; then
            break
        fi
        warning "Invalid selection. Enter 1-5 or comma-separated."
    done

    EXTRAS=()
    IFS=',' read -ra choices <<< "$choice"

    for c in "${choices[@]}"; do
        c=$(echo "$c" | tr -d ' ')
        case $c in
            1) EXTRAS+=("suite2p") ;;
            2) EXTRAS+=("suite3d") ;;
            3) EXTRAS+=("rastermap") ;;
            4) EXTRAS=("all"); break ;;
            5) EXTRAS=(); break ;;
        esac
    done

    # warn if GPU packages selected without toolkit
    if [[ ${#EXTRAS[@]} -gt 0 ]]; then
        local has_gpu_package=false
        for e in "${EXTRAS[@]}"; do
            if [[ "$e" == "suite2p" || "$e" == "suite3d" || "$e" == "all" ]]; then
                has_gpu_package=true
                break
            fi
        done

        if [[ "$has_gpu_package" == true && "$GPU_AVAILABLE" == true && "$TOOLKIT_INSTALLED" == false ]]; then
            echo ""
            warning "CUDA Toolkit not installed. PyTorch will use CPU (slower)."
            warning "Install CUDA Toolkit for GPU acceleration."
        elif [[ "$has_gpu_package" == true && "$GPU_AVAILABLE" == false ]]; then
            echo ""
            warning "No NVIDIA GPU detected. These packages will run in CPU-only mode."
            read -p "Continue anyway? (y/n): " continue_choice
            if [[ "$continue_choice" != "y" ]]; then
                EXTRAS=()
            fi
        fi
    fi
}

install_mbo_environment() {
    echo ""
    info "Creating MBO environment at: $MBO_ENV_PATH"

    # create parent directory
    local env_parent=$(dirname "$MBO_ENV_PATH")
    mkdir -p "$env_parent"

    # create virtual environment
    info "Creating virtual environment..."
    uv venv "$MBO_ENV_PATH" --python 3.12

    local python_path="$MBO_ENV_PATH/bin/python"

    # build the install spec with extras
    local spec
    if [[ "$SOURCE" == "pypi" ]]; then
        if [[ ${#EXTRAS[@]} -gt 0 ]]; then
            local extras_str=$(IFS=','; echo "${EXTRAS[*]}")
            spec="mbo_utilities[$extras_str]"
        else
            spec="mbo_utilities"
        fi
    else
        if [[ ${#EXTRAS[@]} -gt 0 ]]; then
            local extras_str=$(IFS=','; echo "${EXTRAS[*]}")
            spec="${INSTALL_SPEC}[$extras_str]"
        else
            spec="$INSTALL_SPEC"
        fi
    fi

    info "Installing mbo_utilities..."
    info "  Source: $SOURCE"
    if [[ ${#EXTRAS[@]} -gt 0 ]]; then
        info "  Extras: ${EXTRAS[*]}"
    fi

    local needs_pytorch=false
    for e in "${EXTRAS[@]}"; do
        if [[ "$e" == "suite2p" || "$e" == "all" ]]; then
            needs_pytorch=true
            break
        fi
    done

    if [[ "$needs_pytorch" == true && "$TOOLKIT_INSTALLED" == true && -n "$CUDA_VERSION" ]]; then
        local index_url=$(get_pytorch_index_url "$CUDA_VERSION")
        if [[ -n "$index_url" ]]; then
            info "Installing with CUDA-optimized PyTorch for CUDA $CUDA_VERSION..."
            if uv pip install --python "$python_path" "$spec" torch torchvision \
                --index-strategy unsafe-best-match \
                --extra-index-url "$index_url"; then
                success "Installed with CUDA-optimized PyTorch"
            else
                warning "CUDA install failed, falling back to standard install..."
                uv pip install --python "$python_path" "$spec"
            fi
        else
            uv pip install --python "$python_path" "$spec"
        fi
    else
        uv pip install --python "$python_path" "$spec"
    fi

    success "mbo_utilities installed successfully"
}

add_mbo_to_path() {
    info "Configuring PATH..."

    local bin_path="$MBO_ENV_PATH/bin"
    local shell_rc=""

    if [[ -n "${ZSH_VERSION:-}" ]] || [[ "$SHELL" == *"zsh"* ]]; then
        shell_rc="$HOME/.zshrc"
    else
        shell_rc="$HOME/.bashrc"
    fi

    # check if already in rc file
    if grep -q "$bin_path" "$shell_rc" 2>/dev/null; then
        info "MBO already configured in $shell_rc"
    else
        echo "" >> "$shell_rc"
        echo "# MBO Utilities" >> "$shell_rc"
        echo "export PATH=\"$bin_path:\$PATH\"" >> "$shell_rc"
        success "Added $bin_path to PATH in $shell_rc"
    fi

    export PATH="$bin_path:$PATH"
}

create_desktop_entry_linux() {
    info "Creating desktop entry..."

    local app_dir="$HOME/.local/share/applications"
    local icon_dir="$HOME/mbo"
    local desktop_file="$app_dir/mbo-utilities.desktop"
    local icon_path="$icon_dir/mbo_icon.png"

    mkdir -p "$app_dir" "$icon_dir"

    # download icon
    local download_ref="${BRANCH_REF:-master}"
    if curl -LsSf "https://raw.githubusercontent.com/$GITHUB_REPO/$download_ref/mbo_utilities/assets/static/mbo_icon.ico" -o "$icon_path" 2>/dev/null; then
        info "Icon downloaded"
    elif curl -LsSf "https://raw.githubusercontent.com/$GITHUB_REPO/master/mbo_utilities/assets/static/mbo_icon.ico" -o "$icon_path" 2>/dev/null; then
        info "Icon downloaded"
    else
        warning "Could not download icon"
        icon_path=""
    fi

    local mbo_path="$MBO_ENV_PATH/bin/mbo"

    cat > "$desktop_file" << EOF
[Desktop Entry]
Type=Application
Name=MBO Utilities
Comment=Miller Brain Observatory Utilities
Exec=$mbo_path
Icon=$icon_path
Terminal=true
Categories=Science;Education;
EOF

    chmod +x "$desktop_file"

    # copy to desktop if exists
    if [[ -d "$HOME/Desktop" ]]; then
        cp "$desktop_file" "$HOME/Desktop/"
        chmod +x "$HOME/Desktop/mbo-utilities.desktop"
        success "Desktop shortcut created"
    fi

    success "Application menu entry created"
}

create_desktop_entry_macos() {
    info "Creating macOS app bundle..."

    local app_dir="$HOME/Applications"
    local app_path="$app_dir/MBO Utilities.app"
    local icon_dir="$HOME/mbo"

    mkdir -p "$app_dir" "$icon_dir"
    mkdir -p "$app_path/Contents/MacOS"
    mkdir -p "$app_path/Contents/Resources"

    local mbo_path="$MBO_ENV_PATH/bin/mbo"

    # create launcher script
    cat > "$app_path/Contents/MacOS/MBO Utilities" << EOF
#!/bin/bash
export PATH="$MBO_ENV_PATH/bin:\$PATH"
open -a Terminal "$mbo_path"
EOF
    chmod +x "$app_path/Contents/MacOS/MBO Utilities"

    # create Info.plist
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
</dict>
</plist>
EOF

    # download icon
    local download_ref="${BRANCH_REF:-master}"
    if curl -LsSf "https://raw.githubusercontent.com/$GITHUB_REPO/$download_ref/mbo_utilities/assets/static/mbo_icon.ico" -o "$app_path/Contents/Resources/icon.png" 2>/dev/null; then
        info "Icon downloaded (note: .icns format recommended for macOS)"
    fi

    success "App bundle created at: $app_path"

    # create symlink on desktop
    if [[ -d "$HOME/Desktop" ]]; then
        ln -sf "$app_path" "$HOME/Desktop/MBO Utilities.app"
        success "Desktop alias created"
    fi
}

show_usage_instructions() {
    echo ""
    echo -e "${CYAN}Environment Location${NC}"
    echo "  $MBO_ENV_PATH"
    echo ""
    echo -e "${CYAN}Usage${NC}"
    echo ""
    echo "  Desktop shortcut:"
    echo "    Double-click 'MBO Utilities' on your desktop"
    echo ""
    echo "  Command line (after restarting terminal):"
    echo "    mbo"
    echo ""
    echo "  Activate environment:"
    echo "    source $MBO_ENV_PATH/bin/activate"
    echo ""
    echo "  VSCode:"
    echo "    1. Open VSCode"
    echo "    2. Ctrl+Shift+P -> 'Python: Select Interpreter'"
    echo "    3. Choose: $MBO_ENV_PATH/bin/python"
    echo ""
    echo "  Add packages to environment:"
    echo "    uv pip install --python \"$MBO_ENV_PATH/bin/python\" <package>"
    echo ""
}

main() {
    show_banner
    check_platform

    # check/install uv
    if ! check_uv_installed; then
        install_uv
    fi

    # step 1: choose source
    show_source_selection

    # step 2: detect GPU
    check_nvidia_gpu

    # step 3: choose extras
    show_optional_dependencies

    # step 4: create environment and install
    install_mbo_environment

    # step 5: add to PATH
    add_mbo_to_path

    # step 6: create desktop entry
    if [[ "$PLATFORM" == "linux" ]]; then
        create_desktop_entry_linux
    elif [[ "$PLATFORM" == "macos" ]]; then
        create_desktop_entry_macos
    fi

    # show instructions
    show_usage_instructions

    success "Installation completed!"
    echo ""
    echo -e "${YELLOW}You may need to restart your terminal for PATH changes to take effect.${NC}"
}

main "$@"
