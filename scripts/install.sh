#!/usr/bin/env bash

# MBO Utilities Installation Script for Linux/macOS
# installs uv if not present, installs mbo_utilities, and creates desktop entry
#
# usage:
#   curl -LsSf https://raw.githubusercontent.com/.../install.sh | bash           # CLI-only
#   curl -LsSf https://raw.githubusercontent.com/.../install.sh | bash -s -- --env  # full environment
#
# the full environment installs to ~/mbo_env and can be used with VSCode/Jupyter

set -euo pipefail

MBO_ENV_PATH="${MBO_ENV_PATH:-$HOME/mbo_env}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

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
        info "uv is not installed"
        return 1
    fi
}

install_uv() {
    info "Installing uv using the official Astral installer..."

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

install_mbo() {
    info "Installing mbo_utilities using uv..."
    uv tool install mbo_utilities --from git+https://github.com/millerbrainobservatory/mbo_utilities.git
    success "mbo_utilities installed successfully"
}

install_mbo_env() {
    local env_path="${1:-$MBO_ENV_PATH}"

    info "Creating full MBO environment at: $env_path"

    if ! check_uv_installed; then
        install_uv
    fi

    # create venv
    info "Creating virtual environment..."
    uv venv "$env_path" --python 3.11

    # install mbo_utilities with all extras
    info "Installing mbo_utilities (this may take a few minutes)..."
    uv pip install --python "$env_path/bin/python" "mbo_utilities[all] @ git+https://github.com/millerbrainobservatory/mbo_utilities.git"

    # install jupyter for notebook support
    info "Installing Jupyter..."
    uv pip install --python "$env_path/bin/python" jupyterlab ipykernel

    # register kernel for jupyter
    info "Registering Jupyter kernel..."
    "$env_path/bin/python" -m ipykernel install --user --name mbo --display-name "MBO Utilities"

    success "Environment created at: $env_path"
    echo ""
    echo "To use this environment:"
    echo ""
    echo -e "${CYAN}VSCode:${NC}"
    echo "  1. Open VSCode"
    echo "  2. Press Ctrl+Shift+P -> 'Python: Select Interpreter'"
    echo "  3. Choose: $env_path/bin/python"
    echo ""
    echo -e "${CYAN}JupyterLab:${NC}"
    echo "  $env_path/bin/jupyter-lab"
    echo ""
    echo -e "${CYAN}Terminal:${NC}"
    echo "  source $env_path/bin/activate"
    echo ""
}

create_desktop_entry_linux() {
    info "Creating desktop entry..."

    local app_dir="$HOME/.local/share/applications"
    local icon_dir="$HOME/.local/share/icons"
    local desktop_file="$app_dir/mbo-utilities.desktop"
    local icon_path="$icon_dir/mbo_icon.png"

    mkdir -p "$app_dir" "$icon_dir"

    # download icon (png for linux desktop entries)
    if curl -LsSf "https://raw.githubusercontent.com/millerbrainobservatory/mbo_utilities/master/docs/_static/mbo_home_icon.png" -o "$icon_path"; then
        info "Icon downloaded"
    else
        warning "Could not download icon"
        icon_path=""
    fi

    # find mbo executable
    local mbo_path=$(command -v mbo 2>/dev/null || echo "")
    if [[ -z "$mbo_path" ]]; then
        mbo_path="$HOME/.local/bin/mbo"
    fi

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

    mkdir -p "$app_dir"
    mkdir -p "$app_path/Contents/MacOS"
    mkdir -p "$app_path/Contents/Resources"

    # find mbo executable
    local mbo_path=$(command -v mbo 2>/dev/null || echo "$HOME/.local/bin/mbo")

    # create launcher script
    cat > "$app_path/Contents/MacOS/MBO Utilities" << EOF
#!/bin/bash
export PATH="\$HOME/.local/bin:\$PATH"
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

    # download icon (would need conversion to .icns for proper macOS icon)
    if curl -LsSf "https://raw.githubusercontent.com/millerbrainobservatory/mbo_utilities/master/docs/_static/mbo_home_icon.png" -o "$app_path/Contents/Resources/icon.png"; then
        info "Icon downloaded (note: .icns format recommended for macOS)"
    fi

    success "App bundle created at: $app_path"

    # create symlink on desktop
    if [[ -d "$HOME/Desktop" ]]; then
        ln -sf "$app_path" "$HOME/Desktop/MBO Utilities.app"
        success "Desktop alias created"
    fi
}

main() {
    local install_env=false
    local env_path="$MBO_ENV_PATH"

    # parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                install_env=true
                shift
                ;;
            --env-path)
                env_path="$2"
                install_env=true
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    echo ""
    echo -e "${CYAN}  __  __ ____   ___  ${NC}"
    echo -e "${CYAN} |  \\/  | __ ) / _ \\ ${NC}"
    echo -e "${CYAN} | |\\/| |  _ \\| | | |${NC}"
    echo -e "${CYAN} | |  | | |_) | |_| |${NC}"
    echo -e "${CYAN} |_|  |_|____/ \\___/ ${NC}"
    echo ""
    echo "MBO Utilities Installer"
    echo ""

    check_platform

    if ! check_uv_installed; then
        install_uv
    fi

    # install CLI tool
    install_mbo

    if [[ "$PLATFORM" == "linux" ]]; then
        create_desktop_entry_linux
    elif [[ "$PLATFORM" == "macos" ]]; then
        create_desktop_entry_macos
    fi

    # optionally install full environment
    if [[ "$install_env" == true ]]; then
        echo ""
        install_mbo_env "$env_path"
    fi

    if command -v mbo &> /dev/null; then
        success "Installation completed successfully!"
        echo ""
        echo "You can now:"
        echo "  - Double-click the 'MBO Utilities' icon on your desktop"
        echo "  - Or run 'mbo' from any terminal"
        if [[ "$install_env" == false ]]; then
            echo ""
            echo -e "${YELLOW}For VSCode/Jupyter development, re-run with --env:${NC}"
            echo "  curl -LsSf <url>/install.sh | bash -s -- --env"
        fi
        echo ""
    else
        warning "Installation completed but 'mbo' command not found"
        warning "You may need to restart your terminal"
    fi
}

main "$@"
