# MBO Utilities Installation Script for Windows
# installs uv if not present, installs mbo_utilities, and creates a desktop shortcut
#
# usage:
#   irm https://raw.githubusercontent.com/.../install.ps1 | iex           # CLI-only install
#   irm https://raw.githubusercontent.com/.../install.ps1 | iex; Install-MboEnv   # full environment
#
# the full environment installs to ~/mbo_env and can be used with VSCode/Jupyter

$ErrorActionPreference = "Stop"

# default install location for full environment
$MBO_ENV_PATH = Join-Path $env:USERPROFILE "mbo_env"

# colors for output
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warning { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

function Test-UvInstalled {
    try {
        $null = Get-Command uv -ErrorAction Stop
        $version = uv --version
        Write-Info "uv is already installed: $version"
        return $true
    }
    catch {
        Write-Info "uv is not installed"
        return $false
    }
}

function Install-Uv {
    Write-Info "Installing uv using the official Astral installer..."

    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression

        # refresh PATH for current session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

        if (Test-UvInstalled) {
            Write-Success "uv installed successfully"
        }
        else {
            Write-Warning "uv was installed but not found in PATH"
            Write-Warning "You may need to restart your terminal"
        }
    }
    catch {
        Write-Error "Failed to install uv: $_"
        exit 1
    }
}

function Install-MboUtilities {
    Write-Info "Installing mbo_utilities using uv..."

    try {
        # install from github
        uv tool install mbo_utilities --from git+https://github.com/millerbrainobservatory/mbo_utilities.git
        Write-Success "mbo_utilities installed successfully"
    }
    catch {
        Write-Error "Failed to install mbo_utilities: $_"
        exit 1
    }
}

function New-DesktopShortcut {
    Write-Info "Creating desktop shortcut..."

    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktopPath "MBO Utilities.lnk"

    # find uv tools bin directory
    $uvToolsBin = Join-Path $env:LOCALAPPDATA "uv\tools\mbo_utilities\Scripts"
    $mboExe = Join-Path $uvToolsBin "mbo.exe"

    # fallback: check standard uv bin location
    if (-not (Test-Path $mboExe)) {
        $uvBin = Join-Path $env:USERPROFILE ".local\bin"
        $mboExe = Join-Path $uvBin "mbo.exe"
    }

    if (-not (Test-Path $mboExe)) {
        # try to find it via where command
        try {
            $mboExe = (Get-Command mbo -ErrorAction Stop).Source
        }
        catch {
            Write-Warning "Could not locate mbo.exe - shortcut will use 'uv run mbo' instead"
            $mboExe = $null
        }
    }

    # download icon to local appdata
    $iconDir = Join-Path $env:LOCALAPPDATA "mbo_utilities"
    $iconPath = Join-Path $iconDir "mbo_icon.ico"

    if (-not (Test-Path $iconDir)) {
        New-Item -ItemType Directory -Path $iconDir -Force | Out-Null
    }

    try {
        Write-Info "Downloading icon..."
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/millerbrainobservatory/mbo_utilities/master/docs/_static/mbo_icon.ico" -OutFile $iconPath
    }
    catch {
        Write-Warning "Could not download icon, shortcut will use default icon"
        $iconPath = $null
    }

    # create shortcut
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)

    if ($mboExe -and (Test-Path $mboExe)) {
        $shortcut.TargetPath = $mboExe
    }
    else {
        # fallback: use cmd to run uv run mbo
        $shortcut.TargetPath = "cmd.exe"
        $shortcut.Arguments = "/k uv run mbo"
    }

    $shortcut.WorkingDirectory = [Environment]::GetFolderPath("UserProfile")

    if ($iconPath -and (Test-Path $iconPath)) {
        $shortcut.IconLocation = $iconPath
    }

    $shortcut.Description = "MBO Utilities - Miller Brain Observatory"
    $shortcut.Save()

    Write-Success "Desktop shortcut created at: $shortcutPath"
}

function Install-MboEnv {
    <#
    .SYNOPSIS
    Creates a full mbo_utilities environment for use with VSCode/Jupyter.

    .DESCRIPTION
    Creates a Python virtual environment at ~/mbo_env with mbo_utilities installed.
    This environment can be selected in VSCode or used to run Jupyter notebooks.

    .PARAMETER Path
    Installation path. Defaults to ~/mbo_env

    .EXAMPLE
    Install-MboEnv
    Install-MboEnv -Path "C:\projects\mbo_env"
    #>
    param(
        [string]$Path = $MBO_ENV_PATH
    )

    Write-Info "Creating full MBO environment at: $Path"

    # check uv is available
    if (-not (Test-UvInstalled)) {
        Install-Uv
    }

    # create venv
    Write-Info "Creating virtual environment..."
    uv venv $Path --python 3.11

    # install mbo_utilities with all extras using uv pip (targets the venv)
    Write-Info "Installing mbo_utilities (this may take a few minutes)..."
    uv pip install --python "$Path\Scripts\python.exe" "mbo_utilities[all] @ git+https://github.com/millerbrainobservatory/mbo_utilities.git"

    # install jupyter for notebook support
    Write-Info "Installing Jupyter..."
    uv pip install --python "$Path\Scripts\python.exe" jupyterlab ipykernel

    # register kernel for jupyter
    Write-Info "Registering Jupyter kernel..."
    & "$Path\Scripts\python.exe" -m ipykernel install --user --name mbo --display-name "MBO Utilities"

    Write-Success "Environment created at: $Path"
    Write-Host ""
    Write-Host "To use this environment:" -ForegroundColor White
    Write-Host ""
    Write-Host "  VSCode:" -ForegroundColor Cyan
    Write-Host "    1. Open VSCode" -ForegroundColor Gray
    Write-Host "    2. Press Ctrl+Shift+P -> 'Python: Select Interpreter'" -ForegroundColor Gray
    Write-Host "    3. Choose: $Path\Scripts\python.exe" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  JupyterLab:" -ForegroundColor Cyan
    Write-Host "    $Path\Scripts\jupyter-lab.exe" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Terminal:" -ForegroundColor Cyan
    Write-Host "    $Path\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host ""

    return $Path
}

function Main {
    Write-Host ""
    Write-Host "  __  __ ____   ___  " -ForegroundColor Cyan
    Write-Host " |  \/  | __ ) / _ \ " -ForegroundColor Cyan
    Write-Host " | |\/| |  _ \| | | |" -ForegroundColor Cyan
    Write-Host " | |  | | |_) | |_| |" -ForegroundColor Cyan
    Write-Host " |_|  |_|____/ \___/ " -ForegroundColor Cyan
    Write-Host ""
    Write-Host "MBO Utilities Installer" -ForegroundColor White
    Write-Host ""

    # check/install uv
    if (-not (Test-UvInstalled)) {
        Install-Uv
    }

    # install mbo_utilities as CLI tool
    Install-MboUtilities

    # create desktop shortcut
    New-DesktopShortcut

    # verify installation
    try {
        $null = Get-Command mbo -ErrorAction Stop
        Write-Success "Installation completed successfully!"
        Write-Host ""
        Write-Host "You can now:" -ForegroundColor White
        Write-Host "  - Double-click the 'MBO Utilities' icon on your desktop" -ForegroundColor Gray
        Write-Host "  - Or run 'mbo' from any terminal" -ForegroundColor Gray
        Write-Host ""
        Write-Host "For VSCode/Jupyter development, also run:" -ForegroundColor Yellow
        Write-Host "  Install-MboEnv" -ForegroundColor Yellow
        Write-Host ""
    }
    catch {
        Write-Warning "Installation completed but 'mbo' command not found in PATH"
        Write-Warning "You may need to restart your terminal"
        Write-Host ""
        Write-Host "After restarting, you can:" -ForegroundColor White
        Write-Host "  - Double-click the 'MBO Utilities' icon on your desktop" -ForegroundColor Gray
        Write-Host "  - Or run 'mbo' from any terminal" -ForegroundColor Gray
        Write-Host ""
    }
}

# export functions for use after piping
Export-ModuleMember -Function Install-MboEnv, Main -ErrorAction SilentlyContinue

Main