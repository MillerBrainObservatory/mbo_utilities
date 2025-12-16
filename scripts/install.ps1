# MBO Utilities Installation Script for Windows
# installs uv if not present, installs mbo_utilities with user-selected optional dependencies
#
# usage:
#   irm https://raw.githubusercontent.com/.../install.ps1 | iex
#
# after installation, run Install-MboEnv for a full VSCode/Jupyter development environment

$ErrorActionPreference = "Stop"

# default install location for full environment
$MBO_ENV_PATH = Join-Path $env:USERPROFILE "mbo_env"

# colors for output
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

function Show-Banner {
    Write-Host ""
    Write-Host "  __  __ ____   ___  " -ForegroundColor Cyan
    Write-Host " |  \/  | __ ) / _ \ " -ForegroundColor Cyan
    Write-Host " | |\/| |  _ \| | | |" -ForegroundColor Cyan
    Write-Host " | |  | | |_) | |_| |" -ForegroundColor Cyan
    Write-Host " |_|  |_|____/ \___/ " -ForegroundColor Cyan
    Write-Host ""
    Write-Host "MBO Utilities Installer" -ForegroundColor White
    Write-Host ""
}

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
            Write-Warn "uv was installed but not found in PATH"
            Write-Warn "You may need to restart your terminal"
        }
    }
    catch {
        Write-Err "Failed to install uv: $_"
        exit 1
    }
}

function Test-NvidiaGpu {
    <#
    .SYNOPSIS
    Check if NVIDIA GPU and CUDA are available.
    #>
    try {
        $null = Get-Command nvidia-smi -ErrorAction Stop
        $output = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($output) {
            return @{
                Available = $true
                GpuName = $output.Trim()
            }
        }
    }
    catch {}
    return @{ Available = $false; GpuName = $null }
}

function Show-OptionalDependencies {
    <#
    .SYNOPSIS
    Display information about optional dependencies and let user choose.
    #>
    param(
        [hashtable]$GpuInfo
    )

    Write-Host ""
    Write-Host "Optional Processing Pipelines" -ForegroundColor White
    Write-Host ""

    # check GPU
    if ($GpuInfo.Available) {
        Write-Host "  GPU detected: " -NoNewline -ForegroundColor Green
        Write-Host $GpuInfo.GpuName -ForegroundColor White
    }
    else {
        Write-Host "  No NVIDIA GPU detected (GPU features will be slower)" -ForegroundColor Yellow
    }
    Write-Host ""

    # describe options
    Write-Host "  [1] Suite2p   - 2D cell extraction (PyTorch + CUDA)" -ForegroundColor Cyan
    Write-Host "                  Best for: single-plane calcium imaging" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [2] Suite3D   - 3D volumetric registration (CuPy + CUDA)" -ForegroundColor Cyan
    Write-Host "                  Best for: multi-plane/volumetric imaging" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [3] Rastermap - Dimensionality reduction" -ForegroundColor Cyan
    Write-Host "                  Best for: neural activity analysis" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [4] All       - Install all processing pipelines" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [5] None      - Base installation only (fastest)" -ForegroundColor Cyan
    Write-Host "                  Includes: data viewing, format conversion, metadata" -ForegroundColor Gray
    Write-Host ""

    # get user choice
    do {
        $choice = Read-Host "Select option (1-5, or comma-separated like 1,3)"
        $valid = $choice -match '^[1-5](,[1-5])*$'
        if (-not $valid) {
            Write-Warn "Invalid selection. Enter 1-5 or comma-separated (e.g., 1,3)"
        }
    } while (-not $valid)

    # parse selection
    $extras = @()
    $choices = $choice -split ',' | ForEach-Object { $_.Trim() }

    foreach ($c in $choices) {
        switch ($c) {
            "1" { $extras += "suite2p" }
            "2" { $extras += "suite3d" }
            "3" { $extras += "rastermap" }
            "4" { $extras = @("processing"); break }
            "5" { $extras = @(); break }
        }
    }

    # warn if GPU options selected without GPU
    if (-not $GpuInfo.Available) {
        $gpuPackages = @("suite2p", "suite3d", "processing")
        $hasGpuPackage = $false
        foreach ($e in $extras) {
            if ($gpuPackages -contains $e) { $hasGpuPackage = $true; break }
        }

        if ($hasGpuPackage) {
            Write-Host ""
            Write-Warn "You selected GPU-dependent packages but no NVIDIA GPU was detected."
            Write-Warn "These packages will install but may run slowly (CPU-only mode)."
            $continue = Read-Host "Continue anyway? (y/n)"
            if ($continue -ne "y") {
                Write-Info "Removing GPU-dependent packages from selection..."
                $extras = $extras | Where-Object { $gpuPackages -notcontains $_ }
            }
        }
    }

    return $extras
}

function Get-InstallSpec {
    <#
    .SYNOPSIS
    Build the pip install specification string with extras.
    #>
    param(
        [string[]]$Extras
    )

    $baseUrl = "git+https://github.com/millerbrainobservatory/mbo_utilities.git"

    if ($Extras.Count -eq 0) {
        return "mbo_utilities @ $baseUrl"
    }

    $extraStr = $Extras -join ","
    return "mbo_utilities[$extraStr] @ $baseUrl"
}

function Install-MboUtilities {
    <#
    .SYNOPSIS
    Install mbo_utilities with selected optional dependencies.
    #>
    param(
        [string[]]$Extras = @()
    )

    Write-Info "Installing mbo_utilities..."
    if ($Extras.Count -gt 0) {
        Write-Info "  With extras: $($Extras -join ', ')"
    }
    else {
        Write-Info "  Base installation (no optional dependencies)"
    }

    try {
        # install as tool from github
        if ($Extras.Count -eq 0) {
            uv tool install mbo_utilities --from "git+https://github.com/millerbrainobservatory/mbo_utilities.git"
        }
        else {
            $extraStr = $Extras -join ","
            uv tool install mbo_utilities --from "git+https://github.com/millerbrainobservatory/mbo_utilities.git" --with "mbo_utilities[$extraStr]"
        }
        Write-Success "mbo_utilities installed successfully"
    }
    catch {
        Write-Err "Failed to install mbo_utilities: $_"
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
        try {
            $mboExe = (Get-Command mbo -ErrorAction Stop).Source
        }
        catch {
            Write-Warn "Could not locate mbo.exe - shortcut will use 'uv run mbo'"
            $mboExe = $null
        }
    }

    # download icon
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
        Write-Warn "Could not download icon, using default"
        $iconPath = $null
    }

    # create shortcut
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)

    if ($mboExe -and (Test-Path $mboExe)) {
        $shortcut.TargetPath = $mboExe
    }
    else {
        $shortcut.TargetPath = "cmd.exe"
        $shortcut.Arguments = "/k uv run mbo"
    }

    $shortcut.WorkingDirectory = [Environment]::GetFolderPath("UserProfile")

    if ($iconPath -and (Test-Path $iconPath)) {
        $shortcut.IconLocation = $iconPath
    }

    $shortcut.Description = "MBO Utilities - Miller Brain Observatory"
    $shortcut.Save()

    Write-Success "Desktop shortcut created"
}

function Install-MboEnv {
    <#
    .SYNOPSIS
    Creates a full mbo_utilities environment for use with VSCode/Jupyter.

    .DESCRIPTION
    Creates a Python virtual environment with mbo_utilities installed.
    Prompts for optional dependencies if not specified.

    .PARAMETER Path
    Installation path. Defaults to ~/mbo_env

    .PARAMETER Extras
    Optional extras: suite2p, suite3d, rastermap, processing, all.
    If not specified, prompts interactively.

    .EXAMPLE
    Install-MboEnv
    Install-MboEnv -Path "C:\projects\mbo_env" -Extras @("suite2p", "rastermap")
    #>
    param(
        [string]$Path = $MBO_ENV_PATH,
        [string[]]$Extras = $null
    )

    Write-Info "Creating MBO environment at: $Path"

    if (-not (Test-UvInstalled)) {
        Install-Uv
    }

    # prompt for extras if not specified
    if ($null -eq $Extras) {
        $gpuInfo = Test-NvidiaGpu
        $Extras = Show-OptionalDependencies -GpuInfo $gpuInfo
    }

    # create venv
    Write-Info "Creating virtual environment..."
    uv venv $Path --python 3.12

    # build install spec
    $spec = Get-InstallSpec -Extras $Extras

    # install mbo_utilities
    Write-Info "Installing mbo_utilities..."
    if ($Extras.Count -gt 0) {
        Write-Info "  With extras: $($Extras -join ', ')"
        Write-Info "  This may take several minutes for GPU packages..."
    }

    uv pip install --python "$Path\Scripts\python.exe" $spec

    # install jupyter
    Write-Info "Installing Jupyter..."
    uv pip install --python "$Path\Scripts\python.exe" jupyterlab ipykernel

    # register kernel
    Write-Info "Registering Jupyter kernel..."
    & "$Path\Scripts\python.exe" -m ipykernel install --user --name mbo --display-name "MBO Utilities"

    Write-Success "Environment created at: $Path"
    Write-Host ""
    Write-Host "To use this environment:" -ForegroundColor White
    Write-Host ""
    Write-Host "  VSCode:" -ForegroundColor Cyan
    Write-Host "    1. Open VSCode" -ForegroundColor Gray
    Write-Host "    2. Ctrl+Shift+P -> 'Python: Select Interpreter'" -ForegroundColor Gray
    Write-Host "    3. Choose: $Path\Scripts\python.exe" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  JupyterLab:" -ForegroundColor Cyan
    Write-Host "    $Path\Scripts\jupyter-lab.exe" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Terminal:" -ForegroundColor Cyan
    Write-Host "    $Path\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host ""

    # verify installation
    Write-Info "Verifying installation..."
    try {
        & "$Path\Scripts\python.exe" -c "from mbo_utilities.install_checker import check_installation, print_status_cli; print_status_cli(check_installation())"
    }
    catch {
        Write-Warn "Could not verify installation: $_"
    }

    return $Path
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
    Show-Banner

    # check/install uv
    if (-not (Test-UvInstalled)) {
        Install-Uv
    }

    # detect GPU
    $gpuInfo = Test-NvidiaGpu

    # ask user what to install
    $extras = Show-OptionalDependencies -GpuInfo $gpuInfo

    # install mbo_utilities with selected extras
    Install-MboUtilities -Extras $extras

    # create desktop shortcut
    New-DesktopShortcut

    # verify installation
    try {
        $null = Get-Command mbo -ErrorAction Stop
        Write-Success "Installation completed!"
        Write-Host ""

        if ($extras.Count -eq 0) {
            Write-Host "Installed: Base package (viewing, conversion, metadata)" -ForegroundColor Cyan
        }
        else {
            Write-Host "Installed: Base + $($extras -join ', ')" -ForegroundColor Cyan
        }
        Write-Host ""

        Write-Host "You can now:" -ForegroundColor White
        Write-Host "  - Double-click 'MBO Utilities' on your desktop" -ForegroundColor Gray
        Write-Host "  - Or run 'mbo' from any terminal" -ForegroundColor Gray
        Write-Host ""

        Write-Host "For VSCode/Jupyter development:" -ForegroundColor Yellow
        Write-Host "  Install-MboEnv" -ForegroundColor Yellow
        Write-Host ""
    }
    catch {
        Write-Warn "Installation completed but 'mbo' not found in PATH"
        Write-Warn "You may need to restart your terminal"
    }
}

# run main installer
Main

# note: Install-MboEnv remains available after piping
