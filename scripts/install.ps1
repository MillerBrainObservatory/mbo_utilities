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

function Test-CudaToolkit {
    <#
    .SYNOPSIS
    Check if CUDA Toolkit is installed (not just the driver).
    Returns the toolkit version if found, $null otherwise.
    #>

    # check CUDA_PATH environment variable
    if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
        $nvccPath = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        if (Test-Path $nvccPath) {
            try {
                $nvccOutput = & $nvccPath --version 2>$null | Out-String
                if ($nvccOutput -match "release (\d+\.\d+)") {
                    return $matches[1]
                }
            }
            catch {}
        }
    }

    # check common CUDA Toolkit locations
    $cudaPaths = @(
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA"
    )

    foreach ($basePath in $cudaPaths) {
        if (Test-Path $basePath) {
            # find version folders (e.g., v12.1, v11.8)
            $versions = Get-ChildItem -Path $basePath -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^v\d+\.\d+' } |
                Sort-Object Name -Descending

            foreach ($ver in $versions) {
                $nvccPath = Join-Path $ver.FullName "bin\nvcc.exe"
                if (Test-Path $nvccPath) {
                    # extract version from folder name
                    if ($ver.Name -match 'v(\d+\.\d+)') {
                        return $matches[1]
                    }
                }
            }
        }
    }

    # try nvcc in PATH
    try {
        $nvcc = Get-Command nvcc -ErrorAction Stop
        $nvccOutput = & $nvcc.Source --version 2>$null | Out-String
        if ($nvccOutput -match "release (\d+\.\d+)") {
            return $matches[1]
        }
    }
    catch {}

    return $null
}

function Test-NvidiaGpu {
    <#
    .SYNOPSIS
    Check if NVIDIA GPU is available, detect GPU name and CUDA Toolkit version.
    Note: CUDA version from nvidia-smi is driver capability, not toolkit version.
    We need the toolkit for PyTorch CUDA support.
    #>

    # try hardcoded path first (most reliable)
    $nvidiaSmi = "C:\Windows\System32\nvidia-smi.exe"
    if (-not (Test-Path $nvidiaSmi)) {
        # try env var path
        $nvidiaSmi = "$env:SystemRoot\System32\nvidia-smi.exe"
    }
    if (-not (Test-Path $nvidiaSmi)) {
        # try NVSMI location
        $nvidiaSmi = "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    }
    if (-not (Test-Path $nvidiaSmi)) {
        # try PATH
        try {
            $cmd = Get-Command nvidia-smi -ErrorAction Stop
            $nvidiaSmi = $cmd.Source
        }
        catch {
            return @{ Available = $false; GpuName = $null; CudaVersion = $null; ToolkitInstalled = $false }
        }
    }

    try {
        # run nvidia-smi to get GPU info
        $gpuName = & $nvidiaSmi --query-gpu=name --format=csv,noheader 2>$null
        if ($gpuName) {
            # check for actual CUDA Toolkit installation
            $toolkitVersion = Test-CudaToolkit

            return @{
                Available = $true
                GpuName = $gpuName.Trim()
                CudaVersion = $toolkitVersion  # use toolkit version, not driver version
                ToolkitInstalled = ($null -ne $toolkitVersion)
            }
        }
    }
    catch {}
    return @{ Available = $false; GpuName = $null; CudaVersion = $null; ToolkitInstalled = $false }
}

function Get-PyTorchIndexUrl {
    <#
    .SYNOPSIS
    Get the PyTorch index URL for the detected CUDA version.
    #>
    param(
        [string]$CudaVersion
    )

    if (-not $CudaVersion) {
        return $null
    }

    # parse major.minor version
    $parts = $CudaVersion -split '\.'
    $major = [int]$parts[0]
    $minor = [int]$parts[1]

    # map cuda version to pytorch index url
    # pytorch supports: cu118, cu121, cu124 (as of 2025)
    if ($major -eq 11) {
        return "https://download.pytorch.org/whl/cu118"
    }
    elseif ($major -eq 12) {
        if ($minor -le 1) {
            return "https://download.pytorch.org/whl/cu121"
        }
        elseif ($minor -le 4) {
            return "https://download.pytorch.org/whl/cu124"
        }
        else {
            # cuda 12.5+ use cu124 (forward compatible)
            return "https://download.pytorch.org/whl/cu124"
        }
    }
    else {
        # unknown version, let pip figure it out
        return $null
    }
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
        if ($GpuInfo.ToolkitInstalled) {
            Write-Host "  CUDA Toolkit: " -NoNewline -ForegroundColor Green
            Write-Host $GpuInfo.CudaVersion -ForegroundColor White
        }
        else {
            Write-Host "  CUDA Toolkit: " -NoNewline -ForegroundColor Yellow
            Write-Host "Not installed (PyTorch will use CPU)" -ForegroundColor Yellow
            Write-Host "                Install CUDA Toolkit for GPU acceleration" -ForegroundColor Gray
        }
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

    # install from PyPI
    if ($Extras.Count -eq 0) {
        return "mbo_utilities"
    }

    $extraStr = $Extras -join ","
    return "mbo_utilities[$extraStr]"
}

function Install-MboUtilities {
    <#
    .SYNOPSIS
    Install mbo_utilities with selected optional dependencies.
    #>
    param(
        [string[]]$Extras = @(),
        [hashtable]$GpuInfo = @{}
    )

    Write-Info "Installing mbo_utilities..."
    if ($Extras.Count -gt 0) {
        Write-Info "  With extras: $($Extras -join ', ')"
    }
    else {
        Write-Info "  Base installation (no optional dependencies)"
    }

    # check if pytorch is needed (suite2p or processing extras)
    $needsPytorch = $false
    foreach ($e in $Extras) {
        if ($e -eq "suite2p" -or $e -eq "processing") {
            $needsPytorch = $true
            break
        }
    }

    # temporarily allow errors (uv writes progress to stderr which PowerShell treats as error)
    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        # install pytorch with correct cuda version first if needed AND toolkit is installed
        if ($needsPytorch -and $GpuInfo.ToolkitInstalled -and $GpuInfo.CudaVersion) {
            $indexUrl = Get-PyTorchIndexUrl -CudaVersion $GpuInfo.CudaVersion
            if ($indexUrl) {
                Write-Info "Installing PyTorch for CUDA $($GpuInfo.CudaVersion)..."
                Write-Info "  Using index: $indexUrl"
                uv tool install "mbo_utilities[$($Extras -join ',')]" `
                    --python 3.12.9 `
                    --with "torch" --with "torchvision" --with "torchaudio" `
                    --extra-index-url $indexUrl 2>&1 | ForEach-Object { Write-Host $_ }
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "mbo_utilities installed with CUDA-optimized PyTorch"
                    return
                }
                else {
                    throw "uv tool install failed with exit code $LASTEXITCODE"
                }
            }
        }

        # fallback: standard installation from PyPI (includes CPU PyTorch)
        if ($Extras.Count -eq 0) {
            uv tool install mbo_utilities --python 3.12 2>&1 | ForEach-Object { Write-Host $_ }
        }
        else {
            $extraStr = $Extras -join ","
            uv tool install "mbo_utilities[$extraStr]" --python 3.12 2>&1 | ForEach-Object { Write-Host $_ }
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Success "mbo_utilities installed successfully"
        }
        else {
            throw "uv tool install failed with exit code $LASTEXITCODE"
        }
    }
    catch {
        Write-Err "Failed to install mbo_utilities: $_"
        exit 1
    }
    finally {
        $ErrorActionPreference = $prevErrorAction
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
            Write-Warn "Could not locate mbo.exe"
            $mboExe = $null
        }
    }

    # setup local app data directory
    $iconDir = Join-Path $env:LOCALAPPDATA "mbo_utilities"
    if (-not (Test-Path $iconDir)) {
        New-Item -ItemType Directory -Path $iconDir -Force | Out-Null
    }

    # download icon
    $iconPath = Join-Path $iconDir "mbo_icon.ico"
    try {
        Write-Info "Downloading icon..."
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/millerbrainobservatory/mbo_utilities/master/docs/_static/mbo_icon.ico" -OutFile $iconPath
    }
    catch {
        Write-Warn "Could not download icon, using default"
        $iconPath = $null
    }

    # download vbs launcher (runs without console window)
    $launcherPath = Join-Path $iconDir "mbo_launcher.vbs"
    try {
        Write-Info "Downloading launcher..."
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/millerbrainobservatory/mbo_utilities/master/scripts/mbo_launcher.vbs" -OutFile $launcherPath
    }
    catch {
        Write-Warn "Could not download launcher, shortcut will show console"
        $launcherPath = $null
    }

    # create shortcut
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)

    # prefer vbs launcher (no console), fallback to exe
    if ($launcherPath -and (Test-Path $launcherPath)) {
        $shortcut.TargetPath = "wscript.exe"
        $shortcut.Arguments = """$launcherPath"""
    }
    elseif ($mboExe -and (Test-Path $mboExe)) {
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
    Automatically installs PyTorch with correct CUDA version if GPU detected.

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

    # detect GPU and CUDA version
    $gpuInfo = Test-NvidiaGpu

    # prompt for extras if not specified
    if ($null -eq $Extras) {
        $Extras = Show-OptionalDependencies -GpuInfo $gpuInfo
    }

    # temporarily allow errors (uv writes progress to stderr)
    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        # create venv
        Write-Info "Creating virtual environment..."
        uv venv $Path --python 3.12 2>&1 | ForEach-Object { Write-Host $_ }

        # check if pytorch is needed
        $needsPytorch = $false
        foreach ($e in $Extras) {
            if ($e -eq "suite2p" -or $e -eq "processing" -or $e -eq "all") {
                $needsPytorch = $true
                break
            }
        }

        # install pytorch with correct cuda version first
        if ($needsPytorch -and $gpuInfo.CudaVersion) {
            $indexUrl = Get-PyTorchIndexUrl -CudaVersion $gpuInfo.CudaVersion
            if ($indexUrl) {
                Write-Info "Installing PyTorch for CUDA $($gpuInfo.CudaVersion)..."
                Write-Info "  Using index: $indexUrl"
                uv pip install --python "$Path\Scripts\python.exe" torch torchvision torchaudio --extra-index-url $indexUrl 2>&1 | ForEach-Object { Write-Host $_ }
            }
        }

        # build install spec
        $spec = Get-InstallSpec -Extras $Extras

        # install mbo_utilities
        Write-Info "Installing mbo_utilities..."
        if ($Extras.Count -gt 0) {
            Write-Info "  With extras: $($Extras -join ', ')"
            Write-Info "  This may take several minutes for GPU packages..."
        }

        uv pip install --python "$Path\Scripts\python.exe" $spec 2>&1 | ForEach-Object { Write-Host $_ }

        # install jupyter
        Write-Info "Installing Jupyter..."
        uv pip install --python "$Path\Scripts\python.exe" jupyterlab ipykernel 2>&1 | ForEach-Object { Write-Host $_ }
    }
    finally {
        $ErrorActionPreference = $prevErrorAction
    }

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
    Install-MboUtilities -Extras $extras -GpuInfo $gpuInfo

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
