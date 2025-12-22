# MBO Utilities Installation Script for Windows
# installs uv if not present, creates environment, installs mbo_utilities
#
# usage:
#   # default install to ~/mbo/envs/mbo_utilities
#   irm https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.ps1 | iex
#
#   # custom install location
#   $env:MBO_ENV_PATH = "C:\path\to\env"; irm ... | iex
#
#   # install into existing environment (skip venv creation)
#   $env:MBO_ENV_PATH = "C:\existing\.venv"; $env:MBO_USE_EXISTING = "1"; irm ... | iex
#
#   # overwrite existing environment
#   $env:MBO_OVERWRITE = "1"; irm ... | iex

$ErrorActionPreference = "Stop"

$GITHUB_REPO = "MillerBrainObservatory/mbo_utilities"

# default install path (can be overridden via env var or interactive prompt)
$DEFAULT_ENV_PATH = Join-Path $env:USERPROFILE "mbo\envs\mbo_utilities"
$MBO_ENV_PATH = if ($env:MBO_ENV_PATH) { $env:MBO_ENV_PATH } else { $DEFAULT_ENV_PATH }

# check for existing env behavior flags
$USE_EXISTING_ENV = $env:MBO_USE_EXISTING -eq "1"
$OVERWRITE_ENV = $env:MBO_OVERWRITE -eq "1"

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

function Show-InstallLocationPrompt {
    # skip if env var was explicitly set
    if ($env:MBO_ENV_PATH) {
        Write-Info "Install location: $MBO_ENV_PATH (from MBO_ENV_PATH)"
        return $MBO_ENV_PATH
    }

    Write-Host ""
    Write-Host "Install Location" -ForegroundColor White
    Write-Host ""
    Write-Host "  Default: " -NoNewline -ForegroundColor Gray
    Write-Host $DEFAULT_ENV_PATH -ForegroundColor Cyan
    Write-Host ""
    $userInput = Read-Host "Press Enter for default, or enter custom path"

    if ([string]::IsNullOrWhiteSpace($userInput)) {
        return $DEFAULT_ENV_PATH
    }

    # expand ~ to user profile
    $customPath = $userInput.Trim()
    if ($customPath.StartsWith("~")) {
        $customPath = $customPath.Replace("~", $env:USERPROFILE)
    }

    # resolve relative paths
    if (-not [System.IO.Path]::IsPathRooted($customPath)) {
        $customPath = [System.IO.Path]::GetFullPath($customPath)
    }

    return $customPath
}

function Test-UvInstalled {
    try {
        $null = Get-Command uv -ErrorAction Stop
        $version = uv --version
        Write-Info "uv is already installed: $version"
        return $true
    }
    catch {
        return $false
    }
}

function Install-Uv {
    Write-Info "Installing uv..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        if (Test-UvInstalled) {
            Write-Success "uv installed successfully"
        }
        else {
            Write-Warn "uv installed but not found in PATH. Restart terminal after installation."
        }
    }
    catch {
        Write-Err "Failed to install uv: $_"
        exit 1
    }
}

function Get-PyPiVersion {
    try {
        $response = Invoke-RestMethod -Uri "https://pypi.org/pypi/mbo-utilities/json" -TimeoutSec 10
        return $response.info.version
    }
    catch {
        return $null
    }
}

function Get-GitHubBranches {
    try {
        $response = Invoke-RestMethod -Uri "https://api.github.com/repos/$GITHUB_REPO/branches" -TimeoutSec 10
        return $response | ForEach-Object { $_.name }
    }
    catch {
        Write-Warn "Could not fetch branches: $_"
        return @()
    }
}

function Show-SourceSelection {
    Write-Host ""
    Write-Host "Installation Source" -ForegroundColor White
    Write-Host ""

    $pypiVersion = Get-PyPiVersion
    if ($pypiVersion) {
        Write-Host "  [1] PyPI (stable)" -ForegroundColor Cyan
        Write-Host "      Version: $pypiVersion" -ForegroundColor Gray
    }
    else {
        Write-Host "  [1] PyPI (stable)" -ForegroundColor Cyan
        Write-Host "      Version: unknown (could not fetch)" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  [2] GitHub (development)" -ForegroundColor Cyan
    Write-Host "      Install from a specific branch or tag" -ForegroundColor Gray
    Write-Host ""

    do {
        $choice = Read-Host "Select source (1-2)"
        $valid = $choice -match '^[12]$'
        if (-not $valid) {
            Write-Warn "Invalid selection. Enter 1 or 2."
        }
    } while (-not $valid)

    if ($choice -eq "1") {
        return @{ Source = "pypi"; Spec = "mbo_utilities" }
    }

    Write-Host ""
    Write-Host "Fetching available branches..." -ForegroundColor Gray

    $branches = Get-GitHubBranches

    Write-Host ""
    Write-Host "Available Branches:" -ForegroundColor White

    $options = @()
    $idx = 1

    $mainBranch = $branches | Where-Object { $_ -eq "master" -or $_ -eq "main" } | Select-Object -First 1
    if ($mainBranch) {
        Write-Host "  [$idx] $mainBranch" -ForegroundColor Cyan -NoNewline
        Write-Host " (default)" -ForegroundColor Gray
        $options += @{ Type = "branch"; Name = $mainBranch }
        $idx++
    }

    $otherBranches = $branches | Where-Object { $_ -ne "master" -and $_ -ne "main" } | Select-Object -First 10
    foreach ($branch in $otherBranches) {
        Write-Host "  [$idx] $branch" -ForegroundColor Cyan
        $options += @{ Type = "branch"; Name = $branch }
        $idx++
    }

    Write-Host ""
    Write-Host "  [c] Custom branch/tag name" -ForegroundColor Yellow
    Write-Host ""

    do {
        $choice = Read-Host "Select branch/tag (1-$($options.Count) or 'c' for custom)"
        if ($choice -eq "c") {
            $customRef = Read-Host "Enter branch or tag name"
            return @{ Source = "github"; Spec = "git+https://github.com/$GITHUB_REPO@$customRef"; Ref = $customRef }
        }
        $choiceNum = 0
        $valid = [int]::TryParse($choice, [ref]$choiceNum) -and $choiceNum -ge 1 -and $choiceNum -le $options.Count
        if (-not $valid) {
            Write-Warn "Invalid selection."
        }
    } while (-not $valid)

    $selected = $options[$choiceNum - 1]
    $ref = $selected.Name
    return @{ Source = "github"; Spec = "git+https://github.com/$GITHUB_REPO@$ref"; Ref = $ref }
}

function Test-CudaToolkit {
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

    $cudaPaths = @("$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA")
    foreach ($basePath in $cudaPaths) {
        if (Test-Path $basePath) {
            $versions = Get-ChildItem -Path $basePath -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^v\d+\.\d+' } |
                Sort-Object Name -Descending
            foreach ($ver in $versions) {
                $nvccPath = Join-Path $ver.FullName "bin\nvcc.exe"
                if (Test-Path $nvccPath) {
                    if ($ver.Name -match 'v(\d+\.\d+)') {
                        return $matches[1]
                    }
                }
            }
        }
    }

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
    $nvidiaSmi = "C:\Windows\System32\nvidia-smi.exe"
    if (-not (Test-Path $nvidiaSmi)) { $nvidiaSmi = "$env:SystemRoot\System32\nvidia-smi.exe" }
    if (-not (Test-Path $nvidiaSmi)) { $nvidiaSmi = "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe" }
    if (-not (Test-Path $nvidiaSmi)) {
        try { $nvidiaSmi = (Get-Command nvidia-smi -ErrorAction Stop).Source }
        catch { return @{ Available = $false; GpuName = $null; CudaVersion = $null; ToolkitInstalled = $false } }
    }

    try {
        $gpuName = & $nvidiaSmi --query-gpu=name --format=csv,noheader 2>$null
        if ($gpuName) {
            $toolkitVersion = Test-CudaToolkit
            return @{
                Available = $true
                GpuName = $gpuName.Trim()
                CudaVersion = $toolkitVersion
                ToolkitInstalled = ($null -ne $toolkitVersion)
            }
        }
    }
    catch {}
    return @{ Available = $false; GpuName = $null; CudaVersion = $null; ToolkitInstalled = $false }
}

function Get-PyTorchIndexUrl {
    param([string]$CudaVersion)
    if (-not $CudaVersion) { return $null }

    $parts = $CudaVersion -split '\.'
    $major = [int]$parts[0]
    $minor = [int]$parts[1]

    if ($major -eq 11) { return "https://download.pytorch.org/whl/cu118" }
    elseif ($major -eq 12) {
        if ($minor -le 1) { return "https://download.pytorch.org/whl/cu121" }
        else { return "https://download.pytorch.org/whl/cu124" }
    }
    return $null
}

function Show-OptionalDependencies {
    param([hashtable]$GpuInfo)

    Write-Host ""
    Write-Host "Optional Dependencies" -ForegroundColor White
    Write-Host ""

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
        }
    }
    else {
        Write-Host "  No NVIDIA GPU detected (GPU features will be slower)" -ForegroundColor Yellow
    }
    Write-Host ""

    Write-Host "  [1] Suite2p   - 2D cell extraction (PyTorch + CUDA)" -ForegroundColor Cyan
    Write-Host "  [2] Suite3D   - 3D volumetric registration (CuPy + CUDA)" -ForegroundColor Cyan
    Write-Host "  [3] Rastermap - Dimensionality reduction" -ForegroundColor Cyan
    Write-Host "  [4] All       - Install all processing pipelines" -ForegroundColor Cyan
    Write-Host "  [5] None      - Base installation only (fastest)" -ForegroundColor Cyan
    Write-Host ""

    do {
        $choice = Read-Host "Select option (1-5, or comma-separated like 1,3)"
        $valid = $choice -match '^[1-5](,[1-5])*$'
        if (-not $valid) { Write-Warn "Invalid selection. Enter 1-5 or comma-separated." }
    } while (-not $valid)

    $extras = @()
    $choices = $choice -split ',' | ForEach-Object { $_.Trim() } | Select-Object -Unique

    :parseLoop foreach ($c in $choices) {
        switch ($c) {
            "1" { $extras += "suite2p" }
            "2" { $extras += "suite3d" }
            "3" { $extras += "rastermap" }
            "4" { $extras = @("all"); break parseLoop }
            "5" { $extras = @(); break parseLoop }
        }
    }

    # warn if GPU packages selected without toolkit
    if ($extras.Count -gt 0) {
        $gpuPackages = @("suite2p", "suite3d", "all")
        $hasGpuPackage = ($extras | Where-Object { $gpuPackages -contains $_ }).Count -gt 0

        if ($hasGpuPackage -and $GpuInfo.Available -and -not $GpuInfo.ToolkitInstalled) {
            Write-Host ""
            Write-Warn "CUDA Toolkit not installed. PyTorch will use CPU (slower)."
            Write-Warn "Install CUDA Toolkit for GPU acceleration."
        }
        elseif ($hasGpuPackage -and -not $GpuInfo.Available) {
            Write-Host ""
            Write-Warn "No NVIDIA GPU detected. These packages will run in CPU-only mode."
            $continue = Read-Host "Continue anyway? (y/n)"
            if ($continue -ne "y") {
                $extras = $extras | Where-Object { $gpuPackages -notcontains $_ }
            }
        }
    }

    return $extras
}

function Install-MboEnvironment {
    param(
        [string]$InstallSpec,
        [string[]]$Extras = @(),
        [hashtable]$GpuInfo = @{},
        [string]$Source = "pypi"
    )

    Write-Host ""
    Write-Info "Environment path: $MBO_ENV_PATH"

    $pythonPath = Join-Path $MBO_ENV_PATH "Scripts\python.exe"
    $envExists = Test-Path $pythonPath

    if ($envExists) {
        if ($USE_EXISTING_ENV) {
            Write-Info "Using existing environment (MBO_USE_EXISTING=1)"
        }
        elseif ($OVERWRITE_ENV) {
            Write-Warn "Removing existing environment (MBO_OVERWRITE=1)..."
            Remove-Item -Recurse -Force $MBO_ENV_PATH
            $envExists = $false
        }
        else {
            Write-Host ""
            Write-Warn "Environment already exists at: $MBO_ENV_PATH"
            Write-Host ""
            Write-Host "  [1] Overwrite - Delete and recreate environment" -ForegroundColor Cyan
            Write-Host "  [2] Update    - Install into existing environment" -ForegroundColor Cyan
            Write-Host "  [3] Cancel    - Exit without changes" -ForegroundColor Cyan
            Write-Host ""

            do {
                $choice = Read-Host "Select option (1-3)"
                $valid = $choice -match '^[123]$'
                if (-not $valid) { Write-Warn "Invalid selection. Enter 1, 2, or 3." }
            } while (-not $valid)

            switch ($choice) {
                "1" {
                    Write-Info "Removing existing environment..."
                    Remove-Item -Recurse -Force $MBO_ENV_PATH
                    $envExists = $false
                }
                "2" {
                    Write-Info "Installing into existing environment..."
                }
                "3" {
                    Write-Info "Installation cancelled."
                    exit 0
                }
            }
        }
    }

    if (-not $envExists) {
        # create parent directory
        $envParent = Split-Path $MBO_ENV_PATH -Parent
        if (-not (Test-Path $envParent)) {
            New-Item -ItemType Directory -Path $envParent -Force | Out-Null
        }

        # create virtual environment
        Write-Info "Creating virtual environment..."
        uv venv $MBO_ENV_PATH --python 3.12
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Failed to create virtual environment"
            exit 1
        }
    }

    # build the install spec with extras
    if ($Source -eq "pypi") {
        if ($Extras.Count -gt 0) {
            $spec = "mbo_utilities[$($Extras -join ',')]"
        }
        else {
            $spec = "mbo_utilities"
        }
    }
    else {
        if ($Extras.Count -gt 0) {
            $spec = "$InstallSpec[" + ($Extras -join ',') + "]"
        }
        else {
            $spec = $InstallSpec
        }
    }

    Write-Info "Installing mbo_utilities..."
    Write-Info "  Source: $Source"
    if ($Extras.Count -gt 0) {
        Write-Info "  Extras: $($Extras -join ', ')"
    }

    $needsPytorch = ($Extras | Where-Object { $_ -eq "suite2p" -or $_ -eq "all" }).Count -gt 0

    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        if ($needsPytorch -and $GpuInfo.ToolkitInstalled -and $GpuInfo.CudaVersion) {
            $indexUrl = Get-PyTorchIndexUrl -CudaVersion $GpuInfo.CudaVersion
            if ($indexUrl) {
                Write-Info "Installing with CUDA-optimized PyTorch for CUDA $($GpuInfo.CudaVersion)..."
                uv pip install --python $pythonPath $spec torch torchvision `
                    --index-strategy unsafe-best-match `
                    --extra-index-url $indexUrl 2>&1 | ForEach-Object { Write-Host $_ }
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Installed with CUDA-optimized PyTorch"
                }
                else {
                    Write-Warn "CUDA install failed, falling back to standard install..."
                    uv pip install --python $pythonPath $spec 2>&1 | ForEach-Object { Write-Host $_ }
                }
            }
            else {
                uv pip install --python $pythonPath $spec 2>&1 | ForEach-Object { Write-Host $_ }
            }
        }
        else {
            uv pip install --python $pythonPath $spec 2>&1 | ForEach-Object { Write-Host $_ }
        }

        if ($LASTEXITCODE -ne 0) {
            throw "uv pip install failed with exit code $LASTEXITCODE"
        }

        Write-Success "mbo_utilities installed successfully"
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
    param([string]$BranchRef = $null)

    Write-Info "Creating desktop shortcut..."

    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutName = if ($BranchRef -and $BranchRef -ne "master") { "MBO Utilities ($BranchRef).lnk" } else { "MBO Utilities.lnk" }
    $shortcutPath = Join-Path $desktopPath $shortcutName

    # setup icon directory
    $iconDir = Join-Path $env:USERPROFILE "mbo"
    if (-not (Test-Path $iconDir)) { New-Item -ItemType Directory -Path $iconDir -Force | Out-Null }

    $downloadRef = if ($BranchRef) { $BranchRef } else { "master" }

    # download icon
    $iconPath = Join-Path $iconDir "mbo_icon.ico"
    try {
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/$GITHUB_REPO/$downloadRef/mbo_utilities/assets/static/mbo_icon.ico" -OutFile $iconPath -ErrorAction Stop
    }
    catch {
        try { Invoke-WebRequest -Uri "https://raw.githubusercontent.com/$GITHUB_REPO/master/mbo_utilities/assets/static/mbo_icon.ico" -OutFile $iconPath -ErrorAction Stop }
        catch { $iconPath = $null }
    }

    # create launcher script
    $launcherPath = Join-Path $iconDir "launch_mbo.bat"
    $pythonPath = Join-Path $MBO_ENV_PATH "Scripts\python.exe"
    @"
@echo off
"$pythonPath" -m mbo_utilities.cli view --splash
"@ | Set-Content -Path $launcherPath -Encoding ASCII

    # create shortcut
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $launcherPath
    $shortcut.WorkingDirectory = [Environment]::GetFolderPath("UserProfile")
    if ($iconPath -and (Test-Path $iconPath)) { $shortcut.IconLocation = $iconPath }
    $shortcut.Description = "MBO Image Viewer"
    $shortcut.Save()

    Write-Success "Desktop shortcut created: $shortcutName"
}

function Add-MboToPath {
    Write-Info "Adding MBO to user PATH..."

    $scriptsPath = Join-Path $MBO_ENV_PATH "Scripts"
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

    if ($currentPath -notlike "*$scriptsPath*") {
        $newPath = "$scriptsPath;$currentPath"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        $env:Path = "$scriptsPath;$env:Path"
        Write-Success "Added $scriptsPath to PATH"
    }
    else {
        Write-Info "MBO already in PATH"
    }
}

function Show-UsageInstructions {
    Write-Host ""
    Write-Host "Environment Location" -ForegroundColor White
    Write-Host "  $MBO_ENV_PATH" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage" -ForegroundColor White
    Write-Host ""
    Write-Host "  Desktop shortcut:" -ForegroundColor Gray
    Write-Host "    Double-click 'MBO Utilities' on your desktop" -ForegroundColor White
    Write-Host ""
    Write-Host "  Command line (after restarting terminal):" -ForegroundColor Gray
    Write-Host "    mbo" -ForegroundColor White
    Write-Host ""
    Write-Host "  Activate environment:" -ForegroundColor Gray
    Write-Host "    $MBO_ENV_PATH\Scripts\activate" -ForegroundColor White
    Write-Host ""
    Write-Host "  VSCode:" -ForegroundColor Gray
    Write-Host "    1. Open VSCode" -ForegroundColor White
    Write-Host "    2. Ctrl+Shift+P -> 'Python: Select Interpreter'" -ForegroundColor White
    Write-Host "    3. Choose: $MBO_ENV_PATH\Scripts\python.exe" -ForegroundColor White
    Write-Host ""
    Write-Host "  Add packages to environment:" -ForegroundColor Gray
    Write-Host "    uv pip install --python `"$MBO_ENV_PATH\Scripts\python.exe`" <package>" -ForegroundColor White
    Write-Host ""
}

function Main {
    Show-Banner

    # check/install uv
    if (-not (Test-UvInstalled)) {
        Install-Uv
    }

    # step 1: choose install location
    $script:MBO_ENV_PATH = Show-InstallLocationPrompt

    # step 2: choose source
    $sourceInfo = Show-SourceSelection

    # step 3: detect GPU
    $gpuInfo = Test-NvidiaGpu

    # step 4: choose extras
    $extras = Show-OptionalDependencies -GpuInfo $gpuInfo

    # step 5: create environment and install
    Install-MboEnvironment -InstallSpec $sourceInfo.Spec -Extras $extras -GpuInfo $gpuInfo -Source $sourceInfo.Source

    # step 6: add to PATH
    Add-MboToPath

    # step 7: create shortcut
    New-DesktopShortcut -BranchRef $sourceInfo.Ref

    # show instructions
    Show-UsageInstructions

    Write-Success "Installation completed!"
    Write-Host ""
    Write-Host "You may need to restart your terminal for PATH changes to take effect." -ForegroundColor Yellow
}

Main
