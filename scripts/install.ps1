# MBO Utilities Installation Script v2 for Windows
# installs uv if not present, installs mbo_utilities with user-selected version and optional dependencies
#
# usage:
#   irm https://raw.githubusercontent.com/.../install_v2.ps1 | iex

$ErrorActionPreference = "Stop"

$GITHUB_REPO = "MillerBrainObservatory/mbo_utilities"

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
    Write-Host "MBO Utilities Installer v2" -ForegroundColor White
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
    <#
    .SYNOPSIS
    Get the latest version from PyPI.
    #>
    try {
        $response = Invoke-RestMethod -Uri "https://pypi.org/pypi/mbo-utilities/json" -TimeoutSec 10
        return $response.info.version
    }
    catch {
        return $null
    }
}

function Get-GitHubBranches {
    <#
    .SYNOPSIS
    Get list of branches from GitHub repo.
    #>
    try {
        $response = Invoke-RestMethod -Uri "https://api.github.com/repos/$GITHUB_REPO/branches" -TimeoutSec 10
        return $response | ForEach-Object { $_.name }
    }
    catch {
        Write-Warn "Could not fetch branches: $_"
        return @()
    }
}

function Get-GitHubTags {
    <#
    .SYNOPSIS
    Get list of tags (releases) from GitHub repo.
    #>
    try {
        $response = Invoke-RestMethod -Uri "https://api.github.com/repos/$GITHUB_REPO/tags" -TimeoutSec 10
        return $response | ForEach-Object { $_.name }
    }
    catch {
        return @()
    }
}

function Show-SourceSelection {
    <#
    .SYNOPSIS
    Let user choose installation source: PyPI or GitHub branch.
    #>
    Write-Host ""
    Write-Host "Installation Source" -ForegroundColor White
    Write-Host ""

    # get pypi version
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

    # github selection - show branches
    Write-Host ""
    Write-Host "Fetching available branches..." -ForegroundColor Gray

    $branches = Get-GitHubBranches

    Write-Host ""
    Write-Host "Available Branches:" -ForegroundColor White

    $options = @()
    $idx = 1

    # add master/main first if exists
    $mainBranch = $branches | Where-Object { $_ -eq "master" -or $_ -eq "main" } | Select-Object -First 1
    if ($mainBranch) {
        Write-Host "  [$idx] $mainBranch" -ForegroundColor Cyan -NoNewline
        Write-Host " (default)" -ForegroundColor Gray
        $options += @{ Type = "branch"; Name = $mainBranch }
        $idx++
    }

    # add other branches (excluding master/main)
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
            return @{ Source = "github"; Spec = "git+https://github.com/$GITHUB_REPO@$customRef" }
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
            "4" { $extras = @("processing"); break parseLoop }
            "5" { $extras = @(); break parseLoop }
        }
    }

    # warn if GPU packages selected without toolkit
    if ($extras.Count -gt 0) {
        $gpuPackages = @("suite2p", "suite3d", "processing")
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

function Install-MboUtilities {
    param(
        [string]$InstallSpec,
        [string[]]$Extras = @(),
        [hashtable]$GpuInfo = @{},
        [string]$Source = "pypi"
    )

    Write-Host ""
    Write-Info "Installing mbo_utilities..."
    Write-Info "  Source: $Source"
    if ($Extras.Count -gt 0) {
        Write-Info "  Extras: $($Extras -join ', ')"
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
        # github - extras go after the URL
        if ($Extras.Count -gt 0) {
            $spec = "$InstallSpec[" + ($Extras -join ',') + "]"
        }
        else {
            $spec = $InstallSpec
        }
    }

    # check if pytorch needed
    $needsPytorch = ($Extras | Where-Object { $_ -eq "suite2p" -or $_ -eq "processing" }).Count -gt 0

    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        # install with cuda pytorch if toolkit available
        if ($needsPytorch -and $GpuInfo.ToolkitInstalled -and $GpuInfo.CudaVersion) {
            $indexUrl = Get-PyTorchIndexUrl -CudaVersion $GpuInfo.CudaVersion
            if ($indexUrl) {
                Write-Info "Installing PyTorch for CUDA $($GpuInfo.CudaVersion)..."
                # use index-strategy to allow mixing pytorch cuda index with pypi
                # removed torchaudio due to version conflicts on windows
                uv tool install $spec `
                    --reinstall `
                    --python 3.12 `
                    --with "torch" --with "torchvision" `
                    --index-strategy unsafe-best-match `
                    --extra-index-url $indexUrl 2>&1 | ForEach-Object { Write-Host $_ }
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Installed with CUDA-optimized PyTorch"
                    return
                }
                Write-Warn "CUDA install failed, falling back to standard install..."
            }
        }

        # standard installation
        uv tool install $spec --reinstall --python 3.12 2>&1 | ForEach-Object { Write-Host $_ }

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
    param([string]$BranchRef = $null)

    Write-Info "Creating desktop shortcut..."

    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutName = if ($BranchRef -and $BranchRef -ne "master") { "MBO Utilities ($BranchRef).lnk" } else { "MBO Utilities.lnk" }
    $shortcutPath = Join-Path $desktopPath $shortcutName

    # find mbo.exe
    $mboExe = $null
    $searchPaths = @(
        (Join-Path $env:APPDATA "uv\tools\mbo-utilities\Scripts\mbo.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\tools\mbo-utilities\Scripts\mbo.exe"),
        (Join-Path $env:USERPROFILE ".local\bin\mbo.exe")
    )
    foreach ($p in $searchPaths) {
        if (Test-Path $p) { $mboExe = $p; break }
    }
    if (-not $mboExe) {
        try { $mboExe = (Get-Command mbo -ErrorAction Stop).Source }
        catch { Write-Warn "Could not locate mbo.exe"; return }
    }

    # setup icon directory
    $iconDir = Join-Path $env:LOCALAPPDATA "mbo_utilities"
    if (-not (Test-Path $iconDir)) { New-Item -ItemType Directory -Path $iconDir -Force | Out-Null }

    # use branch ref for downloads, fallback to master
    $downloadRef = if ($BranchRef) { $BranchRef } else { "master" }

    # download icon
    $iconPath = Join-Path $iconDir "mbo_icon.ico"
    try {
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/$GITHUB_REPO/$downloadRef/docs/_static/mbo_icon.ico" -OutFile $iconPath -ErrorAction Stop
    }
    catch {
        # fallback to master if branch doesn't have icon
        try { Invoke-WebRequest -Uri "https://raw.githubusercontent.com/$GITHUB_REPO/master/docs/_static/mbo_icon.ico" -OutFile $iconPath -ErrorAction Stop }
        catch { $iconPath = $null }
    }

    # create shortcut directly to mbo.exe (no VBS needed)
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $mboExe
    $shortcut.Arguments = "--splash"
    $shortcut.WorkingDirectory = [Environment]::GetFolderPath("UserProfile")
    if ($iconPath -and (Test-Path $iconPath)) { $shortcut.IconLocation = $iconPath }
    $shortcut.Description = "MBO Image Viewer"
    $shortcut.Save()

    Write-Success "Desktop shortcut created: $shortcutName"
}

function Main {
    Show-Banner

    # check/install uv
    if (-not (Test-UvInstalled)) {
        Install-Uv
    }

    # step 1: choose source (pypi vs github)
    $sourceInfo = Show-SourceSelection

    # step 2: detect GPU
    $gpuInfo = Test-NvidiaGpu

    # step 3: choose extras
    $extras = Show-OptionalDependencies -GpuInfo $gpuInfo

    # step 4: install
    Install-MboUtilities -InstallSpec $sourceInfo.Spec -Extras $extras -GpuInfo $gpuInfo -Source $sourceInfo.Source

    # step 5: create shortcut
    New-DesktopShortcut -BranchRef $sourceInfo.Ref

    # verify
    Write-Host ""
    try {
        $null = Get-Command mbo -ErrorAction Stop
        Write-Success "Installation completed!"
        Write-Host ""
        Write-Host "Source: $($sourceInfo.Source)" -ForegroundColor Cyan
        if ($sourceInfo.Ref) { Write-Host "Branch/Tag: $($sourceInfo.Ref)" -ForegroundColor Cyan }
        if ($extras.Count -gt 0) {
            Write-Host "Extras: $($extras -join ', ')" -ForegroundColor Cyan
        }
        Write-Host ""
        Write-Host "Run 'mbo' or use the desktop shortcut to start." -ForegroundColor Gray
    }
    catch {
        Write-Warn "Installation completed but 'mbo' not found in PATH"
        Write-Warn "You may need to restart your terminal"
    }
}

Main
