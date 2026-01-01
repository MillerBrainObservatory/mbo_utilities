# MBO Utilities Installation Script for Windows
# installs mbo CLI via uv tool, optionally creates a dev environment
#
# usage:
#   irm https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.ps1 | iex
#
# environment variables:
#   MBO_ENV_PATH     - custom path for dev environment (default: ~/mbo/envs/mbo_utilities)
#   MBO_SKIP_ENV     - set to "1" to skip dev environment creation
#   MBO_OVERWRITE    - set to "1" to overwrite existing installations

$ErrorActionPreference = "Stop"

$GITHUB_REPO = "MillerBrainObservatory/mbo_utilities"
$DEFAULT_ENV_PATH = Join-Path $env:USERPROFILE "mbo\envs\mbo_utilities"

# System dependency URLs
$MSVC_BUILD_TOOLS_URL = "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
$FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/"

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

function Test-MsvcBuildTools {
    <#
    .SYNOPSIS
    Check if Microsoft Visual C++ Build Tools are installed.
    Required for compiling Python packages with C extensions.
    #>

    # Check via vswhere (most reliable)
    $vswherePaths = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    )

    foreach ($vswhere in $vswherePaths) {
        if (Test-Path $vswhere) {
            try {
                # Check for any VS installation with C++ build tools
                $result = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
                if ($result) {
                    return @{ Installed = $true; Path = $result }
                }
            }
            catch {}
        }
    }

    # Fallback: check for cl.exe in common paths
    $clPaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio *\VC\bin\cl.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio *\VC\bin\cl.exe"
    )

    foreach ($pattern in $clPaths) {
        $found = Get-Item $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            return @{ Installed = $true; Path = $found.DirectoryName }
        }
    }

    # Check if cl.exe is in PATH
    try {
        $cl = Get-Command cl.exe -ErrorAction Stop
        return @{ Installed = $true; Path = $cl.Source }
    }
    catch {}

    return @{ Installed = $false; Path = $null }
}

function Test-Ffmpeg {
    <#
    .SYNOPSIS
    Check if ffmpeg is installed (optional, for video export).
    #>
    try {
        $ffmpeg = Get-Command ffmpeg -ErrorAction Stop
        $version = & ffmpeg -version 2>&1 | Select-Object -First 1
        if ($version -match "ffmpeg version ([^\s]+)") {
            return @{ Installed = $true; Version = $matches[1] }
        }
        return @{ Installed = $true; Version = "unknown" }
    }
    catch {
        return @{ Installed = $false; Version = $null }
    }
}

function Test-SystemDependencies {
    <#
    .SYNOPSIS
    Check all required and optional system dependencies.
    Returns a hashtable with dependency status.
    #>
    $msvc = Test-MsvcBuildTools
    $ffmpeg = Test-Ffmpeg

    return @{
        Msvc = $msvc
        Ffmpeg = $ffmpeg
    }
}

function Show-SystemDependencyCheck {
    <#
    .SYNOPSIS
    Display system dependency status and prompt user if required deps are missing.
    Returns $true if installation should proceed, $false to abort.
    #>
    param([hashtable]$Deps)

    Write-Host ""
    Write-Host "System Dependencies" -ForegroundColor White
    Write-Host ""

    $hasMissing = $false

    # MSVC Build Tools (required)
    if ($Deps.Msvc.Installed) {
        Write-Host "  [" -NoNewline
        Write-Host "OK" -ForegroundColor Green -NoNewline
        Write-Host "] Microsoft C++ Build Tools" -NoNewline
        Write-Host " (required)" -ForegroundColor Gray
    }
    else {
        Write-Host "  [" -NoNewline
        Write-Host "MISSING" -ForegroundColor Red -NoNewline
        Write-Host "] Microsoft C++ Build Tools" -NoNewline
        Write-Host " (required)" -ForegroundColor Gray
        $hasMissing = $true
    }

    # ffmpeg (optional)
    if ($Deps.Ffmpeg.Installed) {
        Write-Host "  [" -NoNewline
        Write-Host "OK" -ForegroundColor Green -NoNewline
        Write-Host "] ffmpeg" -NoNewline
        Write-Host " (optional, for video export)" -ForegroundColor Gray
    }
    else {
        Write-Host "  [" -NoNewline
        Write-Host "  " -NoNewline
        Write-Host "] ffmpeg" -NoNewline
        Write-Host " (optional, for video export)" -ForegroundColor Gray
    }

    Write-Host ""

    if ($hasMissing) {
        Write-Host ""
        Write-Err "Required system dependencies are missing."
        Write-Host ""
        Write-Host "  Microsoft C++ Build Tools is required to compile Python packages." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Install from:" -ForegroundColor White
        Write-Host "    $MSVC_BUILD_TOOLS_URL" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  During installation, select:" -ForegroundColor White
        Write-Host "    'Desktop development with C++'" -ForegroundColor Cyan
        Write-Host "  or at minimum:" -ForegroundColor White
        Write-Host "    'MSVC v143 - VS 2022 C++ x64/x86 build tools'" -ForegroundColor Cyan
        Write-Host "    'Windows 10/11 SDK'" -ForegroundColor Cyan
        Write-Host ""

        Write-Host "  [1] Open download page and exit" -ForegroundColor Cyan
        Write-Host "  [2] Continue anyway (may fail)" -ForegroundColor Cyan
        Write-Host "  [3] Exit" -ForegroundColor Cyan
        Write-Host ""

        do {
            $choice = Read-Host "Select option (1-3)"
            $valid = $choice -match '^[123]$'
            if (-not $valid) { Write-Warn "Invalid selection." }
        } while (-not $valid)

        switch ($choice) {
            "1" {
                Start-Process $MSVC_BUILD_TOOLS_URL
                Write-Host ""
                Write-Info "After installing Build Tools, restart your terminal and run this script again."
                return $false
            }
            "2" {
                Write-Host ""
                Write-Warn "Continuing without Build Tools. Installation may fail for some packages."
                return $true
            }
            "3" {
                return $false
            }
        }
    }

    return $true
}

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
        return @{ Source = "pypi"; Spec = "mbo_utilities"; Ref = $null }
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
            return @{ Source = "github"; Spec = "mbo_utilities @ git+https://github.com/$GITHUB_REPO@$customRef"; Ref = $customRef }
        }
        $choiceNum = 0
        $valid = [int]::TryParse($choice, [ref]$choiceNum) -and $choiceNum -ge 1 -and $choiceNum -le $options.Count
        if (-not $valid) {
            Write-Warn "Invalid selection."
        }
    } while (-not $valid)

    $selected = $options[$choiceNum - 1]
    $ref = $selected.Name
    return @{ Source = "github"; Spec = "mbo_utilities @ git+https://github.com/$GITHUB_REPO@$ref"; Ref = $ref }
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
            Write-Host "Not installed (GPU packages will use CPU)" -ForegroundColor Yellow
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
            Write-Warn "CUDA Toolkit not installed. GPU packages will use CPU (slower)."
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

function Install-MboTool {
    param(
        [string]$Spec,
        [string[]]$Extras = @()
    )

    Write-Host ""
    Write-Info "Installing mbo CLI tool via uv tool install..."

    # build spec with extras
    # For git URLs: "mbo_utilities @ git+..." -> "mbo_utilities[extras] @ git+..."
    # For PyPI: "mbo_utilities" -> "mbo_utilities[extras]"
    if ($Extras.Count -gt 0) {
        $extrasStr = "[" + ($Extras -join ',') + "]"
        if ($Spec -match '^([^\s@]+)(\s*@\s*.*)$') {
            # Git URL format: insert extras after package name, before @ URL
            $fullSpec = $matches[1] + $extrasStr + $matches[2]
        }
        else {
            # PyPI format: just append extras
            $fullSpec = "$Spec$extrasStr"
        }
    }
    else {
        $fullSpec = $Spec
    }

    Write-Info "  Spec: $fullSpec"

    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        # check if already installed
        $existingTools = uv tool list 2>$null | Out-String
        if ($existingTools -match "mbo[_-]utilities") {
            if ($env:MBO_OVERWRITE -eq "1") {
                Write-Info "Uninstalling existing mbo_utilities..."
                uv tool uninstall mbo_utilities 2>&1 | Out-Null
            }
            else {
                Write-Host ""
                Write-Warn "mbo_utilities is already installed as a tool."
                Write-Host ""
                Write-Host "  [1] Upgrade   - Uninstall and reinstall" -ForegroundColor Cyan
                Write-Host "  [2] Skip      - Keep existing installation" -ForegroundColor Cyan
                Write-Host "  [3] Cancel    - Exit" -ForegroundColor Cyan
                Write-Host ""

                do {
                    $choice = Read-Host "Select option (1-3)"
                    $valid = $choice -match '^[123]$'
                    if (-not $valid) { Write-Warn "Invalid selection." }
                } while (-not $valid)

                switch ($choice) {
                    "1" {
                        Write-Info "Uninstalling existing mbo_utilities..."
                        uv tool uninstall mbo_utilities 2>&1 | Out-Null
                    }
                    "2" {
                        Write-Info "Keeping existing installation."
                        return $true
                    }
                    "3" {
                        Write-Info "Installation cancelled."
                        exit 0
                    }
                }
            }
        }

        # install tool
        uv tool install $fullSpec --python 3.12 2>&1 | ForEach-Object { Write-Host $_ }

        if ($LASTEXITCODE -ne 0) {
            throw "uv tool install failed with exit code $LASTEXITCODE"
        }

        Write-Success "mbo CLI tool installed successfully"
        return $true
    }
    catch {
        Write-Err "Failed to install mbo tool: $_"
        return $false
    }
    finally {
        $ErrorActionPreference = $prevErrorAction
    }
}

function Show-InstallTypePrompt {
    Write-Host ""
    Write-Host "Installation Type" -ForegroundColor White
    Write-Host ""
    Write-Host "  [1] Environment + CLI - Create Python venv with mbo_utilities + global CLI (Recommended)" -ForegroundColor Cyan
    Write-Host "  [2] Environment       - Create Python venv only (for library use)" -ForegroundColor Cyan
    Write-Host "  [3] CLI               - Install global CLI only (mbo command)" -ForegroundColor Cyan
    Write-Host ""

    do {
        $choice = Read-Host "Select installation type (1-3)"
        $valid = $choice -match '^[123]$'
        if (-not $valid) { Write-Warn "Invalid selection." }
    } while (-not $valid)

    switch ($choice) {
        "1" { return @{ InstallCli = $true; InstallEnv = $true } }
        "2" { return @{ InstallCli = $false; InstallEnv = $true } }
        "3" { return @{ InstallCli = $true; InstallEnv = $false } }
    }
}

function Show-EnvLocationPrompt {
    if ($env:MBO_ENV_PATH) {
        Write-Info "Environment location: $($env:MBO_ENV_PATH) (from MBO_ENV_PATH)"
        return $env:MBO_ENV_PATH
    }

    Write-Host ""
    Write-Host "Environment Location" -ForegroundColor White
    Write-Host ""
    Write-Host "  Default: " -NoNewline -ForegroundColor Gray
    Write-Host $DEFAULT_ENV_PATH -ForegroundColor Cyan
    Write-Host ""
    $userInput = Read-Host "Press Enter for default, or enter custom path"

    if ([string]::IsNullOrWhiteSpace($userInput)) {
        return $DEFAULT_ENV_PATH
    }

    $customPath = $userInput.Trim()
    if ($customPath.StartsWith("~")) {
        $customPath = $customPath.Replace("~", $env:USERPROFILE)
    }
    if (-not [System.IO.Path]::IsPathRooted($customPath)) {
        $customPath = [System.IO.Path]::GetFullPath($customPath)
    }

    return $customPath
}

function Install-DevEnvironment {
    param(
        [string]$EnvPath,
        [string]$Spec,
        [string[]]$Extras = @(),
        [hashtable]$GpuInfo = @{}
    )

    Write-Host ""
    Write-Info "Creating development environment at: $EnvPath"

    $pythonPath = Join-Path $EnvPath "Scripts\python.exe"
    $envExists = Test-Path $pythonPath
    $dirExists = Test-Path $EnvPath

    # Check if directory exists but is not a valid venv (likely a project directory)
    if ($dirExists -and -not $envExists) {
        # Check if it looks like a project directory (has pyproject.toml, .git, etc.)
        $isProjectDir = (Test-Path (Join-Path $EnvPath "pyproject.toml")) -or
                        (Test-Path (Join-Path $EnvPath ".git")) -or
                        (Test-Path (Join-Path $EnvPath "setup.py"))

        if ($isProjectDir) {
            Write-Host ""
            Write-Warn "This looks like a project directory: $EnvPath"
            Write-Host ""
            Write-Host "  [1] Create .venv inside - Create environment at $EnvPath\.venv (Recommended)" -ForegroundColor Cyan
            Write-Host "  [2] Skip                - Don't create dev environment" -ForegroundColor Cyan
            Write-Host ""

            do {
                $choice = Read-Host "Select option (1-2)"
                $valid = $choice -match '^[12]$'
                if (-not $valid) { Write-Warn "Invalid selection." }
            } while (-not $valid)

            switch ($choice) {
                "1" {
                    $EnvPath = Join-Path $EnvPath ".venv"
                    Write-Info "Will create environment at: $EnvPath"
                    $pythonPath = Join-Path $EnvPath "Scripts\python.exe"
                    $envExists = Test-Path $pythonPath
                }
                "2" {
                    Write-Info "Skipping dev environment."
                    return $null
                }
            }
        }
        else {
            Write-Host ""
            Write-Warn "Directory exists but is not a virtual environment: $EnvPath"
            Write-Host ""
            Write-Host "  [1] Create .venv inside - Create environment at $EnvPath\.venv" -ForegroundColor Cyan
            Write-Host "  [2] Skip                - Don't create dev environment" -ForegroundColor Cyan
            Write-Host ""

            do {
                $choice = Read-Host "Select option (1-2)"
                $valid = $choice -match '^[12]$'
                if (-not $valid) { Write-Warn "Invalid selection." }
            } while (-not $valid)

            switch ($choice) {
                "1" {
                    $EnvPath = Join-Path $EnvPath ".venv"
                    Write-Info "Will create environment at: $EnvPath"
                    $pythonPath = Join-Path $EnvPath "Scripts\python.exe"
                    $envExists = Test-Path $pythonPath
                }
                "2" {
                    Write-Info "Skipping dev environment."
                    return $null
                }
            }
        }
    }

    if ($envExists) {
        if ($env:MBO_OVERWRITE -eq "1") {
            Write-Warn "Removing existing environment..."
            Remove-Item -Recurse -Force $EnvPath
            $envExists = $false
        }
        else {
            Write-Host ""
            Write-Warn "Environment already exists at: $EnvPath"
            Write-Host ""
            Write-Host "  [1] Overwrite - Delete and recreate" -ForegroundColor Cyan
            Write-Host "  [2] Update    - Install into existing" -ForegroundColor Cyan
            Write-Host "  [3] Skip      - Don't create dev environment" -ForegroundColor Cyan
            Write-Host ""

            do {
                $choice = Read-Host "Select option (1-3)"
                $valid = $choice -match '^[123]$'
                if (-not $valid) { Write-Warn "Invalid selection." }
            } while (-not $valid)

            switch ($choice) {
                "1" {
                    Write-Info "Removing existing environment..."
                    Remove-Item -Recurse -Force $EnvPath
                    $envExists = $false
                }
                "2" {
                    Write-Info "Updating existing environment..."
                }
                "3" {
                    Write-Info "Skipping dev environment."
                    return $null
                }
            }
        }
    }

    if (-not $envExists) {
        $envParent = Split-Path $EnvPath -Parent
        if (-not (Test-Path $envParent)) {
            New-Item -ItemType Directory -Path $envParent -Force | Out-Null
        }

        Write-Info "Creating virtual environment with Python 3.12..."
        uv venv $EnvPath --python 3.12
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Failed to create virtual environment"
            return $null
        }
    }

    # build spec with extras
    # For git URLs: "mbo_utilities @ git+..." -> "mbo_utilities[extras] @ git+..."
    # For PyPI: "mbo_utilities" -> "mbo_utilities[extras]"
    if ($Extras.Count -gt 0) {
        $extrasStr = "[" + ($Extras -join ',') + "]"
        if ($Spec -match '^([^\s@]+)(\s*@\s*.*)$') {
            # Git URL format: insert extras after package name, before @ URL
            $fullSpec = $matches[1] + $extrasStr + $matches[2]
        }
        else {
            # PyPI format: just append extras
            $fullSpec = "$Spec$extrasStr"
        }
    }
    else {
        $fullSpec = $Spec
    }

    Write-Info "Installing mbo_utilities into environment..."
    Write-Info "  Spec: $fullSpec"

    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        uv pip install --python $pythonPath $fullSpec 2>&1 | ForEach-Object { Write-Host $_ }

        if ($LASTEXITCODE -ne 0) {
            throw "uv pip install failed"
        }

        Write-Success "Development environment created successfully"
        return $EnvPath
    }
    catch {
        Write-Err "Failed to install into environment: $_"
        return $null
    }
    finally {
        $ErrorActionPreference = $prevErrorAction
    }
}

function Get-UvToolBinDir {
    try {
        $binDir = uv tool bin-dir 2>$null
        if ($binDir) {
            return $binDir.Trim()
        }
    }
    catch {}

    # fallback to common locations
    $fallbacks = @(
        (Join-Path $env:USERPROFILE ".local\bin"),
        (Join-Path $env:LOCALAPPDATA "uv\bin")
    )
    foreach ($path in $fallbacks) {
        $mboExe = Join-Path $path "mbo.exe"
        if (Test-Path $mboExe) {
            return $path
        }
    }

    return $null
}

function New-DesktopShortcut {
    param([string]$BranchRef = $null)

    Write-Info "Creating desktop shortcut..."

    # get uv tool bin directory
    $binDir = Get-UvToolBinDir
    if (-not $binDir) {
        Write-Warn "Could not find uv tool bin directory. Skipping shortcut."
        return
    }

    $mboExePath = Join-Path $binDir "mbo.exe"
    if (-not (Test-Path $mboExePath)) {
        Write-Warn "mbo.exe not found at $mboExePath. Skipping shortcut."
        return
    }

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

    # create shortcut pointing to uv-managed mbo.exe
    try {
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $mboExePath
        $shortcut.WorkingDirectory = [Environment]::GetFolderPath("UserProfile")
        if ($iconPath -and (Test-Path $iconPath)) { $shortcut.IconLocation = $iconPath }
        $shortcut.Description = "MBO Image Viewer"
        $shortcut.Save()

        Write-Success "Desktop shortcut created: $shortcutName"
        Write-Info "  Target: $mboExePath"
    }
    catch {
        Write-Warn "Could not create desktop shortcut: $_"
        Write-Warn "You can manually create a shortcut to: $mboExePath"
    }
}

function Show-UsageInstructions {
    param(
        [string]$EnvPath = $null,
        [bool]$CliInstalled = $true
    )

    Write-Host ""
    Write-Host "Installation Complete" -ForegroundColor White
    Write-Host ""

    if ($CliInstalled) {
        $binDir = Get-UvToolBinDir
        Write-Host "  CLI Location:" -ForegroundColor Gray
        Write-Host "    $binDir\mbo.exe" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Usage:" -ForegroundColor Gray
        Write-Host "    mbo                    # Open GUI" -ForegroundColor White
        Write-Host "    mbo /path/to/data      # Open specific file" -ForegroundColor White
        Write-Host "    mbo --help             # Show all commands" -ForegroundColor White
        Write-Host ""
    }

    if ($EnvPath) {
        Write-Host "  Development Environment:" -ForegroundColor Gray
        Write-Host "    $EnvPath" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Activate:" -ForegroundColor Gray
        Write-Host "    $EnvPath\Scripts\activate" -ForegroundColor White
        Write-Host ""
        Write-Host "  Use in VSCode:" -ForegroundColor Gray
        Write-Host "    Ctrl+Shift+P -> 'Python: Select Interpreter'" -ForegroundColor White
        Write-Host "    Choose: $EnvPath\Scripts\python.exe" -ForegroundColor White
        Write-Host ""
        Write-Host "  Add packages:" -ForegroundColor Gray
        Write-Host "    uv pip install --python `"$EnvPath\Scripts\python.exe`" <package>" -ForegroundColor White
        Write-Host ""
    }
}

function Main {
    Show-Banner

    # step 0: check system dependencies first
    $sysDeps = Test-SystemDependencies
    $shouldContinue = Show-SystemDependencyCheck -Deps $sysDeps
    if (-not $shouldContinue) {
        exit 0
    }

    # check/install uv
    if (-not (Test-UvInstalled)) {
        Install-Uv
    }

    # step 1: choose installation type (env, CLI, or both)
    $installType = Show-InstallTypePrompt

    # step 2: choose source (pypi or github branch)
    $sourceInfo = Show-SourceSelection

    # step 3: detect GPU
    $gpuInfo = Test-NvidiaGpu

    # step 4: choose extras
    $extras = Show-OptionalDependencies -GpuInfo $gpuInfo

    # step 5: get environment location if needed
    $envPath = $null
    $envLocation = $null
    if ($installType.InstallEnv -and $env:MBO_SKIP_ENV -ne "1") {
        $envLocation = Show-EnvLocationPrompt
    }

    # step 6: install CLI tool if requested
    if ($installType.InstallCli) {
        $toolInstalled = Install-MboTool -Spec $sourceInfo.Spec -Extras $extras
        if (-not $toolInstalled) {
            Write-Err "CLI installation failed."
            exit 1
        }
    }

    # step 7: create dev environment if requested
    if ($installType.InstallEnv -and $envLocation) {
        $envPath = Install-DevEnvironment -EnvPath $envLocation -Spec $sourceInfo.Spec -Extras $extras -GpuInfo $gpuInfo
    }

    # step 8: create desktop shortcut (only if CLI was installed)
    if ($installType.InstallCli) {
        New-DesktopShortcut -BranchRef $sourceInfo.Ref
    }

    # show usage instructions
    Show-UsageInstructions -EnvPath $envPath -CliInstalled $installType.InstallCli

    Write-Success "Installation completed!"
    Write-Host ""
    if ($installType.InstallCli) {
        Write-Host "You may need to restart your terminal for PATH changes to take effect." -ForegroundColor Yellow
    }
}

Main
