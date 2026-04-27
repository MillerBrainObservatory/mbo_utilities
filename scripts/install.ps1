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

# System dependency URLs (informational only)
$FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/"

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

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
    $ffmpeg = Test-Ffmpeg

    return @{
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

function Get-CupyPackage {
    <#
    .SYNOPSIS
    Returns the correct cupy package name for the detected CUDA toolkit version.
    Returns $null if no compatible CUDA toolkit is found.
    #>
    param([hashtable]$GpuInfo)

    if (-not $GpuInfo.ToolkitInstalled -or -not $GpuInfo.CudaVersion) {
        return $null
    }

    $cudaVersion = $GpuInfo.CudaVersion
    if ($cudaVersion -match '^(\d+)') {
        $major = [int]$matches[1]
        if ($major -ge 12) { return "cupy-cuda12x" }
        if ($major -eq 11) { return "cupy-cuda11x" }
    }

    return $null
}

function Get-PytorchIndexUrl {
    <#
    .SYNOPSIS
    Returns the pytorch wheel index URL matching the detected CUDA toolkit.
    Pytorch doesn't publish GPU wheels to PyPI — they live on its own
    index. Without `--index-url` pointing there, the main install pulls
    the CPU `torch` package and cellpose/suite2p run on CPU regardless
    of whether cuda + cupy are available. Returns $null if no suitable
    mapping (no GPU, no toolkit, or CUDA version outside the supported
    set).
    #>
    param([hashtable]$GpuInfo)

    if (-not $GpuInfo.ToolkitInstalled -or -not $GpuInfo.CudaVersion) {
        return $null
    }

    if ($GpuInfo.CudaVersion -match '^(\d+)\.(\d+)') {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        # pick the highest pytorch-published cu12X that's <= installed
        # toolkit. wheels are driver-backward-compatible within a major,
        # so a cu126 wheel works fine on a 12.8 toolkit machine.
        if ($major -eq 12) {
            if ($minor -ge 6)     { return "https://download.pytorch.org/whl/cu126" }
            elseif ($minor -ge 4) { return "https://download.pytorch.org/whl/cu124" }
            elseif ($minor -ge 1) { return "https://download.pytorch.org/whl/cu121" }
            else                  { return "https://download.pytorch.org/whl/cu118" }
        }
        if ($major -eq 11) { return "https://download.pytorch.org/whl/cu118" }
    }
    return $null
}

function Get-UvToolPythonPath {
    <#
    .SYNOPSIS
    Returns the path to the python interpreter inside a given uv tool's
    environment. Returns $null if the tool isn't installed or the path
    can't be located.
    #>
    param([string]$ToolName)

    try {
        $toolDir = uv tool dir 2>$null
        if ($toolDir) {
            $toolDir = $toolDir.Trim()
            # uv normalizes package names — try both underscore and hyphen forms
            foreach ($name in @($ToolName, $ToolName.Replace("_", "-"), $ToolName.Replace("-", "_"))) {
                $candidate = Join-Path $toolDir "$name\Scripts\python.exe"
                if (Test-Path $candidate) { return $candidate }
            }
        }
    }
    catch {}
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
            Write-Host "Not installed (GPU packages will use CPU)" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  No NVIDIA GPU detected (GPU features will be slower)" -ForegroundColor Yellow
    }
    Write-Host ""

    Write-Host "  [1] Suite2p   - 2D cell extraction (PyTorch + CUDA)" -ForegroundColor Cyan
    Write-Host "  [2] Rastermap - Dimensionality reduction" -ForegroundColor Cyan
    Write-Host "  [3] All       - Install all processing pipelines" -ForegroundColor Cyan
    Write-Host "  [4] None      - Base installation only (fastest)" -ForegroundColor Cyan
    Write-Host ""

    do {
        $choice = Read-Host "Select option (1-4, or comma-separated like 1,2)"
        $valid = $choice -match '^[1-4](,[1-4])*$'
        if (-not $valid) { Write-Warn "Invalid selection. Enter 1-4 or comma-separated." }
    } while (-not $valid)

    $extras = @()
    $choices = $choice -split ',' | ForEach-Object { $_.Trim() } | Select-Object -Unique

    :parseLoop foreach ($c in $choices) {
        switch ($c) {
            "1" { $extras += "suite2p" }
            "2" { $extras += "rastermap" }
            "3" { $extras = @("all"); break parseLoop }
            "4" { $extras = @(); break parseLoop }
        }
    }

    # warn if GPU packages selected without toolkit
    if ($extras.Count -gt 0) {
        $gpuPackages = @("suite2p", "all")
        $hasGpuPackage = ($extras | Where-Object { $gpuPackages -contains $_ }).Count -gt 0
        $needsCupy = $false  # cupy is now a pure-optional accelerator, not tied to an extra

        if ($hasGpuPackage -and $GpuInfo.Available -and -not $GpuInfo.ToolkitInstalled) {
            Write-Host ""
            Write-Warn "CUDA Toolkit not installed. GPU packages will use CPU (slower)."
            Write-Warn "Install CUDA Toolkit for GPU acceleration."
        }
        elseif ($hasGpuPackage -and -not $GpuInfo.Available) {
            Write-Host ""
            Write-Warn "No NVIDIA GPU detected. These packages will run in CPU-only mode."
            if ($needsCupy) {
                Write-Warn "Suite3D requires CuPy + NVIDIA GPU. CuPy will NOT be installed."
            }
            $continue = Read-Host "Continue anyway? (y/n)"
            if ($continue -ne "y") {
                $extras = $extras | Where-Object { $gpuPackages -notcontains $_ }
            }
        }
        elseif ($needsCupy -and $GpuInfo.ToolkitInstalled) {
            $cupyPkg = Get-CupyPackage -GpuInfo $GpuInfo
            if ($cupyPkg) {
                Write-Host ""
                Write-Info "CuPy will be installed for CUDA $($GpuInfo.CudaVersion): $cupyPkg"
            }
            else {
                Write-Host ""
                Write-Warn "Unsupported CUDA version $($GpuInfo.CudaVersion). CuPy will NOT be installed."
                Write-Warn "Suite3D requires CUDA 11.x or 12.x."
            }
        }
    }

    return $extras
}

function Install-MboTool {
    param(
        [string]$Spec,
        [string[]]$Extras = @(),
        [string]$CupyPackage = $null,
        [string]$PytorchIndexUrl = $null
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

    # add cupy + NVRTC helpers as --with when the caller asks for gpu
    # acceleration. the nvrtc/runtime wheels isolate cupy from the user's
    # system CUDA toolkit so a driver > toolkit version skew can't break
    # kernel compilation (fixes the "__nv_fp8_e8m0 incomplete type" class
    # of bug). mirrors the dev-env install so both paths behave the same.
    $withArgs = @()
    if ($CupyPackage) {
        $withArgs += @("--with", $CupyPackage,
                       "--with", "nvidia-cuda-nvrtc-cu12",
                       "--with", "nvidia-cuda-runtime-cu12")
        Write-Info "  CuPy: $CupyPackage + bundled NVRTC (via --with)"
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

        # install tool (with cupy if gpu acceleration is wanted)
        # --reinstall forces uv to re-fetch and rebuild even if the same
        # branch name is already cached. without it, pushing fixes to a
        # branch and re-running the script would silently keep the stale
        # version in the tool environment.
        $installArgs = @($fullSpec, "--python", "3.12", "--reinstall") + $withArgs
        uv tool install @installArgs 2>&1 | ForEach-Object { Write-Host $_ }

        if ($LASTEXITCODE -ne 0) {
            throw "uv tool install failed with exit code $LASTEXITCODE"
        }

        # ensure the tool bin dir is on User PATH. uv's own install-time
        # PATH wiring doesn't always fire (fresh windows machines, locked-
        # down execution policies, certain shell configs), which leaves
        # `mbo` unreachable from a new terminal even though the tool
        # itself installed fine. `uv tool update-shell` is idempotent —
        # safe to run even when PATH is already correct.
        $updateOut = uv tool update-shell 2>&1 | Out-String
        if ($updateOut.Trim()) { Write-Host $updateOut.Trim() }

        # make `mbo` reachable in THIS shell too, so the user doesn't
        # need to restart their terminal to try it. update-shell modifies
        # the User PATH in the registry, but the current session inherits
        # its PATH from when it was launched — we need to refresh it
        # explicitly. merge Machine PATH + updated User PATH.
        $binDir = Get-UvToolBinDir
        if ($binDir -and $env:Path -notlike "*$binDir*") {
            $env:Path = "$binDir;$env:Path"
        }

        # replace the CPU torch that came from PyPI with the GPU build
        # from pytorch's own index. uv tool install can't express
        # "use alt index for one package only", so we do this as a
        # post-install reinstall into the tool's own venv.
        if ($PytorchIndexUrl) {
            $toolPy = Get-UvToolPythonPath -ToolName "mbo_utilities"
            if ($toolPy) {
                Write-Info "Replacing CPU torch with CUDA build ($PytorchIndexUrl)..."
                uv pip install --python $toolPy --reinstall torch torchvision --index-url $PytorchIndexUrl 2>&1 | ForEach-Object { Write-Host $_ }
                if ($LASTEXITCODE -ne 0) {
                    Write-Warn "GPU torch install failed. Tool will use CPU torch."
                }
                else {
                    Write-Success "GPU torch installed in tool env"
                }
            }
            else {
                Write-Warn "Could not locate tool's python; GPU torch not installed."
            }
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
    Write-Host "  CLI             - global 'mbo' command on your PATH, just runs the GUI." -ForegroundColor DarkGray
    Write-Host "                    Self-contained; no activation, no imports, no notebooks." -ForegroundColor DarkGray
    Write-Host "  Local env       - project-local venv you 'cd' into and run 'uv run ...' or" -ForegroundColor DarkGray
    Write-Host "                    import from. Use this for scripts, notebooks, development." -ForegroundColor DarkGray
    Write-Host "  Both            - pick this if you want the GUI anywhere AND a local env to code in." -ForegroundColor DarkGray
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

function Get-MboVersionFromEnv {
    <#
    .SYNOPSIS
    Query the installed mbo_utilities version from a virtual environment.
    Returns hashtable with version, source (pypi/git), and commit if applicable.
    #>
    param([string]$PythonPath)

    if (-not (Test-Path $PythonPath)) {
        return $null
    }

    try {
        # Get version and install location
        $versionOutput = & $PythonPath -c "
import sys
try:
    from importlib.metadata import version, distribution
    v = version('mbo_utilities')
    d = distribution('mbo_utilities')
    # Check if installed from git
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
" 2>$null

        if ($versionOutput -and $versionOutput -notmatch '^error') {
            $parts = $versionOutput.Trim() -split '\|'
            return @{
                Version = $parts[0]
                Source = $parts[1]
                Branch = $parts[2]
                Commit = $parts[3]
            }
        }
    }
    catch {}

    return $null
}

function Install-DevEnvironment {
    param(
        [string]$EnvPath,
        [string]$Spec,
        [string[]]$Extras = @(),
        [hashtable]$GpuInfo = @{},
        [string]$CupyPackage = $null,
        [string]$PytorchIndexUrl = $null
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
            # Check if .venv already exists inside project
            $venvPath = Join-Path $EnvPath ".venv"
            $venvPythonPath = Join-Path $venvPath "Scripts\python.exe"
            $venvExists = Test-Path $venvPythonPath

            Write-Host ""
            Write-Warn "This looks like a project directory: $EnvPath"

            if ($venvExists) {
                # Query existing installation
                $installedInfo = Get-MboVersionFromEnv -PythonPath $venvPythonPath

                Write-Host ""
                Write-Host "  Existing .venv found:" -ForegroundColor White
                if ($installedInfo) {
                    Write-Host "    mbo_utilities: " -NoNewline -ForegroundColor Gray
                    Write-Host "v$($installedInfo.Version)" -NoNewline -ForegroundColor Cyan
                    if ($installedInfo.Source -eq "git") {
                        Write-Host " (git: $($installedInfo.Branch)@$($installedInfo.Commit))" -ForegroundColor Gray
                    } else {
                        Write-Host " (PyPI)" -ForegroundColor Gray
                    }
                } else {
                    Write-Host "    mbo_utilities: " -NoNewline -ForegroundColor Gray
                    Write-Host "not installed or error reading" -ForegroundColor Yellow
                }
                Write-Host "    Python: $venvPythonPath" -ForegroundColor Gray
                Write-Host ""
                Write-Host "  [1] Overwrite - Delete .venv and recreate" -ForegroundColor Cyan
                Write-Host "  [2] Update    - Install/upgrade mbo_utilities in existing .venv" -ForegroundColor Cyan
                Write-Host "  [3] Skip      - Don't modify dev environment" -ForegroundColor Cyan
                Write-Host ""

                do {
                    $choice = Read-Host "Select option (1-3)"
                    $valid = $choice -match '^[123]$'
                    if (-not $valid) { Write-Warn "Invalid selection." }
                } while (-not $valid)

                switch ($choice) {
                    "1" {
                        Write-Info "Removing existing .venv..."
                        Remove-Item -Recurse -Force $venvPath
                        $EnvPath = $venvPath
                        $pythonPath = $venvPythonPath
                        $envExists = $false
                    }
                    "2" {
                        Write-Info "Updating existing .venv..."
                        $EnvPath = $venvPath
                        $pythonPath = $venvPythonPath
                        $envExists = $true
                    }
                    "3" {
                        Write-Info "Skipping dev environment."
                        return $null
                    }
                }
            } else {
                # No .venv exists, offer to create one
                Write-Host ""
                Write-Host "  [1] Create .venv - Create environment at $venvPath (Recommended)" -ForegroundColor Cyan
                Write-Host "  [2] Skip         - Don't create dev environment" -ForegroundColor Cyan
                Write-Host ""

                do {
                    $choice = Read-Host "Select option (1-2)"
                    $valid = $choice -match '^[12]$'
                    if (-not $valid) { Write-Warn "Invalid selection." }
                } while (-not $valid)

                switch ($choice) {
                    "1" {
                        $EnvPath = $venvPath
                        $pythonPath = $venvPythonPath
                        $envExists = $false
                    }
                    "2" {
                        Write-Info "Skipping dev environment."
                        return $null
                    }
                }
            }
        }
        else {
            # Not a project directory, but directory exists without venv
            $venvPath = Join-Path $EnvPath ".venv"
            Write-Host ""
            Write-Warn "Directory exists but is not a virtual environment: $EnvPath"
            Write-Host ""
            Write-Host "  [1] Create .venv inside - Create environment at $venvPath" -ForegroundColor Cyan
            Write-Host "  [2] Skip                - Don't create dev environment" -ForegroundColor Cyan
            Write-Host ""

            do {
                $choice = Read-Host "Select option (1-2)"
                $valid = $choice -match '^[12]$'
                if (-not $valid) { Write-Warn "Invalid selection." }
            } while (-not $valid)

            switch ($choice) {
                "1" {
                    $EnvPath = $venvPath
                    $pythonPath = Join-Path $EnvPath "Scripts\python.exe"
                    $envExists = $false
                }
                "2" {
                    Write-Info "Skipping dev environment."
                    return $null
                }
            }
        }
    }
    elseif ($envExists) {
        # EnvPath itself is a valid venv
        if ($env:MBO_OVERWRITE -eq "1") {
            Write-Warn "Removing existing environment..."
            Remove-Item -Recurse -Force $EnvPath
            $envExists = $false
        }
        else {
            # Query existing installation
            $installedInfo = Get-MboVersionFromEnv -PythonPath $pythonPath

            Write-Host ""
            Write-Warn "Environment already exists at: $EnvPath"
            Write-Host ""
            if ($installedInfo) {
                Write-Host "  Installed: " -NoNewline -ForegroundColor Gray
                Write-Host "mbo_utilities v$($installedInfo.Version)" -NoNewline -ForegroundColor Cyan
                if ($installedInfo.Source -eq "git") {
                    Write-Host " (git: $($installedInfo.Branch)@$($installedInfo.Commit))" -ForegroundColor Gray
                } else {
                    Write-Host " (PyPI)" -ForegroundColor Gray
                }
                Write-Host ""
            }
            Write-Host "  [1] Overwrite - Delete and recreate" -ForegroundColor Cyan
            Write-Host "  [2] Update    - Install into existing" -ForegroundColor Cyan
            Write-Host "  [3] Skip      - Don't modify dev environment" -ForegroundColor Cyan
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
        # --reinstall (not --reinstall-package) rewrites files for ALL
        # packages in the resolution, not just mbo-utilities. this
        # clobbers any stale transitive-dep files left over from prior
        # installs — the mbo-fpl → mbo-fastplotlib rename pattern where
        # two packages owned overlapping module directories, or the
        # napari/app-model/vispy version-skew where the lockfile version
        # differs from what's already on disk. for a "refresh" flow we
        # want every file rewritten to match the current resolution.
        uv pip install --python $pythonPath --reinstall $fullSpec 2>&1 | ForEach-Object { Write-Host $_ }

        if ($LASTEXITCODE -ne 0) {
            throw "uv pip install failed"
        }

        # install cupy separately when gpu acceleration is wanted. --reinstall
        # here too so an existing cupy-cuda12x install doesn't mask the version
        # we want. also pull in the NVRTC/runtime wheels so cupy uses
        # pip-managed CUDA headers instead of whatever system toolkit is on PATH
        # — fixes the "cuda_fp8.h missing __nv_fp8_e8m0" class of bug on
        # machines where driver > toolkit.
        if ($CupyPackage) {
            Write-Info "Installing $CupyPackage + bundled NVRTC (enables GPU axial registration)..."
            uv pip install --python $pythonPath --reinstall $CupyPackage nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 2>&1 | ForEach-Object { Write-Host $_ }
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "CuPy installation failed. Axial registration will fall back to CPU."
            }
            else {
                Write-Success "CuPy + NVRTC installed: $CupyPackage"
            }
        }

        # replace the CPU torch that came from PyPI with the GPU build
        # from pytorch's own index. pytorch only publishes GPU wheels to
        # its dedicated index, so without this step cellpose / suite2p
        # run on CPU despite cuda + cupy being present.
        if ($PytorchIndexUrl) {
            Write-Info "Replacing CPU torch with CUDA build ($PytorchIndexUrl)..."
            uv pip install --python $pythonPath --reinstall torch torchvision --index-url $PytorchIndexUrl 2>&1 | ForEach-Object { Write-Host $_ }
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "GPU torch install failed. Environment will use CPU torch."
            }
            else {
                Write-Success "GPU torch installed from $PytorchIndexUrl"
            }
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
    # `uv tool dir --bin` is the canonical query. `uv tool bin-dir`
    # was a typo — not a real subcommand, always fell through to the
    # fallback below (which works on Windows default but masked the bug).
    try {
        $binDir = uv tool dir --bin 2>$null
        if ($binDir -and $LASTEXITCODE -eq 0) {
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
    $shortcutName = if ($BranchRef -and $BranchRef -ne "master") { "Miller Brain Studio ($BranchRef).lnk" } else { "Miller Brain Studio.lnk" }
    $shortcutPath = Join-Path $desktopPath $shortcutName

    # setup mbo directory for launcher and icon
    $mboDir = Join-Path $env:USERPROFILE "mbo"
    if (-not (Test-Path $mboDir)) { New-Item -ItemType Directory -Path $mboDir -Force | Out-Null }

    $downloadRef = if ($BranchRef) { $BranchRef } else { "master" }

    # download icon (same as taskbar icon)
    $iconPath = Join-Path $mboDir "icon.ico"
    try {
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/$GITHUB_REPO/$downloadRef/mbo_utilities/assets/app_settings/icon.ico" -OutFile $iconPath -ErrorAction Stop
    }
    catch {
        try { Invoke-WebRequest -Uri "https://raw.githubusercontent.com/$GITHUB_REPO/master/mbo_utilities/assets/app_settings/icon.ico" -OutFile $iconPath -ErrorAction Stop }
        catch { $iconPath = $null }
    }

    # create VBScript launcher to hide terminal window
    # The "0" argument to Run means hidden window, "False" means don't wait
    $launcherPath = Join-Path $mboDir "mbo_launcher.vbs"
    $vbsContent = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """$mboExePath""", 0, False
"@
    try {
        Set-Content -Path $launcherPath -Value $vbsContent -Encoding ASCII
    }
    catch {
        Write-Warn "Could not create launcher script: $_"
        # Fall back to direct shortcut
        $launcherPath = $null
    }

    # create shortcut pointing to VBScript launcher (no terminal window)
    try {
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($shortcutPath)

        if ($launcherPath -and (Test-Path $launcherPath)) {
            # Use wscript.exe to run the VBS launcher (fully hidden)
            $shortcut.TargetPath = "wscript.exe"
            $shortcut.Arguments = """$launcherPath"""
        }
        else {
            # Fallback: direct to mbo.exe (will show terminal)
            $shortcut.TargetPath = $mboExePath
        }

        $shortcut.WorkingDirectory = [Environment]::GetFolderPath("UserProfile")
        if ($iconPath -and (Test-Path $iconPath)) { $shortcut.IconLocation = $iconPath }
        $shortcut.Description = "MBO Image Viewer"
        $shortcut.Save()

        Write-Success "Desktop shortcut created: $shortcutName"
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

    # two distinct usage modes depending on what was installed. the CLI
    # tool is available from any directory; the environment is a
    # project-local venv that needs a `cd` (or activation) first. label
    # them clearly so users running both don't conflate the two.
    $sectionNum = 0
    $showBoth = $CliInstalled -and $EnvPath

    if ($CliInstalled) {
        $sectionNum++
        $binDir = Get-UvToolBinDir
        $header = if ($showBoth) { "(${sectionNum}) CLI - available system-wide" } else { "CLI - available system-wide" }
        Write-Host "  $header" -ForegroundColor Gray
        Write-Host "    mbo                    # open GUI" -ForegroundColor White
        Write-Host "    mbo /path/to/data      # open specific file" -ForegroundColor White
        Write-Host "    mbo --help             # show all commands" -ForegroundColor White
        Write-Host "    Location: $binDir\mbo.exe" -ForegroundColor DarkGray
        Write-Host ""
    }

    if ($EnvPath) {
        $sectionNum++
        # $EnvPath points at the actual venv dir (ends in \.venv when the
        # user pointed at a project root). `uv run` wants the project dir,
        # not the venv dir — strip the trailing .venv so `cd` lands on the
        # project, and all the other `uv ...` commands below Just Work.
        $cdPath = $EnvPath
        if ($cdPath -match '[\\/]\.venv[\\/]?$') {
            $cdPath = Split-Path $cdPath -Parent
        }

        $header = if ($showBoth) {
            "(${sectionNum}) Local environment - use from the env directory"
        } else {
            "Local environment - use from the env directory"
        }
        Write-Host "  $header" -ForegroundColor Gray
        Write-Host "    cd $cdPath" -ForegroundColor White
        Write-Host "    uv run mbo             # open GUI (uses this env)" -ForegroundColor White
        Write-Host "    uv run mbo --help      # show all commands" -ForegroundColor White
        Write-Host "    uv run python          # interactive session with this env" -ForegroundColor White
        Write-Host "    uv pip install <pkg>   # add a package to this env" -ForegroundColor White
        Write-Host "    uv pip list            # show installed packages + versions" -ForegroundColor White
        Write-Host ""
        Write-Host "  Use in VSCode:" -ForegroundColor Gray
        Write-Host "    Ctrl+Shift+P -> 'Python: Select Interpreter'" -ForegroundColor White
        Write-Host "    Choose: $EnvPath\Scripts\python.exe" -ForegroundColor White
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

    # step 4.5: cupy is no longer tied to a specific extra — it's a pure
    # optional accelerator for axial registration. user can install it
    # manually if they want GPU; the installer doesn't pull it in.
    $needsCupy = $false
    $cupyPackage = $null
    if ($needsCupy) {
        $cupyPackage = Get-CupyPackage -GpuInfo $gpuInfo
    }

    # step 4.6: determine pytorch wheel index based on CUDA version.
    # torch is pulled in by suite2p/all, and PyPI only ships the CPU
    # build — GPU wheels live on pytorch's own index. fire this whenever
    # an extras value brings torch along AND we've got a CUDA toolkit.
    $needsTorch = ($extras | Where-Object { @("suite2p", "all", "processing") -contains $_ }).Count -gt 0
    $pytorchIndexUrl = $null
    if ($needsTorch) {
        $pytorchIndexUrl = Get-PytorchIndexUrl -GpuInfo $gpuInfo
        if ($pytorchIndexUrl) {
            Write-Info "GPU torch will be installed from $pytorchIndexUrl"
        }
    }

    # step 5: get environment location if needed
    $envPath = $null
    $envLocation = $null
    if ($installType.InstallEnv -and $env:MBO_SKIP_ENV -ne "1") {
        $envLocation = Show-EnvLocationPrompt
    }

    # step 6: install CLI tool if requested
    if ($installType.InstallCli) {
        $toolInstalled = Install-MboTool -Spec $sourceInfo.Spec -Extras $extras -CupyPackage $cupyPackage -PytorchIndexUrl $pytorchIndexUrl
        if (-not $toolInstalled) {
            Write-Err "CLI installation failed."
            exit 1
        }
    }

    # step 7: create dev environment if requested
    if ($installType.InstallEnv -and $envLocation) {
        $envPath = Install-DevEnvironment -EnvPath $envLocation -Spec $sourceInfo.Spec -Extras $extras -GpuInfo $gpuInfo -CupyPackage $cupyPackage -PytorchIndexUrl $pytorchIndexUrl
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
