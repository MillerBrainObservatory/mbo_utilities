# MBO Utilities Installation Script for Windows
# Installs the global 'mbo' CLI (uv tool) and/or a local dev environment,
# with an optional desktop shortcut for each.
#
# usage:
#   irm https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/scripts/install.ps1 | iex
#
# environment variables:
#   MBO_ENV_PATH     - custom path for dev environment (default: ~/<repos|code|software|projects>/mbo_utilities)
#   MBO_SKIP_ENV     - set to "1" to skip dev environment creation
#   MBO_OVERWRITE    - set to "1" to overwrite existing installations
#   MBO_ASSUME_YES   - set to "1" to accept default answers (non-interactive)
#   MBO_PYTHON       - Python version for the install (default: 3.12)

$ErrorActionPreference = "Stop"

$GITHUB_REPO = "MillerBrainObservatory/mbo_utilities"

# accept default answers without prompting (CI / unattended re-provisioning)
$ASSUME_YES = ($env:MBO_ASSUME_YES -eq "1")
# python version used for `uv tool install` and `uv venv`
$MBO_PYTHON = if ($env:MBO_PYTHON) { $env:MBO_PYTHON } else { "3.12" }

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
    Display system dependency status (informational; no required deps on Windows).
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
        throw
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
    if ($ASSUME_YES) {
        Write-Info "MBO_ASSUME_YES: installing from PyPI (stable)."
        return @{ Source = "pypi"; Spec = "mbo_utilities"; Ref = $null }
    }

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
        catch { return @{ Available = $false; GpuName = $null; CudaVersion = $null; DriverCudaVersion = $null; ToolkitInstalled = $false } }
    }

    try {
        $gpuName = & $nvidiaSmi --query-gpu=name --format=csv,noheader 2>$null
        if ($gpuName) {
            # the nvidia-smi header reports the max CUDA version the
            # installed DRIVER supports. this is what pytorch GPU wheels
            # need — they bundle their own CUDA runtime, so no system
            # toolkit (nvcc) is required.
            $driverCuda = $null
            try {
                $smiText = & $nvidiaSmi 2>$null | Out-String
                if ($smiText -match 'CUDA Version:\s*(\d+\.\d+)') { $driverCuda = $matches[1] }
            }
            catch {}

            $toolkitVersion = Test-CudaToolkit
            return @{
                Available = $true
                GpuName = $gpuName.Trim()
                CudaVersion = $toolkitVersion
                DriverCudaVersion = $driverCuda
                ToolkitInstalled = ($null -ne $toolkitVersion)
            }
        }
    }
    catch {}
    return @{ Available = $false; GpuName = $null; CudaVersion = $null; DriverCudaVersion = $null; ToolkitInstalled = $false }
}

function Get-CupyPackages {
    <#
    .SYNOPSIS
    Returns the cupy wheel + matching NVRTC/runtime wheels for the detected
    GPU/driver, as an array ready to install. CuPy is the optional GPU
    backend for axial registration; the NVRTC + runtime wheels supply
    pip-managed CUDA so cupy's JIT kernels compile without a system CUDA
    toolkit. Returns an empty array when there is no NVIDIA GPU. Mirrors
    mbo_utilities.install.recommended_cupy_package.
    #>
    param([hashtable]$GpuInfo)

    if (-not $GpuInfo.Available) {
        return @()
    }

    $cudaVersion = $GpuInfo.DriverCudaVersion
    if (-not $cudaVersion) { $cudaVersion = $GpuInfo.CudaVersion }

    # cupy ships one wheel per CUDA major (cupy-cuda11x/12x/13x). default to
    # 12x (broadest coverage) when the version can't be read.
    $major = "12"
    if ($cudaVersion -match '^(\d+)') {
        $m = [int]$matches[1]
        if ($m -ge 13)     { $major = "13" }
        elseif ($m -ge 12) { $major = "12" }
        else               { $major = "11" }
    }

    return @("cupy-cuda${major}x",
             "nvidia-cuda-nvrtc-cu${major}",
             "nvidia-cuda-runtime-cu${major}")
}

function Get-PytorchIndexUrl {
    <#
    .SYNOPSIS
    Returns the pytorch wheel index URL matching the GPU/driver.

    PyTorch doesn't publish GPU wheels to PyPI — they live on its own
    index. Without `--index-url` pointing there, the install pulls the
    CPU `torch` and suite2p/cellpose run on CPU. PyTorch CUDA wheels
    bundle their own CUDA runtime, so only an NVIDIA DRIVER is required,
    NOT a system CUDA toolkit (nvcc). Pick the wheel from the driver's
    max supported CUDA version (nvidia-smi), falling back to the toolkit
    version. Returns $null only when there is no NVIDIA GPU.
    #>
    param([hashtable]$GpuInfo)

    if (-not $GpuInfo.Available) {
        return $null
    }

    $cudaVersion = $GpuInfo.DriverCudaVersion
    if (-not $cudaVersion) { $cudaVersion = $GpuInfo.CudaVersion }

    # GPU present but CUDA version undetectable — use a broadly compatible
    # default rather than silently falling back to the CPU build.
    if (-not $cudaVersion) {
        return "https://download.pytorch.org/whl/cu121"
    }

    if ($cudaVersion -match '^(\d+)\.(\d+)') {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        # pick the highest pytorch-published cuXXX that's <= the driver's
        # supported CUDA. wheels are backward-compatible, so a cu126 wheel
        # runs fine on a 12.8-capable driver.
        if ($major -ge 13) { return "https://download.pytorch.org/whl/cu128" }
        if ($major -eq 12) {
            if ($minor -ge 8)     { return "https://download.pytorch.org/whl/cu128" }
            elseif ($minor -ge 6) { return "https://download.pytorch.org/whl/cu126" }
            elseif ($minor -ge 4) { return "https://download.pytorch.org/whl/cu124" }
            elseif ($minor -ge 1) { return "https://download.pytorch.org/whl/cu121" }
            else                  { return "https://download.pytorch.org/whl/cu118" }
        }
        if ($major -eq 11) { return "https://download.pytorch.org/whl/cu118" }
    }
    return "https://download.pytorch.org/whl/cu121"
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

    if ($ASSUME_YES) {
        Write-Info "MBO_ASSUME_YES: no optional extras (base install)."
        $extras = @()
        return $extras
    }

    Write-Host ""
    Write-Host "Optional Dependencies" -ForegroundColor White
    Write-Host ""

    if ($GpuInfo.Available) {
        Write-Host "  GPU detected: " -NoNewline -ForegroundColor Green
        Write-Host $GpuInfo.GpuName -ForegroundColor White
        if ($GpuInfo.DriverCudaVersion) {
            Write-Host "  Driver CUDA:  " -NoNewline -ForegroundColor Green
            Write-Host "$($GpuInfo.DriverCudaVersion) (CUDA torch/CuPy used with the Suite2p extra)" -ForegroundColor White
        }
        else {
            Write-Host "  Driver CUDA:  " -NoNewline -ForegroundColor Yellow
            Write-Host "unknown (default CUDA build used with the Suite2p extra)" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  No NVIDIA GPU detected" -ForegroundColor Yellow
    }
    Write-Host ""

    Write-Host "  The viewer, I/O, and metadata tools are always installed." -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  [1] Suite2p - suite2p/cellpose pipeline + rastermap + z-registration (pytorch)" -ForegroundColor Cyan
    Write-Host "  [2] Napari  - alternate viewer" -ForegroundColor Cyan
    Write-Host "  [3] Isoview - light-sheet pipeline" -ForegroundColor Cyan
    Write-Host "  [4] All     - everything (Recommended)" -ForegroundColor Cyan
    Write-Host "  [5] None    - base viewer only" -ForegroundColor Cyan
    Write-Host ""

    do {
        $choice = Read-Host "Select option (1-5, or comma-separated like 1,2)"
        $valid = $choice -match '^[1-5](,[1-5])*$'
        if (-not $valid) { Write-Warn "Invalid selection. Enter 1-5 or comma-separated." }
    } while (-not $valid)

    $extras = @()
    $choices = $choice -split ',' | ForEach-Object { $_.Trim() } | Select-Object -Unique

    :parseLoop foreach ($c in $choices) {
        switch ($c) {
            "1" { $extras += "suite2p" }
            "2" { $extras += "napari" }
            "3" { $extras += "isoview" }
            "4" { $extras = @("all"); break parseLoop }
            "5" { $extras = @(); break parseLoop }
        }
    }

    return $extras
}

function Install-MboTool {
    param(
        [string]$Spec,
        [string[]]$Extras = @(),
        [string[]]$CupyPackages = @(),
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

                if ($ASSUME_YES) {
                    $choice = "2"
                    Write-Info "MBO_ASSUME_YES: keeping existing installation."
                }
                else {
                    do {
                        $choice = Read-Host "Select option (1-3)"
                        $valid = $choice -match '^[123]$'
                        if (-not $valid) { Write-Warn "Invalid selection." }
                    } while (-not $valid)
                }

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
                        $script:Cancelled = $true
                        return $false
                    }
                }
            }
        }

        # install tool.
        # --reinstall forces uv to re-fetch and rebuild even if the same
        # branch name is already cached. without it, pushing fixes to a
        # branch and re-running the script would silently keep the stale
        # version in the tool environment.
        $installArgs = @($fullSpec, "--python", $MBO_PYTHON, "--reinstall")
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

        # install cupy + its NVRTC/runtime wheels into the tool's venv: the
        # optional GPU backend for axial registration. done as a post-install
        # step (not --with) so a cupy resolution hiccup only warns instead of
        # aborting the whole tool install.
        if ($CupyPackages.Count -gt 0) {
            $toolPy = Get-UvToolPythonPath -ToolName "mbo_utilities"
            if ($toolPy) {
                Write-Info "Installing CuPy for GPU axial registration: $($CupyPackages -join ', ')..."
                uv pip install --python $toolPy --reinstall @CupyPackages 2>&1 | ForEach-Object { Write-Host $_ }
                if ($LASTEXITCODE -ne 0) {
                    Write-Warn "CuPy install failed. Axial registration will use CPU."
                }
                else {
                    Write-Success "CuPy installed in tool env: $($CupyPackages[0])"
                }
            }
            else {
                Write-Warn "Could not locate tool's python; CuPy not installed."
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
    if ($ASSUME_YES) {
        Write-Info "MBO_ASSUME_YES: installing both global CLI + local environment."
        return @{ InstallCli = $true; InstallEnv = $true }
    }

    Write-Host ""
    Write-Host "Installation Type" -ForegroundColor White
    Write-Host ""
    Write-Host "  Global - one 'mbo' command that works in any terminal." -ForegroundColor DarkGray
    Write-Host "           Opens the GUI viewer. Nothing to activate. Choose this to just use the app." -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Local  - a Python venv you install into and work from." -ForegroundColor DarkGray
    Write-Host "           For writing scripts, running notebooks, 'import mbo_utilities', or development." -ForegroundColor DarkGray
    Write-Host "           You 'cd' into its folder and run things with 'uv run ...'." -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Both   - the 'mbo' command everywhere, plus an environment to write code in." -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  [1] Both   - global mbo command + local Python environment (Recommended)" -ForegroundColor Cyan
    Write-Host "  [2] Local  - Python environment only (scripts, notebooks, imports)" -ForegroundColor Cyan
    Write-Host "  [3] Global - mbo command only (just the GUI)" -ForegroundColor Cyan
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

function Get-DefaultEnvPath {
    # put the env next to the user's code: first existing of these common
    # parent dirs, else suggest ~/projects.
    foreach ($base in @("repos", "code", "software", "projects")) {
        $candidate = Join-Path $env:USERPROFILE $base
        if (Test-Path $candidate) {
            return Join-Path $candidate "mbo_utilities"
        }
    }
    return Join-Path $env:USERPROFILE "projects\mbo_utilities"
}

function Show-EnvLocationPrompt {
    if ($env:MBO_ENV_PATH) {
        Write-Info "Environment location: $($env:MBO_ENV_PATH) (from MBO_ENV_PATH)"
        return $env:MBO_ENV_PATH
    }

    $defaultPath = Get-DefaultEnvPath

    if ($ASSUME_YES) {
        Write-Info "MBO_ASSUME_YES: environment location $defaultPath"
        return $defaultPath
    }

    Write-Host ""
    Write-Host "Environment Location" -ForegroundColor White
    Write-Host ""
    Write-Host "  Default: " -NoNewline -ForegroundColor Gray
    Write-Host $defaultPath -ForegroundColor Cyan
    Write-Host ""
    $userInput = Read-Host "Press Enter for default, or enter custom path"

    if ([string]::IsNullOrWhiteSpace($userInput)) {
        return $defaultPath
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
        [string[]]$CupyPackages = @(),
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

                if ($ASSUME_YES) {
                    $choice = "2"
                    Write-Info "MBO_ASSUME_YES: updating existing .venv."
                }
                else {
                    do {
                        $choice = Read-Host "Select option (1-3)"
                        $valid = $choice -match '^[123]$'
                        if (-not $valid) { Write-Warn "Invalid selection." }
                    } while (-not $valid)
                }

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

                if ($ASSUME_YES) {
                    $choice = "1"
                    Write-Info "MBO_ASSUME_YES: creating .venv."
                }
                else {
                    do {
                        $choice = Read-Host "Select option (1-2)"
                        $valid = $choice -match '^[12]$'
                        if (-not $valid) { Write-Warn "Invalid selection." }
                    } while (-not $valid)
                }

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

            if ($ASSUME_YES) {
                $choice = "1"
                Write-Info "MBO_ASSUME_YES: creating .venv inside."
            }
            else {
                do {
                    $choice = Read-Host "Select option (1-2)"
                    $valid = $choice -match '^[12]$'
                    if (-not $valid) { Write-Warn "Invalid selection." }
                } while (-not $valid)
            }

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

            if ($ASSUME_YES) {
                $choice = "2"
                Write-Info "MBO_ASSUME_YES: updating existing environment."
            }
            else {
                do {
                    $choice = Read-Host "Select option (1-3)"
                    $valid = $choice -match '^[123]$'
                    if (-not $valid) { Write-Warn "Invalid selection." }
                } while (-not $valid)
            }

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

        Write-Info "Creating virtual environment with Python $MBO_PYTHON..."
        uv venv $EnvPath --python $MBO_PYTHON
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

        # install cupy + its NVRTC/runtime wheels: the optional GPU backend
        # for axial registration. --reinstall so an existing cupy build
        # doesn't mask the wanted one; the nvrtc/runtime wheels give cupy
        # pip-managed CUDA so its jit kernels compile without a system toolkit.
        if ($CupyPackages.Count -gt 0) {
            Write-Info "Installing CuPy for GPU axial registration: $($CupyPackages -join ', ')..."
            uv pip install --python $pythonPath --reinstall @CupyPackages 2>&1 | ForEach-Object { Write-Host $_ }
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "CuPy installation failed. Axial registration will use CPU."
            }
            else {
                Write-Success "CuPy installed: $($CupyPackages[0])"
            }
        }

        # replace the CPU torch that came from PyPI with the GPU build
        # from pytorch's own index. pytorch only publishes GPU wheels to
        # its dedicated index, so without this step cellpose / suite2p
        # run on CPU.
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

function Read-YesNo {
    param([string]$Prompt, [bool]$DefaultYes = $true)

    if ($ASSUME_YES) { return $DefaultYes }

    $suffix = if ($DefaultYes) { "[Y/n]" } else { "[y/N]" }
    while ($true) {
        $ans = Read-Host "$Prompt $suffix"
        if ([string]::IsNullOrWhiteSpace($ans)) { return $DefaultYes }
        switch ($ans.Trim().ToLower()) {
            "y"   { return $true }
            "yes" { return $true }
            "n"   { return $false }
            "no"  { return $false }
            default { Write-Warn "Please answer y or n." }
        }
    }
}

function Add-MboShortcut {
    param([string]$MboExe, [string]$Name)

    if (-not (Test-Path $MboExe)) {
        # fall back to PATH lookup (e.g. the bin dir wasn't resolvable)
        $resolved = Get-Command $MboExe -ErrorAction SilentlyContinue
        if ($resolved) {
            $MboExe = $resolved.Source
        }
        else {
            Write-Warn "mbo not found ($MboExe); skipping shortcut."
            return
        }
    }
    Write-Info "Creating desktop shortcut..."
    # `mbo shortcut` creates the .lnk using the bundled icon and prints the path
    & $MboExe shortcut --name $Name
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Could not create desktop shortcut (requires a newer mbo_utilities)."
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
        $header = if ($showBoth) { "(${sectionNum}) Global - available system-wide" } else { "Global - available system-wide" }
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
    $script:Cancelled = $false
    Show-Banner

    # step 0: show system dependencies (informational)
    $sysDeps = Test-SystemDependencies
    Show-SystemDependencyCheck -Deps $sysDeps

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

    # step 4.5: GPU packages only matter when the suite2p extra is selected
    # (it pulls torch for suite2p and uses cupy for axial registration). a
    # base/viewer install stays slim — no torch, no cupy. wheels bundle their
    # own CUDA runtime, so only an NVIDIA driver is required.
    $wantsSuite2p = ($extras -contains "suite2p") -or ($extras -contains "all")
    $cupyPackages = if ($wantsSuite2p) { Get-CupyPackages -GpuInfo $gpuInfo } else { @() }
    if ($cupyPackages.Count -gt 0) {
        Write-Info "GPU CuPy will be installed: $($cupyPackages[0])"
    }

    $pytorchIndexUrl = if ($wantsSuite2p) { Get-PytorchIndexUrl -GpuInfo $gpuInfo } else { $null }
    if ($pytorchIndexUrl) {
        Write-Info "GPU torch will be installed from $pytorchIndexUrl"
    }

    # step 5: get environment location if needed
    $envPath = $null
    $envLocation = $null
    if ($installType.InstallEnv -and $env:MBO_SKIP_ENV -ne "1") {
        $envLocation = Show-EnvLocationPrompt
    }

    # step 6: install global CLI tool if requested
    if ($installType.InstallCli) {
        $toolInstalled = Install-MboTool -Spec $sourceInfo.Spec -Extras $extras -CupyPackages $cupyPackages -PytorchIndexUrl $pytorchIndexUrl
        if ($script:Cancelled) {
            return
        }
        if (-not $toolInstalled) {
            Write-Err "Global installation failed."
            return
        }
        if (Read-YesNo "Add a desktop shortcut for the global app?" $true) {
            $binDir = Get-UvToolBinDir
            $globalMbo = if ($binDir) { Join-Path $binDir "mbo.exe" } else { "mbo" }
            Add-MboShortcut -MboExe $globalMbo -Name "Miller Brain Studio"
        }
    }

    # step 7: create local environment if requested
    if ($installType.InstallEnv -and $envLocation) {
        $envPath = Install-DevEnvironment -EnvPath $envLocation -Spec $sourceInfo.Spec -Extras $extras -GpuInfo $gpuInfo -CupyPackages $cupyPackages -PytorchIndexUrl $pytorchIndexUrl
        if ($envPath) {
            if (Read-YesNo "Add a desktop shortcut for the local environment?" $false) {
                $localMbo = Join-Path $envPath "Scripts\mbo.exe"
                Add-MboShortcut -MboExe $localMbo -Name "Miller Brain Studio (local)"
            }
        }
    }

    # show usage instructions
    Show-UsageInstructions -EnvPath $envPath -CliInstalled $installType.InstallCli

    Write-Success "Installation completed!"
    Write-Host ""
    if ($installType.InstallCli) {
        Write-Host "Open a NEW terminal to use 'mbo' (this session's PATH is already updated)." -ForegroundColor Yellow
    }
}

Main
