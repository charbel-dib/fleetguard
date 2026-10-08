param([switch]$Research, [switch]$Serve)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (Get-Command deactivate -ErrorAction SilentlyContinue) {
    deactivate
}

# Keep package files and cache outside the OneDrive-synced repository.
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA "FleetGuard\venvs\py312-copy"
$env:UV_CACHE_DIR = Join-Path $env:LOCALAPPDATA "FleetGuard\uv-cache-copy"
$env:UV_LINK_MODE = "copy"
$env:MLFLOW_DISABLE_TELEMETRY = "true"
$env:MLFLOW_DISABLE_AGENT_HINT = "true"

$syncArguments = @("sync", "--python", "3.12", "--frozen", "--extra", "dev")
if ($Research) {
    $syncArguments += @("--extra", "research")
}

if ($Serve) {
    $syncArguments += @("--extra", "serve")
}

& uv @syncArguments
if ($LASTEXITCODE -ne 0) {
    throw "Environment installation failed. Do not run checks or training yet."
}

$fleetPython = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\python.exe"
& $fleetPython -c "import sys, fleetguard; print(sys.executable); print(sys.version); print('FleetGuard', fleetguard.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "FleetGuard import failed in the selected environment."
}

Write-Host "Environment ready. In this terminal use: uv run --no-sync python -m fleetguard ..."
