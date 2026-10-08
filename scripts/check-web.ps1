param([switch]$Browser)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Install Node.js 24 LTS before checking the frontend."
}
# Configure the parent terminal with use-environment.ps1 -Serve before this script.
& uv run --no-sync python -c "import fleetguard, fastapi; print('FleetGuard', fleetguard.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Configure the Python serving environment first." }

Push-Location (Join-Path $projectRoot "frontend")
try {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
    foreach ($step in @("check", "test", "build")) {
        & npm.cmd run $step
        if ($LASTEXITCODE -ne 0) { throw "Frontend check failed: $step." }
    }
    if ($Browser) {
        & npx.cmd playwright install chromium
        if ($LASTEXITCODE -ne 0) { throw "Browser installation failed." }
        & npm.cmd run test:e2e
        if ($LASTEXITCODE -ne 0) { throw "Browser workflows failed." }
    }
}
finally {
    Pop-Location
}
Write-Host "All selected frontend checks passed."
