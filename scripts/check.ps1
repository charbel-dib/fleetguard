param([switch]$Research)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\use-environment.ps1" -Research:$Research
$fleetPython = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\python.exe"

& $fleetPython -m ruff format --check .
if ($LASTEXITCODE -ne 0) { throw "Formatting check failed." }

& $fleetPython -m ruff check .
if ($LASTEXITCODE -ne 0) { throw "Lint check failed." }

$modules = @("pytest", "fleetguard.smoke", "fleetguard.comparison_smoke")
if ($Research) { $modules += "fleetguard.optimization_smoke" }
$modules += "fleetguard.release_smoke"
$modules += "build"

foreach ($module in $modules) {
    & $fleetPython -m $module
    if ($LASTEXITCODE -ne 0) { throw "Check failed: $module. Remaining checks stopped." }
}

Write-Host "All selected checks passed."
