param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not (Get-Command terraform -ErrorAction SilentlyContinue)) {
    throw "Install Terraform 1.13 or later (before 2.0) before checking infrastructure."
}
$savedDataDir = $env:TF_DATA_DIR
$savedCacheDir = $env:TF_PLUGIN_CACHE_DIR
$env:TF_PLUGIN_CACHE_DIR = Join-Path $env:LOCALAPPDATA "FleetGuard/terraform-plugin-cache"
New-Item -ItemType Directory -Path $env:TF_PLUGIN_CACHE_DIR -Force | Out-Null
$checkRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("fleetguard-cloud-check-" + [guid]::NewGuid().ToString("N"))
try {
    & terraform "-chdir=$projectRoot" fmt -check -recursive infra
    if ($LASTEXITCODE -ne 0) { throw "Terraform format check failed." }
    foreach ($module in @("bootstrap", "runtime")) {
        $env:TF_DATA_DIR = Join-Path $checkRoot $module
        $moduleRoot = Join-Path $projectRoot "infra/$module"
        & terraform "-chdir=$moduleRoot" init -backend=false -input=false -lockfile=readonly
        if ($LASTEXITCODE -ne 0) { throw "Terraform provider initialization failed: $module." }
        & terraform "-chdir=$moduleRoot" validate
        if ($LASTEXITCODE -ne 0) { throw "Terraform validation failed: $module." }
        & terraform "-chdir=$moduleRoot" test
        if ($LASTEXITCODE -ne 0) { throw "Terraform mock tests failed: $module." }
    }
}
finally {
    $env:TF_DATA_DIR = $savedDataDir
    $env:TF_PLUGIN_CACHE_DIR = $savedCacheDir
    if (Test-Path $checkRoot) { Remove-Item -LiteralPath $checkRoot -Recurse -Force }
}
Write-Host "Cloud configuration checks passed. No live AWS resources were created."
