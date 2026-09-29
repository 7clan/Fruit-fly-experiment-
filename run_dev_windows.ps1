# run_dev_windows.ps1 — dev launcher (PASSIVE by default; autonomy is
# hard-gated and CANNOT be enabled from this script).
#
#   .\run_dev_windows.ps1                      # synthetic source, mock brain
#   .\run_dev_windows.ps1 -Canonical           # synthetic source, canonical brain
#   .\run_dev_windows.ps1 -Seconds 60

param(
    [switch]$Canonical,
    [int]$Seconds = 30
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) { throw "Run .\setup_windows.ps1 first" }

$runtime = "mock"
if ($Canonical) {
    $runtime = "canonical"
    Write-Host "Canonical brain: init-once ~10 s + ~3 GB RAM; chunked stepping." -ForegroundColor Yellow
}

& .\.venv\Scripts\python.exe -m lab.app --runtime $runtime --seconds $Seconds --dashboard
exit $LASTEXITCODE
