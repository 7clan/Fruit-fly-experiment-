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
$python = ".\.venv\Scripts\python.exe"
$chunkMs = 50
$brainHz = 10.0

if ($Canonical) {
    $runtime = "canonical"
    $python = ".\brain\.venv\Scripts\python.exe"
    if (-not (Test-Path $python)) { throw "Brain venv missing; run .\setup_windows.ps1 first" }

    if (Test-Path "config\live_settings.json") {
        $live = Get-Content "config\live_settings.json" -Raw | ConvertFrom-Json
        if ($live.brain_chunk_ms) { $chunkMs = [int]$live.brain_chunk_ms }
        if ($live.brain_hz) { $brainHz = [double]$live.brain_hz }
    }
    Write-Host ("Canonical brain: measured settings chunk={0} ms, target={1} Hz; PASSIVE/SHADOW only." -f $chunkMs, $brainHz) -ForegroundColor Yellow
}

& $python -m lab.app --runtime $runtime --seconds $Seconds --chunk-ms $chunkMs --brain-hz $brainHz --dashboard
exit $LASTEXITCODE
