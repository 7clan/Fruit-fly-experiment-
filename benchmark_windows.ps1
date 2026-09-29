# benchmark_windows.ps1 — hardware + latency benchmarks (FINAL_ARCHITECTURE §8)
# Runs on the target Windows laptop BEFORE selecting final live rates.
# Produces WINDOWS_HARDWARE_REPORT.md + config/live_settings.json.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host "== DigitalFlyLab Windows hardware benchmark ==" -ForegroundColor Cyan

# full benchmark incl. canonical 138k-neuron brain chunks (brain venv)
& brain\.venv\Scripts\python.exe benchmark_windows.py --seconds 10 --json
if ($LASTEXITCODE -ne 0) { throw "benchmark failed" }

Write-Host ""
Write-Host "Report: WINDOWS_HARDWARE_REPORT.md" -ForegroundColor Green
Write-Host "Live settings (auto-chosen, conservative): config\live_settings.json"
Write-Host "Review the P95 sensory-to-action line against the 250 ms target."
