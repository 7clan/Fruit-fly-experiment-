# run_gate5v2_windows.ps1 — one hardware-realistic Gate-5 v2 assessment.
# PASSIVE ONLY. No keyboard/mouse input is emitted.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

# Avoid nested native thread pools fighting the canonical brain on the
# target two-core laptop. This changes engineering throughput only.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$mainPython = ".\.venv\Scripts\python.exe"
$brainPython = ".\brain\.venv\Scripts\python.exe"

if (-not (Test-Path $mainPython)) {
    throw "Main venv missing; run .\setup_windows.ps1 first"
}
if (-not (Test-Path $brainPython)) {
    throw "Brain venv missing; run .\setup_windows.ps1 first"
}

$transportCheck = & $mainPython -c "from lab.brain.subprocess_runtime import CanonicalBrainSubprocessRuntime as R; print(R.transport_label)"
if ($LASTEXITCODE -ne 0 -or $transportCheck.Trim() -ne "subprocess_pipe") {
    throw "Expected current named-pipe brain transport. Pull origin/main first."
}

Write-Host "== Gate 5 v2 PASSIVE assessment ==" -ForegroundColor Cyan
Write-Host "120 seconds starts only after canonical brain READY." -ForegroundColor Yellow
Write-Host "DO NOT press Ctrl+C unless you want to abort; this assessment stops by itself." -ForegroundColor Red
Write-Host "Low-load mode: capture/fast CV 4 Hz, 480px detector, dashboard 1 Hz. Show quest NPC, Bandits, and scenery." -ForegroundColor Yellow

& $mainPython -m lab.app `
    --runtime canonical `
    --capture windows `
    --seconds 120 `
    --capture-fps 4 `
    --chunk-ms 50 `
    --brain-codegen cython `
    --brain-transport subprocess `
    --fast-hz 4 `
    --heavy-hz 0.25 `
    --fast-detect-width 480 `
    --dashboard-ui `
    --dashboard-hz 1

exit $LASTEXITCODE
