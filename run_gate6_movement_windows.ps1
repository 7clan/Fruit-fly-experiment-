# run_gate6_movement_windows.ps1 — first active movement-only fly test.
# HARD FILTER: only W/A/D can reach Windows input. No combat/interact/mouse.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$mainPython = ".\.venv\Scripts\python.exe"
$brainPython = ".\brain\.venv\Scripts\python.exe"

if (-not (Test-Path $mainPython)) { throw "Main venv missing" }
if (-not (Test-Path $brainPython)) { throw "Brain venv missing" }

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

Write-Host "== Gate 6A MOVEMENT-ONLY autonomy ==" -ForegroundColor Cyan
Write-Host "ACTIVE INPUT: only W/A/D are allowed by a hard backend filter." -ForegroundColor Yellow
Write-Host "No attacks, block, interact, abilities, mouse input, or combat actions." -ForegroundColor Yellow
Write-Host "Keep the character in an open low-risk area. Ctrl+C aborts." -ForegroundColor Red

& $mainPython -m lab.app `
    --runtime canonical `
    --capture windows `
    --seconds 45 `
    --capture-fps 4 `
    --chunk-ms 50 `
    --brain-codegen cython `
    --brain-transport subprocess `
    --fast-hz 4 `
    --heavy-hz 0.25 `
    --fast-detect-width 480 `
    --dashboard-ui `
    --dashboard-hz 1 `
    --movement-only-autonomy

exit $LASTEXITCODE
