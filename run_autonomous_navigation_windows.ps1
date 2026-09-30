# run_autonomous_navigation_windows.ps1
# Canonical fly autonomously follows GPO's recommended-quest waypoint.
# No attacks/interact/abilities. F12 is the global emergency stop.

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

Write-Host "== AUTONOMOUS QUEST-WAYPOINT NAVIGATION ==" -ForegroundColor Cyan
Write-Host "Do not steer the character. The fly will turn/approach by itself." -ForegroundColor Yellow
Write-Host "Roblox is focused automatically after brain READY." -ForegroundColor Yellow
Write-Host "F12 = EMERGENCY STOP. This run auto-stops after 180 seconds." -ForegroundColor Red

& $mainPython -m lab.app `
    --runtime canonical `
    --capture windows `
    --seconds 180 `
    --capture-fps 4 `
    --chunk-ms 50 `
    --brain-codegen cython `
    --brain-transport subprocess `
    --fast-hz 4 `
    --heavy-hz 0.25 `
    --fast-detect-width 480 `
    --movement-only-autonomy

exit $LASTEXITCODE
