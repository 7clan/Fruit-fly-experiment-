# run_teacher_windows.ps1
# HUMAN TEACHER MODE: records gameplay demonstrations only.
# No AI, fruit-fly brain, or SendInput gameplay control is enabled.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Main venv missing. Run .\setup_windows.ps1 first."
}

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

Write-Host "== HUMAN TEACHER DEMONSTRATION MODE ==" -ForegroundColor Cyan
Write-Host "You control the game. DigitalFlyLab only records verified gameplay controls + perception." -ForegroundColor Green
Write-Host "No autonomous keyboard/mouse input is emitted." -ForegroundColor Yellow
Write-Host "Play normal quest/combat/navigation examples. Press Ctrl+C when finished." -ForegroundColor Yellow

$appArgs = @(
    "-m", "lab.app",
    "--runtime", "mock",
    "--capture", "windows",
    "--seconds", "0",
    "--capture-fps", "12",
    "--capture-downsample", "1",
    "--fast-hz", "8",
    "--heavy-hz", "0.25",
    "--fast-detect-width", "480",
    "--teacher-mode",
    "--gpo-loadout", "default_melee"
)
& $python @appArgs
exit $LASTEXITCODE
