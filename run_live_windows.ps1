# run_live_windows.ps1 — persistent PASSIVE GPO + canonical fly dashboard.
# No keyboard/mouse input is emitted. Ctrl+C stops the session.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$mainPython = ".\\.venv\\Scripts\\python.exe"
$brainPython = ".\\brain\\.venv\\Scripts\\python.exe"

if (-not (Test-Path $mainPython)) {
    throw "Main venv missing; run .\\setup_windows.ps1 first"
}
if (-not (Test-Path $brainPython)) {
    throw "Brain venv missing; run .\\setup_windows.ps1 first"
}

Write-Host "== DigitalFlyLab LIVE PASSIVE dashboard ==" -ForegroundColor Cyan
Write-Host "Canonical brain runs in its own process. No game input is emitted." -ForegroundColor Yellow
Write-Host "Press Ctrl+C in this PowerShell window to stop." -ForegroundColor Yellow

& $mainPython -m lab.app `
    --runtime canonical `
    --capture windows `
    --seconds 0 `
    --capture-fps 5 `
    --chunk-ms 50 `
    --brain-codegen cython `
    --brain-transport subprocess `
    --fast-hz 5 `
    --heavy-hz 1 `
    --dashboard-ui `
    --dashboard-hz 3

exit $LASTEXITCODE
