# Passive Gate-5 Windows.Graphics.Capture probe. No input emission.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Main venv missing; run .\setup_windows.ps1 first" }

& $python -c "import windows_capture" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing pinned windows-capture 2.0.1 ..." -ForegroundColor Cyan
    & $python -m pip install --timeout 120 --retries 10 --prefer-binary windows-capture==2.0.1
    if ($LASTEXITCODE -ne 0) { throw "windows-capture install failed" }
}

Write-Host "== PASSIVE Windows game-window capture probe ==" -ForegroundColor Cyan
& $python probe_capture_windows.py --seconds 10 --fps 30
if ($LASTEXITCODE -ne 0) { throw "capture probe failed" }
