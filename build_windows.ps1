# build_windows.ps1 — package DigitalFlyLab.exe (PyInstaller)
# FINAL_ARCHITECTURE §29: user experience = ONE application; no Python
# terminals for the finished experiment.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) { throw "Run .\setup_windows.ps1 first" }

& .\.venv\Scripts\python.exe -m pip install pyinstaller --quiet

# One-file windowed app. The brain venv (brian2) stays external: the
# packaged app drives the canonical brain via subprocess to
# brain\.venv\Scripts\python.exe (init-once worker process), so the exe
# does not bundle the 138k-neuron stack. Synthetic/mock mode is fully
# self-contained for bring-up.
& .\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean `
    --name DigitalFlyLab --onefile --windowed `
    --add-data "config;config" `
    --add-data "brain\data\d7_d10\*.json;brain\data\d7_d10" `
    --add-data "brain\data\io_map;brain\data\io_map" `
    lab\app.py

Write-Host ""
Write-Host "Built dist\DigitalFlyLab.exe" -ForegroundColor Green
Write-Host "NOTE: canonical-brain live mode expects brain\.venv next to the exe "`
    "(or set DIGITALFLYLAB_BRAIN_PY). Run .\benchmark_windows.ps1 first." -ForegroundColor Yellow
