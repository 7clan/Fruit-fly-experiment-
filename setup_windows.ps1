# setup_windows.ps1 — DigitalFlyLab environment setup (Windows)
# FINAL_ARCHITECTURE §29: the finished experiment should not require
# manually opening Python terminals. Run ONCE per machine:
#
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
#
# What it does:
#   1. checks Python 3.12 specifically (required by pinned Brian2/NumPy/SciPy stack)
#   2. creates the main .venv + installs requirements.txt
#   3. clones + verifies the pinned third-party brain model (fail-closed)
#   4. creates the brain venv with pinned brian2 (via setup_model.py env)
#   5. optional heavy-vision extras (tesseract OCR) — checked, not fatal
#   6. runs the application self-test (PASSIVE, mock runtime)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host "== DigitalFlyLab setup ==" -ForegroundColor Cyan

# --- 1. Python -----------------------------------------------------------------
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pyLauncher) {
    throw "Python launcher 'py' not found. Install Python 3.12 from python.org with the launcher enabled."
}
$py312 = & $pyLauncher.Source -3.12 -c "import sys; print(sys.executable)" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $py312) {
    throw "Python 3.12 is required by the pinned Brian2 2.10.1 / NumPy 2.5.3 / SciPy 1.18.1 stack. Install Python 3.12, then rerun setup."
}
$py312 = $py312.Trim()
& $py312 --version
Write-Host "Python 3.12 OK" -ForegroundColor Green

# --- 2. main venv -------------------------------------------------------------
$recreateMainVenv = $false
if (Test-Path ".venv\Scripts\python.exe") {
    $mainVer = & .\.venv\Scripts\python.exe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($mainVer.Trim() -ne "3.12") { $recreateMainVenv = $true }
}
if ($recreateMainVenv) {
    Write-Host "Recreating main venv with Python 3.12 (old venv used $mainVer) ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}
if (-not (Test-Path ".venv")) {
    & $py312 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Main venv OK" -ForegroundColor Green

# --- 3. pinned third-party brain model (fail-closed verification) ---------------
Write-Host "Cloning/verifying pinned Shiu brain model (91bdd1e) ..."
& .\.venv\Scripts\python.exe brain\setup\setup_model.py clone
if ($LASTEXITCODE -ne 0) { throw "model clone failed" }
& .\.venv\Scripts\python.exe brain\setup\setup_model.py verify
if ($LASTEXITCODE -ne 0) { throw "MODEL VERIFICATION FAILED - do not proceed" }
Write-Host "Pinned model verified" -ForegroundColor Green

# --- 4. brain venv (pinned brian2) ----------------------------------------------
& .\.venv\Scripts\python.exe brain\setup\setup_model.py env
if ($LASTEXITCODE -ne 0) { throw "brain venv setup failed" }
Write-Host "Brain venv OK (brian2 pinned)" -ForegroundColor Green

# --- 5. heavy vision extras (optional, non-fatal) --------------------------------
$tess = Get-Command tesseract -ErrorAction SilentlyContinue
if ($tess) {
    Write-Host "tesseract found at $($tess.Source) (heavy vision OCR enabled)"
} else {
    Write-Host "tesseract NOT found - heavy vision OCR will use the mock backend. " `
        "Install from https://github.com/UB-Mannheim/tesseract/wiki if needed " `
        "(non-Roblox source, allowed per docs/GPO_PLAN.md)." -ForegroundColor Yellow
}

# --- 6. self-test (PASSIVE, mock runtime — no game required) ----------------------
Write-Host "Running application self-test (passive, mock) ..."
& .\.venv\Scripts\python.exe -m lab.app --self-test --runtime mock
if ($LASTEXITCODE -ne 0) { throw "SELF-TEST FAILED" }
Write-Host "SELF-TEST PASS" -ForegroundColor Green

Write-Host ""
Write-Host "Setup complete. Next: .\benchmark_windows.ps1  (run BEFORE Gate 5)" -ForegroundColor Cyan
