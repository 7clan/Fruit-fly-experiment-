# run_autonomous_questing_windows.ps1
# Canonical fly navigation + Meta Llama API semantic coach + quest/PvE.
# Cloud semantic reasoning avoids competing with Brian2 on the old laptop.
# F12 is the global emergency stop.

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

Write-Host "== GATE 7D CLOUD: DIGITAL FLY + META LLAMA API COACH ==" -ForegroundColor Cyan

& $mainPython -c "from lab.action.windows_input import _INPUT,_EXPECTED_INPUT_SIZE; import ctypes; s=ctypes.sizeof(_INPUT); print(f'[preflight] Win32 INPUT size={s} expected={_EXPECTED_INPUT_SIZE}'); raise SystemExit(0 if s==_EXPECTED_INPUT_SIZE else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "Windows input layout preflight failed; brain startup aborted."
}

if ([string]::IsNullOrWhiteSpace($env:LLAMA_API_KEY)) {
    $env:LLAMA_API_KEY = [Environment]::GetEnvironmentVariable("LLAMA_API_KEY", "User")
}
if ([string]::IsNullOrWhiteSpace($env:LLAMA_MODEL)) {
    $env:LLAMA_MODEL = [Environment]::GetEnvironmentVariable("LLAMA_MODEL", "User")
}
if ([string]::IsNullOrWhiteSpace($env:LLAMA_API_BASE)) {
    $env:LLAMA_API_BASE = [Environment]::GetEnvironmentVariable("LLAMA_API_BASE", "User")
}
if ([string]::IsNullOrWhiteSpace($env:LLAMA_API_BASE)) {
    $env:LLAMA_API_BASE = "https://api.llama.com/compat/v1"
}
if ([string]::IsNullOrWhiteSpace($env:LLAMA_MODEL)) {
    $env:LLAMA_MODEL = "auto"
}

if ([string]::IsNullOrWhiteSpace($env:LLAMA_API_KEY)) {
    Write-Host "Llama API key is not configured; running one-time secure setup..." -ForegroundColor Yellow
    & ".\setup_llama_api_windows.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "Llama API setup failed."
    }
}

Write-Host "Fly brain owns biological approach/retreat/escape signals." -ForegroundColor Green
Write-Host "CLOUD AI: Meta Llama API selects high-level skills and can inspect occasional game frames." -ForegroundColor Green
Write-Host "Preferred model: Llama 4 Maverick; Scout is automatic fallback if needed." -ForegroundColor Yellow
Write-Host "No local language/vision model will run beside Roblox + Brian2." -ForegroundColor Yellow
Write-Host "Camera policy: autonomy NEVER rotates/drags your camera." -ForegroundColor Yellow
Write-Host "Dashboard shows PERCEPTION -> AI, AI -> FLY, WHY, and executor command." -ForegroundColor Yellow
Write-Host "Dashboard: ENABLE, DISABLE, REFOCUS, RELEASE KEYS, END RUN, EMERGENCY STOP." -ForegroundColor Yellow
Write-Host "Backup hotkeys: F8 enable, F9 disable, F10 refocus, F11 release keys, F12 emergency stop." -ForegroundColor Yellow
Write-Host "Persistent run: END RUN/F12 saves the report and ZIP." -ForegroundColor Red

Write-Host "Testing Meta Llama API before brain startup..." -ForegroundColor Cyan
$probeArgs = @(
    "-m", "lab.coach.llama_probe",
    "--model", $env:LLAMA_MODEL,
    "--base-url", $env:LLAMA_API_BASE,
    "--timeout", "20"
)
$probeOutput = @(& $mainPython @probeArgs)
$probeExit = $LASTEXITCODE
$probeOutput | ForEach-Object { Write-Host $_ }
if ($probeExit -ne 0) {
    throw "Meta Llama API preflight failed; brain startup aborted."
}

$modelLine = $probeOutput | Where-Object { $_ -like "COACH_MODEL=*" } | Select-Object -Last 1
if ([string]::IsNullOrWhiteSpace($modelLine)) {
    throw "Llama preflight passed but did not return a model."
}
$selectedModel = $modelLine.Substring("COACH_MODEL=".Length).Trim()
$env:LLAMA_MODEL = $selectedModel
[Environment]::SetEnvironmentVariable("LLAMA_MODEL", $selectedModel, "User")

Write-Host "Meta Llama coach READY: $selectedModel" -ForegroundColor Green
Write-Host "Low-power profile: local capture/CV 3 Hz; semantic API calls are scene/rate gated." -ForegroundColor Yellow

$appArgs = @(
    "-m", "lab.app",
    "--runtime", "canonical",
    "--capture", "windows",
    "--seconds", "0",
    "--capture-fps", "3",
    "--capture-downsample", "2",
    "--chunk-ms", "50",
    "--brain-codegen", "cython",
    "--brain-transport", "subprocess",
    "--fast-hz", "3",
    "--heavy-hz", "0.05",
    "--fast-detect-width", "480",
    "--dashboard-ui",
    "--dashboard-hz", "0.25",
    "--quest-autonomy",
    "--semantic-coach",
    "--coach-provider", "llama",
    "--coach-model", $selectedModel,
    "--coach-url", $env:LLAMA_API_BASE,
    "--coach-hz", "1.0",
    "--gpo-loadout", "default_melee"
)
& $mainPython @appArgs
exit $LASTEXITCODE
