# run_ai_only_windows.ps1
# Experimental branch: Ollama Cloud AI is the sole gameplay decision-maker.
# No canonical fruit-fly/Brian2 brain is started or consulted.
# F12 = immediate emergency stop and end run.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$mainPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $mainPython)) { throw "Main venv missing. Run .\setup_windows.ps1 first." }

# Keep the old laptop responsive. There is no Brian2 process in this mode.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

Write-Host "== AI-ONLY AUTOPILOT EXPERIMENT ==" -ForegroundColor Cyan
Write-Host "Decision owner: Ollama Cloud AI ONLY." -ForegroundColor Green
Write-Host "Fruit-fly brain: DISABLED / NOT STARTED." -ForegroundColor Yellow
Write-Host "Local CV only measures the screen and realizes the AI-selected target." -ForegroundColor Yellow
Write-Host "The AI owns quests, navigation goals, combat, equipment, abilities, shops and progression." -ForegroundColor Yellow
Write-Host "Vision profile: 736px fast cloud frames, 960px for UI/dialogue; local detector 560px/6Hz." -ForegroundColor Yellow
Write-Host "F8 enable, F9 disable, F10 refocus, F11 release keys, F12 emergency stop." -ForegroundColor Red

& $mainPython -c "from lab.action.windows_input import _INPUT,_EXPECTED_INPUT_SIZE; import ctypes; s=ctypes.sizeof(_INPUT); print(f'[preflight] Win32 INPUT size={s} expected={_EXPECTED_INPUT_SIZE}'); raise SystemExit(0 if s==_EXPECTED_INPUT_SIZE else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "Windows input layout preflight failed."
}

if ([string]::IsNullOrWhiteSpace($env:OLLAMA_API_KEY)) {
    $env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable("OLLAMA_API_KEY", "User")
}
if ([string]::IsNullOrWhiteSpace($env:OLLAMA_CLOUD_MODEL)) {
    $env:OLLAMA_CLOUD_MODEL = [Environment]::GetEnvironmentVariable("OLLAMA_CLOUD_MODEL", "User")
}
if ([string]::IsNullOrWhiteSpace($env:OLLAMA_CLOUD_BASE)) {
    $env:OLLAMA_CLOUD_BASE = [Environment]::GetEnvironmentVariable("OLLAMA_CLOUD_BASE", "User")
}
if ([string]::IsNullOrWhiteSpace($env:OLLAMA_CLOUD_BASE)) {
    $env:OLLAMA_CLOUD_BASE = "https://ollama.com/v1"
}
if ([string]::IsNullOrWhiteSpace($env:OLLAMA_CLOUD_MODEL)) {
    $env:OLLAMA_CLOUD_MODEL = "gemma4:cloud"
}

if ([string]::IsNullOrWhiteSpace($env:OLLAMA_API_KEY)) {
    Write-Host "Ollama Cloud key missing; opening one-time setup..." -ForegroundColor Yellow
    & ".\setup_ollama_cloud_windows.ps1"
    if ($LASTEXITCODE -ne 0) { throw "Ollama Cloud setup failed." }
}

Write-Host "Testing REQUIRED image vision; preferring a smaller/faster model when available..." -ForegroundColor Cyan
$probeArgs = @(
    "-m", "lab.coach.ollama_cloud_probe",
    "--model", $env:OLLAMA_CLOUD_MODEL,
    "--base-url", $env:OLLAMA_CLOUD_BASE,
    "--timeout", "25",
    "--require-vision",
    "--prefer-fast"
)
$probeOutput = @(& $mainPython @probeArgs)
$probeExit = $LASTEXITCODE
$probeOutput | ForEach-Object { Write-Host $_ }
if ($probeExit -ne 0) {
    throw "Ollama Cloud vision preflight failed; AI-only run will not start blind."
}
$modelLine = $probeOutput | Where-Object { $_ -like "COACH_MODEL=*" } | Select-Object -Last 1
if ([string]::IsNullOrWhiteSpace($modelLine)) {
    throw "Cloud preflight passed but returned no model."
}
$selectedModel = $modelLine.Substring("COACH_MODEL=".Length).Trim()
$env:OLLAMA_CLOUD_MODEL = $selectedModel
[Environment]::SetEnvironmentVariable("OLLAMA_CLOUD_MODEL", $selectedModel, "User")
Write-Host "AI controller READY: $selectedModel" -ForegroundColor Green

$appArgs = @(
    "-m", "lab.app",
    "--runtime", "mock",
    "--capture", "windows",
    "--seconds", "0",
    "--capture-fps", "6",
    "--capture-downsample", "1",
    "--fast-hz", "6",
    "--heavy-hz", "0.10",
    "--fast-detect-width", "560",
    "--ai-only-autonomy",
    "--semantic-coach",
    "--coach-provider", "ollama_cloud",
    "--coach-model", $selectedModel,
    "--coach-url", $env:OLLAMA_CLOUD_BASE,
    "--coach-hz", "4.0",
    "--gpo-loadout", "default_melee"
)

Write-Host "Starting. Do not steer/click during the experiment if you want a clean AI-only result." -ForegroundColor Cyan
Write-Host "The run auto-focuses Roblox and starts the autopilot. Press F12 at any time to stop and save the ZIP." -ForegroundColor Red
& $mainPython @appArgs
exit $LASTEXITCODE
