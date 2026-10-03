# run_ai_only_windows.ps1
# Experimental branch: a multimodal cloud AI is the sole gameplay decision-maker.
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
Write-Host "Decision owner: Gemini Flash-Lite AI ONLY." -ForegroundColor Green
Write-Host "Fruit-fly brain: DISABLED / NOT STARTED." -ForegroundColor Yellow
Write-Host "Local CV only measures the screen and realizes the AI-selected target." -ForegroundColor Yellow
Write-Host "The AI owns quests, navigation goals, combat, equipment, abilities, shops and progression." -ForegroundColor Yellow
Write-Host "Vision profile: Gemini Flash-Lite preferred; local CV 8Hz + local tracking between cloud replies." -ForegroundColor Yellow
Write-Host "F8 enable, F9 disable, F10 refocus, F11 release keys, F12 emergency stop." -ForegroundColor Red

& $mainPython -c "from lab.action.windows_input import _INPUT,_EXPECTED_INPUT_SIZE; import ctypes; s=ctypes.sizeof(_INPUT); print(f'[preflight] Win32 INPUT size={s} expected={_EXPECTED_INPUT_SIZE}'); raise SystemExit(0 if s==_EXPECTED_INPUT_SIZE else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "Windows input layout preflight failed."
}

# This experiment intentionally stays on Gemini. Do not silently fall back
# to the much slower Ollama 31B path; if Gemini hits quota later we will test
# the local SmolVLM branch separately.
$provider = "gemini"
$selectedModel = $null

if ([string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
    $env:GEMINI_API_KEY = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")
}
if ([string]::IsNullOrWhiteSpace($env:GEMINI_MODEL)) {
    $env:GEMINI_MODEL = [Environment]::GetEnvironmentVariable("GEMINI_MODEL", "User")
}
if ([string]::IsNullOrWhiteSpace($env:GEMINI_MODEL)) {
    $env:GEMINI_MODEL = "gemini-3.5-flash-lite"
}

if ([string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
    throw "Gemini API key missing. Run .\setup_semantic_coach_windows.ps1 once, then retry."
}

Write-Host "Benchmarking available Gemini Flash-Lite vision models for the lowest startup latency..." -ForegroundColor Cyan
$gemProbeArgs = @(
    "-m", "lab.coach.probe",
    "--model", $env:GEMINI_MODEL,
    "--timeout", "15",
    "--require-vision",
    "--fastest-vision"
)
$gemProbe = @(& $mainPython @gemProbeArgs)
$gemExit = $LASTEXITCODE
$gemProbe | ForEach-Object { Write-Host $_ }
if ($gemExit -ne 0) {
    throw "Gemini vision preflight failed. AI-only run will not fall back to a slower provider."
}
$modelLine = $gemProbe | Where-Object { $_ -like "COACH_MODEL=*" } | Select-Object -Last 1
if ([string]::IsNullOrWhiteSpace($modelLine)) {
    throw "Gemini preflight passed but returned no model."
}
$selectedModel = $modelLine.Substring("COACH_MODEL=".Length).Trim()
$env:GEMINI_MODEL = $selectedModel
[Environment]::SetEnvironmentVariable("GEMINI_MODEL", $selectedModel, "User")

Write-Host "AI controller READY: provider=$provider model=$selectedModel" -ForegroundColor Green

$appArgs = @(
    "-m", "lab.app",
    "--runtime", "mock",
    "--capture", "windows",
    "--seconds", "0",
    "--capture-fps", "8",
    "--capture-downsample", "1",
    "--fast-hz", "8",
    "--heavy-hz", "0.10",
    "--fast-detect-width", "560",
    "--ai-only-autonomy",
    "--semantic-coach",
    "--coach-provider", $provider,
    "--coach-model", $selectedModel,
    "--coach-hz", "4.0",
    "--gpo-loadout", "default_melee"
)
Write-Host "Starting. Do not steer/click during the experiment if you want a clean AI-only result." -ForegroundColor Cyan
Write-Host "The run auto-focuses Roblox and starts the autopilot. Press F12 at any time to stop and save the ZIP." -ForegroundColor Red
& $mainPython @appArgs
exit $LASTEXITCODE
