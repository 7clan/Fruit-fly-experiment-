# run_autonomous_questing_windows.ps1
# Canonical fly navigation + engineered quest/starter-PvE supervisor.
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

Write-Host "== GATE 7C: DIGITAL FLY + SEMANTIC GAME COACH ==" -ForegroundColor Cyan

# Cheap structural preflight before the ~1 minute canonical brain startup.
# SendInput requires cbSize to equal the native Win32 INPUT size.
& $mainPython -c "from lab.action.windows_input import _INPUT,_EXPECTED_INPUT_SIZE; import ctypes; s=ctypes.sizeof(_INPUT); print(f'[preflight] Win32 INPUT size={s} expected={_EXPECTED_INPUT_SIZE}'); raise SystemExit(0 if s==_EXPECTED_INPUT_SIZE else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "Windows input layout preflight failed; brain startup aborted."
}

Write-Host "Fly brain owns turn/approach/retreat navigation." -ForegroundColor Green
Write-Host "Semantic coach: understands quests, progression, shops, ships, gear, fruit/style/weapon planning and visible UI." -ForegroundColor Yellow
Write-Host "Camera policy: autonomy NEVER rotates/drags your camera; fly steers with character movement." -ForegroundColor Yellow
Write-Host "Obstacle recovery: semantic coach may choose jump, climb, backtrack, sprint or reobserve." -ForegroundColor Yellow
Write-Host "Starter PvE: F block, Q evade, M1, E Gut Punch, R Ground Smash; broader skill keys are catalogued but context-gated." -ForegroundColor Yellow
Write-Host "Keep the observed default Melee loadout equipped for the first semantic-coach test." -ForegroundColor Yellow
Write-Host "Defense learning is ENGINEERED ValueTable learning, not biological MB learning." -ForegroundColor DarkYellow
Write-Host "Dashboard: ENABLE, DISABLE, REFOCUS, RELEASE KEYS, END RUN, EMERGENCY STOP." -ForegroundColor Yellow
Write-Host "Backup hotkeys: F8 enable, F9 disable, F10 refocus, F11 release keys, F12 emergency stop." -ForegroundColor Yellow
Write-Host "Roblox is focused automatically after brain READY." -ForegroundColor Yellow
Write-Host "Low-power profile: capture/CV 3 Hz, x2 capture downsample, 480px detector, dashboard 0.25 Hz." -ForegroundColor Yellow
Write-Host "Persistent run: use END RUN or F12 to stop and save the report/ZIP." -ForegroundColor Red
if ([string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
    Write-Host "Semantic coach API key: MISSING (run .\setup_semantic_coach_windows.ps1 first)." -ForegroundColor Red
} else {
    Write-Host "Semantic coach API key: configured." -ForegroundColor Green
}
$coachModel = $env:GEMINI_MODEL
if ([string]::IsNullOrWhiteSpace($coachModel)) { $coachModel = "gemini-2.5-flash-lite" }
Write-Host "Coach model: $coachModel" -ForegroundColor DarkCyan

& $mainPython -m lab.app `
    --runtime canonical `
    --capture windows `
    --seconds 0 `
    --capture-fps 3 `
    --capture-downsample 2 `
    --chunk-ms 50 `
    --brain-codegen cython `
    --brain-transport subprocess `
    --fast-hz 3 `
    --heavy-hz 0.05 `
    --fast-detect-width 480 `
    --dashboard-ui `
    --dashboard-hz 0.25 `
    --quest-autonomy `
    --semantic-coach `
    --coach-model $coachModel `
    --coach-hz 1 `
    --gpo-loadout default_melee

exit $LASTEXITCODE
