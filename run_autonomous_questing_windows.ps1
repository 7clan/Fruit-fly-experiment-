$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$mainPython = Join-Path $root ".venv\Scripts\python.exe"
$brainPython = Join-Path $root "brain\.venv\Scripts\python.exe"

if (-not (Test-Path $mainPython)) {
    throw "Main venv missing: $mainPython"
}
if (-not (Test-Path $brainPython)) {
    throw "Brain venv missing: $brainPython"
}

Write-Host "== DIGITAL FLY — AUTONOMOUS QUEST + STARTER PVE ==" -ForegroundColor Cyan

& $mainPython -c "from lab.action.windows_input import _INPUT,_EXPECTED_INPUT_SIZE; import ctypes; s=ctypes.sizeof(_INPUT); print(f\'[preflight] Win32 INPUT size={s} expected={_EXPECTED_INPUT_SIZE}\'); raise SystemExit(0 if s==_EXPECTED_INPUT_SIZE else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "Windows input layout preflight failed; brain startup aborted."
}

Write-Host "Fly brain: turn / approach / retreat navigation." -ForegroundColor Green
Write-Host "Engineered quest helper: yellow QUEST -> T interact; green tracker -> travel; red quest marker -> starter PvE." -ForegroundColor Yellow
Write-Host "Obstacle recovery: jump, climb, and bounded camera search." -ForegroundColor Yellow
Write-Host "Starter PvE hard allowlist: F block, Q evade, M1, E Gut Punch, R Ground Smash." -ForegroundColor Yellow
Write-Host "Defense learning is engineered ValueTable learning, NOT biological mushroom-body learning." -ForegroundColor DarkYellow
Write-Host "Current loadout profile: default_melee." -ForegroundColor Yellow
Write-Host "Low-power profile: capture/CV 3 Hz, x2 capture downsample, 480px detector, dashboard 0.25 Hz." -ForegroundColor Yellow
Write-Host "Dashboard: ENABLE, DISABLE, REFOCUS, RELEASE KEYS, END RUN, EMERGENCY STOP." -ForegroundColor Yellow
Write-Host "Backup hotkeys: F8 enable, F9 disable, F10 refocus, F11 release keys, F12 emergency stop." -ForegroundColor Yellow
Write-Host "F12 = EMERGENCY STOP. Persistent run; END RUN saves report/ZIP." -ForegroundColor Red

$env:DIGITALFLYLAB_BRAIN_PY = $brainPython

& $mainPython -u -m lab.app `
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
    --gpo-loadout default_melee
