# run_autonomous_navigation_windows.ps1
# Canonical fly autonomously follows GPO's recommended-quest waypoint.
# No attacks/interact/abilities. F12 is the global emergency stop.

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

Write-Host "== AUTONOMOUS QUEST-WAYPOINT NAVIGATION ==" -ForegroundColor Cyan

# Cheap structural preflight before the ~1 minute canonical brain startup.
# SendInput requires cbSize to equal the native Win32 INPUT size.
& $mainPython -c "from lab.action.windows_input import _INPUT,_EXPECTED_INPUT_SIZE; import ctypes; s=ctypes.sizeof(_INPUT); print(f'[preflight] Win32 INPUT size={s} expected={_EXPECTED_INPUT_SIZE}'); raise SystemExit(0 if s==_EXPECTED_INPUT_SIZE else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "Windows input layout preflight failed; brain startup aborted."
}

Write-Host "Dashboard: ENABLE, DISABLE, REFOCUS, RELEASE KEYS, END RUN, EMERGENCY STOP." -ForegroundColor Yellow
Write-Host "Backup hotkeys: F8 enable, F9 disable, F10 refocus, F11 release keys, F12 emergency stop." -ForegroundColor Yellow
Write-Host "Roblox is focused automatically after brain READY." -ForegroundColor Yellow
Write-Host "F12 = EMERGENCY STOP. This verification run auto-stops after 45 seconds." -ForegroundColor Red

& $mainPython -m lab.app `
    --runtime canonical `
    --capture windows `
    --seconds 45 `
    --capture-fps 3 `
    --chunk-ms 50 `
    --brain-codegen cython `
    --brain-transport subprocess `
    --fast-hz 3 `
    --heavy-hz 0.1 `
    --fast-detect-width 480 `
    --dashboard-ui `
    --dashboard-hz 0.5 `
    --movement-only-autonomy

exit $LASTEXITCODE
