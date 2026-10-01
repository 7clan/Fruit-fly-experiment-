# run_autonomous_questing_windows.ps1
# Canonical fly navigation + LOCAL SmolVLM2 semantic coach + quest/PvE supervisor.
# F12 is the global emergency stop. No cloud API key is required.

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

$modelSpec = "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q4_K_M"
$modelAlias = "smolvlm2-256m"
$port = 18080
$baseUrl = "http://127.0.0.1:$port"
$apiUrl = "$baseUrl/v1"
$logDir = Join-Path $root "runtime_state\local_coach"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Refresh-LocalPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $links = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
    $env:Path = "$machine;$user;$links"
}

function Find-LlamaServer {
    $cmd = Get-Command "llama-server" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command "llama-server.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Test-Health {
    try {
        $null = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get -TimeoutSec 2
        return $true
    }
    catch {
        return $false
    }
}

Write-Host "== GATE 7C LOCAL: DIGITAL FLY + SMOLVLM2-256M COACH ==" -ForegroundColor Cyan

& $mainPython -c "from lab.action.windows_input import _INPUT,_EXPECTED_INPUT_SIZE; import ctypes; s=ctypes.sizeof(_INPUT); print(f'[preflight] Win32 INPUT size={s} expected={_EXPECTED_INPUT_SIZE}'); raise SystemExit(0 if s==_EXPECTED_INPUT_SIZE else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "Windows input layout preflight failed; brain startup aborted."
}

Refresh-LocalPath
$serverExe = Find-LlamaServer
if (-not $serverExe) {
    Write-Host "Local coach is not installed yet; running one-time setup..." -ForegroundColor Yellow
    & ".\setup_local_semantic_coach_windows.ps1"
    Refresh-LocalPath
    $serverExe = Find-LlamaServer
}
if (-not $serverExe) {
    throw "llama-server unavailable after local coach setup."
}

Write-Host "Fly brain contributes approach/retreat/escape; local AI selects skills and fresh CV servo handles time-critical target steering." -ForegroundColor Green
Write-Host "LOCAL AI: SmolVLM2-256M Q4_K_M via llama.cpp; no Gemini/API key." -ForegroundColor Green
Write-Host "Local AI now performs real text skill-selection on scene changes/~30s; expensive image reasoning remains rare." -ForegroundColor Yellow
Write-Host "Camera policy: autonomy NEVER rotates/drags your camera." -ForegroundColor Yellow
Write-Host "Local AI may reason about quests, obstacles, combat, visible shops, equipment and ships using the offline GPO playbook." -ForegroundColor Yellow
Write-Host "CPU policy: llama.cpp uses 2 low-priority CPU threads in short bursts; zero GPU layers." -ForegroundColor Yellow
Write-Host "Dashboard: ENABLE, DISABLE, REFOCUS, RELEASE KEYS, END RUN, EMERGENCY STOP." -ForegroundColor Yellow
Write-Host "Backup hotkeys: F8 enable, F9 disable, F10 refocus, F11 release keys, F12 emergency stop." -ForegroundColor Yellow
Write-Host "Persistent run: END RUN/F12 saves the report and ZIP." -ForegroundColor Red

$serverProc = $null
if (-not (Test-Health)) {
    $stdout = Join-Path $logDir "live_server.stdout.log"
    $stderr = Join-Path $logDir "live_server.stderr.log"
    Remove-Item $stdout,$stderr -Force -ErrorAction SilentlyContinue
    $serverArgs = @("-hf", $modelSpec, "--alias", $modelAlias, "--host", "127.0.0.1", "--port", "$port", "--threads", "2", "--threads-batch", "2", "--ctx-size", "2048", "--parallel", "1", "--n-gpu-layers", "0", "--no-mmproj-offload", "--no-warmup", "--no-webui")
    Write-Host "Starting local SmolVLM server..." -ForegroundColor Cyan
    $serverProc = Start-Process -FilePath $serverExe -ArgumentList $serverArgs -PassThru -WindowStyle Minimized -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    try {
        $serverProc.PriorityClass = "BelowNormal"
    } catch {
        Write-Host "[local-ai] could not lower process priority; continuing." -ForegroundColor DarkYellow
    }

    $ready = $false
    $maxWaitSeconds = 600
    for ($i = 0; $i -lt $maxWaitSeconds; $i++) {
        if ($serverProc.HasExited) { break }
        if (Test-Health) {
            $ready = $true
            break
        }
        if (($i % 10) -eq 0) {
            $lastLine = ""
            if (Test-Path $stderr) {
                $lastLine = Get-Content $stderr -Tail 1 -ErrorAction SilentlyContinue
            }
            if ([string]::IsNullOrWhiteSpace($lastLine) -and (Test-Path $stdout)) {
                $lastLine = Get-Content $stdout -Tail 1 -ErrorAction SilentlyContinue
            }
            if ([string]::IsNullOrWhiteSpace($lastLine)) {
                Write-Host ("[local-ai] still loading/downloading... {0}s" -f $i) -ForegroundColor DarkCyan
            } else {
                Write-Host ("[local-ai] {0}s | {1}" -f $i, $lastLine) -ForegroundColor DarkCyan
            }
        }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) {
        $tailErr = ""
        $tailOut = ""
        if (Test-Path $stderr) {
            $tailErr = (Get-Content $stderr -Tail 30 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        }
        if (Test-Path $stdout) {
            $tailOut = (Get-Content $stdout -Tail 30 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        }
        $tail = "STDERR: " + $tailErr + " | STDOUT: " + $tailOut
        if ($serverProc -and -not $serverProc.HasExited) {
            Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
        }
        throw "Local SmolVLM server failed to start. Last log: $tail"
    }
}

Write-Host "Testing local coach model before brain startup..." -ForegroundColor Cyan
& $mainPython -m lab.coach.local_probe --url $apiUrl --model $modelAlias --timeout 30
if ($LASTEXITCODE -ne 0) {
    if ($serverProc -and -not $serverProc.HasExited) {
        Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
    }
    throw "Local semantic coach model preflight failed; brain startup aborted."
}

Write-Host "Local coach READY: $modelAlias" -ForegroundColor Green
Write-Host "Low-power profile: capture/CV 3 Hz, x2 downsample, AI skill calls scene-gated/~30s, visual reasoning rare." -ForegroundColor Yellow

$exitCode = 1
try {
    & $mainPython -m lab.app --runtime canonical --capture windows --seconds 0 --capture-fps 3 --capture-downsample 2 --chunk-ms 50 --brain-codegen cython --brain-transport subprocess --fast-hz 3 --heavy-hz 0.05 --fast-detect-width 480 --dashboard-ui --dashboard-hz 0.25 --quest-autonomy --semantic-coach --coach-provider local --coach-model $modelAlias --coach-url $apiUrl --coach-hz 0.5 --gpo-loadout default_melee
    $exitCode = $LASTEXITCODE
}
finally {
    if ($serverProc -and -not $serverProc.HasExited) {
        Write-Host "Stopping local SmolVLM server..." -ForegroundColor DarkGray
        Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
        $serverProc.WaitForExit()
    }
}

exit $exitCode
