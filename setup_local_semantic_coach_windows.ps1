# setup_local_semantic_coach_windows.ps1
# Installs llama.cpp and downloads/tests the tiny local SmolLM2-135M model.
# No API key is used. Model inference stays on this PC.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$mainPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $mainPython)) { throw "Main venv missing" }

$modelSpec = "lmstudio-community/SmolLM2-135M-Instruct-GGUF:Q4_K_M"
$modelAlias = "smollm2-135m"
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

Write-Host "== LOCAL SEMANTIC COACH SETUP ==" -ForegroundColor Cyan
Write-Host "Model: SmolLM2-135M-Instruct Q4_K_M" -ForegroundColor Yellow
Write-Host "Runtime: llama.cpp, CPU-only, ONE low-priority inference thread" -ForegroundColor Yellow
Write-Host "No Gemini/API key is required." -ForegroundColor Green
Write-Host "First setup downloads only the tiny text model; there is no vision projector." -ForegroundColor DarkYellow

Refresh-LocalPath
$serverExe = Find-LlamaServer
if (-not $serverExe) {
    $winget = Get-Command "winget" -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget is missing. Install Microsoft App Installer, then rerun this script."
    }
    Write-Host "Installing llama.cpp with WinGet..." -ForegroundColor Cyan
    & winget install --id ggml.llamacpp -e --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "llama.cpp installation failed (winget exit $LASTEXITCODE)."
    }
    Refresh-LocalPath
    $serverExe = Find-LlamaServer
}
if (-not $serverExe) {
    throw "llama-server was installed but is not visible on PATH. Open a new PowerShell and rerun this setup."
}

Write-Host "llama-server: $serverExe" -ForegroundColor Green

if (Test-Health) {
    Write-Host "A local coach server is already running on port $port." -ForegroundColor Yellow
    & $mainPython -m lab.coach.local_probe --url $apiUrl --model $modelAlias --timeout 60
    if ($LASTEXITCODE -ne 0) {
        throw "Existing local server on port $port is not the expected SmolLM coach."
    }
}
else {
    $stdout = Join-Path $logDir "setup_server.stdout.log"
    $stderr = Join-Path $logDir "setup_server.stderr.log"
    Remove-Item $stdout,$stderr -Force -ErrorAction SilentlyContinue

    $args = @("-hf", $modelSpec, "--alias", $modelAlias, "--host", "127.0.0.1", "--port", "$port", "--threads", "1", "--threads-batch", "1", "--ctx-size", "1024", "--parallel", "1", "--n-gpu-layers", "0", "--no-warmup", "--no-webui")

    Write-Host "Downloading/loading local model. First run can take a few minutes..." -ForegroundColor Cyan
    $proc = Start-Process -FilePath $serverExe -ArgumentList $args -PassThru -WindowStyle Minimized -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    try {
        $proc.PriorityClass = "Idle"
    } catch {
        Write-Host "[local-ai] could not lower process priority; continuing." -ForegroundColor DarkYellow
    }

    try {
        $ready = $false
        $maxWaitSeconds = 600
        for ($i = 0; $i -lt $maxWaitSeconds; $i++) {
            if ($proc.HasExited) { break }
            if (Test-Health) {
                $ready = $true
                break
            }
            if (($i % 10) -eq 0) {
                $lastLine = ""
                if (Test-Path $stderr) {
                    $lastLine = Get-Content $stderr -Tail 1 -ErrorAction SilentlyContinue
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
            throw "Local SmolLM server did not become ready. Last server log: $tail"
        }

        Write-Host "Local server ready; testing the schema-constrained text selector..." -ForegroundColor Cyan
        & $mainPython -m lab.coach.local_probe --url $apiUrl --model $modelAlias --timeout 30
        if ($LASTEXITCODE -ne 0) {
            throw "Local SmolLM model probe failed. See $stderr"
        }
    }
    finally {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            $proc.WaitForExit()
        }
    }
}

[Environment]::SetEnvironmentVariable("LOCAL_COACH_MODEL", $modelAlias, "User")
[Environment]::SetEnvironmentVariable("LOCAL_COACH_URL", $apiUrl, "User")
$env:LOCAL_COACH_MODEL = $modelAlias
$env:LOCAL_COACH_URL = $apiUrl

Write-Host ""
Write-Host "LOCAL COACH READY." -ForegroundColor Green
Write-Host "Model: $modelAlias" -ForegroundColor Green
Write-Host "Endpoint: $apiUrl" -ForegroundColor Green
Write-Host "Gemini and the old SmolVLM vision model are not required." -ForegroundColor Green
Write-Host ""
Write-Host "Next command:" -ForegroundColor Cyan
Write-Host ".\run_autonomous_questing_windows.ps1" -ForegroundColor White
