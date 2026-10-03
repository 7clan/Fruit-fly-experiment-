# setup_semantic_coach_windows.ps1
# One-time Gemini API-key setup for AI-only mode.
# The secret is NEVER written to git. It is protected with Windows DPAPI
# (current-user scope) under gitignored runtime_state/ and also mirrored into
# the current process/user environment for compatibility with the Python code.

param(
    [switch]$ReplaceKey
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
$mainPython = ".\.venv\Scripts\python.exe"
$secretDir = Join-Path $root "runtime_state"
$secretPath = Join-Path $secretDir "gemini_key.dpapi"

Write-Host "== DigitalFlyLab Gemini key setup ==" -ForegroundColor Cyan
Write-Host "The key stays LOCAL to this Windows user; it is never committed to GitHub." -ForegroundColor Yellow
Write-Host "New AI Studio keys may begin with AQ. (authorization key) or AIza (legacy/standard key)." -ForegroundColor DarkYellow

function Normalize-GeminiKey([string]$Value) {
    if ($null -eq $Value) { return "" }
    $k = [string]$Value
    $k = $k.Trim()

    foreach ($prefix in @(
        "GEMINI_API_KEY=",
        "GOOGLE_API_KEY=",
        "x-goog-api-key:"
    )) {
        if ($k.StartsWith(
                $prefix,
                [System.StringComparison]::OrdinalIgnoreCase)) {
            $k = $k.Substring($prefix.Length).Trim()
            break
        }
    }

    if ($k.Length -ge 2) {
        $first = $k.Substring(0, 1)
        $last = $k.Substring($k.Length - 1, 1)
        if (($first -eq '"' -and $last -eq '"') -or
            ($first -eq "'" -and $last -eq "'")) {
            $k = $k.Substring(1, $k.Length - 2).Trim()
        }
    }

    # String overloads avoid the PowerShell 5.1 char/string Replace bug that
    # previously crashed after a correct hidden paste.
    return $k.Replace([string][char]13, "").Replace([string][char]10, "").Trim()
}

function Test-GeminiKeyShape([string]$Value) {
    $k = Normalize-GeminiKey $Value
    if ([string]::IsNullOrWhiteSpace($k)) { return $false }
    if ($k.Length -lt 20) { return $false }
    return ($k.StartsWith("AQ.") -or $k.StartsWith("AIza"))
}

function Read-HiddenGeminiKey {
    $secure = Read-Host "Paste the COMPLETE Gemini API key (hidden)" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        return Normalize-GeminiKey (
            [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr))
    }
    finally {
        if ($ptr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
        }
    }
}

function Save-DpapiSecret([string]$Key) {
    New-Item -ItemType Directory -Path $secretDir -Force | Out-Null
    $secure = ConvertTo-SecureString $Key -AsPlainText -Force
    $protected = ConvertFrom-SecureString $secure
    [IO.File]::WriteAllText(
        $secretPath,
        $protected,
        [Text.Encoding]::UTF8)
}

$key = ""

# Fast path: if the user copied the key from AI Studio, consume it directly.
try {
    $clip = [string](Get-Clipboard -Raw)
    $candidate = Normalize-GeminiKey $clip
    if (Test-GeminiKeyShape $candidate) {
        $key = $candidate
        Write-Host "Found a complete Gemini key on the clipboard." -ForegroundColor Green
    }
}
catch {}

if (-not (Test-GeminiKeyShape $key)) {
    Write-Host ""
    Write-Host "Copy the COMPLETE key from Google AI Studio, then paste it once below." -ForegroundColor Yellow
    $key = Read-HiddenGeminiKey
}

try { Set-Clipboard -Value "" } catch {}

if (-not (Test-GeminiKeyShape $key)) {
    throw "The value is not a complete Gemini API key. Nothing was saved."
}

$keyKind = if ($key.StartsWith("AQ.")) {
    "authorization key"
} else {
    "standard/legacy key"
}
Write-Host "Key shape accepted: $keyKind, length=$($key.Length) (secret not printed)." -ForegroundColor DarkCyan

$model = $env:GEMINI_MODEL
if ([string]::IsNullOrWhiteSpace($model)) {
    $model = [Environment]::GetEnvironmentVariable("GEMINI_MODEL", "User")
}
if ([string]::IsNullOrWhiteSpace($model)) {
    $model = "gemini-3.5-flash-lite"
}

# Test in THIS process before persisting anything.
$env:GEMINI_API_KEY = $key
$env:GEMINI_MODEL = $model

if (Test-Path $mainPython) {
    Write-Host "Testing Gemini key + image vision before saving..." -ForegroundColor Cyan
    $probeArgs = @(
        "-m", "lab.coach.probe",
        "--model", $model,
        "--timeout", "18",
        "--require-vision"
    )
    $probe = @(& $mainPython @probeArgs)
    $probeExit = $LASTEXITCODE
    $probe | ForEach-Object { Write-Host $_ }
    if ($probeExit -ne 0) {
        Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
        throw "Gemini rejected this key/model. Nothing was saved. If this key was exposed, create a fresh auth key in AI Studio and retry."
    }

    $modelLine = $probe |
        Where-Object { $_ -like "COACH_MODEL=*" } |
        Select-Object -Last 1
    if (-not [string]::IsNullOrWhiteSpace($modelLine)) {
        $model = $modelLine.Substring("COACH_MODEL=".Length).Trim()
        $env:GEMINI_MODEL = $model
    }
}

Save-DpapiSecret $key
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $key, "User")
[Environment]::SetEnvironmentVariable("GEMINI_MODEL", $model, "User")

Write-Host ""
Write-Host "Gemini is integrated locally." -ForegroundColor Green
Write-Host "Encrypted local secret: runtime_state\gemini_key.dpapi (gitignored, current Windows user only)." -ForegroundColor Green
Write-Host "Model: $model" -ForegroundColor Green
Write-Host "The key was NOT written into source code or committed to GitHub." -ForegroundColor Green
Write-Host ""
Write-Host "Run:" -ForegroundColor Cyan
Write-Host ".\run_ai_only_windows.ps1" -ForegroundColor White
Write-Host ""
Write-Host "When finished permanently, remove it with:" -ForegroundColor DarkYellow
Write-Host ".\remove_gemini_key_windows.ps1" -ForegroundColor White
