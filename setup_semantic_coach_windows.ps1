# setup_semantic_coach_windows.ps1
# One-time Gemini API-key setup for the low-rate semantic coach.
# The key is stored in the current Windows USER environment, never in git.

param(
    [switch]$ReplaceKey
)

$ErrorActionPreference = "Stop"

Write-Host "== DigitalFlyLab semantic coach setup ==" -ForegroundColor Cyan
Write-Host "Provider: Google Gemini API" -ForegroundColor Yellow
Write-Host "Default model: gemini-3.5-flash-lite (stable, multimodal, lower-cost current model)." -ForegroundColor Yellow
Write-Host "Gemini 3.5 Flash-Lite is the preferred low-latency multimodal model for new projects." -ForegroundColor DarkYellow
Write-Host "The key is saved as a USER environment variable, not in the repository." -ForegroundColor DarkYellow
Write-Host ""
Write-Host "Opening Google AI Studio API-key page..." -ForegroundColor Cyan
Start-Process "https://aistudio.google.com/apikey"

function Normalize-GeminiKey([string]$Value) {
    if ($null -eq $Value) { return "" }
    $k = $Value.Trim()
    foreach ($prefix in @("GEMINI_API_KEY=", "GOOGLE_API_KEY=", "x-goog-api-key:")) {
        if ($k.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            $k = $k.Substring($prefix.Length).Trim()
            break
        }
    }
    if ($k.Length -ge 2) {
        $first = $k.Substring(0,1)
        $last = $k.Substring($k.Length-1,1)
        if (($first -eq [char]34 -and $last -eq [char]34) -or
            ($first -eq [char]39 -and $last -eq [char]39)) {
            $k = $k.Substring(1, $k.Length-2).Trim()
        }
    }
    return $k.Replace([char]13, "").Replace([char]10, "").Trim()
}

function Test-GeminiKeyShape([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return $false }
    if ($Value.Length -lt 20) { return $false }
    return ($Value.StartsWith("AIza") -or $Value.StartsWith("AQ."))
}

Write-Host ""
Write-Host "In AI Studio, COPY the complete API key." -ForegroundColor Yellow
Write-Host "Then come back here and press Enter. I will read it from the clipboard, save it securely as a USER environment variable, and clear the clipboard." -ForegroundColor DarkYellow
[void](Read-Host "Press Enter after copying the full key")

$key = ""
try {
    $key = Normalize-GeminiKey ([string](Get-Clipboard -Raw))
}
catch {
    $key = ""
}
try { Set-Clipboard -Value "" } catch {}

if (-not (Test-GeminiKeyShape $key)) {
    Write-Host "Clipboard did not contain a complete Gemini key. Falling back to hidden paste." -ForegroundColor Yellow
    $secure = Read-Host "Paste the COMPLETE Gemini API key here (input hidden)" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $key = Normalize-GeminiKey ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr))
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

if (-not (Test-GeminiKeyShape $key)) {
    throw "The value is not a complete Gemini API key. Expected a full AI Studio key beginning with AIza or AQ. (not '*', not a masked value, and not a one-character placeholder). Nothing was saved."
}

# The key was already normalized and validated above.
$keyKind = if ($key.StartsWith("AIza")) {
    "Google standard API key"
} elseif ($key.StartsWith("AQ.")) {
    "Google authorization API key"
} else {
    "unrecognized key format"
}
Write-Host "Key shape: $keyKind, length=$($key.Length) (secret not printed)." -ForegroundColor DarkCyan

$model = Read-Host "Model [gemini-3.5-flash-lite]"
if ([string]::IsNullOrWhiteSpace($model)) {
    $model = "gemini-3.5-flash-lite"
}

[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $key, "User")
[Environment]::SetEnvironmentVariable("GEMINI_MODEL", $model, "User")
$env:GEMINI_API_KEY = $key
$env:GEMINI_MODEL = $model

Write-Host ""
Write-Host "Semantic coach configured for this PowerShell and future terminals." -ForegroundColor Green
Write-Host "Model: $model" -ForegroundColor Green
Write-Host "The API key was NOT written to a project file." -ForegroundColor Green
Write-Host ""
Write-Host "Next command:" -ForegroundColor Cyan
Write-Host ".\run_ai_only_windows.ps1" -ForegroundColor White
