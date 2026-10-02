# setup_semantic_coach_windows.ps1
# One-time Gemini API-key setup for the low-rate semantic coach.
# The key is stored in the current Windows USER environment, never in git.

$ErrorActionPreference = "Stop"

Write-Host "== DigitalFlyLab semantic coach setup ==" -ForegroundColor Cyan
Write-Host "Provider: Google Gemini API" -ForegroundColor Yellow
Write-Host "Default model: gemini-3.5-flash-lite (stable, multimodal, lower-cost current model)." -ForegroundColor Yellow
Write-Host "Gemini 3.5 Flash-Lite is the preferred low-latency multimodal model for new projects." -ForegroundColor DarkYellow
Write-Host "The key is saved as a USER environment variable, not in the repository." -ForegroundColor DarkYellow
Write-Host ""
Write-Host "Opening Google AI Studio API-key page..." -ForegroundColor Cyan
Start-Process "https://aistudio.google.com/apikey"

$secure = Read-Host "Paste your Gemini API key here (input hidden)" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
}

if ([string]::IsNullOrWhiteSpace($key)) {
    throw "No API key entered."
}

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
