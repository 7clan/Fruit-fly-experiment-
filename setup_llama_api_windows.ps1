# setup_llama_api_windows.ps1
# One-time Meta Llama API setup for the cloud semantic coach.
# The key is stored in the current Windows USER environment, never in git.

$ErrorActionPreference = "Stop"

Write-Host "== DigitalFlyLab Meta Llama API setup ==" -ForegroundColor Cyan
Write-Host "Provider: Meta Llama API (OpenAI-compatible endpoint)" -ForegroundColor Yellow
Write-Host "Preferred model: Llama 4 Maverick; Scout is automatic fallback." -ForegroundColor Yellow
Write-Host "The key is saved as a USER environment variable, not in the repository." -ForegroundColor DarkYellow
Write-Host ""
Write-Host "Opening Meta Llama developer portal..." -ForegroundColor Cyan
Start-Process "https://llama.developer.meta.com/"

$secure = Read-Host "Paste your Llama API key here (input hidden)" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
}

if ([string]::IsNullOrWhiteSpace($key)) {
    throw "No Llama API key entered."
}

$model = Read-Host "Model [AUTO: Maverick -> Scout]"
if ([string]::IsNullOrWhiteSpace($model)) {
    $model = "auto"
}

$base = "https://api.llama.com/compat/v1"

[Environment]::SetEnvironmentVariable("LLAMA_API_KEY", $key, "User")
[Environment]::SetEnvironmentVariable("LLAMA_MODEL", $model, "User")
[Environment]::SetEnvironmentVariable("LLAMA_API_BASE", $base, "User")
$env:LLAMA_API_KEY = $key
$env:LLAMA_MODEL = $model
$env:LLAMA_API_BASE = $base

Write-Host ""
Write-Host "Llama API coach configured for this PowerShell and future terminals." -ForegroundColor Green
Write-Host "Model preference: $model" -ForegroundColor Green
Write-Host "Base URL: $base" -ForegroundColor Green
Write-Host "The API key was NOT written to a project file." -ForegroundColor Green
Write-Host ""
Write-Host "Next command:" -ForegroundColor Cyan
Write-Host ".\run_autonomous_questing_windows.ps1" -ForegroundColor White
