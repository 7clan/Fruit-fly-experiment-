# setup_meta_model_api_windows.ps1
# One-time current Meta Model API setup for Muse Spark.
# Key is stored in the Windows USER environment and never committed.

$ErrorActionPreference = "Stop"

Write-Host "== DigitalFlyLab Meta Model API setup ==" -ForegroundColor Cyan
Write-Host "Provider: Meta Model API" -ForegroundColor Yellow
Write-Host "Default model: muse-spark-1.3" -ForegroundColor Yellow
Write-Host "The key is stored as MODEL_API_KEY in your Windows USER environment." -ForegroundColor DarkYellow
Write-Host ""
Write-Host "Opening current Meta Model API developer site..." -ForegroundColor Cyan
Start-Process "https://dev.meta.ai/"

$secure = Read-Host "Paste your NEW Meta Model API key here (input hidden)" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
}

if ([string]::IsNullOrWhiteSpace($key)) {
    throw "No Meta Model API key entered."
}

$model = Read-Host "Model [muse-spark-1.3]"
if ([string]::IsNullOrWhiteSpace($model)) {
    $model = "muse-spark-1.3"
}
$base = "https://api.meta.ai/v1"

[Environment]::SetEnvironmentVariable("MODEL_API_KEY", $key, "User")
[Environment]::SetEnvironmentVariable("META_MODEL", $model, "User")
[Environment]::SetEnvironmentVariable("META_MODEL_API_BASE", $base, "User")
$env:MODEL_API_KEY = $key
$env:META_MODEL = $model
$env:META_MODEL_API_BASE = $base

Write-Host ""
Write-Host "Testing key/model before saving the workflow..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m lab.coach.meta_probe --model $model --base-url $base --timeout 20
if ($LASTEXITCODE -ne 0) {
    throw "Meta Model API probe failed. The key/model was not usable."
}

Write-Host ""
Write-Host "Meta Model API coach configured." -ForegroundColor Green
Write-Host "Model: $model" -ForegroundColor Green
Write-Host "Base URL: $base" -ForegroundColor Green
Write-Host "The API key was NOT written to a project file." -ForegroundColor Green
Write-Host ""
Write-Host "Next command:" -ForegroundColor Cyan
Write-Host ".\run_autonomous_questing_windows.ps1" -ForegroundColor White
