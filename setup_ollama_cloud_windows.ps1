param([switch]$ReplaceKey)

# setup_ollama_cloud_windows.ps1
# One-time Ollama Cloud API setup.
# Stores the API key in the Windows USER environment, never in git.

$ErrorActionPreference = "Stop"

Write-Host "== DigitalFlyLab Ollama Cloud setup ==" -ForegroundColor Cyan
Write-Host "Provider: Ollama Cloud" -ForegroundColor Yellow
Write-Host "Preferred model: gemma4:31b (included free usage)" -ForegroundColor Yellow
Write-Host "The key is stored in your Windows USER environment, not the repository." -ForegroundColor DarkYellow
Write-Host ""

if ([string]::IsNullOrWhiteSpace($env:OLLAMA_API_KEY)) {
    $env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable("OLLAMA_API_KEY", "User")
}

if ($ReplaceKey -or [string]::IsNullOrWhiteSpace($env:OLLAMA_API_KEY)) {
    $secure = Read-Host "Paste your Ollama API key here (input hidden)" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
    if ([string]::IsNullOrWhiteSpace($key)) {
        throw "No Ollama API key entered."
    }
    [Environment]::SetEnvironmentVariable("OLLAMA_API_KEY", $key, "User")
    $env:OLLAMA_API_KEY = $key
}

$model = [Environment]::GetEnvironmentVariable("OLLAMA_CLOUD_MODEL", "User")
$staleModels = @(
    "qwen3-vl:235b-cloud", "qwen3-vl:235b", "qwen3.5", "qwen3.6",
    "qwen3.8", "kimi-k3"
)
if ($ReplaceKey -or [string]::IsNullOrWhiteSpace($model) -or $staleModels -contains $model) {
    $model = "gemma4:31b"
    [Environment]::SetEnvironmentVariable("OLLAMA_CLOUD_MODEL", $model, "User")
}
$env:OLLAMA_CLOUD_MODEL = $model

$base = "https://ollama.com/api"
[Environment]::SetEnvironmentVariable("OLLAMA_CLOUD_BASE", $base, "User")
$env:OLLAMA_CLOUD_BASE = $base

Write-Host ""
Write-Host "Ollama Cloud coach configured." -ForegroundColor Green
Write-Host "Model: $model" -ForegroundColor Green
Write-Host "Base URL: $base" -ForegroundColor Green
Write-Host "The API key was NOT written to a project file." -ForegroundColor Green
Write-Host ""
Write-Host "Next command:" -ForegroundColor Cyan
Write-Host ".\run_autonomous_questing_windows.ps1" -ForegroundColor White
