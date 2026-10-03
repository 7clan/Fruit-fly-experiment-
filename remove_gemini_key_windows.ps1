# remove_gemini_key_windows.ps1
# Removes every local Gemini credential used by DigitalFlyLab on this PC.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$secretPath = Join-Path $root "runtime_state\gemini_key.dpapi"

Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $null, "User")
[Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", $null, "User")

if (Test-Path $secretPath) {
    Remove-Item $secretPath -Force
}

Write-Host "DigitalFlyLab Gemini key removed from this process, USER environment, and local DPAPI secret." -ForegroundColor Green
Write-Host "This does not revoke the key at Google; revoke/delete it in AI Studio too if you are done with it." -ForegroundColor Yellow
