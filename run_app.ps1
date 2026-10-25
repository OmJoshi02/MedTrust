# MedTrust V1 Clinical System Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "MEDTRUST V1 - TRUST-AWARE EXPLAINABLE AI" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Start Python GPU Inference API from backend/ in background
Write-Host "[1/2] Starting MedTrust GPU Inference API (http://localhost:8000)..." -ForegroundColor Green
$serverJob = Start-Process -FilePath "python" -ArgumentList "backend/server.py" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

# 2. Start React + TypeScript Frontend
Write-Host "[2/2] Starting React + TypeScript Frontend..." -ForegroundColor Green
Set-Location -Path "frontend"
cmd /c npm run dev

# Cleanup when user stops frontend
if ($serverJob -and -not $serverJob.HasExited) {
    Stop-Process -Id $serverJob.Id -Force
}
