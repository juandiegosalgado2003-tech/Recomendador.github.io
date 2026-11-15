$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "                   CRITERIO                         " -ForegroundColor Yellow
Write-Host "  Sistema de Recomendacion por Afinidad de Gustos   " -ForegroundColor White
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    Write-Host "Iniciando servidor en http://127.0.0.1:8000 ..." -ForegroundColor Green
    Start-Process "http://127.0.0.1:8000"
    & $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
} else {
    Write-Host "Entorno virtual no encontrado en .venv. Probando python del sistema..." -ForegroundColor Yellow
    Start-Process "http://127.0.0.1:8000"
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
}
