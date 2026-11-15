@echo off
title Criterio - Sistema de Recomendacion
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0run.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ocurrio un error al ejecutar la aplicacion.
    pause
)
