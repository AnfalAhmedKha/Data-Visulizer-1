@echo off
title DataSci Studio Pro - One Click Setup
color 0D

echo.
echo ========================================
echo    DataSci Studio Pro
echo    One-Click Setup & Launch
echo ========================================
echo.

:: Check if dependencies are installed
python -c "import streamlit, pandas, sklearn" >nul 2>&1

if errorlevel 1 (
    echo [INFO] Dependencies not found. Installing now...
    echo.
    call requirements.bat
) else (
    echo [OK] Dependencies already installed.
)

echo.
echo Launching Dashboard...
echo.

python -m streamlit run dashboard.py

pause