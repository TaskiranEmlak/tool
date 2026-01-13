@echo off
title Trading Brain Dashboard

echo.
echo ========================================
echo   Trading Brain Dashboard Baslatiliyor
echo ========================================
echo.

REM Conda environment aktive et
call C:\Users\Windows\anaconda3\Scripts\activate.bat base

REM Proje klasorune git
cd /d "C:\Users\Windows\Desktop\kriptol\web"

echo Tarayicida ac: http://localhost:8000
echo.

REM Tarayiciyi ac
timeout /t 2 /nobreak >nul
start "" "http://localhost:8000"

REM Dashboard baslat
python brain_web.py

pause
