@echo off
cd /d "%~dp0"
if not exist "reports\index.html" (
  ".venv\Scripts\python.exe" visualize.py
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
start "" "reports\index.html"
