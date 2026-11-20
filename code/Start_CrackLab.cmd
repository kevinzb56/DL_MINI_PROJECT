@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the Python environment first. See docs\PHASE_6.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\run_inspection.py
if errorlevel 1 pause
