@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 python -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" train_model.py --validate-only
if errorlevel 1 goto failed
".venv\Scripts\python.exe" train_model.py
if errorlevel 1 goto failed
echo Setup complete. Launch run_project.bat.
exit /b 0
:failed
echo Setup failed. Check Python 3.12 availability, package access, and dataset errors above.
pause
exit /b 1
