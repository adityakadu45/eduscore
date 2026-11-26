@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup_required
".venv\Scripts\python.exe" -c "import streamlit, pandas, numpy, sklearn, plotly, matplotlib, seaborn, joblib"
if errorlevel 1 goto setup_required
for %%F in ("models\best_model.joblib" "models\candidate_models.joblib" "models\model_metadata.json" "results\model_comparison.csv" "results\classification_reports.json" "results\evaluation_results.json") do (
    if not exist %%F goto train
)
goto serve
:train
".venv\Scripts\python.exe" train_model.py
if errorlevel 1 goto failed
:serve
".venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost --server.headless false
if errorlevel 1 goto failed
exit /b 0
:setup_required
echo Run setup.bat to create the virtual environment and install dependencies.
pause
exit /b 1
:failed
echo EduScore could not start. See the error above.
pause
exit /b 1
