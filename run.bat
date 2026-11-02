@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 goto failure
)
fc /b requirements.txt .venv\installed-requirements.txt >nul 2>&1
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto failure
  copy /y requirements.txt .venv\installed-requirements.txt >nul
)
.venv\Scripts\python.exe -m streamlit run app.py --server.headless false
goto :eof
:failure
echo Установите Python 3.12 с python.org и проверьте подключение к интернету.
pause
