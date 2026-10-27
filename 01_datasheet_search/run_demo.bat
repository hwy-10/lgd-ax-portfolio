@echo off
rem Datasheet search demo launcher.
rem Kept ASCII-only: a code page switch (chcp) inside a batch file breaks it
rem when the folder path contains non-ASCII characters.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or later is required: https://www.python.org/downloads/
  pause
  exit /b 1
)

rem The virtual environment goes to a short fixed path. Package files nested
rem under a deep extraction folder exceed the Windows 260-character path limit.
if "%LGD_AX_VENV%"=="" set "LGD_AX_VENV=%LOCALAPPDATA%\lgd_ax_venv"
set "PY=%LGD_AX_VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo [1/3] Creating virtual environment: %LGD_AX_VENV%
  python -m venv "%LGD_AX_VENV%" || goto :fail
)

echo [2/3] Installing packages
"%PY%" -m pip install -q -r requirements.txt || goto :fail

if not exist .env (
  copy .env.example .env >nul
  echo Created .env. Answers use a local model via Ollama: ollama pull gemma3:4b
  echo Without a model, search, blocking decisions and the results tab still work.
)

echo [3/3] Starting demo: http://localhost:8501
rem Streamlit runs headless, so the browser is opened once the server answers.
if "%LGD_AX_NO_BROWSER%"=="" start "" /b "%PY%" tools\open_browser.py
"%PY%" -m streamlit run app.py
exit /b 0

:fail
echo Failed. See the error above.
pause
exit /b 1
