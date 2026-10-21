@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: py -3 -m venv .venv
  echo Then: .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8000
.venv\Scripts\python -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
