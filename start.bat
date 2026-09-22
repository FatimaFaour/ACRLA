@echo off
title ACRLA - Learning Assistant
color 0A

echo.
echo  ================================================
echo   ACRLA - Adaptive Learning Assistant
echo   Powered by phi3:mini (Ollama - free, local)
echo  ================================================
echo.

set OLLAMA=ollama
set PYTHON=py -3.14
set PGBIN="C:\Program Files\PostgreSQL\17\bin"

REM Check Ollama
echo [1/4] Checking Ollama...
%OLLAMA% list >nul 2>&1
if errorlevel 1 (
    echo  Starting Ollama...
    start "" %OLLAMA% serve
    timeout /t 3 >nul
)
echo  OK - Ollama is running

REM Check phi3:mini
echo [2/4] Checking phi3:mini model...
%OLLAMA% list | findstr "phi3:mini" >nul 2>&1
if errorlevel 1 (
    echo  Downloading phi3:mini...
    %OLLAMA% pull phi3:mini
)
echo  OK - phi3:mini ready

REM Check nomic-embed-text
%OLLAMA% list | findstr "nomic-embed-text" >nul 2>&1
if errorlevel 1 (
    echo  Downloading nomic-embed-text...
    %OLLAMA% pull nomic-embed-text
)
echo  OK - Embeddings ready

REM Setup Python venv
echo [3/4] Setting up Python environment...
if not exist "backend\venv\Scripts\activate.bat" (
    echo  Creating virtual environment with Python 3.14...
    cd backend
    %PYTHON% -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    cd ..
) else (
    call backend\venv\Scripts\activate.bat
)
echo  OK - Python environment ready

REM Copy .env if missing
if not exist "backend\.env" (
    copy "backend\.env.example" "backend\.env"
    echo  Created backend\.env
)

REM Start backend
echo [4/4] Starting ACRLA backend...
echo.
echo  ================================================
echo   Backend:   http://localhost:8000
echo   API docs:  http://localhost:8000/docs
echo   Frontend:  open frontend\index.html in browser
echo  ================================================
echo.
echo  Press Ctrl+C to stop.
echo.

cd backend
call venv\Scripts\activate.bat
uvicorn main:app --reload --port 8000 --host 0.0.0.0
