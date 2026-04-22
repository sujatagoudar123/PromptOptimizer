@echo off
REM Local run script (Windows).
setlocal

pushd "%~dp0\.."

if not exist ".venv" (
    echo [run_local] Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo [run_local] Installing dependencies...
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

if not exist ".env" (
    if exist ".env.example" (
        echo [run_local] No .env found, copying .env.example
        copy /Y ".env.example" ".env" > nul
    )
)

echo [run_local] Starting LLM Optimization Gateway...
python -m llm_gateway.main

popd
endlocal
