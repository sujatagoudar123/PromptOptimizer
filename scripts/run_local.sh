#!/usr/bin/env bash
# Local run script (Linux/macOS).
#
# Creates a venv if missing, installs dependencies, loads .env if present,
# and starts the gateway on http://localhost:8080.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  echo "[run_local] Creating virtual environment..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo "[run_local] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  echo "[run_local] No .env found — copying .env.example"
  cp .env.example .env
fi

echo "[run_local] Starting LLM Optimization Gateway..."
exec python -m llm_gateway.main
