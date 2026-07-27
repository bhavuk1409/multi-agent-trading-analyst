#!/usr/bin/env bash
# ============================================================
# NEXUS — Multi-Agent Trading Analyst: Quick Start
# ============================================================
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# This script:
#   1. Starts the Python API server on port 8000
#   2. Starts the Vite dev server (React frontend) on port 5174
#   3. Opens http://localhost:5174 in your browser
#   4. On Ctrl-C, kills both servers cleanly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Colours ──────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RESET="\033[0m"

echo -e "${CYAN}"
echo "  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗"
echo "  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝"
echo "  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗"
echo "  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║"
echo "  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║"
echo "  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝"
echo -e "${RESET}"
echo "  Multi-Agent LLM Trading Analyst"
echo ""

# ─── Check .env ───────────────────────────────────────────
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
  echo -e "${YELLOW}⚠  .env not found — copying .env.example → .env${RESET}"
  cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
  echo -e "${YELLOW}   Edit .env and add your GROQ_API_KEY before running again.${RESET}"
  exit 1
fi

if ! grep -q "^GROQ_API_KEY=.\+" "${SCRIPT_DIR}/.env" 2>/dev/null; then
  echo -e "${YELLOW}⚠  GROQ_API_KEY is not set in .env${RESET}"
  echo "   Get a free key at https://console.groq.com and add it to .env"
  exit 1
fi

# ─── Locate Python virtualenv ─────────────────────────────
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
  echo "Virtual environment not found at .venv/"
  echo "Create it with:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# ─── Start API server ─────────────────────────────────────
echo -e "${GREEN}▶  Starting API server on http://127.0.0.1:8000 …${RESET}"
"$PYTHON" "${SCRIPT_DIR}/api_server.py" 8000 &
API_PID=$!

# ─── Start frontend dev server ────────────────────────────
echo -e "${GREEN}▶  Starting React frontend on http://localhost:5174 …${RESET}"
cd "${SCRIPT_DIR}/frontend"
npm run dev &
FRONTEND_PID=$!
cd "${SCRIPT_DIR}"

# ─── Cleanup on exit ──────────────────────────────────────
cleanup() {
  echo -e "\n${YELLOW}Shutting down …${RESET}"
  kill "$API_PID"      2>/dev/null || true
  kill "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo ""
echo -e "${CYAN}  ✓ NEXUS is running at http://localhost:5174${RESET}"
echo "  Press Ctrl-C to stop."
echo ""

# Wait for either process to exit
wait "$API_PID" "$FRONTEND_PID"
