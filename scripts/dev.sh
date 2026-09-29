#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export VITE_PROXY_TARGET="${VITE_PROXY_TARGET:-http://${BACKEND_HOST}:${BACKEND_PORT}}"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ ! -f "$ROOT_DIR/frontend/package.json" ]]; then
  echo "frontend/package.json not found: $ROOT_DIR/frontend" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
echo "Starting backend: http://${BACKEND_HOST}:${BACKEND_PORT}"
"$PYTHON_BIN" -m uvicorn api.main:app \
  --reload \
  --host "$BACKEND_HOST" \
  --port "$BACKEND_PORT" &
BACKEND_PID=$!

cd "$ROOT_DIR/frontend"
echo "Starting frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
