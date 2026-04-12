#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

QDRANT_CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-zotero-llm-qdrant}"
OLLAMA_PID_FILE="${OLLAMA_PID_FILE:-${ROOT_DIR}/.ollama-serve.pid}"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

echo "Stopping services for zotero-llm..."

if [[ -f "${OLLAMA_PID_FILE}" ]]; then
  OLLAMA_PID="$(cat "${OLLAMA_PID_FILE}")"
  if [[ -n "${OLLAMA_PID}" ]] && kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
      kill -9 "${OLLAMA_PID}" >/dev/null 2>&1 || true
    fi
    echo "Stopped Ollama process ${OLLAMA_PID}."
  else
    echo "Ollama PID file existed, but process was not running."
  fi
  rm -f "${OLLAMA_PID_FILE}"
else
  echo "No managed Ollama PID file found; leaving any existing Ollama server as-is."
fi

if have_cmd docker; then
  if docker ps --format '{{.Names}}' | grep -Fxq "${QDRANT_CONTAINER_NAME}"; then
    docker stop "${QDRANT_CONTAINER_NAME}" >/dev/null
    echo "Stopped Qdrant container '${QDRANT_CONTAINER_NAME}'."
  else
    echo "Qdrant container '${QDRANT_CONTAINER_NAME}' is not running."
  fi
else
  echo "Docker not found; skipping Qdrant shutdown."
fi

echo "Done."
