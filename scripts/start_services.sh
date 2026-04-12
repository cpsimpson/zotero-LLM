#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

QDRANT_CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-zotero-llm-qdrant}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant}"
QDRANT_HOST_PORT="${QDRANT_HOST_PORT:-6333}"
QDRANT_STORAGE_DIR="${QDRANT_STORAGE_DIR:-${ROOT_DIR}/qdrant-storage}"

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_PID_FILE="${OLLAMA_PID_FILE:-${ROOT_DIR}/.ollama-serve.pid}"
OLLAMA_LOG_FILE="${OLLAMA_LOG_FILE:-${ROOT_DIR}/.ollama-serve.log}"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

wait_for_ollama() {
  local timeout_seconds=20
  local start_time
  start_time="$(date +%s)"
  while true; do
    if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start_time >= timeout_seconds )); then
      return 1
    fi
    sleep 1
  done
}

echo "Starting services for zotero-llm..."

if have_cmd docker; then
  mkdir -p "${QDRANT_STORAGE_DIR}"

  if docker ps --format '{{.Names}}' | grep -Fxq "${QDRANT_CONTAINER_NAME}"; then
    echo "Qdrant container '${QDRANT_CONTAINER_NAME}' is already running."
  elif docker ps -a --format '{{.Names}}' | grep -Fxq "${QDRANT_CONTAINER_NAME}"; then
    docker start "${QDRANT_CONTAINER_NAME}" >/dev/null
    echo "Started existing Qdrant container '${QDRANT_CONTAINER_NAME}'."
  else
    docker run -d \
      --name "${QDRANT_CONTAINER_NAME}" \
      -p "${QDRANT_HOST_PORT}:6333" \
      -v "${QDRANT_STORAGE_DIR}:/qdrant/storage" \
      "${QDRANT_IMAGE}" >/dev/null
    echo "Created and started Qdrant container '${QDRANT_CONTAINER_NAME}'."
  fi
  echo "Qdrant URL: http://localhost:${QDRANT_HOST_PORT}"
else
  echo "Docker not found; skipping Qdrant startup."
fi

if have_cmd ollama; then
  if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ollama server already running at ${OLLAMA_HOST}."
  else
    nohup ollama serve >"${OLLAMA_LOG_FILE}" 2>&1 &
    echo $! >"${OLLAMA_PID_FILE}"
    if wait_for_ollama; then
      echo "Started Ollama server at ${OLLAMA_HOST}."
    else
      echo "Started Ollama process, but server did not become healthy in time."
      echo "Check logs at: ${OLLAMA_LOG_FILE}"
    fi
  fi
else
  echo "Ollama CLI not found; skipping Ollama startup."
fi

echo "Done."
