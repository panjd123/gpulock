#!/usr/bin/env bash
# Install the gpulock Python package and the gpulock guard service.
#
# Both the package (with all guard dependencies) and the guard service are
# installed unconditionally. Use `gpulock service uninstall` to remove the
# service later if you change your mind.
#
# Usage:
#   ./install.sh                                # install + auto-detect backend
#   ./install.sh --gpu-ids 0,1                  # watch specific GPUs
#   ./install.sh --backend systemd-user
#   ./install.sh --backend supervisor
#   ./install.sh --no-start                     # install but don't start
#
# Environment overrides:
#   GPULOCK_INSTALLER=uv|pip|auto       # default auto: uv > pip
#   UV_LINK_MODE=copy                   # uv tool install link mode (default copy)
#   PYTHON_BIN=/path/to/python          # only used by the pip fallback
#   GPULOCK_SERVICE_BACKEND=auto|systemd-user|supervisor
#   GPULOCK_SERVICE_GPU_IDS="0,1"
#   GPULOCK_SERVICE_IDLE_TIMEOUT=5400
#   GPULOCK_SERVICE_NO_START=1
#   GPULOCK_SERVICE_NO_ENABLE=1
#   GPULOCK_SERVICE_NO_PLACEHOLDER_LOAD=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="${GPULOCK_INSTALLER:-auto}"

SERVICE_BACKEND="${GPULOCK_SERVICE_BACKEND:-auto}"
SERVICE_GPU_IDS="${GPULOCK_SERVICE_GPU_IDS:-}"
SERVICE_IDLE_TIMEOUT="${GPULOCK_SERVICE_IDLE_TIMEOUT:-}"
SERVICE_NO_START="${GPULOCK_SERVICE_NO_START:-0}"
SERVICE_NO_ENABLE="${GPULOCK_SERVICE_NO_ENABLE:-0}"
SERVICE_NO_PLACEHOLDER_LOAD="${GPULOCK_SERVICE_NO_PLACEHOLDER_LOAD:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)              SERVICE_BACKEND="$2"; shift ;;
    --backend=*)            SERVICE_BACKEND="${1#*=}" ;;
    --gpu-ids)              SERVICE_GPU_IDS="$2"; shift ;;
    --gpu-ids=*)            SERVICE_GPU_IDS="${1#*=}" ;;
    --idle-timeout)         SERVICE_IDLE_TIMEOUT="$2"; shift ;;
    --idle-timeout=*)       SERVICE_IDLE_TIMEOUT="${1#*=}" ;;
    --no-start)             SERVICE_NO_START=1 ;;
    --no-enable)            SERVICE_NO_ENABLE=1 ;;
    --no-placeholder-load)  SERVICE_NO_PLACEHOLDER_LOAD=1 ;;
    -h|--help)              sed -n '1,30p' "$0"; exit 0 ;;
    *)                      echo "[install.sh] unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

cd "${SCRIPT_DIR}"

# --- 1. install the python package ----------------------------------------
case "${INSTALLER}" in
  auto|uv) want_uv=1 ;;
  pip)     want_uv=0 ;;
  *) echo "[install.sh] invalid GPULOCK_INSTALLER: ${INSTALLER}" >&2; exit 2 ;;
esac

if (( want_uv )) && command -v uv >/dev/null 2>&1; then
  UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv tool install . --force
  echo "[install.sh] installed package with uv: gpulock"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "[install.sh] missing python interpreter: ${PYTHON_BIN}" >&2; exit 1
  }
  "${PYTHON_BIN}" -m pip install --user .
  echo "[install.sh] installed package with pip: gpulock"
fi

# --- 2. install the guard service -----------------------------------------
command -v gpulock >/dev/null 2>&1 || {
  echo "[install.sh] gpulock binary not in PATH after install; service step skipped." >&2
  echo "[install.sh] add ~/.local/bin (pip --user) or your uv tool path to PATH and re-run." >&2
  exit 1
}

SERVICE_ARGS=("install" "--backend" "${SERVICE_BACKEND}")
[[ -n "${SERVICE_GPU_IDS}"      ]] && SERVICE_ARGS+=("--gpu-ids" "${SERVICE_GPU_IDS}")
[[ -n "${SERVICE_IDLE_TIMEOUT}" ]] && SERVICE_ARGS+=("--idle-timeout" "${SERVICE_IDLE_TIMEOUT}")
[[ "${SERVICE_NO_START}"             == "1" || "${SERVICE_NO_START}"             == "true" ]] && SERVICE_ARGS+=("--no-start")
[[ "${SERVICE_NO_ENABLE}"            == "1" || "${SERVICE_NO_ENABLE}"            == "true" ]] && SERVICE_ARGS+=("--no-enable")
[[ "${SERVICE_NO_PLACEHOLDER_LOAD}"  == "1" || "${SERVICE_NO_PLACEHOLDER_LOAD}"  == "true" ]] && SERVICE_ARGS+=("--no-placeholder-load")

echo "[install.sh] gpulock service ${SERVICE_ARGS[*]}"
gpulock service "${SERVICE_ARGS[@]}"
