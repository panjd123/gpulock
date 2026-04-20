#!/usr/bin/env bash
# Install the gpulock Python package and write a default guard service
# config (`gpulock service install --no-start`). The guard service is NOT
# started.
#
# After this script:
#   gpulock service config show              # see the default config
#   gpulock service config set gpu_ids=0,1   # tweak which GPUs to watch
#   gpulock service start                    # start the guard daemon
#
# Usage:
#   ./install.sh
#
# Environment overrides (only for the package install step):
#   GPULOCK_INSTALLER=uv|pip|auto       # default: auto (uv > pip)
#   UV_LINK_MODE=copy                   # kept for compatibility; ignored in editable installs
#   PYTHON_BIN=/path/to/python          # only used by the pip fallback
set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "[install.sh] this script takes no arguments." >&2
  echo "[install.sh] use \`gpulock service ...\` to configure / start the guard." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="${GPULOCK_INSTALLER:-auto}"

case "${INSTALLER}" in
  auto|uv) want_uv=1 ;;
  pip)     want_uv=0 ;;
  *) echo "[install.sh] invalid GPULOCK_INSTALLER: ${INSTALLER}" >&2; exit 2 ;;
esac

cd "${SCRIPT_DIR}"

# --- 1. install the python package ----------------------------------------
# Use editable installs for repo-local setup. `uv tool install .` has shown
# stale-build behavior with local paths; editable mode keeps the installed
# entrypoint bound to the current checkout.
if (( want_uv )) && command -v uv >/dev/null 2>&1; then
  UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv tool install -e . --force --reinstall --refresh
  echo "[install.sh] installed package with uv: gpulock"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "[install.sh] missing python interpreter: ${PYTHON_BIN}" >&2; exit 1
  }
  "${PYTHON_BIN}" -m pip install --user --editable .
  echo "[install.sh] installed package with pip: gpulock"
fi

command -v gpulock >/dev/null 2>&1 || {
  echo "[install.sh] gpulock binary not in PATH after install; service step skipped." >&2
  echo "[install.sh] add ~/.local/bin (pip --user) or your uv tool path to PATH and re-run." >&2
  exit 1
}

# --- 2. write the default guard service config (do NOT start) -------------
gpulock service install --no-start

cat <<'EOF'

[install.sh] done. next steps:
  gpulock service config show                         # inspect default config
  gpulock service config set gpu_ids=0,1              # pick GPUs to watch
  gpulock service config set idle_timeout=5400        # tweak any other field
  gpulock service start                               # start the guard daemon
EOF
