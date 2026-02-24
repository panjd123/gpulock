#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/gpulock"
DST="/usr/local/bin/gpulock"

if [[ ! -x "${SRC}" ]]; then
  echo "missing executable: ${SRC}" >&2
  exit 1
fi

ln -sfn "${SRC}" "${DST}"
chmod +x "${SRC}"
echo "installed: ${DST} -> ${SRC}"

UNLOCK_DST="/usr/local/bin/gpuunlock"
ln -sfn "${SRC}" "${UNLOCK_DST}"
echo "installed: ${UNLOCK_DST} -> ${SRC}"
