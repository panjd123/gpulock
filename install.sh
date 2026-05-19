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
#   GPULOCK_INSTALLER=uv|pip|auto           # default: auto (uv > pip)
#   GPULOCK_TORCH_CANDIDATES="2.9.1:cu129 ..." # ordered torch CUDA wheel fallbacks
#   GPULOCK_TORCH_VERSION=2.9.1             # force one torch version
#   GPULOCK_TORCH_BACKEND=cu129             # force one CUDA wheel backend
#   UV_LINK_MODE=copy                       # kept for compatibility; ignored in editable installs
#   PYTHON_BIN=/path/to/python              # only used by the pip fallback
set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "[install.sh] this script takes no arguments." >&2
  echo "[install.sh] use \`gpulock service ...\` to configure / start the guard." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="${GPULOCK_INSTALLER:-auto}"
FALLBACK_TORCH_CANDIDATES=(
  "2.9.1:cu129"
  "2.9.1:cu130"
  "2.9.1:cu128"
  "2.7.1:cu128"
  "2.6.0:cu126"
  "2.5.1:cu124"
  "2.0.1:cu121"
  "2.7.1:cu118"
)

case "${INSTALLER}" in
  auto|uv) want_uv=1 ;;
  pip)     want_uv=0 ;;
  *) echo "[install.sh] invalid GPULOCK_INSTALLER: ${INSTALLER}" >&2; exit 2 ;;
esac

cd "${SCRIPT_DIR}"

detect_max_compute_cap() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1

  nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null |
    awk '
      /^[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*$/ {
        gsub(/[[:space:]]/, "", $0)
        split($0, parts, ".")
        major = parts[1] + 0
        minor = parts[2] + 0
        score = major * 100 + minor
        if (score > best) {
          best = score
          cap = major "." minor
        }
      }
      END {
        if (best > 0) print cap
      }
    '
}

torch_candidates_for_cap() {
  local cap="$1"
  local major="${cap%%.*}"

  if [[ -z "${cap}" || "${major}" == "${cap}" ]]; then
    printf '%s\n' "${FALLBACK_TORCH_CANDIDATES[@]}"
    return
  fi

  # Blackwell / GB200 (sm100+) needs PyTorch wheels with sm_100 kernels.
  if (( major >= 10 )); then
    printf '%s\n' \
      "2.9.1:cu129" \
      "2.9.1:cu130" \
      "2.9.1:cu128" \
      "2.8.0:cu129" \
      "2.8.0:cu128"
    return
  fi

  # Hopper (sm90) is well covered by CUDA 12.4 wheels and newer.
  if (( major == 9 )); then
    printf '%s\n' \
      "2.5.1:cu124" \
      "2.6.0:cu126" \
      "2.7.1:cu128" \
      "2.9.1:cu128" \
      "2.9.1:cu129" \
      "2.9.1:cu130"
    return
  fi

  # Ada / Ampere and older do not need Blackwell-only wheels.
  if (( major == 8 )); then
    printf '%s\n' \
      "2.5.1:cu124" \
      "2.6.0:cu126" \
      "2.7.1:cu128" \
      "2.9.1:cu128" \
      "2.0.1:cu121" \
      "2.7.1:cu118"
    return
  fi

  printf '%s\n' "${FALLBACK_TORCH_CANDIDATES[@]}"
}

python_for_gpulock_entrypoint() {
  local entrypoint shebang
  entrypoint="$(command -v gpulock || true)"
  [[ -n "${entrypoint}" && -f "${entrypoint}" ]] || return 1
  IFS= read -r shebang <"${entrypoint}" || return 1
  [[ "${shebang}" == '#!'* ]] || return 1
  printf '%s\n' "${shebang#\#!}"
}

validate_installed_torch_arch() {
  local cap="$1"
  local py="${2:-}"
  [[ -n "${cap}" ]] || return 0

  local major="${cap%%.*}"
  local minor="${cap#*.}"
  [[ -n "${major}" && "${major}" != "${cap}" ]] || return 0

  local expected_arch="sm_${major}${minor}"
  if [[ -z "${py}" ]]; then
    py="$(python_for_gpulock_entrypoint)" || {
      echo "[install.sh] could not resolve gpulock entrypoint Python for torch validation" >&2
      return 1
    }
  fi

  echo "[install.sh] validating installed torch supports ${expected_arch}"
  "${py}" -c '
import sys
expected = sys.argv[1]
try:
    import torch
except Exception as exc:
    print(f"failed to import torch: {exc}", file=sys.stderr)
    raise SystemExit(1)
arches = list(getattr(torch.cuda, "get_arch_list", lambda: [])())
print(f"[install.sh] installed torch={torch.__version__} cuda={torch.version.cuda} arches={arches}")
if expected not in arches:
    print(f"missing required CUDA arch {expected}", file=sys.stderr)
    raise SystemExit(1)
' "${expected_arch}"
}

detected_cap="$(detect_max_compute_cap || true)"
if [[ -n "${detected_cap}" ]]; then
  echo "[install.sh] detected max NVIDIA compute capability: ${detected_cap}"
fi

if [[ -n "${GPULOCK_TORCH_VERSION:-}" || -n "${GPULOCK_TORCH_BACKEND:-}" ]]; then
  TORCH_VERSION="${GPULOCK_TORCH_VERSION:-2.9.1}"
  TORCH_BACKEND="${GPULOCK_TORCH_BACKEND:-cu129}"
  TORCH_CANDIDATES=("${TORCH_VERSION}:${TORCH_BACKEND}")
elif [[ -n "${GPULOCK_TORCH_CANDIDATES:-}" ]]; then
  read -r -a TORCH_CANDIDATES <<<"${GPULOCK_TORCH_CANDIDATES}"
else
  if [[ -n "${detected_cap}" ]]; then
    mapfile -t TORCH_CANDIDATES < <(torch_candidates_for_cap "${detected_cap}")
  else
    echo "[install.sh] could not detect NVIDIA compute capability; using conservative torch candidates" >&2
    TORCH_CANDIDATES=("${FALLBACK_TORCH_CANDIDATES[@]}")
  fi
fi

try_uv_install() {
  local version="$1"
  local backend="$2"

  echo "[install.sh] trying uv install with torch ${version} (${backend})"
  UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv tool install -e . \
    --force --reinstall --refresh \
    --with "torch==${version}" \
    --torch-backend "${backend}"
}

try_pip_install_torch() {
  local version="$1"
  local backend="$2"
  local index_url="https://download.pytorch.org/whl/${backend}"

  echo "[install.sh] trying pip install with torch ${version} from ${index_url}"
  "${PYTHON_BIN}" -m pip install --user --force-reinstall \
    "torch==${version}" \
    --index-url "${index_url}"
}

# --- 1. install the python package ----------------------------------------
# Use editable installs for repo-local setup. `uv tool install .` has shown
# stale-build behavior with local paths; editable mode keeps the installed
# entrypoint bound to the current checkout.
if (( want_uv )) && command -v uv >/dev/null 2>&1; then
  installed_torch=""
  for candidate in "${TORCH_CANDIDATES[@]}"; do
    IFS=: read -r TORCH_VERSION TORCH_BACKEND <<<"${candidate}"
    if [[ -z "${TORCH_VERSION}" || -z "${TORCH_BACKEND}" ]]; then
      echo "[install.sh] invalid torch candidate: ${candidate}" >&2
      exit 2
    fi
    if try_uv_install "${TORCH_VERSION}" "${TORCH_BACKEND}" &&
       validate_installed_torch_arch "${detected_cap:-}"; then
      installed_torch="${TORCH_VERSION} (${TORCH_BACKEND})"
      break
    fi
    echo "[install.sh] torch ${TORCH_VERSION} (${TORCH_BACKEND}) is unavailable or incompatible here; trying next candidate." >&2
  done
  if [[ -z "${installed_torch}" ]]; then
    echo "[install.sh] failed to install torch from candidates: ${TORCH_CANDIDATES[*]}" >&2
    exit 1
  fi
  echo "[install.sh] installed package with uv: gpulock"
  echo "[install.sh] installed torch ${installed_torch} into the uv tool env"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "[install.sh] missing python interpreter: ${PYTHON_BIN}" >&2; exit 1
  }
  "${PYTHON_BIN}" -m pip install --user --editable .
  installed_torch=""
  for candidate in "${TORCH_CANDIDATES[@]}"; do
    IFS=: read -r TORCH_VERSION TORCH_BACKEND <<<"${candidate}"
    if [[ -z "${TORCH_VERSION}" || -z "${TORCH_BACKEND}" ]]; then
      echo "[install.sh] invalid torch candidate: ${candidate}" >&2
      exit 2
    fi
    if try_pip_install_torch "${TORCH_VERSION}" "${TORCH_BACKEND}" &&
       validate_installed_torch_arch "${detected_cap:-}" "${PYTHON_BIN}"; then
      installed_torch="${TORCH_VERSION} (${TORCH_BACKEND})"
      break
    fi
    echo "[install.sh] torch ${TORCH_VERSION} (${TORCH_BACKEND}) is unavailable or incompatible here; trying next candidate." >&2
  done
  if [[ -z "${installed_torch}" ]]; then
    echo "[install.sh] failed to install torch from candidates: ${TORCH_CANDIDATES[*]}" >&2
    exit 1
  fi
  echo "[install.sh] installed package with pip: gpulock"
  echo "[install.sh] installed torch ${installed_torch}"
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
