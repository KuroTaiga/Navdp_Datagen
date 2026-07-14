#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_DIR="${ENV_DIR:-${SCRIPT_DIR}/.venv-navdp-render}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
INSTALL_TORCH="${INSTALL_TORCH:-true}"
USE_VENV="${USE_VENV:-auto}"

missing=()
for required in \
  "render_label_paths_telesim.py" \
  "unpack_3dgs_compressed.py" \
  "gaussian_renderer" \
  "arguments" \
  "scene" \
  "utils" \
  "lighting" \
  "TeleSim3D/tele_sim"; do
  if [ ! -e "${SCRIPT_DIR}/${required}" ]; then
    missing+=("${required}")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo "[ERROR] Release package is missing required renderer files:" >&2
  for item in "${missing[@]}"; do
    echo "  - ${item}" >&2
  done
  cat >&2 <<EOF

This release folder should contain all required renderer modules. If these files
are missing, rebuild or recopy the release/navdp_path_renderer folder.
EOF
  exit 2
fi

if [ "${USE_VENV}" = "auto" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ]; then
    USE_VENV=false
  else
    USE_VENV=true
  fi
fi

if [ "${USE_VENV}" = "true" ]; then
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel ninja

if [ "${INSTALL_TORCH}" = "true" ]; then
  python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

python -m pip install -e "${SCRIPT_DIR}/TeleSim3D"

python - <<'PY'
import importlib

for name in ("torch", "gsplat", "imageio", "plyfile", "tele_sim"):
    importlib.import_module(name)
print("NavDP render environment OK")
PY

cat <<EOF

Install target:
  $(python -c 'import sys; print(sys.executable)')

EOF

if [ "${USE_VENV}" = "true" ]; then
  cat <<EOF
Activate it with:
  source "${ENV_DIR}/bin/activate"
EOF
fi
