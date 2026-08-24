#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_DIR="${ENV_DIR:-${SCRIPT_DIR}/.venv-navdp-render}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_ENV="${CONDA_ENV:-navdp-render}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
INSTALL_TORCH="${INSTALL_TORCH:-true}"
INSTALL_TELESIM="${INSTALL_TELESIM:-true}"
USE_CONDA="${USE_CONDA:-auto}"
USE_VENV="${USE_VENV:-auto}"
CREATE_CONDA_ENV="${CREATE_CONDA_ENV:-auto}"
REQUIRE_CUDA="${REQUIRE_CUDA:-auto}"

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

have_conda() {
  command -v conda >/dev/null 2>&1
}

conda_env_exists() {
  local env_name="$1"
  conda env list | awk '{print $1}' | grep -Fxq "${env_name}"
}

activate_conda_env() {
  local env_name="$1"
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${env_name}"
}

if [ "${USE_CONDA}" = "auto" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ]; then
    USE_CONDA=true
    CREATE_CONDA_ENV=false
  elif have_conda && [ "${USE_VENV}" != "true" ]; then
    USE_CONDA=true
  else
    USE_CONDA=false
  fi
fi

if [ "${USE_VENV}" = "auto" ]; then
  if [ "${USE_CONDA}" = "true" ]; then
    USE_VENV=false
  else
    USE_VENV=true
  fi
fi

if [ "${USE_CONDA}" = "true" ]; then
  if ! have_conda; then
    echo "[ERROR] USE_CONDA=true but conda was not found in PATH." >&2
    exit 1
  fi
  if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ]; then
    echo "[INSTALL] Using active conda env: ${CONDA_DEFAULT_ENV}" >&2
  else
    if [ "${CREATE_CONDA_ENV}" = "auto" ]; then
      CREATE_CONDA_ENV=true
    fi
    if ! conda_env_exists "${CONDA_ENV}"; then
      if [ "${CREATE_CONDA_ENV}" != "true" ]; then
        echo "[ERROR] Conda env ${CONDA_ENV} does not exist. Set CREATE_CONDA_ENV=true or activate an env first." >&2
        exit 1
      fi
      echo "[INSTALL] Creating conda env ${CONDA_ENV} with Python ${PYTHON_VERSION}" >&2
      conda create -y -n "${CONDA_ENV}" "python=${PYTHON_VERSION}"
    fi
    activate_conda_env "${CONDA_ENV}"
    echo "[INSTALL] Using conda env: ${CONDA_DEFAULT_ENV}" >&2
  fi
elif [ "${USE_VENV}" = "true" ]; then
  echo "[INSTALL] Using venv: ${ENV_DIR}" >&2
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
else
  echo "[INSTALL] Installing into current Python: $(command -v python)" >&2
fi

python -m pip install --upgrade pip setuptools wheel ninja

if [ "${INSTALL_TORCH}" = "true" ]; then
  echo "[INSTALL] Installing PyTorch ${TORCH_VERSION} (${TORCH_CUDA})" >&2
  python -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

if [ "${INSTALL_TELESIM}" = "true" ]; then
  python -m pip install -e "${SCRIPT_DIR}/TeleSim3D"
fi

export INSTALL_TELESIM REQUIRE_CUDA
python - <<'PY'
import importlib
import os
import sys

for name in ("torch", "gsplat", "imageio", "imageio_ffmpeg", "numpy", "plyfile"):
    importlib.import_module(name)
try:
    import cv2  # noqa: F401
except Exception as exc:  # pylint: disable=broad-except
    raise RuntimeError(f"opencv-python import failed: {exc}") from exc
if os.environ.get("INSTALL_TELESIM", "true").lower() == "true":
    importlib.import_module("tele_sim")

torch = importlib.import_module("torch")
require_cuda = os.environ.get("REQUIRE_CUDA", "auto").lower()
cuda_available = bool(torch.cuda.is_available())
if require_cuda == "true" and not cuda_available:
    raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

print(f"NavDP render environment OK: python={sys.executable}")
print(f"torch={torch.__version__} cuda_available={cuda_available} torch_cuda={torch.version.cuda}")
if cuda_available:
    print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY

if [ "${REQUIRE_CUDA}" = "auto" ]; then
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    print("[WARN] CUDA is not available to PyTorch in this environment.")
PY
fi

cat <<EOF

Install target:
  $(python -c 'import sys; print(sys.executable)')

EOF

if [ "${USE_VENV}" = "true" ]; then
  cat <<EOF
Activate it with:
  source "${ENV_DIR}/bin/activate"
EOF
elif [ "${USE_CONDA}" = "true" ]; then
  cat <<EOF
Activate it with:
  conda activate "${CONDA_DEFAULT_ENV:-${CONDA_ENV}}"
EOF
fi
