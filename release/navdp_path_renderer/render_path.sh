#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [ -n "${CONDA_DEFAULT_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    CONDA_ENV_PREFIX="$(
      conda env list 2>/dev/null | awk -v env="${CONDA_DEFAULT_ENV}" '
        $1 == env { print $NF; exit }
        $1 == "*" && $2 == env { print $NF; exit }
      '
    )"
    if [ -n "${CONDA_ENV_PREFIX}" ] && [ -x "${CONDA_ENV_PREFIX}/bin/python" ]; then
      PYTHON_BIN="${CONDA_ENV_PREFIX}/bin/python"
    fi
  fi
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[ERROR] python is required but was not found in PATH." >&2
    exit 1
  fi
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import imageio  # noqa: F401
PY
then
  echo "[ERROR] Selected Python cannot import imageio: ${PYTHON_BIN}" >&2
  echo "Activate the release env or pass PYTHON_BIN=/path/to/env/bin/python." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/render_path.py" "$@"
