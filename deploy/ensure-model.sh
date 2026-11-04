#!/usr/bin/env bash
# Idempotent model bootstrap for the local dev server and the DC/prod instance.
# If the runtime ONNX graph is already present, this exits immediately.
# Otherwise it runs export/ensure_model.py, which calls download_model.py locally.
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)

if [[ -d "$script_dir/../export" ]]; then
  root=$(cd "$script_dir/.." && pwd)
elif [[ -d /opt/maskarapay/export ]]; then
  root=/opt/maskarapay
else
  echo "cannot find MaskaraPay export/ directory" >&2
  exit 1
fi

export_dir="${MASKARA_EXPORT_DIR:-$root/export}"
assets_dir="${MASKARA_ASSETS_DIR:-$root/assets}"
export_python="${MASKARA_EXPORT_PYTHON:-3.12}"
export MASKARA_ASSETS_DIR="$assets_dir"
export MASKARA_EXPORT_DIR="$export_dir"

onnx_ready() {
  if [[ -n "${MASKARA_MODEL_PATH:-}" ]]; then
    [[ -f "$MASKARA_MODEL_PATH" ]]
    return
  fi
  [[ -f "$assets_dir/maskara-int8.onnx" || -f "$assets_dir/maskara-f32.onnx" ]]
}

if [[ "${MASKARA_SKIP_MODEL_DOWNLOAD:-}" == "1" ]]; then
  echo "MASKARA_SKIP_MODEL_DOWNLOAD=1; not fetching the model"
  exit 0
fi

if onnx_ready; then
  echo "maskara runtime model already present under $assets_dir"
  exit 0
fi

echo "maskara runtime model missing; downloading on this instance via download_model.py"
cd "$export_dir"

ensure_venv_python() {
  local venv_python="$export_dir/.venv/bin/python"
  if [[ ! -x "$venv_python" ]]; then
    return
  fi

  local venv_version
  venv_version=$("$venv_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [[ "$venv_version" != "$export_python" ]]; then
    echo "recreating export virtual environment with Python $export_python (found $venv_version)"
    rm -rf "$export_dir/.venv"
  fi
}

if command -v uv >/dev/null 2>&1; then
  ensure_venv_python
  uv venv --clear --python "$export_python" .venv
  uv pip install -r requirements.txt
  uv run python ensure_model.py
elif command -v "python$export_python" >/dev/null 2>&1; then
  ensure_venv_python
  "python$export_python" -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -r requirements.txt
  .venv/bin/python ensure_model.py
else
  echo "Python $export_python is required for the pinned PyTorch export dependencies; install uv or python$export_python" >&2
  exit 1
fi
