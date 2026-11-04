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

if command -v uv >/dev/null 2>&1; then
  [[ -d .venv ]] || uv venv .venv
  uv pip install -r requirements.txt
  uv run python ensure_model.py
elif [[ -x .venv/bin/python ]]; then
  .venv/bin/pip install -r requirements.txt
  .venv/bin/python ensure_model.py
else
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -r requirements.txt
  .venv/bin/python ensure_model.py
fi
