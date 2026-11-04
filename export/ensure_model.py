"""Ensure Hugging Face weights and ONNX graphs exist on this machine.

Dev (`cargo run`) and prod (systemd) both invoke this when the runtime model
is missing. Weights are fetched with download_model.py on the instance; ONNX
is then exported locally so the Rust service can start.
"""

from __future__ import annotations

import os

import download_model
import export_onnx
import quantize_int8

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.join(os.path.dirname(HERE), "assets")


def assets_dir() -> str:
    return os.environ.get("MASKARA_ASSETS_DIR", DEFAULT_ASSETS)


def onnx_present(root: str | None = None) -> bool:
    explicit = os.environ.get("MASKARA_MODEL_PATH")
    if explicit:
        return os.path.isfile(explicit)
    base = root or assets_dir()
    return os.path.isfile(os.path.join(base, "maskara-int8.onnx")) or os.path.isfile(
        os.path.join(base, "maskara-f32.onnx")
    )


def main() -> None:
    assets = assets_dir()
    os.makedirs(assets, exist_ok=True)
    if onnx_present(assets):
        print(f"runtime ONNX already present under {assets}")
        return

    print("runtime ONNX missing; ensuring Hugging Face weights on this instance")
    download_model.main()

    print("exporting ONNX graphs")
    f32 = export_onnx.build_f32(assets_dir=assets)
    quantize_int8.build_int8(
        f32_path=f32,
        int8_path=os.path.join(assets, "maskara-int8.onnx"),
    )
    if not onnx_present(assets):
        raise SystemExit(f"model export finished but no ONNX file is under {assets}")
    print(f"runtime model ready under {assets}")


if __name__ == "__main__":
    main()
