"""Step 2: download ONLY the pinned-repo files for somukandula/maskara."""

import os

from huggingface_hub import snapshot_download

REVISION = "c49d5ecb82871f796776b058e4ff0ad1c89e0bcb"
ALLOW = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "model.safetensors",
]


def source_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-source")


def is_downloaded(out: str | None = None) -> bool:
    root = out or source_dir()
    for name in ALLOW:
        path = os.path.join(root, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
    return True


def _print_listing(path: str) -> None:
    print("downloaded to", path)
    for name in sorted(os.listdir(path)):
        print(f"  {name}: {os.path.getsize(os.path.join(path, name))} bytes")


def main():
    out = source_dir()
    os.makedirs(out, exist_ok=True)
    if is_downloaded(out):
        print("huggingface weights already present at", out)
        _print_listing(out)
        return
    path = snapshot_download(
        repo_id="somukandula/maskara",
        revision=REVISION,
        allow_patterns=ALLOW,
        local_dir=out,
    )
    if not is_downloaded(path):
        raise SystemExit(f"download incomplete under {path}")
    _print_listing(path)


if __name__ == "__main__":
    main()
