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


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-source")
    os.makedirs(out, exist_ok=True)
    path = snapshot_download(
        repo_id="somukandula/maskara",
        revision=REVISION,
        allow_patterns=ALLOW,
        local_dir=out,
    )
    print("downloaded to", path)
    for f in sorted(os.listdir(path)):
        print(f"  {f}: {os.path.getsize(os.path.join(path, f))} bytes")


if __name__ == "__main__":
    main()
