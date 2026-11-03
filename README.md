# MaskaraPay

PII-redaction inference service for [`somukandula/maskara`](https://huggingface.co/somukandula/maskara) — Rust + ONNX Runtime, with x402 payments planned next.

Status: **Phase 0–1 — local inference, in progress.**

| Item | Value |
| --- | --- |
| Model | `somukandula/maskara` v2 (repo root), pinned commit `c49d5ecb82871f796776b058e4ff0ad1c89e0bcb` |
| Architecture | BertForTokenClassification, 109M params, 35 BIO labels (17 PII types + `O`) |
| Engine (decision) | Rust `ort 2.0.0-rc.15` + INT8 ONNX (fallback F32 ONNX; Python is one-off export only) |
| Pipeline spec | [`docs/chunking_spec.md`](docs/chunking_spec.md) — windowing 510/stride 256, word-mean logits, BIO merge, reconcile rule |
| Parity guard | `assets/golden.json` — Python reference pipeline vs Rust service, must match |

## Layout

```
docs/            spec + hosting notes
assets/          ONNX artifacts, tokenizer, config, golden.json (binaries gitignored)
export/          Python one-off: ONNX export, INT8 quantize, golden file (uv venv)
service/         Rust axum service (ort + tokenizers)
```

## Quickstart

(To be finalized as phases land.)

### Python export (phase 0, one-off)

```bash
cd export && uv venv .venv && uv pip install -r requirements.txt
uv run python build_all.py   # download → export → quantize → golden file
```

### Rust service (phase 1, local)

```bash
cd service
cargo test --release         # includes golden parity tests when assets/golden.json exists
cargo run --release          # serves on :8080
```

## Limitations

Demo quality. v2 real-world F1 is 0.8868; this is NOT production-safe PII
protection. CREDIT_CARD and PASSWORD labels are known-weak in real-world data.
INT8 quantization is only shipped if the measured drop is acceptable.

## License

Model: Apache-2.0. Code: TBD.
