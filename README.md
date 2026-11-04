# MaskaraPay

MaskaraPay is a local HTTP service for detecting and redacting PII with the
[`somukandula/maskara`](https://huggingface.co/somukandula/maskara) token-classification model.
It runs the model through ONNX Runtime in Rust and exposes one straightforward
endpoint: `POST /api/v1/redact`.

> This is a demo-quality PII detector, not a production-safe data-protection
> system. Do not use it as the sole safeguard for sensitive data.

## API

Send a typical inference payload with a model id and input text:

```bash
curl --fail-with-body -sS http://127.0.0.1:8080/api/v1/redact \
  -H 'content-type: application/json' \
  --data '{
    "model": "somukandula/maskara",
    "text": "Email ravi.sharma@example.com about the invoice."
  }'
```

Example response:

```json
{
  "model": "somukandula/maskara",
  "redacted_text": "Email [EMAIL] about the invoice.",
  "spans": [
    { "label": "EMAIL", "start": 6, "end": 29 }
  ],
  "usage": {
    "input_chars": 48,
    "input_tokens": 14,
    "windows": 1
  }
}
```

`start` and `end` are Unicode character offsets: `start` is inclusive and
`end` is exclusive. The bundled service supports only the exact model id
`somukandula/maskara`.

| Route | Purpose |
| --- | --- |
| `POST /api/v1/redact` | Redact PII from a `{ "model", "text" }` JSON payload. |
| `GET /healthz` | Check that the HTTP service is running. |

Requests with another model id return `400`; text longer than 20,000 Unicode
characters returns `413` by default. Request text and redacted output are never
written to application logs.

## Run locally

Prerequisites: Rust stable, Python 3.10+ (or `uv`), and outbound access to
Hugging Face for the first start.

```bash
cd service
cargo test --release
cargo run --release
```

The service listens on `127.0.0.1:8080` by default. On first start it checks
for `assets/maskara-int8.onnx` or `assets/maskara-f32.onnx`. If neither file
is present, it runs `deploy/ensure-model.sh` on this machine, which calls
`export/download_model.py` and then exports ONNX. Later starts skip the
download. In a second terminal, run the curl command above.

### Model artifacts

The ONNX model files are deliberately not committed. The service loads
`assets/maskara-int8.onnx` first and falls back to `assets/maskara-f32.onnx`.
`assets/config.json` and `assets/tokenizer.json` are included in the repository.

To prepare the artifacts yourself instead of letting the server do it:

```bash
./deploy/ensure-model.sh
```

That script is a no-op when the runtime model is already present. You can
still run the export steps individually:

```bash
cd export
uv venv .venv
uv pip install -r requirements.txt
uv run python download_model.py
uv run python export_onnx.py
uv run python quantize_int8.py
```

The raw Hugging Face weights, Python environment, and generated ONNX binaries
are ignored by Git.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MASKARA_BIND_ADDR` | `127.0.0.1:8080` | Address the HTTP server binds to. |
| `MASKARA_ASSETS_DIR` | `../assets` relative to `service/` | Directory containing model artifacts and tokenizer/config files. |
| `MASKARA_MODEL_PATH` | unset | Use one explicit ONNX file; disables the INT8/F32 fallback. |
| `MASKARA_MAX_TEXT_CHARS` | `20000` | Maximum Unicode characters accepted in `text`. |
| `MASKARA_EXPORT_DIR` | `export/` in this repo, or `/opt/maskarapay/export` | Python download/export tooling used when the runtime model is missing. |
| `MASKARA_EXPORT_PYTHON` | `3.12` | Python minor version used for the pinned PyTorch export dependencies. |
| `MASKARA_ENSURE_MODEL` | `deploy/ensure-model.sh` or `/opt/maskarapay/bin/ensure-model` | Bootstrap script the server runs on this instance if ONNX is absent. |
| `MASKARA_SKIP_MODEL_DOWNLOAD` | unset | Set to `1` to disable the on-instance download. |

## Deployment

The supported first deployment is a loopback-only AWS EC2 service published
through Cloudflare Tunnel. Follow the [AWS + Cloudflare deployment
runbook](docs/aws-cloudflare-deployment.md); it does not require Route 53.

## Implementation notes

| Item | Value |
| --- | --- |
| Model | `somukandula/maskara` v2, pinned at `c49d5ecb82871f796776b058e4ff0ad1c89e0bcb` |
| Architecture | BERT token classification; 35 BIO labels across 17 PII types plus `O` |
| Runtime | Rust, Axum, `ort 2.0.0-rc.15`, and an INT8 ONNX graph with F32 fallback |
| Pipeline | [Chunking and reconciliation specification](docs/chunking_spec.md) |
| Parity | Python reference pipeline exists; generating `assets/golden.json` is still pending |

## Limitations

`CREDIT_CARD` and `PASSWORD` are known weaker labels. x402 payments are not
implemented.

## License

Model: Apache-2.0. Code: not yet specified.
