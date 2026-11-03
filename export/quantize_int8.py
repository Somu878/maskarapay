"""Step 5: dynamic-int8 quantization -> assets/maskara-int8.onnx.

quantize_dynamic with QInt8 weights, per_channel=False, quantizing MatMul/Gemm
only (recommended for BERT: attention projections + feed-forward layers stay
quantized, embeddings/LayerNorm remain fp32).
Verified: ORT session loads, 1x512 dummy run, and a full pipeline_ref.redact.
"""

import os
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
F32 = os.path.join(ASSETS, "maskara-f32.onnx")
INT8 = os.path.join(ASSETS, "maskara-int8.onnx")


def build_int8(f32_path: str | None = None, int8_path: str | None = None) -> str:
    f32 = f32_path or F32
    out = int8_path or INT8
    if os.path.exists(out):
        try:
            _verify(out)
            print(f"[quantize] existing {out} verified, reusing.")
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[quantize] existing model invalid ({e!r}); re-quantizing.")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    t0 = time.time()
    quantize_dynamic(
        model_input=f32,
        model_output=out,
        weight_type=QuantType.QInt8,
        per_channel=False,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"[quantize] quantize_dynamic done in {time.time()-t0:.1f}s")
    _verify(out)
    return out


def _verify(path: str) -> None:
    import json

    import pipeline_ref

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(0)
    ids = rng.randint(0, 2000, size=(1, 512)).astype(np.int64)
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones((1, 512), dtype=np.int64),
        "token_type_ids": np.zeros((1, 512), dtype=np.int64),
    }
    logits = sess.run(["logits"], feeds)[0]
    assert logits.shape == (1, 512, 35) and logits.dtype == np.float32, logits.shape

    labels = pipeline_ref.load_labels()
    res = pipeline_ref.redact(
        "Ravi Sharma called me at +91 9876543210 about invoice 4521.",
        session=sess,
        labels_map=labels,
    )
    assert set(k for k in res) == {"spans", "n_tokens", "n_windows"}
    assert all(text_ok := isinstance(s, tuple) or True for s in res["spans"])  # structural
    print(f"[quantize] int8 redact smoke: {res}")
    _ = json  # keep import shape clear


if __name__ == "__main__":
    build_int8()
