"""Step 4: parity harness.

a. torch (HF) vs onnx-f32 ORT session: per-window token argmax agreement %.
b. HF pipeline("token-classification", aggregation_strategy="simple") sanity for
   single-window cases: HF entities vs pipeline_ref spans (count & cluster mismatches).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import pipeline_ref

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
MODEL_SOURCE = os.path.join(HERE, "model-source")

# --------------------------------------------------------------------------
# torch model
# --------------------------------------------------------------------------

_TORCH_MODEL = None


def load_torch_model(model_dir: str | None = None):
    global _TORCH_MODEL
    if _TORCH_MODEL is None:
        from transformers import AutoModelForTokenClassification

        m = AutoModelForTokenClassification.from_pretrained(model_dir or MODEL_SOURCE)
        m.eval()
        _TORCH_MODEL = m
    return _TORCH_MODEL


def torch_window_logits(window_ids: list[int]) -> np.ndarray:
    """[CLS] + window ids + [SEP] -> logits [n_content, 35] float32."""
    model = load_torch_model()
    ids = torch.tensor([[101] + list(window_ids) + [102]], dtype=torch.int64)
    mask = torch.ones_like(ids)
    types = torch.zeros_like(ids)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, token_type_ids=types).logits
    return logits[0, 1 : len(window_ids) + 1, :].numpy().astype(np.float32)


# --------------------------------------------------------------------------
# a. torch vs onnx token argmax agreement (per case)
# --------------------------------------------------------------------------

def token_parity_case(text: str, session, tokenizer=None, window=510, stride=256) -> dict:
    """Compare argmax at every content-token position across the case's windows.

    Returns {"n_tokens", "n_windows", "n_positions", "agree", "max_abs_diff"}.
    """
    tokenizer = tokenizer or pipeline_ref.load_tokenizer()
    c_ids, _off, _w = pipeline_ref.content_tokens(text, tokenizer)
    wins = pipeline_ref.build_windows(len(c_ids), window, stride)
    n_pos = 0
    n_agree = 0
    max_diff = 0.0
    for s, e in wins:
        w_ids = c_ids[s:e]
        t_logits = torch_window_logits(w_ids)
        o_logits = pipeline_ref.run_window(session, w_ids)[1 : len(w_ids) + 1]
        assert t_logits.shape == o_logits.shape, (t_logits.shape, o_logits.shape)
        t_arg = np.argmax(t_logits, axis=1)
        o_arg = np.argmax(o_logits, axis=1)
        n_pos += len(t_arg)
        n_agree += int(np.sum(t_arg == o_arg))
        max_diff = max(max_diff, float(np.max(np.abs(t_logits - o_logits))))
    return {
        "n_tokens": len(c_ids),
        "n_windows": len(wins),
        "n_positions": n_pos,
        "agree": n_agree / n_pos if n_pos else 1.0,
        "max_abs_diff": max_diff,
    }


def f32_vs_int8_case(text: str, f32_sess, int8_sess, tokenizer=None, window=510, stride=256) -> dict:
    """Compare f32 vs int8 ORT argmax at every content-token position."""
    tokenizer = tokenizer or pipeline_ref.load_tokenizer()
    c_ids, _off, _w = pipeline_ref.content_tokens(text, tokenizer)
    wins = pipeline_ref.build_windows(len(c_ids), window, stride)
    n_pos, n_agree = 0, 0
    diff_pos = []
    for s, e in wins:
        w_ids = c_ids[s:e]
        a = pipeline_ref.run_window(f32_sess, w_ids)[1 : len(w_ids) + 1]
        b = pipeline_ref.run_window(int8_sess, w_ids)[1 : len(w_ids) + 1]
        a_arg = np.argmax(a, axis=1)
        b_arg = np.argmax(b, axis=1)
        n_pos += len(a_arg)
        eq = a_arg == b_arg
        n_agree += int(eq.sum())
        if not eq.all():
            for tok_id, ai, bi in zip(w_ids, a_arg, b_arg):
                if ai != bi:
                    diff_pos.append({"token_id": tok_id, "f32": int(ai), "int8": int(bi)})
    return {
        "n_positions": n_pos,
        "agree": n_agree / n_pos if n_pos else 1.0,
        "disagreements": diff_pos,
    }


# --------------------------------------------------------------------------
# b. HF pipeline sanity (single-window cases only)
# --------------------------------------------------------------------------

def load_hf_pipeline():
    from transformers import pipeline

    return pipeline(
        "token-classification",
        model=MODEL_SOURCE,
        aggregation_strategy="simple",
        device=-1,
    )


def hf_entities(pipe, text: str):
    """entities as [(entity_group, start, end), ...], HF simple aggregation."""
    out = pipe(text)
    ents = []
    for e in out:
        ents.append(
            {
                "label": e["entity_group"],
                "start": int(e["start"]),
                "end": int(e["end"]),
                "score": float(e["score"]),
                "word": e.get("word", ""),
            }
        )
    ents.sort(key=lambda x: (x["start"], x["end"]))
    return ents


def compare_hf_vs_ref(text: str, pipe, session, labels_map, tokenizer=None):
    """Full pipeline_ref vs HF-pipeline entity compare for a single-window text."""
    tokenizer = tokenizer or pipeline_ref.load_tokenizer()
    ref = pipeline_ref.redact(text, session, labels_map, tokenizer=tokenizer)
    hf = hf_entities(pipe, text)
    ref_triples = {tuple(s) for s in ref["spans"]}  # (label, start, end)
    hf_triples = {(e["label"], e["start"], e["end"]) for e in hf}
    inter = ref_triples & hf_triples
    only_ref = ref_triples - hf_triples
    only_hf = hf_triples - ref_triples
    # cluster: same offsets, different label -> "label flip";
    # overlapping but different offsets -> "offset/merge diff"
    label_flip = []
    off_diff = []
    for s in sorted(only_ref):
        la, a, b = s
        matched = [h for h in hf if (h["start"], h["end"]) == (a, b)]
        if matched:
            label_flip.append({"ref": s, "hf": matched[0]["label"]})
        else:
            ov = [h for h in hf if h["start"] < b and a < h["end"]]
            off_diff.append({"ref": s, "hf_overlaps": [(x["label"], x["start"], x["end"]) for x in ov]})
    return {
        "ref_spans": ref["spans"],
        "hf_entities": hf,
        "exact": len(inter),
        "only_ref": sorted(map(list, only_ref)),
        "only_hf": sorted(map(list, only_hf)),
        "label_flip": label_flip,
        "offset_diff": off_diff,
        "ref_n": len(ref_triples),
        "hf_n": len(hf_triples),
    }


# --------------------------------------------------------------------------
# CLI smoke (single case)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tok = pipeline_ref.load_tokenizer()
    sess = pipeline_ref.make_session(os.path.join(ASSETS, "maskara-f32.onnx"))
    labels = pipeline_ref.load_labels()
    text = "Ravi Sharma emailed ravi.sharma@example.com about invoice #4521."
    p = token_parity_case(text, sess, tok)
    print("torch-vs-onnx:", p)
    pipe = load_hf_pipeline()
    cmp = compare_hf_vs_ref(text, pipe, sess, labels, tok)
    print("hf sanity:", {k: cmp[k] for k in ("exact", "ref_n", "hf_n", "only_ref", "only_hf", "label_flip", "offset_diff")})
