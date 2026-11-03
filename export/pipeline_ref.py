"""Reference implementation of docs/chunking_spec.md (v1) for MaskaraPay.

Importable module; the Rust service (service/src/pipeline.rs) must agree with
this implementation char-for-char on the golden file (assets/golden.json).

Pipeline (exactly per spec):
  1. full-text encode (add_special_tokens=True, truncation/padding disabled)
  2. content tokens (word_id != None, i.e. strip [CLS]/[SEP])
  3. windowing: window=510 content tokens, stride=256, ascending starts
  4. per-window ORT run (int64 inputs, token_type zeros)
  5. word aggregation: f32 mean of subtoken logits (summed in token order),
     argmax (ties -> lower id), softmax score, word char span = min start..max end
  6. BIO merge per window (I-L without matching current open span -> treated as B-L)
  7. cross-window reconciliation: ascending window start, replace only a single
     overlapping final when score strictly greater and overlap_ratio >= 0.5.

Fully deterministic: no threading, no randomness.
"""

from __future__ import annotations

import json
import os
import numpy as np
import onnxruntime as ort
import tokenizers as tk

# --------------------------------------------------------------------------
# Constants (computed lazily from the tokenizer file; CLS=101/SEP=102 for BERT)
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
TOKENIZER_PATH = os.path.join(ASSETS, "tokenizer.json")


def load_tokenizer(path: str | None = None) -> tk.Tokenizer:
    """Load the fast tokenizers.Tokenizer from assets/tokenizer.json."""
    return tk.Tokenizer.from_file(path or TOKENIZER_PATH)


def encode_full(text: str, tokenizer: tk.Tokenizer):
    """Encode the FULL text. Returns (ids, offsets, word_ids).

    add_special_tokens=True (one [CLS] prefix + one [SEP] suffix),
    truncation and padding explicitly disabled.
    Offsets are CHAR offsets into the original string:
    text[start:end] == the token's source text.
    """
    enc = tokenizer.encode(text, add_special_tokens=True)
    ids = list(enc.ids)
    offsets = list(enc.offsets)
    word_ids = list(enc.word_ids)
    assert len(ids) == len(offsets) == len(word_ids)
    return ids, offsets, word_ids


def content_tokens(text: str, tokenizer: tk.Tokenizer):
    """Strip special tokens; keep content tokens in order.

    Returns (content_ids, content_offsets, content_word_ids) where
      - content_ids: token ids without [CLS]/[SEP]
      - content_offsets: per-token (start, end) char offsets
      - content_word_ids: word id per content token (>= 0)
    """
    ids, offsets, word_ids = encode_full(text, tokenizer)
    c_ids, c_off, c_word = [], [], []
    for i, w in enumerate(word_ids):
        if w is not None:
            c_ids.append(ids[i])
            c_off.append(offsets[i])
            c_word.append(w)
    return c_ids, c_off, c_word


def build_windows(n: int, window: int = 510, stride: int = 256):
    """Window starts over content tokens, per spec.

    If n <= window: single window [0, n).
    Else starts s = 0, stride, 2*stride, ... while s < n;
    each window covers [s, min(s+window, n)).
    """
    if n <= window:
        return [(0, n)]
    starts = list(range(0, n, stride))
    return [(s, min(s + window, n)) for s in starts]


# --------------------------------------------------------------------------
# ORT session running
# --------------------------------------------------------------------------

def make_session(
    model_path: str,
    intra_threads: int | None = None,
) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if intra_threads is not None:
        opts.intra_op_num_threads = intra_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


def run_window(session: ort.InferenceSession, window_ids: list[int]):
    """Run one window on an ORT session; returns logits [n_tok, 35] float32.

    Model input: [CLS] + window_ids + [SEP]; attention ones; token_type zeros.
    """
    input_ids = np.asarray([[101] + list(window_ids) + [102]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    token_type_ids = np.zeros_like(input_ids)
    outputs = session.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    logits = outputs[0][0]  # [S, 35]
    assert logits.shape[0] == input_ids.shape[1]
    return logits  # positions 1..(len(window_ids)) are the content logits


# --------------------------------------------------------------------------
# Word aggregation (per spec; f32 math with defined summation order)
# --------------------------------------------------------------------------

def _f32_sum(values) -> float:
    """Sequential f32 sum in token order (matches Rust f32 accumulation)."""
    total = np.float32(0.0)
    for v in values:
        total = np.float32(total + np.float32(v))
    return total


def _f32_softmax(logits: np.ndarray) -> np.ndarray:
    """Plain softmax over all classes, f32 with sequential sum (Rust-mirrored)."""
    v = np.asarray(logits, dtype=np.float32)
    m = v.max()
    e = np.exp(v - m).astype(np.float32)
    s = np.float32(_f32_sum([np.float32(x) for x in e]))
    return (e / s).astype(np.float32)


def aggregate_words(
    window_logits: np.ndarray,
    sub_word_ids: list[int],
    sub_offsets: list[tuple[int, int]],
) -> list[dict]:
    """Group window CONTENT tokens by word id, aggregate logits/score/span.

    window_logits: [n_content_tokens_in_window, 35] float32 (CLS/SEP excluded).
    sub_word_ids:  word id per content token in the window.
    sub_offsets:   per-token (start, end) char offsets.

    Returns words in original order:
      {word_id, label_id, score, start, end, logits}
    """
    n = len(sub_word_ids)
    assert window_logits.shape[0] == n
    words = {}
    for i in range(n):
        wid = sub_word_ids[i]
        words.setdefault(wid, {"idx": []})
        words[wid]["idx"].append(i)

    out = []
    for wid in sorted(words):
        idxs = words[wid]["idx"]
        mean_logits = np.zeros(window_logits.shape[1], dtype=np.float32)
        for c in range(window_logits.shape[1]):
            mean_logits[c] = np.float32(_f32_sum(window_logits[i, c] for i in idxs) / len(idxs))
        label_id = int(np.argmax(mean_logits))  # numpy argmax -> first (lowest) id on ties
        score = float(_f32_softmax(mean_logits)[label_id])
        starts = [sub_offsets[i][0] for i in idxs]
        ends = [sub_offsets[i][1] for i in idxs]
        out.append(
            {
                "word_id": int(wid),
                "label_id": label_id,
                "score": score,
                "start": min(starts),
                "end": max(ends),
                "logits": mean_logits,
            }
        )
    return out


# --------------------------------------------------------------------------
# BIO merge (per window)
# --------------------------------------------------------------------------

def bio_merge(words: list[dict], labels_map: dict[int, str]) -> list[dict]:
    """Walk words in order; emit candidate spans.

    - O      -> close current span
    - B-L    -> close current span; open new span L
    - I-L    -> extend when current span has label L, else close + open L
    span = (label, first_word_start, last_word_end, mean word score)
    """
    spans = []
    cur = None  # {label, first_start, last_end, scores}
    for w in words:
        lab = labels_map[w["label_id"]]
        if lab == "O":
            if cur is not None:
                spans.append(_close_span(cur))
                cur = None
            continue
        prefix, _, typ = lab.partition("-")
        if prefix == "B" or cur is None or cur["label"] != typ:
            if cur is not None:
                spans.append(_close_span(cur))
            cur = {"label": typ, "first_start": w["start"], "last_end": w["end"], "scores": [w["score"]]}
        else:  # I- and matching current label
            cur["last_end"] = w["end"]
            cur["scores"].append(w["score"])
    if cur is not None:
        spans.append(_close_span(cur))
    return spans


def _close_span(cur: dict) -> dict:
    scores = cur["scores"]
    mean_score = float(_f32_sum(scores) / len(scores))
    return {
        "label": cur["label"],
        "start": cur["first_start"],
        "end": cur["last_end"],
        "score": mean_score,
    }


# --------------------------------------------------------------------------
# Cross-window reconciliation
# --------------------------------------------------------------------------

def _overlap_ratio(a: dict, b: dict) -> float:
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    if union <= 0:
        return 0.0
    return inter / union


def reconcile(window_spans: list[list[dict]]) -> list[dict]:
    """Process windows ascending by start s; candidates in emission order.

    - no overlapping finals           -> append
    - exactly one overlap, S.score > C.score strictly, overlap_ratio >= 0.5 -> replace
    - else                            -> drop (ties keep the earlier window)
    """
    finals: list[dict] = []
    for cand_list in window_spans:
        for S in cand_list:
            overlapping = [i for i, C in enumerate(finals) if C["start"] < S["end"] and S["start"] < C["end"]]
            if not overlapping:
                finals.append(S)
            elif len(overlapping) == 1 and S["score"] > finals[overlapping[0]]["score"] and _overlap_ratio(S, finals[overlapping[0]]) >= 0.5:
                finals[overlapping[0]] = S
            # else: drop S
    finals.sort(key=lambda x: x["start"])
    return finals


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def predict_words(
    text: str,
    session: ort.InferenceSession,
    tokenizer: tk.Tokenizer,
    window: int = 510,
    stride: int = 256,
):
    """Run the model over all windows; return (windows, n_content_tokens).

    windows: list of {"s", "e", "words"} ascending by s.
    words: aggregated word dicts from aggregate_words().
    """
    c_ids, c_off, c_word = content_tokens(text, tokenizer)
    n = len(c_ids)
    windows = []
    for s, e in build_windows(n, window, stride):
        window_ids = c_ids[s:e]
        logits = run_window(session, window_ids)
        content_logits = logits[1 : len(window_ids) + 1]  # drop CLS; SEP at end too
        sub_word_ids = c_word[s:e]
        sub_offsets = c_off[s:e]
        words = aggregate_words(content_logits, sub_word_ids, sub_offsets)
        windows.append({"s": s, "e": e, "words": words})
    return windows, n


def redact(
    text: str,
    session: ort.InferenceSession,
    labels_map: dict,
    window: int = 510,
    stride: int = 256,
    tokenizer: tk.Tokenizer | None = None,
) -> dict:
    """Chunked PII redaction per spec.

    labels_map: {int id -> str label} ("O", "B-NAME", ...).
    Returns {"spans": [(label, start, end), ...], "n_tokens": int, "n_windows": int}.
    Spans are char offsets into the ORIGINAL string, sorted by start,
    non-overlapping, text[start:end] == the span's source text.
    """
    labels_map = {int(k): v for k, v in labels_map.items()}
    tok = tokenizer or load_tokenizer()
    windows, n = predict_words(text, session, tok, window, stride)

    window_spans = []
    for wdw in windows:
        words = wdw["words"]
        # Drop degenerate words (shouldn't happen, defensive): start < end required
        words = [w for w in words if w["start"] < w["end"]]
        window_spans.append(bio_merge(words, labels_map))

    finals = reconcile(window_spans)
    spans = [(S["label"], int(S["start"]), int(S["end"])) for S in finals if S["start"] < S["end"]]
    spans.sort(key=lambda x: x[1])
    return {"spans": spans, "n_tokens": n, "n_windows": len(windows)}


def apply_redaction(text: str, spans) -> str:
    """Apply spans (ascending, non-overlapping): replace each with [<LABEL>]."""
    out = []
    last = 0
    for label, start, end in spans:
        out.append(text[last:start])
        out.append(f"[{label}]")
        last = end
    out.append(text[last:])
    return "".join(out)


def load_labels(config_path: str | None = None) -> dict[int, str]:
    """id2label from config.json -> {int id: str label}."""
    path = config_path or os.path.join(ASSETS, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {int(k): v for k, v in cfg["id2label"].items()}
