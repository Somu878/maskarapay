# Chunking, word-aggregation and BIO-merge spec (v1)

Shared source of truth for the Python reference implementation
(`export/pipeline_ref.py`) and the Rust service (`service/src/pipeline.rs`).
Both MUST implement this exactly. The golden file (`assets/golden.json`)
verifies that the two implementations agree.

- Model: `somukandula/maskara` @ `c49d5ecb82871f796776b058e4ff0ad1c89e0bcb`
  (v2, repo root).
- Max positions 512. Every model call is ≤ 512 tokens total (including CLS/SEP).

## Model interface

- Inputs (int64): `input_ids [B, S]`, `attention_mask [B, S]`, `token_type_ids [B, S]` (zeros).
- Output: `logits [B, S, 35]` float32.
- `id2label` comes from `config.json` (35 labels; `O` = 0; 17 PII types × B/I).

## Tokenizer

- `tokenizer.json` from the repo root (bert-base-uncased WordPiece, uncased).
- Encode the FULL original text once with `add_special_tokens = true`,
  `truncation = none`, `padding = none`.
- The encoding gives: ids, attention (all 1s), offsets `(start, end)` into the
  ORIGINAL text, and word_ids (`None` for `[CLS]`/`[SEP]`).
- `[CLS]` is token 0; `[SEP]` is the last token (one pair per encoding).
- Offsets are CHAR offsets (unicode codepoint indices) into the original string:
  `text[start:end]` must equal the token's source text.
  Rust note: the `tokenizers` crate may expose byte offsets — convert to char
  offsets against the original string; the golden multibyte cases are the test.

## Windowing (content tokens)

- `content` = tokens excluding `[CLS]`/`[SEP]`. Let `n = |content|`.
- If `n ≤ 510`: one window covering `content[0..n]`.
- Else: window starts `s = 0, 256, 512, …` (stride 256) while `s < n`; each
  window covers `content[s .. min(s+510, n)]`
  (510 content tokens + CLS + SEP = 512).
- Model input for a window: `[CLS] + content[s..e] + [SEP]`, attention ones,
  token_type zeros (length `e − s + 2 ≤ 512`).
- Only logits at positions 1..(e−s+1) (excluding CLS/SEP) attach to content
  tokens, in order.

## Word aggregation (per window)

- Group tokens by word_id. Special tokens are excluded.
- For each word: aggregated logits = MEAN of its subtoken logits (f32, summed
  in token order); `label = argmax`; `score = softmax(aggregated)[label]`
  (plain softmax over all 35 classes).
- Word char offsets: `start = min(subtoken starts)`, `end = max(subtoken ends)`.
- Words stay in original order. Ties in argmax resolve to the lower label id.

## BIO → spans (per window, over words)

- Walk words in order, maintaining the current span `C(label, first_start, last_end)`:
  - `O` → close `C` (emit); skip word.
  - `B-L` → close `C`; open new span `L` at this word.
  - `I-L`: if `C` exists and `C.label == L` → extend `C.end = word.end`;
    else → close `C` and open a new span `L` (treated like B).
- Span range = `[first word start, last word end)` (exclusive end).
- Span `score = mean(its words' scores)` — used only for reconciliation.

## Cross-window reconciliation

- Process windows in ascending start `s`. Final list `F` starts empty.
- For each candidate span `S(start, end, label, score)`:
  - Overlapping finals: all `C ∈ F` with `C.start < S.end && S.start < C.end`.
  - If none → append `S` to `F`.
  - If exactly one `C` overlaps AND `S.score > C.score` AND
    `overlap_ratio = |intersection| / |union| ≥ 0.5` → replace `C` with `S`.
  - Else (`0` or `≥ 2` overlapping finals, or the replace condition fails,
    including ties) → drop `S`. Earlier windows win ties.

## Output

- Redaction replaces each final span's substring with `[<LABEL>]`, in ascending
  start order; all other text is preserved byte-for-byte.
- Spans returned sorted by `start`; char offsets; `start` inclusive, `end` exclusive.
- Empty text: no tokens → no spans → redacted text equals input.

## Limits & errors

- Request text longer than `MAX_TEXT_CHARS` (default 20000) is rejected
  (service responds 413).
- No logging of request text or redacted output — only lengths, latency, ids.
