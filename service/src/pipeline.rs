use std::{
    fmt, fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

pub const MODEL_ID: &str = "somukandula/maskara";
pub const DEFAULT_MAX_TEXT_CHARS: usize = 20_000;

const CLS_TOKEN_ID: i64 = 101;
const SEP_TOKEN_ID: i64 = 102;
const WINDOW_SIZE: usize = 510;
const WINDOW_STRIDE: usize = 256;
const LABEL_COUNT: usize = 35;

#[derive(Debug)]
pub struct PipelineError(String);

impl PipelineError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for PipelineError {}

#[derive(Debug, Clone, Serialize)]
pub struct Span {
    pub label: String,
    /// Unicode code-point offset, inclusive.
    pub start: usize,
    /// Unicode code-point offset, exclusive.
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub spans: Vec<Span>,
    pub redacted_text: String,
    pub input_tokens: usize,
    pub windows: usize,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    id2label: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct ContentToken {
    id: i64,
    word_id: u32,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct WordPrediction {
    label_id: usize,
    score: f32,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct CandidateSpan {
    label: String,
    start: usize,
    end: usize,
    score: f32,
}

/// A single preloaded model session. ONNX Runtime sessions require mutable access
/// while running, so inference is serialized through the mutex.
pub struct Pipeline {
    tokenizer: Tokenizer,
    labels: Vec<String>,
    session: Mutex<Session>,
    model_path: PathBuf,
}

impl Pipeline {
    /// Loads tokenization/configuration artifacts and the preferred INT8 model.
    /// If the bundled INT8 graph cannot be used, loading falls back to the F32
    /// graph. An explicit model path is never silently replaced.
    pub fn load(
        assets_dir: &Path,
        explicit_model_path: Option<PathBuf>,
    ) -> Result<Self, PipelineError> {
        let tokenizer_path = assets_dir.join("tokenizer.json");
        let config_path = assets_dir.join("config.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            PipelineError::new(format!(
                "could not load tokenizer from {}: {error}",
                tokenizer_path.display()
            ))
        })?;
        let labels = load_labels(&config_path)?;

        let environment = ort::init()
            .with_name("maskara-pay")
            .with_telemetry(false)
            .build()
            .map_err(|error| {
                PipelineError::new(format!("could not initialize ONNX Runtime: {error}"))
            })?;

        let candidates = match explicit_model_path {
            Some(path) => vec![path],
            None => vec![
                assets_dir.join("maskara-int8.onnx"),
                assets_dir.join("maskara-f32.onnx"),
            ],
        };

        let explicit = candidates.len() == 1;
        let mut failures = Vec::new();
        for candidate in candidates {
            if !candidate.is_file() {
                failures.push(format!("{} does not exist", candidate.display()));
                continue;
            }

            match create_session(&environment, &candidate) {
                Ok(session) => {
                    return Ok(Self {
                        tokenizer,
                        labels,
                        session: Mutex::new(session),
                        model_path: candidate,
                    });
                }
                Err(error) => {
                    if explicit {
                        return Err(error);
                    }
                    failures.push(error.to_string());
                }
            }
        }

        Err(PipelineError::new(format!(
            "could not load any bundled ONNX model: {}",
            failures.join("; ")
        )))
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Runs the chunking, word aggregation, BIO merging, and reconciliation
    /// rules in docs/chunking_spec.md.
    pub fn predict(&self, text: &str) -> Result<Prediction, PipelineError> {
        let tokens = self.content_tokens(text)?;
        let windows = build_windows(tokens.len());
        let mut window_spans = Vec::with_capacity(windows.len());

        for (start, end) in &windows {
            let window = &tokens[*start..*end];
            let logits = self.run_window(window)?;
            let words = aggregate_words(&logits, window)?;
            window_spans.push(bio_merge(&words, &self.labels)?);
        }

        let candidates = reconcile(window_spans);
        let spans = candidates
            .into_iter()
            .filter(|span| span.start < span.end)
            .map(|span| Span {
                label: span.label,
                start: span.start,
                end: span.end,
            })
            .collect::<Vec<_>>();
        let redacted_text = apply_redaction(text, &spans)?;

        Ok(Prediction {
            spans,
            redacted_text,
            input_tokens: tokens.len(),
            windows: windows.len(),
        })
    }

    fn content_tokens(&self, text: &str) -> Result<Vec<ContentToken>, PipelineError> {
        // encode_char_offsets is intentional: the public API, spec, and Python
        // reference all use Unicode character offsets rather than UTF-8 bytes.
        let encoding = self
            .tokenizer
            .encode_char_offsets(text, true)
            .map_err(|error| PipelineError::new(format!("could not tokenize input: {error}")))?;

        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets();
        let word_ids = encoding.get_word_ids();
        if ids.len() != offsets.len() || ids.len() != word_ids.len() {
            return Err(PipelineError::new(
                "tokenizer returned inconsistent token metadata",
            ));
        }

        let mut tokens = Vec::new();
        for index in 0..ids.len() {
            let Some(word_id) = word_ids[index] else {
                continue;
            };
            let (start, end) = offsets[index];
            tokens.push(ContentToken {
                id: i64::from(ids[index]),
                word_id,
                start,
                end,
            });
        }
        Ok(tokens)
    }

    /// Returns only content-token logits, flattened as [token][label].
    fn run_window(&self, tokens: &[ContentToken]) -> Result<Vec<f32>, PipelineError> {
        let sequence_len = tokens.len() + 2;
        let mut input_ids = Vec::with_capacity(sequence_len);
        input_ids.push(CLS_TOKEN_ID);
        input_ids.extend(tokens.iter().map(|token| token.id));
        input_ids.push(SEP_TOKEN_ID);
        let attention_mask = vec![1_i64; sequence_len];
        let token_type_ids = vec![0_i64; sequence_len];

        let shape = vec![1_usize, sequence_len];
        let input_ids = Tensor::<i64>::from_array((shape.clone(), input_ids)).map_err(|error| {
            PipelineError::new(format!("could not create input_ids tensor: {error}"))
        })?;
        let attention_mask =
            Tensor::<i64>::from_array((shape.clone(), attention_mask)).map_err(|error| {
                PipelineError::new(format!("could not create attention_mask tensor: {error}"))
            })?;
        let token_type_ids =
            Tensor::<i64>::from_array((shape, token_type_ids)).map_err(|error| {
                PipelineError::new(format!("could not create token_type_ids tensor: {error}"))
            })?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| PipelineError::new("ONNX session lock was poisoned"))?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            })
            .map_err(|error| PipelineError::new(format!("ONNX inference failed: {error}")))?;
        let output = outputs
            .get("logits")
            .ok_or_else(|| PipelineError::new("ONNX model did not return logits"))?;
        let (output_shape, output_values) =
            output.try_extract_tensor::<f32>().map_err(|error| {
                PipelineError::new(format!("could not read logits tensor: {error}"))
            })?;

        let expected_values = sequence_len * LABEL_COUNT;
        if output_shape.as_ref() != [1_i64, sequence_len as i64, LABEL_COUNT as i64]
            || output_values.len() != expected_values
        {
            return Err(PipelineError::new(format!(
                "unexpected logits shape {:?}; expected [1, {sequence_len}, {LABEL_COUNT}]",
                output_shape
            )));
        }

        let mut content_logits = Vec::with_capacity(tokens.len() * LABEL_COUNT);
        for token_index in 0..tokens.len() {
            let offset = (token_index + 1) * LABEL_COUNT;
            content_logits.extend_from_slice(&output_values[offset..offset + LABEL_COUNT]);
        }
        Ok(content_logits)
    }
}

fn load_labels(config_path: &Path) -> Result<Vec<String>, PipelineError> {
    let config = fs::read_to_string(config_path).map_err(|error| {
        PipelineError::new(format!("could not read {}: {error}", config_path.display()))
    })?;
    let config: ModelConfig = serde_json::from_str(&config).map_err(|error| {
        PipelineError::new(format!(
            "could not parse {}: {error}",
            config_path.display()
        ))
    })?;

    let mut labels = Vec::with_capacity(LABEL_COUNT);
    for id in 0..LABEL_COUNT {
        let label = config.id2label.get(&id.to_string()).ok_or_else(|| {
            PipelineError::new(format!("config id2label is missing label id {id}"))
        })?;
        labels.push(label.clone());
    }
    if config.id2label.len() != LABEL_COUNT || labels.first().is_none_or(|label| label != "O") {
        return Err(PipelineError::new(
            "config id2label must contain exactly ids 0..34 with O at id 0",
        ));
    }
    Ok(labels)
}

fn create_session(
    environment: &ort::environment::Environment,
    model_path: &Path,
) -> Result<Session, PipelineError> {
    let builder = Session::builder(environment).map_err(|error| {
        PipelineError::new(format!("could not configure ONNX session: {error}"))
    })?;
    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|error| PipelineError::new(format!("could not optimize ONNX session: {error}")))?;
    builder.commit_from_file(model_path).map_err(|error| {
        PipelineError::new(format!(
            "could not load ONNX model {}: {error}",
            model_path.display()
        ))
    })
}

fn build_windows(token_count: usize) -> Vec<(usize, usize)> {
    if token_count <= WINDOW_SIZE {
        return vec![(0, token_count)];
    }
    (0..token_count)
        .step_by(WINDOW_STRIDE)
        .map(|start| (start, (start + WINDOW_SIZE).min(token_count)))
        .collect()
}

fn aggregate_words(
    logits: &[f32],
    tokens: &[ContentToken],
) -> Result<Vec<WordPrediction>, PipelineError> {
    if logits.len() != tokens.len() * LABEL_COUNT {
        return Err(PipelineError::new("logits do not match window token count"));
    }

    let mut words = Vec::new();
    let mut token_index = 0;
    while token_index < tokens.len() {
        let group_start = token_index;
        let word_id = tokens[token_index].word_id;
        let mut start = tokens[token_index].start;
        let mut end = tokens[token_index].end;
        token_index += 1;
        while token_index < tokens.len() && tokens[token_index].word_id == word_id {
            start = start.min(tokens[token_index].start);
            end = end.max(tokens[token_index].end);
            token_index += 1;
        }

        let token_count = token_index - group_start;
        let mut mean_logits = vec![0.0_f32; LABEL_COUNT];
        // Deliberately sum in token order to mirror the reference pipeline.
        for current_token in group_start..token_index {
            let offset = current_token * LABEL_COUNT;
            for label in 0..LABEL_COUNT {
                mean_logits[label] += logits[offset + label];
            }
        }
        for value in &mut mean_logits {
            *value /= token_count as f32;
        }

        let label_id = argmax(&mean_logits);
        let score = softmax_probability(&mean_logits, label_id);
        words.push(WordPrediction {
            label_id,
            score,
            start,
            end,
        });
    }
    Ok(words)
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0;
    for index in 1..values.len() {
        // Strictly greater means ties resolve to the smaller id.
        if values[index] > values[best_index] {
            best_index = index;
        }
    }
    best_index
}

fn softmax_probability(logits: &[f32], target: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0_f32;
    let mut numerator = 0.0_f32;
    for (index, value) in logits.iter().enumerate() {
        let exponent = (*value - max).exp();
        denominator += exponent;
        if index == target {
            numerator = exponent;
        }
    }
    numerator / denominator
}

fn bio_merge(
    words: &[WordPrediction],
    labels: &[String],
) -> Result<Vec<CandidateSpan>, PipelineError> {
    #[derive(Debug)]
    struct OpenSpan {
        label: String,
        start: usize,
        end: usize,
        scores: Vec<f32>,
    }

    fn close(open: OpenSpan) -> CandidateSpan {
        let mut total = 0.0_f32;
        for score in &open.scores {
            total += score;
        }
        CandidateSpan {
            label: open.label,
            start: open.start,
            end: open.end,
            score: total / open.scores.len() as f32,
        }
    }

    let mut spans = Vec::new();
    let mut current: Option<OpenSpan> = None;
    for word in words {
        let label = labels.get(word.label_id).ok_or_else(|| {
            PipelineError::new(format!("unknown predicted label id {}", word.label_id))
        })?;

        if label == "O" {
            if let Some(open) = current.take() {
                spans.push(close(open));
            }
            continue;
        }

        let (is_begin, entity) = if let Some(entity) = label.strip_prefix("B-") {
            (true, entity)
        } else if let Some(entity) = label.strip_prefix("I-") {
            (false, entity)
        } else {
            return Err(PipelineError::new(format!(
                "invalid BIO label in config: {label}"
            )));
        };

        let should_open = is_begin || current.as_ref().is_none_or(|open| open.label != entity);
        if should_open {
            if let Some(open) = current.take() {
                spans.push(close(open));
            }
            current = Some(OpenSpan {
                label: entity.to_owned(),
                start: word.start,
                end: word.end,
                scores: vec![word.score],
            });
        } else if let Some(open) = current.as_mut() {
            open.end = word.end;
            open.scores.push(word.score);
        }
    }
    if let Some(open) = current {
        spans.push(close(open));
    }
    Ok(spans)
}

fn reconcile(window_spans: Vec<Vec<CandidateSpan>>) -> Vec<CandidateSpan> {
    let mut final_spans: Vec<CandidateSpan> = Vec::new();
    for candidates in window_spans {
        for candidate in candidates {
            let overlaps = final_spans
                .iter()
                .enumerate()
                .filter_map(|(index, existing)| {
                    (existing.start < candidate.end && candidate.start < existing.end)
                        .then_some(index)
                })
                .collect::<Vec<_>>();
            match overlaps.as_slice() {
                [] => final_spans.push(candidate),
                [index]
                    if candidate.score > final_spans[*index].score
                        && overlap_ratio(&candidate, &final_spans[*index]) >= 0.5 =>
                {
                    final_spans[*index] = candidate;
                }
                _ => {}
            }
        }
    }
    final_spans.sort_by_key(|span| span.start);
    final_spans
}

fn overlap_ratio(first: &CandidateSpan, second: &CandidateSpan) -> f32 {
    let intersection = first
        .end
        .min(second.end)
        .saturating_sub(first.start.max(second.start));
    let union = (first.end - first.start) + (second.end - second.start) - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn apply_redaction(text: &str, spans: &[Span]) -> Result<String, PipelineError> {
    // Maps the character indices used by the API to valid UTF-8 byte boundaries
    // needed for Rust string slicing.
    let mut boundaries = text
        .char_indices()
        .map(|(byte, _)| byte)
        .collect::<Vec<_>>();
    boundaries.push(text.len());

    let mut output = String::with_capacity(text.len());
    let mut previous = 0;
    for span in spans {
        if span.start > span.end || span.end >= boundaries.len() || span.start < previous {
            return Err(PipelineError::new(
                "redaction spans are invalid or overlapping",
            ));
        }
        output.push_str(&text[boundaries[previous]..boundaries[span.start]]);
        output.push('[');
        output.push_str(&span.label);
        output.push(']');
        previous = span.end;
    }
    output.push_str(&text[boundaries[previous]..]);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{
        apply_redaction, build_windows, overlap_ratio, CandidateSpan, Span, WINDOW_SIZE,
        WINDOW_STRIDE,
    };

    #[test]
    fn windows_follow_the_spec() {
        assert_eq!(build_windows(0), vec![(0, 0)]);
        assert_eq!(build_windows(WINDOW_SIZE), vec![(0, WINDOW_SIZE)]);
        assert_eq!(
            build_windows(WINDOW_SIZE + 1),
            vec![(0, WINDOW_SIZE), (WINDOW_STRIDE, WINDOW_SIZE + 1)]
        );
    }

    #[test]
    fn redaction_uses_character_not_byte_offsets() {
        let text = "Asha 👩🏽‍💻 emailed ravi@example.com.";
        let email_start = text.chars().position(|character| character == 'r').unwrap();
        let email_end = text.chars().count() - 1;
        let result = apply_redaction(
            text,
            &[Span {
                label: "EMAIL".to_owned(),
                start: email_start,
                end: email_end,
            }],
        )
        .unwrap();
        assert_eq!(result, "Asha 👩🏽‍💻 emailed [EMAIL].");
    }

    #[test]
    fn overlap_ratio_is_intersection_over_union() {
        let first = CandidateSpan {
            label: "EMAIL".to_owned(),
            start: 0,
            end: 10,
            score: 1.0,
        };
        let second = CandidateSpan {
            label: "EMAIL".to_owned(),
            start: 5,
            end: 15,
            score: 1.0,
        };
        assert!((overlap_ratio(&first, &second) - (1.0 / 3.0)).abs() < f32::EPSILON);
    }
}
