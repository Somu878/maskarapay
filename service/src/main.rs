mod pipeline;

use std::{
    env,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use pipeline::{Pipeline, Prediction, Span, DEFAULT_MAX_TEXT_CHARS, MODEL_ID};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tracing::{error, info};

const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8080";

#[derive(Clone)]
struct AppState {
    pipeline: Arc<Pipeline>,
    max_text_chars: usize,
    next_request_id: Arc<AtomicU64>,
}

#[derive(Debug, Deserialize)]
struct InferenceRequest {
    model: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct InferenceResponse {
    model: &'static str,
    redacted_text: String,
    spans: Vec<Span>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Usage {
    input_chars: usize,
    input_tokens: usize,
    windows: usize,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    model: &'static str,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
        }
    }

    fn payload_too_large(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            code: "text_too_long",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "inference_failed",
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct ErrorResponse {
            error: ErrorDetail,
        }
        #[derive(Serialize)]
        struct ErrorDetail {
            code: &'static str,
            message: String,
        }

        (
            self.status,
            Json(ErrorResponse {
                error: ErrorDetail {
                    code: self.code,
                    message: self.message,
                },
            }),
        )
            .into_response()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let assets_dir = env::var_os("MASKARA_ASSETS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_assets_dir);
    let explicit_model_path = env::var_os("MASKARA_MODEL_PATH").map(PathBuf::from);
    let max_text_chars = env_usize("MASKARA_MAX_TEXT_CHARS", DEFAULT_MAX_TEXT_CHARS)?;
    let bind_addr: SocketAddr = env::var("MASKARA_BIND_ADDR")
        .unwrap_or_else(|_| DEFAULT_BIND_ADDR.to_owned())
        .parse()?;

    let pipeline = Pipeline::load(&assets_dir, explicit_model_path)?;
    info!(
        model = MODEL_ID,
        model_path = %pipeline.model_path().display(),
        max_text_chars,
        "inference model loaded"
    );

    // A JSON string can use up to four bytes per Unicode scalar plus a small
    // JSON envelope. This keeps oversized bodies out before deserialization.
    let max_body_bytes = max_text_chars
        .checked_mul(4)
        .and_then(|value| value.checked_add(1_024))
        .ok_or("MASKARA_MAX_TEXT_CHARS is too large")?;
    let state = AppState {
        pipeline: Arc::new(pipeline),
        max_text_chars,
        next_request_id: Arc::new(AtomicU64::new(1)),
    };
    let app = Router::new()
        .route("/healthz", get(health))
        .route("/redact", post(inference))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state);

    let listener = TcpListener::bind(bind_addr).await?;
    info!(address = %bind_addr, "Maskara inference service listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model: MODEL_ID,
    })
}

async fn inference(
    State(state): State<AppState>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, ApiError> {
    if request.model != MODEL_ID {
        return Err(ApiError::bad_request(
            "unsupported_model",
            format!("model must be {MODEL_ID}"),
        ));
    }

    let input_chars = request.text.chars().count();
    if input_chars > state.max_text_chars {
        return Err(ApiError::payload_too_large(format!(
            "text exceeds the {max} character limit",
            max = state.max_text_chars
        )));
    }

    let request_id = state.next_request_id.fetch_add(1, Ordering::Relaxed);
    let input_bytes = request.text.len();
    let started = Instant::now();
    let pipeline = Arc::clone(&state.pipeline);
    let prediction = tokio::task::spawn_blocking(move || pipeline.predict(&request.text))
        .await
        .map_err(|error| {
            error!(request_id, error = %error, "inference task failed");
            ApiError::internal("inference task failed")
        })?
        .map_err(|error| {
            error!(request_id, error = %error, "inference failed");
            ApiError::internal("inference failed")
        })?;

    log_completed_request(
        request_id,
        input_chars,
        input_bytes,
        &prediction,
        started.elapsed(),
    );
    Ok(Json(InferenceResponse {
        model: MODEL_ID,
        redacted_text: prediction.redacted_text,
        spans: prediction.spans,
        usage: Usage {
            input_chars,
            input_tokens: prediction.input_tokens,
            windows: prediction.windows,
        },
    }))
}

fn log_completed_request(
    request_id: u64,
    input_chars: usize,
    input_bytes: usize,
    prediction: &Prediction,
    elapsed: std::time::Duration,
) {
    // Never include original or redacted request text in application logs.
    info!(
        request_id,
        input_chars,
        input_bytes,
        input_tokens = prediction.input_tokens,
        windows = prediction.windows,
        spans = prediction.spans.len(),
        latency_ms = elapsed.as_millis(),
        "inference completed"
    );
}

fn default_assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets")
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match env::var(name) {
        Ok(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                Err(format!("{name} must be greater than zero").into())
            } else {
                Ok(parsed)
            }
        }
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "service=info".into()),
        )
        .without_time()
        .init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
