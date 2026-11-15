//! Local x402 payment gate for `POST /api/v1/redact`.
//!
//! Hand-written implementation of the x402 HTTP flow (no `x402-axum`
//! crate): a 402 challenge for both protocol versions, validation of the
//! incoming signed payment, a facilitator HTTP client (`/verify`, `/settle`)
//! and an in-memory replay cache. Settle-first: the payment is verified and
//! settled *before* the wrapped handler runs, so the gate is handler-agnostic
//! and works around any Axum handler.
//!
//! Protocol reference: `docs/x402.md`, `scripts/x402/pay_client.py` and
//! `scripts/x402/mock_facilitator.py` define the exact wire contract:
//! challenges/payments/receipts are base64-encoded compact JSON exchanged in
//! `PAYMENT-REQUIRED` / `PAYMENT-SIGNATURE` / `PAYMENT-RESPONSE` headers
//! (v2) and the 402 body / `X-PAYMENT` / `X-PAYMENT-RESPONSE` (v1).
//!
//! Logging rule (repo rule): never log request text, redacted output or
//! payment payloads/signatures. Addresses, transaction hashes, amounts,
//! outcome codes and latencies are fine.

use std::{
    collections::HashMap,
    env,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use base64::{
    engine::general_purpose::{
        GeneralPurpose, STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD,
    },
    Engine,
};
use serde_json::{json, Map, Value};
use tracing::{error, info, warn};

// --- wire constants ----------------------------------------------------------

const HEADER_CHALLENGE_V2: &str = "payment-required";
const HEADER_PAYMENT_V2: &str = "payment-signature";
const HEADER_RECEIPT_V2: &str = "payment-response";
const HEADER_PAYMENT_V1: &str = "x-payment";
const HEADER_RECEIPT_V1: &str = "x-payment-response";

const RESOURCE_DESCRIPTION: &str = "MaskaraPay PII redaction";
const RESOURCE_MIME_TYPE: &str = "application/json";
const DEFAULT_MAX_TIMEOUT_SECONDS: u64 = 60;

const DEFAULT_PRICE_USD: &str = "0.01";
const DEFAULT_NETWORK: &str = "base-sepolia";
const DEFAULT_FACILITATOR_URL: &str = "https://x402.org/facilitator";

/// Facilitator calls fail closed after this long.
const FACILITATOR_TIMEOUT_SECONDS: u64 = 10;
/// Seen payment nonces stay cached this long past their own expiry, so a
/// client that lost a paid response cannot replay the same authorization.
const REPLAY_GRACE_SECONDS: u64 = 10 * 60;
const REPLAY_CACHE_CAPACITY: usize = 10_000;

struct NetworkSpec {
    key: &'static str,
    caip2: &'static str,
    /// Kept for parity with the documented network table (clients use it for
    /// EIP-712 signing); the server itself never needs the chain id.
    #[allow(dead_code)]
    chain_id: u64,
    usdc: &'static str,
    domain_name: &'static str,
    domain_version: &'static str,
}

const BASE_SEPOLIA: NetworkSpec = NetworkSpec {
    key: "base-sepolia",
    caip2: "eip155:84532",
    chain_id: 84532,
    usdc: "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    domain_name: "USDC",
    domain_version: "2",
};

const BASE: NetworkSpec = NetworkSpec {
    key: "base",
    caip2: "eip155:8453",
    chain_id: 8453,
    usdc: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    domain_name: "USD Coin",
    domain_version: "2",
};

fn network_spec(key: &str) -> Option<&'static NetworkSpec> {
    match key {
        "base-sepolia" => Some(&BASE_SEPOLIA),
        "base" => Some(&BASE),
        _ => None,
    }
}

// --- configuration -----------------------------------------------------------

/// Everything the gate needs, built once at startup (from the environment in
/// `main`, or directly in tests — never read the environment inside tests).
#[derive(Debug, Clone)]
pub struct PaymentConfig {
    /// `MASKARA_PAYEE_ADDRESS` set and non-empty. Unset means the route runs
    /// free with no payment headers at all.
    pub enabled: bool,
    pub payee: String,
    /// USDC atomic units (6 decimals) as a decimal string, e.g. "10000".
    pub amount_atomic: String,
    /// The original `MASKARA_PRICE_USD` string, for logging only.
    pub price_usd: String,
    /// `base-sepolia` or `base`.
    pub network: String,
    /// Facilitator base URL (no trailing slash); `/verify` and `/settle` are
    /// appended.
    pub facilitator_url: String,
    /// Precomputed `Basic <base64(user:secret)>` header value, if configured.
    pub facilitator_auth_header: Option<String>,
    /// URL advertised in 402 challenges.
    pub resource_url: String,
    pub max_timeout_seconds: u64,
}

impl PaymentConfig {
    /// Read the payment environment variables. Called from `main` only.
    pub fn from_env(bind_addr: &str) -> Result<PaymentConfig, String> {
        let payee = env::var("MASKARA_PAYEE_ADDRESS")
            .unwrap_or_default()
            .trim()
            .to_string();
        let price_usd =
            env::var("MASKARA_PRICE_USD").unwrap_or_else(|_| DEFAULT_PRICE_USD.to_string());
        let amount_atomic = parse_price_usd(&price_usd)?;
        let network = env::var("MASKARA_X402_NETWORK")
            .unwrap_or_else(|_| DEFAULT_NETWORK.to_string())
            .trim()
            .to_string();
        if network_spec(&network).is_none() {
            return Err(format!(
                "MASKARA_X402_NETWORK {network:?} is not supported (use \"base-sepolia\" or \"base\")"
            ));
        }
        let facilitator_url = env::var("MASKARA_FACILITATOR_URL")
            .unwrap_or_else(|_| DEFAULT_FACILITATOR_URL.to_string());
        let facilitator_url = facilitator_url.trim().trim_end_matches('/').to_string();
        if facilitator_url.is_empty() {
            return Err("MASKARA_FACILITATOR_URL is empty".to_string());
        }
        let facilitator_auth_header = match env::var("MASKARA_FACILITATOR_AUTH") {
            Ok(raw) if !raw.trim().is_empty() => Some(basic_auth_header(raw.trim())?),
            _ => None,
        };
        let resource_url = match env::var("MASKARA_RESOURCE_URL") {
            Ok(raw) if !raw.trim().is_empty() => raw.trim().to_string(),
            _ => format!("http://{bind_addr}/api/v1/redact"),
        };
        let enabled = !payee.is_empty();
        if enabled && !is_evm_address(&payee) {
            return Err(format!(
                "MASKARA_PAYEE_ADDRESS {payee:?} is not a valid EVM address (0x + 40 hex chars)"
            ));
        }
        Ok(PaymentConfig {
            enabled,
            payee,
            amount_atomic,
            price_usd,
            network,
            facilitator_url,
            facilitator_auth_header,
            resource_url,
            max_timeout_seconds: DEFAULT_MAX_TIMEOUT_SECONDS,
        })
    }

    fn network(&self) -> &'static NetworkSpec {
        network_spec(&self.network).expect("network key validated at startup")
    }
}

fn basic_auth_header(user_secret: &str) -> Result<String, String> {
    let Some((user, secret)) = user_secret.split_once(':') else {
        return Err("MASKARA_FACILITATOR_AUTH must be \"user:secret\"".to_string());
    };
    Ok(format!(
        "Basic {}",
        STANDARD.encode(format!("{user}:{secret}"))
    ))
}

/// Convert a decimal USD price into USDC atomic units (6 decimals) as a
/// string: "0.01" -> "10000", "0.001" -> "1000".
///
/// Strict parsing: plain decimal with an integer part and at most 6
/// fractional digits; no sign, exponent, empty or zero values. Rounding is
/// never applied — "0.0000001" is an error.
pub fn parse_price_usd(raw: &str) -> Result<String, String> {
    let text = raw.trim();
    let invalid = || format!("MASKARA_PRICE_USD {raw:?} is not a decimal USD amount like \"0.01\"");
    if text.is_empty() {
        return Err(invalid());
    }
    let (integer, fraction) = match text.split_once('.') {
        Some((integer, fraction)) => {
            if fraction.is_empty() {
                return Err(invalid());
            }
            (integer, fraction)
        }
        None => (text, ""),
    };
    if integer.is_empty() || !integer.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }
    if !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }
    if fraction.len() > 6 {
        return Err(format!(
            "MASKARA_PRICE_USD {raw:?} has more than 6 decimal places; USDC only has 6"
        ));
    }
    let whole: u128 = integer.parse().map_err(|_| invalid())?;
    let fractional: u128 = if fraction.is_empty() {
        0
    } else {
        format!("{fraction:0<6}").parse().map_err(|_| invalid())?
    };
    let atomic = whole
        .checked_mul(1_000_000)
        .and_then(|value| value.checked_add(fractional))
        .ok_or_else(invalid)?;
    if atomic == 0 {
        return Err("MASKARA_PRICE_USD must be greater than zero".to_string());
    }
    Ok(atomic.to_string())
}

// --- base64 JSON helpers -----------------------------------------------------

/// Compact JSON -> standard base64 (padded), exactly what the client expects.
fn encode_b64_json(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).expect("serde_json Value serialization cannot fail");
    STANDARD.encode(bytes)
}

/// base64 (standard or urlsafe, with or without padding) -> JSON value.
/// Tolerates missing padding like the Python client does.
fn decode_b64_json(raw: &str) -> Result<Value, String> {
    let text = raw.trim();
    if text.is_empty() {
        return Err("payment header is empty".to_string());
    }
    let engines: [&GeneralPurpose; 4] = [&STANDARD, &STANDARD_NO_PAD, &URL_SAFE, &URL_SAFE_NO_PAD];
    for engine in engines {
        if let Some(value) = decode_with(engine, text) {
            return Ok(value);
        }
    }
    Err("payment header is not valid base64 JSON".to_string())
}

fn decode_with<E: Engine>(engine: &E, text: &str) -> Option<Value> {
    let bytes = engine.decode(text).ok()?;
    serde_json::from_slice(&bytes).ok()
}

// --- challenge and receipt builders ------------------------------------------

fn v2_requirement(config: &PaymentConfig) -> Value {
    let spec = config.network();
    json!({
        "scheme": "exact",
        "network": spec.caip2,
        "amount": config.amount_atomic,
        "asset": spec.usdc,
        "payTo": config.payee,
        "maxTimeoutSeconds": config.max_timeout_seconds,
        "extra": {"name": spec.domain_name, "version": spec.domain_version},
    })
}

fn v1_requirement(config: &PaymentConfig, with_version: bool) -> Value {
    let spec = config.network();
    let mut value = json!({
        "scheme": "exact",
        "network": spec.key,
        "maxAmountRequired": config.amount_atomic,
        "asset": spec.usdc,
        "payTo": config.payee,
        "maxTimeoutSeconds": config.max_timeout_seconds,
        "resource": config.resource_url,
        "description": RESOURCE_DESCRIPTION,
        "mimeType": RESOURCE_MIME_TYPE,
    });
    if with_version {
        value["x402Version"] = json!(1);
    }
    value
}

/// The v2 402 challenge carried in the `PAYMENT-REQUIRED` header.
fn v2_challenge(config: &PaymentConfig, error: &str) -> Value {
    json!({
        "x402Version": 2,
        "error": error,
        "resource": {
            "url": config.resource_url,
            "description": RESOURCE_DESCRIPTION,
            "mimeType": RESOURCE_MIME_TYPE,
        },
        "accepts": [v2_requirement(config)],
    })
}

/// The v1 402 challenge: the 402 response body itself. Also carries the same
/// requirement under `accepts` so the bundled Python client can pick it the
/// same way it picks v2 entries.
fn v1_challenge_body(config: &PaymentConfig, error: &str) -> Value {
    let mut body = v1_requirement(config, true);
    body["error"] = json!(error);
    body["accepts"] = json!([v1_requirement(config, false)]);
    body
}

fn success_receipt(network: &str, transaction: &str, payer: &str) -> Value {
    json!({
        "success": true,
        "transaction": transaction,
        "network": network,
        "payer": payer,
    })
}

fn failure_receipt(network: &str, error_reason: &str, payer: &str) -> Value {
    json!({
        "success": false,
        "errorReason": error_reason,
        "transaction": "",
        "network": network,
        "payer": payer,
    })
}

// --- facilitator client ------------------------------------------------------

struct VerifyResponse {
    is_valid: bool,
    invalid_reason: String,
    payer: String,
}

struct SettleResponse {
    success: bool,
    error_reason: String,
    transaction: String,
    network: String,
    payer: String,
}

struct FacilitatorClient {
    http: reqwest::Client,
    base_url: String,
    auth_header: Option<String>,
}

impl FacilitatorClient {
    fn new(base_url: String, auth_header: Option<String>) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(FACILITATOR_TIMEOUT_SECONDS))
            .build()
            .map_err(|error| format!("failed to build facilitator HTTP client: {error}"))?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_header,
        })
    }

    async fn post_json(&self, path: &str, version: u8, body: &Value) -> Result<Value, String> {
        // v1 payments additionally carry their version at the top level; v2
        // stays with the plain two-key body. Extra keys are tolerated.
        let mut body = body.clone();
        if version == 1 {
            body["x402Version"] = json!(1);
        }
        let mut request = self
            .http
            .post(format!("{}{path}", self.base_url))
            .json(&body);
        if let Some(auth) = &self.auth_header {
            request = request.header(reqwest::header::AUTHORIZATION, auth);
        }
        let response = request
            .send()
            .await
            .map_err(|error| format!("facilitator request failed: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("facilitator returned HTTP {status}"));
        }
        let value: Value = response
            .json()
            .await
            .map_err(|error| format!("facilitator response was not valid JSON: {error}"))?;
        if !value.is_object() {
            return Err("facilitator response was not a JSON object".to_string());
        }
        Ok(value)
    }

    async fn verify(
        &self,
        version: u8,
        payment: &Value,
        requirements: &Value,
    ) -> Result<VerifyResponse, String> {
        let body = json!({
            "paymentPayload": payment,
            "paymentRequirements": requirements,
        });
        let value = self.post_json("/verify", version, &body).await?;
        let is_valid = value
            .get("isValid")
            .and_then(Value::as_bool)
            .ok_or_else(|| "facilitator /verify response has no usable isValid".to_string())?;
        Ok(VerifyResponse {
            is_valid,
            invalid_reason: string_field(&value, "invalidReason"),
            payer: string_field(&value, "payer"),
        })
    }

    async fn settle(
        &self,
        version: u8,
        payment: &Value,
        requirements: &Value,
    ) -> Result<SettleResponse, String> {
        let body = json!({
            "paymentPayload": payment,
            "paymentRequirements": requirements,
        });
        let value = self.post_json("/settle", version, &body).await?;
        let success = value
            .get("success")
            .and_then(Value::as_bool)
            .ok_or_else(|| "facilitator /settle response has no usable success".to_string())?;
        Ok(SettleResponse {
            success,
            error_reason: string_field(&value, "errorReason"),
            transaction: string_field(&value, "transaction"),
            network: string_field(&value, "network"),
            payer: string_field(&value, "payer"),
        })
    }
}

fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

// --- replay protection -------------------------------------------------------

/// Remembers the nonce of every payment that passed verification and went to
/// settlement, so the same signed authorization can never be settled (or
/// served) twice even if the client retries after losing a response.
struct ReplayCache {
    entries: Mutex<HashMap<String, u64>>,
    max_entries: usize,
}

impl ReplayCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries: max_entries.max(1),
        }
    }

    fn contains(&self, key: &str, now: u64) -> bool {
        let mut entries = self.lock();
        Self::prune(&mut entries, now, self.max_entries);
        entries.contains_key(key)
    }

    /// Returns false when the key was already present (concurrent duplicate).
    fn insert(&self, key: String, valid_before: u64, now: u64) -> bool {
        let mut entries = self.lock();
        Self::prune(&mut entries, now, self.max_entries);
        match entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(_) => false,
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(valid_before);
                true
            }
        }
    }

    /// Test/introspection helper (asserted in unit tests).
    #[cfg(test)]
    fn len(&self) -> usize {
        self.lock().len()
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<String, u64>> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn prune(entries: &mut HashMap<String, u64>, now: u64, max_entries: usize) {
        entries.retain(|_, valid_before| valid_before.saturating_add(REPLAY_GRACE_SECONDS) >= now);
        // Hard cap: best-effort eviction of arbitrary entries so the cache
        // cannot grow without bound under a flood of payments.
        while entries.len() >= max_entries {
            let Some(key) = entries.keys().next().cloned() else {
                break;
            };
            entries.remove(&key);
        }
    }
}

// --- payment validation ------------------------------------------------------

/// A payment header value that passed all local checks and may go to the
/// facilitator.
struct VerifiedPayment {
    /// The full decoded payment object, forwarded verbatim as `paymentPayload`.
    payload: Value,
    /// Payer address (`authorization.from`).
    payer: String,
    /// Lowercase nonce hex, the replay-cache key.
    nonce_key: String,
    /// `authorization.validBefore`, unix seconds.
    valid_before: u64,
}

enum GateReject {
    /// Bad base64/JSON/shape/format — client error, retrying helps nobody.
    Malformed(String),
    /// Well-formed but does not match the server's current requirements.
    Mismatch(String),
    /// 402 with the challenge (plus error text) and no receipt header.
    Challenge(String),
    /// Fail closed: the facilitator could not be reached or spoke nonsense.
    FacilitatorUnavailable(String),
    /// Settlement ran and failed; the client gets a failure receipt.
    SettleFailed {
        reason: String,
        receipt: Value,
        receipt_header: &'static str,
    },
}

fn require_hex_bytes(value: &str, expected_len: usize, what: &str) -> Result<(), GateReject> {
    let hex = value
        .strip_prefix("0x")
        .ok_or_else(|| GateReject::Malformed(format!("{what} must start with 0x")))?;
    if hex.len() != expected_len || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(GateReject::Malformed(format!(
            "{what} must be 0x followed by {expected_len} hex characters"
        )));
    }
    Ok(())
}

fn is_evm_address(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("0x") else {
        return false;
    };
    hex.len() == 40 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Decimal amount from a JSON string (or bare number, tolerated) field.
fn decimal_field(object: &Map<String, Value>, key: &str) -> Result<u128, GateReject> {
    match object.get(key) {
        Some(Value::String(text)) => text.parse::<u128>().map_err(|_| {
            GateReject::Malformed(format!("payment field {key:?} must be a decimal integer"))
        }),
        Some(Value::Number(number)) => number.as_u64().map(u128::from).ok_or_else(|| {
            GateReject::Malformed(format!("payment field {key:?} must be a decimal integer"))
        }),
        _ => Err(GateReject::Malformed(format!(
            "payment payload is missing {key:?}"
        ))),
    }
}

fn check_scheme_network(
    requirement: &Map<String, Value>,
) -> Result<&'static NetworkSpec, GateReject> {
    let scheme = requirement
        .get("scheme")
        .and_then(Value::as_str)
        .ok_or_else(|| GateReject::Mismatch("requirement has no \"scheme\"".to_string()))?;
    if scheme != "exact" {
        return Err(GateReject::Mismatch(format!(
            "unsupported scheme {scheme:?} (only \"exact\")"
        )));
    }
    let network = requirement
        .get("network")
        .and_then(Value::as_str)
        .ok_or_else(|| GateReject::Mismatch("requirement has no \"network\"".to_string()))?
        .to_ascii_lowercase();
    for spec in [&BASE_SEPOLIA, &BASE] {
        if network == spec.key || network == spec.caip2 {
            return Ok(spec);
        }
    }
    Err(GateReject::Mismatch(format!(
        "unsupported network {network:?}"
    )))
}

fn check_amount(
    config: &PaymentConfig,
    requirement: &Map<String, Value>,
    field: &str,
) -> Result<(), GateReject> {
    let offered = decimal_field(requirement, field)?;
    let required: u128 = config
        .amount_atomic
        .parse()
        .expect("amount_atomic produced by parse_price_usd");
    if offered != required {
        return Err(GateReject::Mismatch(format!(
            "requirement {field} {offered} does not match the requested price {required}"
        )));
    }
    Ok(())
}

fn check_asset_and_pay_to(
    config: &PaymentConfig,
    requirement: &Map<String, Value>,
) -> Result<(), GateReject> {
    let spec = config.network();
    let asset = requirement
        .get("asset")
        .and_then(Value::as_str)
        .ok_or_else(|| GateReject::Mismatch("requirement has no \"asset\"".to_string()))?;
    if !asset.eq_ignore_ascii_case(spec.usdc) {
        return Err(GateReject::Mismatch(format!(
            "requirement asset {asset:?} is not the network's USDC"
        )));
    }
    let pay_to = requirement
        .get("payTo")
        .and_then(Value::as_str)
        .ok_or_else(|| GateReject::Mismatch("requirement has no \"payTo\"".to_string()))?;
    if !pay_to.eq_ignore_ascii_case(&config.payee) {
        return Err(GateReject::Mismatch(
            "requirement payTo does not match this server's payee".to_string(),
        ));
    }
    Ok(())
}

/// Locally validate a decoded payment header value before anything is sent
/// to the facilitator. `header_version` comes from which header carried the
/// payment (2 = PAYMENT-SIGNATURE, 1 = X-PAYMENT) and must agree with the
/// payload's `x402Version`.
fn validate_payment(
    config: &PaymentConfig,
    header_version: u8,
    raw: &str,
    now: u64,
) -> Result<VerifiedPayment, GateReject> {
    let decoded = decode_b64_json(raw).map_err(GateReject::Malformed)?;
    let Some(object) = decoded.as_object() else {
        return Err(GateReject::Malformed(
            "payment header must decode to a JSON object".to_string(),
        ));
    };
    let version = object
        .get("x402Version")
        .and_then(Value::as_u64)
        .filter(|value| *value == 1 || *value == 2)
        .ok_or_else(|| GateReject::Malformed("payment x402Version must be 1 or 2".to_string()))?
        as u8;
    if version != header_version {
        return Err(GateReject::Malformed(format!(
            "payment x402Version {version} does not match the payment header used"
        )));
    }

    match version {
        2 => {
            let accepted = object
                .get("accepted")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    GateReject::Malformed(
                        "v2 payment is missing the accepted requirement".to_string(),
                    )
                })?;
            check_scheme_network(accepted)?;
            check_amount(config, accepted, "amount")?;
            check_asset_and_pay_to(config, accepted)?;
        }
        1 => {
            // v1 envelopes carry scheme/network at the top level instead of an
            // accepted requirement object.
            check_scheme_network(object)?;
        }
        _ => unreachable!("version filtered above"),
    }

    let payload = object
        .get("payload")
        .and_then(Value::as_object)
        .ok_or_else(|| GateReject::Malformed("payment is missing \"payload\"".to_string()))?;
    let signature = payload
        .get("signature")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            GateReject::Malformed("payment payload is missing \"signature\"".to_string())
        })?;
    require_hex_bytes(signature, 130, "payment signature")?;

    let authorization = payload
        .get("authorization")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            GateReject::Malformed("payment payload is missing \"authorization\"".to_string())
        })?;
    for field in ["from", "to", "value", "validAfter", "validBefore", "nonce"] {
        if !authorization.contains_key(field) {
            return Err(GateReject::Malformed(format!(
                "payment authorization is missing {field:?}"
            )));
        }
    }
    let payer = authorization["from"]
        .as_str()
        .ok_or_else(|| GateReject::Malformed("authorization.from must be a string".to_string()))?
        .to_string();
    if !is_evm_address(&payer) {
        return Err(GateReject::Malformed(
            "authorization.from must be an EVM address (0x + 40 hex chars)".to_string(),
        ));
    }
    let to = authorization["to"]
        .as_str()
        .ok_or_else(|| GateReject::Malformed("authorization.to must be a string".to_string()))?;
    if !is_evm_address(to) {
        return Err(GateReject::Malformed(
            "authorization.to must be an EVM address (0x + 40 hex chars)".to_string(),
        ));
    }
    if !to.eq_ignore_ascii_case(&config.payee) {
        return Err(GateReject::Mismatch(
            "authorization.to does not match this server's payee".to_string(),
        ));
    }
    let value = decimal_field(authorization, "value")?;
    let required: u128 = config
        .amount_atomic
        .parse()
        .expect("amount_atomic produced by parse_price_usd");
    if value != required {
        return Err(GateReject::Mismatch(format!(
            "authorization.value {value} does not match the requested price {required}"
        )));
    }
    let valid_after = decimal_field(authorization, "validAfter")?;
    let valid_before = decimal_field(authorization, "validBefore")?;
    let nonce = authorization["nonce"]
        .as_str()
        .ok_or_else(|| GateReject::Malformed("authorization.nonce must be a string".to_string()))?;
    require_hex_bytes(nonce, 64, "authorization nonce")?;

    // Validity-window basics: the authorization must not be expired yet, and
    // the window must be ordered. A past `validAfter` is allowed (clock skew);
    // no upper bound on `validBefore` is enforced.
    let now_wide = u128::from(now);
    if valid_before <= now_wide {
        return Err(GateReject::Challenge("payment_expired".to_string()));
    }
    if valid_after >= valid_before {
        return Err(GateReject::Challenge(
            "authorization_window_invalid".to_string(),
        ));
    }

    let nonce_key = nonce[2..].to_ascii_lowercase();
    Ok(VerifiedPayment {
        payload: decoded,
        payer,
        nonce_key,
        valid_before: u64::try_from(valid_before).unwrap_or(u64::MAX),
    })
}

// --- middleware --------------------------------------------------------------

/// Shared gate state: config, replay cache and facilitator client.
#[derive(Clone)]
pub struct PaymentState {
    config: Arc<PaymentConfig>,
    replay: Arc<ReplayCache>,
    facilitator: Arc<FacilitatorClient>,
}

impl PaymentState {
    pub fn new(config: PaymentConfig) -> Result<PaymentState, String> {
        let facilitator = FacilitatorClient::new(
            config.facilitator_url.clone(),
            config.facilitator_auth_header.clone(),
        )?;
        Ok(PaymentState {
            config: Arc::new(config),
            replay: Arc::new(ReplayCache::new(REPLAY_CACHE_CAPACITY)),
            facilitator: Arc::new(facilitator),
        })
    }
}

/// The payment gate. Apply per-route with
/// `middleware::from_fn_with_state(state, payment::gate)`; handler-agnostic —
/// the wrapped handler's body extraction works untouched. Payments disabled
/// (`enabled == false`) is a pure pass-through.
pub async fn gate(State(state): State<PaymentState>, request: Request, next: Next) -> Response {
    if !state.config.enabled {
        return next.run(request).await;
    }
    let started = Instant::now();

    let header_version = match (
        request.headers().get(HEADER_PAYMENT_V2),
        request.headers().get(HEADER_PAYMENT_V1),
    ) {
        (Some(_), Some(_)) => {
            warn!(
                outcome = "malformed",
                "payment gate: both payment headers present"
            );
            return GateReject::Malformed(
                "only one of PAYMENT-SIGNATURE or X-PAYMENT may be sent".to_string(),
            )
            .into_response_with(&state.config);
        }
        (Some(_), None) => 2u8,
        (None, Some(_)) => 1u8,
        (None, None) => {
            info!(
                outcome = "challenge",
                network = %state.config.network,
                amount = %state.config.amount_atomic,
                latency_ms = started.elapsed().as_millis(),
                "payment gate: no payment, challenge issued"
            );
            return GateReject::Challenge(String::new()).into_response_with(&state.config);
        }
    };

    let raw = match request
        .headers()
        .get(if header_version == 2 {
            HEADER_PAYMENT_V2
        } else {
            HEADER_PAYMENT_V1
        })
        .and_then(|value| value.to_str().ok())
    {
        Some(raw) => raw,
        None => {
            warn!(
                outcome = "malformed",
                "payment gate: payment header is not ASCII"
            );
            return GateReject::Malformed("payment header is not valid".to_string())
                .into_response_with(&state.config);
        }
    };

    let now = unix_now();
    let outcome = process_payment(&state, header_version, raw, now).await;
    let latency_ms = started.elapsed().as_millis();
    let success = match outcome {
        Ok(success) => {
            info!(
                outcome = "paid",
                version = header_version,
                payer = %success.payer,
                transaction = %success.transaction,
                network = %state.config.network,
                amount = %state.config.amount_atomic,
                latency_ms = latency_ms,
                "payment gate: settled, serving request"
            );
            success
        }
        Err(reject) => {
            match &reject {
                GateReject::Malformed(message) => {
                    error!(outcome = "malformed", version = header_version, message = %message, "payment gate: malformed payment")
                }
                GateReject::Mismatch(message) => {
                    warn!(outcome = "mismatch", version = header_version, message = %message, "payment gate: requirements mismatch")
                }
                GateReject::Challenge(error) => {
                    warn!(outcome = "payment_rejected", version = header_version, error = %error, "payment gate: payment rejected")
                }
                GateReject::FacilitatorUnavailable(message) => {
                    error!(outcome = "facilitator_unavailable", version = header_version, message = %message, "payment gate: facilitator unavailable")
                }
                GateReject::SettleFailed { reason, .. } => {
                    warn!(outcome = "settle_failed", version = header_version, reason = %reason, "payment gate: settlement failed")
                }
            }
            return reject.into_response_with(&state.config);
        }
    };

    // Settlement succeeded — now (and only now) run the wrapped handler. If
    // inference fails afterwards the charge stays; no refund flow exists.
    let mut response = next.run(request).await;
    if let Ok(value) = HeaderValue::from_str(&success.receipt_value) {
        response.headers_mut().insert(success.receipt_header, value);
    }
    response
}

struct SuccessPayment {
    receipt_header: &'static str,
    receipt_value: String,
    payer: String,
    transaction: String,
}

async fn process_payment(
    state: &PaymentState,
    header_version: u8,
    raw: &str,
    now: u64,
) -> Result<SuccessPayment, GateReject> {
    let config = &state.config;
    let payment = validate_payment(config, header_version, raw, now)?;

    if state.replay.contains(&payment.nonce_key, now) {
        return Err(GateReject::Challenge("payment_already_used".to_string()));
    }

    let requirements = match header_version {
        2 => v2_requirement(config),
        1 => v1_requirement(config, true),
        _ => unreachable!("header_version is 1 or 2"),
    };

    let verification = state
        .facilitator
        .verify(header_version, &payment.payload, &requirements)
        .await
        .map_err(GateReject::FacilitatorUnavailable)?;
    if !verification.is_valid {
        let reason = if verification.invalid_reason.is_empty() {
            "verification_failed".to_string()
        } else {
            verification.invalid_reason
        };
        return Err(GateReject::Challenge(reason));
    }

    // Mark the nonce used *before* settling: a retry of this exact payment —
    // even after a lost settle response — is refused here without another
    // facilitator round trip (no duplicate charge, no free result).
    if !state
        .replay
        .insert(payment.nonce_key.clone(), payment.valid_before, now)
    {
        return Err(GateReject::Challenge("payment_already_used".to_string()));
    }

    let settlement = state
        .facilitator
        .settle(header_version, &payment.payload, &requirements)
        .await
        .map_err(GateReject::FacilitatorUnavailable)?;

    let spec = config.network();
    // The facilitator's recovered payer is the most authoritative, then its
    // verify payer, then our own parsed authorization.from.
    let payer_owned = if !settlement.payer.is_empty() {
        settlement.payer.clone()
    } else if !verification.payer.is_empty() {
        verification.payer.clone()
    } else {
        payment.payer.clone()
    };
    let payer = payer_owned.as_str();
    let receipt_network = if settlement.network.is_empty() {
        match header_version {
            2 => spec.caip2,
            1 => spec.key,
            _ => unreachable!(),
        }
    } else {
        settlement.network.as_str()
    };

    if !settlement.success {
        let reason = if settlement.error_reason.is_empty() {
            "settlement_failed".to_string()
        } else {
            settlement.error_reason.clone()
        };
        return Err(GateReject::SettleFailed {
            reason: reason.clone(),
            receipt: failure_receipt(receipt_network, &reason, payer),
            receipt_header: receipt_header_name(header_version),
        });
    }

    let receipt_value = encode_b64_json(&success_receipt(
        receipt_network,
        &settlement.transaction,
        payer,
    ));
    Ok(SuccessPayment {
        receipt_header: receipt_header_name(header_version),
        receipt_value,
        payer: payment.payer,
        transaction: settlement.transaction,
    })
}

fn receipt_header_name(version: u8) -> &'static str {
    if version == 2 {
        HEADER_RECEIPT_V2
    } else {
        HEADER_RECEIPT_V1
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or_default()
}

// --- reject responses --------------------------------------------------------

impl GateReject {
    /// Responses that carry the challenge (v2 header + v1 body) get it built
    /// here; the others are plain JSON errors.
    fn into_response_with(self, config: &PaymentConfig) -> Response {
        let error_json = |status: StatusCode, code: &'static str, message: String| {
            (
                status,
                Json(json!({"error": {"code": code, "message": message}})),
            )
                .into_response()
        };
        match self {
            GateReject::Malformed(message) => {
                error_json(StatusCode::BAD_REQUEST, "malformed_payment", message)
            }
            GateReject::Mismatch(message) => error_json(
                StatusCode::BAD_REQUEST,
                "payment_requirements_mismatch",
                message,
            ),
            GateReject::Challenge(error) => {
                let mut response = (
                    StatusCode::PAYMENT_REQUIRED,
                    Json(v1_challenge_body(config, &error)),
                )
                    .into_response();
                if let Ok(value) =
                    HeaderValue::from_str(&encode_b64_json(&v2_challenge(config, &error)))
                {
                    response.headers_mut().insert(HEADER_CHALLENGE_V2, value);
                }
                response
            }
            GateReject::FacilitatorUnavailable(message) => error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                "facilitator_unavailable",
                message,
            ),
            GateReject::SettleFailed {
                reason,
                receipt,
                receipt_header,
            } => {
                let mut response = (
                    StatusCode::PAYMENT_REQUIRED,
                    Json(json!({"error": {"code": "settlement_failed", "message": reason}})),
                )
                    .into_response();
                if let Ok(value) = HeaderValue::from_str(&encode_b64_json(&receipt)) {
                    response.headers_mut().insert(receipt_header, value);
                }
                response
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- price parsing -------------------------------------------------------

    #[test]
    fn price_parsing_converts_usd_to_usdc_atomic_units() {
        assert_eq!(parse_price_usd("0.01").unwrap(), "10000");
        assert_eq!(parse_price_usd("0.001").unwrap(), "1000");
        assert_eq!(parse_price_usd("0.000001").unwrap(), "1");
        assert_eq!(parse_price_usd("1").unwrap(), "1000000");
        assert_eq!(parse_price_usd("1.5").unwrap(), "1500000");
        assert_eq!(parse_price_usd("0.123456").unwrap(), "123456");
        assert_eq!(parse_price_usd(" 0.01 ").unwrap(), "10000");
    }

    #[test]
    fn price_parsing_rejects_nonsense() {
        for bad in [
            "",
            "   ",
            "0",
            "0.0",
            "0.000000",
            "abc",
            "-0.01",
            "+0.01",
            "1e-3",
            "0.0000001",
            ".5",
            "5.",
            "1.2.3",
            "nan",
            "inf",
        ] {
            assert!(parse_price_usd(bad).is_err(), "expected error for {bad:?}");
        }
    }

    // --- base64 helpers ------------------------------------------------------

    #[test]
    fn base64_json_round_trips_compact() {
        let value = json!({
            "x402Version": 2,
            "nested": {"list": [1, 2, 3], "text": "héllo ↔ ð"},
            "amount": "10000",
        });
        let encoded = encode_b64_json(&value);
        // Compact separators, standard padded alphabet.
        assert!(!encoded.contains('-') && !encoded.contains('_'));
        assert_eq!(
            encoded,
            base64::engine::general_purpose::STANDARD.encode(serde_json::to_vec(&value).unwrap())
        );
        assert_eq!(decode_b64_json(&encoded).unwrap(), value);
    }

    #[test]
    fn base64_json_decodes_urlsafe_and_unpadded_variants() {
        let value = json!({"ok": true});
        let encoded = encode_b64_json(&value);
        let padded_len = encoded.len();

        // URL-safe alphabet must still decode (only differs on +/ chars).
        let urlsafe =
            base64::engine::general_purpose::URL_SAFE.encode(serde_json::to_vec(&value).unwrap());
        assert_eq!(decode_b64_json(&urlsafe).unwrap(), value);

        // Missing padding tolerated.
        let unpadded: String = encoded.trim_end_matches('=').to_string();
        assert!(unpadded.len() < padded_len || encoded.ends_with('='));
        assert_eq!(decode_b64_json(&unpadded).unwrap(), value);

        // Garbage is rejected.
        assert!(decode_b64_json("not base64 !!").is_err());
        assert!(decode_b64_json("").is_err());
        // Valid base64 that is not JSON is rejected.
        let not_json = STANDARD.encode(b"plain bytes");
        assert!(decode_b64_json(&not_json).is_err());
    }

    // --- network table -------------------------------------------------------

    #[test]
    fn network_table_matches_the_documented_constants() {
        let sepolia = network_spec("base-sepolia").unwrap();
        assert_eq!(sepolia.caip2, "eip155:84532");
        assert_eq!(sepolia.chain_id, 84532);
        assert_eq!(sepolia.usdc, "0x036CbD53842c5426634e7929541eC2318f3dCF7e");
        assert_eq!(sepolia.domain_name, "USDC");
        assert_eq!(sepolia.domain_version, "2");

        let mainnet = network_spec("base").unwrap();
        assert_eq!(mainnet.caip2, "eip155:8453");
        assert_eq!(mainnet.chain_id, 8453);
        assert_eq!(mainnet.usdc, "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913");
        assert_eq!(mainnet.domain_name, "USD Coin");
        assert_eq!(mainnet.domain_version, "2");

        assert!(network_spec("ethereum").is_none());
        assert!(network_spec("base-sepolia ").is_none()); // must be trimmed upstream
    }

    // --- replay cache --------------------------------------------------------

    #[test]
    fn replay_cache_rejects_a_second_use_of_the_same_nonce() {
        let cache = ReplayCache::new(8);
        assert!(cache.insert("aa01".into(), 1_000, 500));
        assert!(cache.contains("aa01", 500));
        // Same nonce again -> rejected (concurrent duplicate or replay).
        assert!(!cache.insert("aa01".into(), 1_000, 500));
        // Unknown nonces stay unknown.
        assert!(!cache.contains("bb02", 500));
    }

    #[test]
    fn replay_cache_prunes_expired_entries() {
        let cache = ReplayCache::new(8);
        // valid_before 1000 stays live for REPLAY_GRACE_SECONDS past expiry.
        assert!(cache.insert("old".into(), 1_000, 1_100));
        // At 1_100 the entry is still live (1000 + 600 >= 1100).
        assert!(cache.contains("old", 1_100));
        assert_eq!(cache.len(), 1);
        // At 1_601 the grace window has passed -> pruned.
        assert!(cache.insert("fresh".into(), 9_999, 1_601));
        assert!(!cache.contains("old", 1_601));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.lock().get("fresh"), Some(&9_999));
    }

    #[test]
    fn replay_cache_is_capped() {
        let cache = ReplayCache::new(3);
        for index in 0..10 {
            assert!(cache.insert(format!("nonce-{index:04}"), 9_999, index));
        }
        assert!(cache.len() <= 3, "cache exceeded its cap: {}", cache.len());
    }
}

#[cfg(test)]
mod integration {
    use super::*;
    use axum::{
        middleware,
        routing::{get, post},
        Router,
    };
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    const PAYER: &str = "0x1111111111111111111111111111111111111111";
    const PAYEE: &str = "0x209693Bc6afc0C5328bA36FaF03C514EF312287C";
    const SEPOLIA_USDC: &str = "0x036CbD53842c5426634e7929541eC2318f3dCF7e";
    const AMOUNT: &str = "10000";

    // --- in-process mock facilitator (mirrors scripts/x402/mock_facilitator.py)

    #[derive(Default)]
    struct MockFacilitatorState {
        verify_calls: AtomicU64,
        settle_calls: AtomicU64,
        verify_fail: AtomicBool,
        settle_fail: AtomicBool,
        last_verify_body: Mutex<Option<Value>>,
        last_settle_body: Mutex<Option<Value>>,
    }

    fn mock_payer(body: &Value) -> String {
        body.pointer("/paymentPayload/payload/authorization/from")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    }

    async fn mock_verify(
        State(state): State<Arc<MockFacilitatorState>>,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        state.verify_calls.fetch_add(1, Ordering::SeqCst);
        *state.last_verify_body.lock().unwrap() = Some(body.clone());
        let payer = mock_payer(&body);
        if state.verify_fail.load(Ordering::SeqCst) {
            Json(json!({"isValid": false, "invalidReason": "insufficient_funds", "payer": payer}))
        } else {
            Json(json!({"isValid": true, "invalidReason": "", "payer": payer}))
        }
    }

    async fn mock_settle(
        State(state): State<Arc<MockFacilitatorState>>,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        state.settle_calls.fetch_add(1, Ordering::SeqCst);
        *state.last_settle_body.lock().unwrap() = Some(body.clone());
        let payer = mock_payer(&body);
        let network = body
            .pointer("/paymentRequirements/network")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if state.settle_fail.load(Ordering::SeqCst) {
            Json(json!({
                "success": false,
                "errorReason": "transaction_failed",
                "transaction": "",
                "network": network,
                "payer": payer,
            }))
        } else {
            Json(json!({
                "success": true,
                "errorReason": "",
                "transaction": format!("0x{}", "ef".repeat(32)),
                "network": network,
                "payer": payer,
            }))
        }
    }

    async fn spawn_mock_facilitator(state: Arc<MockFacilitatorState>) -> String {
        let app = Router::new()
            .route("/verify", post(mock_verify))
            .route("/settle", post(mock_settle))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("mock facilitator server");
        });
        format!("http://{address}")
    }

    // --- test service: the gate wrapped around a trivial stub handler ---------

    fn test_config(enabled: bool) -> PaymentConfig {
        PaymentConfig {
            enabled,
            payee: PAYEE.to_string(),
            amount_atomic: AMOUNT.to_string(),
            price_usd: "0.01".to_string(),
            network: "base-sepolia".to_string(),
            facilitator_url: "http://127.0.0.1:9".to_string(),
            facilitator_auth_header: None,
            resource_url: "http://127.0.0.1:9/api/v1/redact".to_string(),
            max_timeout_seconds: 60,
        }
    }

    async fn stub_redact() -> Json<Value> {
        Json(json!({"model": "stub", "redacted_text": "[REDACTED]", "spans": []}))
    }

    /// Binds the service on 127.0.0.1:0 with the gate layered over a stub
    /// handler and an in-process mock facilitator; returns the base URL and
    /// the mock's shared state.
    async fn spawn_service(
        mut config: PaymentConfig,
        facilitator_url: Option<String>,
    ) -> (String, Arc<MockFacilitatorState>) {
        let facilitator_state = Arc::new(MockFacilitatorState::default());
        let mock_url = spawn_mock_facilitator(Arc::clone(&facilitator_state)).await;
        config.facilitator_url = facilitator_url.unwrap_or(mock_url);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        config.resource_url = format!("http://{address}/api/v1/redact");
        let payment_state = PaymentState::new(config).unwrap();

        let app = Router::new()
            .route("/healthz", get(|| async { Json(json!({"status": "ok"})) }))
            .route(
                "/api/v1/redact",
                post(stub_redact).layer(middleware::from_fn_with_state(payment_state, gate)),
            )
            .with_state(());
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("test service");
        });
        (format!("http://{address}"), facilitator_state)
    }

    // --- payment builders -----------------------------------------------------

    fn dummy_signature() -> String {
        format!("0x{}", "ab".repeat(65))
    }

    fn nonce_hex(counter: u64) -> String {
        format!("0x{counter:064x}")
    }

    fn authorization(counter: u64, value: &str) -> Value {
        let now = unix_now();
        json!({
            "from": PAYER,
            "to": PAYEE,
            "value": value,
            "validAfter": (now.saturating_sub(30)).to_string(),
            "validBefore": (now + 600).to_string(),
            "nonce": nonce_hex(counter),
        })
    }

    fn v2_payment(requirement: &Value, counter: u64) -> Value {
        json!({
            "x402Version": 2,
            "resource": {
                "url": "http://resource.test/api/v1/redact",
                "description": RESOURCE_DESCRIPTION,
                "mimeType": RESOURCE_MIME_TYPE,
            },
            "accepted": requirement.clone(),
            "payload": {
                "signature": dummy_signature(),
                "authorization": authorization(counter, AMOUNT),
            },
        })
    }

    fn v1_payment(network: &str, counter: u64) -> Value {
        json!({
            "x402Version": 1,
            "scheme": "exact",
            "network": network,
            "payload": {
                "signature": dummy_signature(),
                "authorization": authorization(counter, AMOUNT),
            },
        })
    }

    async fn post_redact(
        client: &reqwest::Client,
        base_url: &str,
        payment_headers: &[(&'static str, String)],
    ) -> reqwest::Response {
        let mut request = client
            .post(format!("{base_url}/api/v1/redact"))
            .json(&json!({"model": "stub", "text": "hello world"}));
        for (name, value) in payment_headers {
            request = request.header(*name, value);
        }
        request.send().await.unwrap()
    }

    /// Initial unpaid request: the 402 response whose PAYMENT-REQUIRED header
    /// carries the v2 challenge and whose body carries the v1 one.
    async fn fetch_challenge(client: &reqwest::Client, base_url: &str) -> reqwest::Response {
        post_redact(client, base_url, &[]).await
    }

    fn decode_header(response: &reqwest::Response, name: &'static str) -> Value {
        let raw = response.headers().get(name).unwrap().to_str().unwrap();
        decode_b64_json(raw).unwrap()
    }

    // --- tests ----------------------------------------------------------------

    #[tokio::test]
    async fn payments_disabled_runs_free_with_no_payment_headers() {
        let (base_url, facilitator) = spawn_service(test_config(false), None).await;
        let client = reqwest::Client::new();
        let response = post_redact(&client, &base_url, &[]).await;

        assert_eq!(response.status(), 200);
        assert!(response.headers().get("payment-required").is_none());
        assert!(response.headers().get("payment-response").is_none());
        assert!(response.headers().get("x-payment-response").is_none());
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["redacted_text"], "[REDACTED]");
        assert_eq!(facilitator.verify_calls.load(Ordering::SeqCst), 0);
        assert_eq!(facilitator.settle_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn unpaid_request_gets_v2_header_challenge_and_v1_body_challenge() {
        let (base_url, _facilitator) = spawn_service(test_config(true), None).await;
        let client = reqwest::Client::new();
        let response = fetch_challenge(&client, &base_url).await;

        assert_eq!(response.status(), 402);
        assert_eq!(response.url().as_str(), format!("{base_url}/api/v1/redact"));

        // v2 challenge in the PAYMENT-REQUIRED header.
        let challenge = decode_header(&response, "payment-required");
        assert_eq!(challenge["x402Version"], 2);
        assert_eq!(challenge["error"], "");
        assert_eq!(
            challenge["resource"]["url"],
            format!("{base_url}/api/v1/redact")
        );
        assert_eq!(challenge["resource"]["description"], RESOURCE_DESCRIPTION);
        assert_eq!(challenge["resource"]["mimeType"], RESOURCE_MIME_TYPE);
        let requirement = &challenge["accepts"][0];
        assert_eq!(requirement["scheme"], "exact");
        assert_eq!(requirement["network"], "eip155:84532");
        assert_eq!(requirement["amount"], AMOUNT);
        assert_eq!(requirement["asset"], SEPOLIA_USDC);
        assert_eq!(requirement["payTo"], PAYEE);
        assert_eq!(requirement["maxTimeoutSeconds"], 60);
        assert_eq!(requirement["extra"]["name"], "USDC");
        assert_eq!(requirement["extra"]["version"], "2");

        // v1 challenge in the 402 body.
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["x402Version"], 1);
        assert_eq!(body["scheme"], "exact");
        assert_eq!(body["network"], "base-sepolia");
        assert_eq!(body["maxAmountRequired"], AMOUNT);
        assert_eq!(body["asset"], SEPOLIA_USDC);
        assert_eq!(body["payTo"], PAYEE);
        assert_eq!(body["resource"], format!("{base_url}/api/v1/redact"));
        assert_eq!(body["description"], RESOURCE_DESCRIPTION);
        assert_eq!(body["mimeType"], RESOURCE_MIME_TYPE);
        assert_eq!(body["accepts"][0]["maxAmountRequired"], AMOUNT);
        assert_eq!(body["accepts"][0]["network"], "base-sepolia");
    }

    #[tokio::test]
    async fn malformed_payments_rejected_without_redaction() {
        let (base_url, facilitator) = spawn_service(test_config(true), None).await;
        let client = reqwest::Client::new();
        let (_, requirement) = challenge_requirement(&client, &base_url).await;

        // (label, header name, raw header value)
        let cases: Vec<(&'static str, &'static str, String)> = vec![
            (
                "bad base64",
                "payment-signature",
                "!!!not-base64!!!".to_string(),
            ),
            (
                "valid base64, not JSON",
                "payment-signature",
                STANDARD.encode(b"plain bytes"),
            ),
            (
                "JSON but not an object",
                "payment-signature",
                encode_b64_json(&json!("hello")),
            ),
            (
                "missing accepted and payload",
                "payment-signature",
                encode_b64_json(&json!({"x402Version": 2})),
            ),
            (
                "bad signature format",
                "payment-signature",
                encode_b64_json(&with_signature(v2_payment(&requirement, 1), "0x1234")),
            ),
            (
                "bad nonce format",
                "payment-signature",
                encode_b64_json(&with_nonce(v2_payment(&requirement, 1), "0xzz")),
            ),
            (
                "missing authorization field",
                "payment-signature",
                encode_b64_json(&without_auth_field(
                    v2_payment(&requirement, 1),
                    "validBefore",
                )),
            ),
            ("v1: bad base64", "x-payment", "@@@garbage@@@".to_string()),
            (
                "v1: missing payload",
                "x-payment",
                encode_b64_json(
                    &json!({"x402Version": 1, "scheme": "exact", "network": "base-sepolia"}),
                ),
            ),
        ];

        for (label, header, value) in cases {
            let response = post_redact(&client, &base_url, &[(header, value)]).await;
            assert_eq!(response.status(), 400, "case {label}");
            let body: Value = response.json().await.unwrap();
            assert_eq!(body["error"]["code"], "malformed_payment", "case {label}");
            assert!(body.get("redacted_text").is_none(), "case {label}");
        }
        assert_eq!(facilitator.verify_calls.load(Ordering::SeqCst), 0);
        assert_eq!(facilitator.settle_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn mismatched_accepted_requirements_rejected_without_redaction() {
        let (base_url, facilitator) = spawn_service(test_config(true), None).await;
        let client = reqwest::Client::new();
        let (_, requirement) = challenge_requirement(&client, &base_url).await;

        let mut wrong_amount = requirement.clone();
        wrong_amount["amount"] = json!("20000");
        let mut wrong_network = requirement.clone();
        wrong_network["network"] = json!("eip155:1");
        let mut wrong_asset = requirement.clone();
        wrong_asset["asset"] = json!("0x9999999999999999999999999999999999999999");
        let mut wrong_pay_to = requirement.clone();
        wrong_pay_to["payTo"] = json!("0x9999999999999999999999999999999999999999");

        for (label, requirement) in [
            ("wrong amount", &wrong_amount),
            ("wrong network", &wrong_network),
            ("wrong asset", &wrong_asset),
            ("wrong payTo", &wrong_pay_to),
        ] {
            let response = post_redact(
                &client,
                &base_url,
                &[(
                    "payment-signature",
                    encode_b64_json(&v2_payment(requirement, 1)),
                )],
            )
            .await;
            assert_eq!(response.status(), 400, "case {label}");
            let body: Value = response.json().await.unwrap();
            assert_eq!(
                body["error"]["code"], "payment_requirements_mismatch",
                "case {label}"
            );
            assert!(body.get("redacted_text").is_none(), "case {label}");
        }
        assert_eq!(facilitator.verify_calls.load(Ordering::SeqCst), 0);
        assert_eq!(facilitator.settle_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn authorization_value_mismatch_is_rejected() {
        let (base_url, facilitator) = spawn_service(test_config(true), None).await;
        let client = reqwest::Client::new();
        let (_, requirement) = challenge_requirement(&client, &base_url).await;

        let mut payment = v2_payment(&requirement, 1);
        payment["payload"]["authorization"]["value"] = json!("20000");
        let response = post_redact(
            &client,
            &base_url,
            &[("payment-signature", encode_b64_json(&payment))],
        )
        .await;

        assert_eq!(response.status(), 400);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "payment_requirements_mismatch");
        assert!(body.get("redacted_text").is_none());
        assert_eq!(facilitator.verify_calls.load(Ordering::SeqCst), 0);
        assert_eq!(facilitator.settle_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn replayed_payment_rejected_without_second_settlement() {
        let (base_url, facilitator) = spawn_service(test_config(true), None).await;
        let client = reqwest::Client::new();
        let (_, requirement) = challenge_requirement(&client, &base_url).await;

        let header = encode_b64_json(&v2_payment(&requirement, 1));
        let first = post_redact(&client, &base_url, &[("payment-signature", header.clone())]).await;
        assert_eq!(first.status(), 200);

        let replay = post_redact(&client, &base_url, &[("payment-signature", header)]).await;
        assert_eq!(replay.status(), 402);
        assert!(replay.headers().get("payment-response").is_none());
        let challenge = decode_header(&replay, "payment-required");
        assert_eq!(challenge["error"], "payment_already_used");
        let body: Value = replay.json().await.unwrap();
        assert!(body.get("redacted_text").is_none());

        // Exactly one verify and one settle, from the first request only.
        assert_eq!(facilitator.verify_calls.load(Ordering::SeqCst), 1);
        assert_eq!(facilitator.settle_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn verify_failure_rejects_with_challenge_and_no_receipt() {
        let (base_url, facilitator) = spawn_service(test_config(true), None).await;
        facilitator.verify_fail.store(true, Ordering::SeqCst);
        let client = reqwest::Client::new();
        let (_, requirement) = challenge_requirement(&client, &base_url).await;

        let response = post_redact(
            &client,
            &base_url,
            &[(
                "payment-signature",
                encode_b64_json(&v2_payment(&requirement, 1)),
            )],
        )
        .await;

        assert_eq!(response.status(), 402);
        assert!(response.headers().get("payment-response").is_none());
        let challenge = decode_header(&response, "payment-required");
        assert_eq!(challenge["error"], "insufficient_funds");
        let body: Value = response.json().await.unwrap();
        assert!(body.get("redacted_text").is_none());
        assert_eq!(facilitator.verify_calls.load(Ordering::SeqCst), 1);
        assert_eq!(facilitator.settle_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn settle_failure_returns_failure_receipt_and_no_redaction() {
        let (base_url, facilitator) = spawn_service(test_config(true), None).await;
        facilitator.settle_fail.store(true, Ordering::SeqCst);
        let client = reqwest::Client::new();
        let (_, requirement) = challenge_requirement(&client, &base_url).await;

        let response = post_redact(
            &client,
            &base_url,
            &[(
                "payment-signature",
                encode_b64_json(&v2_payment(&requirement, 1)),
            )],
        )
        .await;

        assert_eq!(response.status(), 402);
        let receipt = decode_header(&response, "payment-response");
        assert_eq!(receipt["success"], false);
        assert_eq!(receipt["errorReason"], "transaction_failed");
        assert_eq!(receipt["transaction"], "");
        assert_eq!(receipt["network"], "eip155:84532");
        assert_eq!(receipt["payer"], PAYER);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "settlement_failed");
        assert!(body.get("redacted_text").is_none());
        assert_eq!(facilitator.settle_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unreachable_facilitator_fails_closed() {
        // Port 9 (discard) on loopback: nothing listens, connect is refused.
        let (base_url, _facilitator) =
            spawn_service(test_config(true), Some("http://127.0.0.1:9".to_string())).await;
        let client = reqwest::Client::new();
        let (_, requirement) = challenge_requirement(&client, &base_url).await;

        let response = post_redact(
            &client,
            &base_url,
            &[(
                "payment-signature",
                encode_b64_json(&v2_payment(&requirement, 1)),
            )],
        )
        .await;

        assert_eq!(response.status(), 500);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "facilitator_unavailable");
        assert!(body.get("redacted_text").is_none());
    }

    #[tokio::test]
    async fn healthz_stays_free_while_payments_are_enabled() {
        let (base_url, _facilitator) = spawn_service(test_config(true), None).await;
        let client = reqwest::Client::new();
        let response = client
            .get(format!("{base_url}/healthz"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert!(response.headers().get("payment-required").is_none());
        // /healthz never triggers the gate even on a paid-enabled server.
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn happy_path_v2_charges_then_serves_redaction() {
        let (base_url, facilitator) = spawn_service(test_config(true), None).await;
        let client = reqwest::Client::new();
        let (challenge, requirement) = challenge_requirement(&client, &base_url).await;

        let payment = v2_payment(&requirement, 1);
        let response = post_redact(
            &client,
            &base_url,
            &[("payment-signature", encode_b64_json(&payment))],
        )
        .await;

        assert_eq!(response.status(), 200);
        let receipt = decode_header(&response, "payment-response");
        assert_eq!(receipt["success"], true);
        assert_eq!(receipt["network"], "eip155:84532");
        assert_eq!(receipt["payer"], PAYER);
        let transaction = receipt["transaction"].as_str().unwrap();
        assert!(transaction.starts_with("0x") && transaction.len() == 66);

        let body: Value = response.json().await.unwrap();
        assert_eq!(body["redacted_text"], "[REDACTED]");

        // The facilitator saw one verify and one settle with the documented
        // request shape.
        assert_eq!(facilitator.verify_calls.load(Ordering::SeqCst), 1);
        assert_eq!(facilitator.settle_calls.load(Ordering::SeqCst), 1);
        let verify_body = facilitator
            .last_verify_body
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        assert!(verify_body.get("paymentPayload").is_some());
        assert!(verify_body.get("paymentRequirements").is_some());
        assert_eq!(
            verify_body["paymentPayload"]["payload"]["authorization"]["nonce"],
            payment["payload"]["authorization"]["nonce"]
        );
        assert_eq!(verify_body["paymentRequirements"]["amount"], AMOUNT);
        assert_eq!(
            verify_body["paymentRequirements"]["network"],
            "eip155:84532"
        );
        assert_eq!(verify_body["paymentRequirements"]["payTo"], PAYEE);
        assert_eq!(verify_body["paymentRequirements"]["asset"], SEPOLIA_USDC);
        // The challenge resource echoed by the client is irrelevant to the
        // facilitator; the server sends its own requirement object.
        assert_eq!(verify_body["paymentRequirements"], challenge["accepts"][0]);
        let settle_body = facilitator
            .last_settle_body
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        assert_eq!(
            settle_body["paymentRequirements"]["network"],
            "eip155:84532"
        );
    }

    #[tokio::test]
    async fn happy_path_v1_via_x_payment_header() {
        let (base_url, facilitator) = spawn_service(test_config(true), None).await;
        let client = reqwest::Client::new();
        let response = fetch_challenge(&client, &base_url).await;
        assert_eq!(response.status(), 402);
        let challenge: Value = response.json().await.unwrap();
        let network = challenge["network"].as_str().unwrap().to_string();

        let payment = v1_payment(&network, 1);
        let retry = post_redact(
            &client,
            &base_url,
            &[("x-payment", encode_b64_json(&payment))],
        )
        .await;

        assert_eq!(retry.status(), 200);
        let receipt = decode_header(&retry, "x-payment-response");
        assert_eq!(receipt["success"], true);
        assert_eq!(receipt["network"], "base-sepolia");
        assert_eq!(receipt["payer"], PAYER);
        let body: Value = retry.json().await.unwrap();
        assert_eq!(body["redacted_text"], "[REDACTED]");

        assert_eq!(facilitator.verify_calls.load(Ordering::SeqCst), 1);
        assert_eq!(facilitator.settle_calls.load(Ordering::SeqCst), 1);
        let settle_body = facilitator
            .last_settle_body
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        assert_eq!(
            settle_body["paymentRequirements"]["network"],
            "base-sepolia"
        );
        assert_eq!(
            settle_body["paymentRequirements"]["maxAmountRequired"],
            AMOUNT
        );
        // v1 facilitator bodies additionally carry the version at the top level.
        assert_eq!(settle_body["x402Version"], 1);
        assert!(facilitator
            .last_verify_body
            .lock()
            .unwrap()
            .clone()
            .unwrap()
            .get("x402Version")
            .is_some());
    }

    // --- small local helpers ---------------------------------------------------

    /// Performs the unpaid request and returns the decoded v2 challenge plus
    /// its first (and only) accepted requirement.
    async fn challenge_requirement(client: &reqwest::Client, base_url: &str) -> (Value, Value) {
        let response = fetch_challenge(client, base_url).await;
        assert_eq!(response.status(), 402);
        let challenge = decode_header(&response, "payment-required");
        let requirement = challenge["accepts"][0].clone();
        (challenge, requirement)
    }

    fn with_signature(mut payment: Value, signature: &str) -> Value {
        payment["payload"]["signature"] = json!(signature);
        payment
    }

    fn with_nonce(mut payment: Value, nonce: &str) -> Value {
        payment["payload"]["authorization"]["nonce"] = json!(nonce);
        payment
    }

    fn without_auth_field(mut payment: Value, field: &str) -> Value {
        payment["payload"]["authorization"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        payment
    }
}
