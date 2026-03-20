//! Anthropic provider adapter — Messages API (`POST /v1/messages`).
//!
//! Handles Anthropic's strict user/assistant alternation requirement, system
//! message extraction, thinking/redacted_thinking blocks, and optional prompt
//! caching (F-013).

use std::collections::HashMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::StreamExt as _;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::{Value, json};

use crate::{
    error::UnifiedLlmError,
    providers::ProviderAdapter,
    sse::process_sse_line,
    streaming::{EventStream, StreamEvent, StreamEventType},
    types::{
        ContentKind, ContentPart, FinishReason, Message, RateLimitInfo, Request, Response, Role,
        ThinkingData, ToolCall, ToolCallData, Usage, Warning,
    },
};

// ---------------------------------------------------------------------------
// AnthropicAdapter
// ---------------------------------------------------------------------------

/// Anthropic provider adapter using the Messages API.
#[derive(Debug)]
pub struct AnthropicAdapter {
    api_key: String,
    base_url: String,
    anthropic_version: String,
    http_client: reqwest::Client,
    /// When true, inject `cache_control` breakpoints automatically (F-013).
    prompt_caching: bool,
}

impl AnthropicAdapter {
    /// Construct from an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            anthropic_version: "2023-06-01".to_string(),
            http_client: reqwest::Client::new(),
            prompt_caching: false,
        }
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Override the `anthropic-version` header value.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = version.into();
        self
    }

    /// Enable or disable automatic prompt-caching injection (F-013).
    pub fn with_prompt_caching(mut self, enabled: bool) -> Self {
        self.prompt_caching = enabled;
        self
    }

    /// Construct from environment variables.
    ///
    /// - `ANTHROPIC_API_KEY` (required)
    /// - `ANTHROPIC_BASE_URL` (optional, default `"https://api.anthropic.com"`)
    pub fn from_env() -> Result<Self, UnifiedLlmError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            return Err(UnifiedLlmError::Configuration {
                message: "ANTHROPIC_API_KEY environment variable is not set or empty".to_string(),
            });
        }
        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
        Ok(Self::new(api_key).with_base_url(base_url))
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Build the HTTP headers for a request.
    /// `extra_betas` is merged with any `beta_headers` from provider_options.
    fn build_headers(&self, beta_headers: &[String]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Ok(v) = HeaderValue::from_str(&self.api_key) {
            headers.insert("x-api-key", v);
        }
        if let Ok(v) = HeaderValue::from_str(&self.anthropic_version) {
            headers.insert("anthropic-version", v);
        }
        if !beta_headers.is_empty() {
            let joined = beta_headers.join(",");
            if let Ok(v) = HeaderValue::from_str(&joined) {
                headers.insert("anthropic-beta", v);
            }
        }
        headers
    }

    /// Collect beta headers from provider_options and optionally add caching.
    fn collect_beta_headers(&self, request: &Request) -> Vec<String> {
        let mut betas: Vec<String> = Vec::new();

        if let Some(opts) = &request.provider_options {
            if let Some(arr) = opts["beta_headers"].as_array() {
                for v in arr {
                    if let Some(s) = v.as_str() {
                        betas.push(s.to_string());
                    }
                }
            } else if let Some(s) = opts["beta_headers"].as_str() {
                // Single string (edge case per spec)
                betas.push(s.to_string());
            }
        }

        if self.prompt_caching && !betas.contains(&"prompt-caching-2024-07-31".to_string()) {
            betas.push("prompt-caching-2024-07-31".to_string());
        }
        betas
    }

    /// Build the JSON request body (shared by `complete()` and `stream()`).
    fn build_request_body(&self, request: &Request, stream: bool) -> Value {
        // --- Extract system messages ---
        let system_text: Option<String> = {
            let texts: Vec<&str> = request
                .messages
                .iter()
                .filter(|m| matches!(m.role, Role::System | Role::Developer))
                .flat_map(|m| m.content.iter())
                .filter(|p| p.kind == ContentKind::Text)
                .filter_map(|p| p.text.as_deref())
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n\n"))
            }
        };

        // --- Normalize and translate messages ---
        let non_system: Vec<&Message> = request
            .messages
            .iter()
            .filter(|m| !matches!(m.role, Role::System | Role::Developer))
            .collect();
        let mut messages = normalize_messages_anthropic(&non_system);

        // --- Determine tool choice (none → omit tools entirely) ---
        let tool_choice_is_none = request
            .tool_choice
            .as_ref()
            .map(|tc| tc.mode == "none")
            .unwrap_or(false);

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(4096),
        });

        if let Some(sys) = system_text.as_ref() {
            body["system"] = json!(sys);
        }

        if stream {
            body["stream"] = json!(true);
        }

        // --- Tools (omit if tool_choice is "none") ---
        if !tool_choice_is_none {
            if let Some(tools) = &request.tools {
                if !tools.is_empty() {
                    let tools_json: Vec<Value> = tools
                        .iter()
                        .map(|t| {
                            json!({
                                "name": t.name,
                                "description": t.description,
                                "input_schema": t.parameters,
                            })
                        })
                        .collect();
                    body["tools"] = json!(tools_json);

                    // Tool choice (only if tools present and not none)
                    if let Some(tc) = &request.tool_choice {
                        body["tool_choice"] = translate_tool_choice_anthropic(tc);
                    }
                }
            }
        }

        // --- Sampling parameters ---
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(stops) = &request.stop_sequences {
            body["stop_sequences"] = json!(stops);
        }
        if let Some(meta) = &request.metadata {
            if let Some(user_id) = meta.get("user_id") {
                body["metadata"] = json!({ "user_id": user_id });
            }
        }

        // --- Provider options (escape hatch, skip beta_headers which is handled separately) ---
        if let Some(opts) = &request.provider_options {
            if let Some(map) = opts.as_object() {
                for (k, v) in map {
                    if k != "beta_headers" {
                        body[k] = v.clone();
                    }
                }
            }
        }

        // --- Inject prompt caching (F-013) ---
        if self.prompt_caching {
            inject_cache_control(system_text.as_deref(), &mut body, &mut messages);
        }

        body
    }

    fn parse_rate_limit_headers(headers: &reqwest::header::HeaderMap) -> Option<RateLimitInfo> {
        let requests_limit = headers
            .get("anthropic-ratelimit-requests-limit")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());
        let requests_remaining = headers
            .get("anthropic-ratelimit-requests-remaining")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());
        let tokens_limit = headers
            .get("anthropic-ratelimit-tokens-limit")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());
        let tokens_remaining = headers
            .get("anthropic-ratelimit-tokens-remaining")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());

        if requests_limit.is_none()
            && requests_remaining.is_none()
            && tokens_limit.is_none()
            && tokens_remaining.is_none()
        {
            return None;
        }
        Some(RateLimitInfo {
            requests_remaining,
            requests_limit,
            tokens_remaining,
            tokens_limit,
            reset_at: None,
        })
    }

    fn parse_response_body(
        body: Value,
        rate_limit: Option<RateLimitInfo>,
    ) -> Result<Response, UnifiedLlmError> {
        let id = body["id"].as_str().unwrap_or("unknown").to_string();
        let model = body["model"].as_str().unwrap_or("").to_string();
        let mut content: Vec<ContentPart> = Vec::new();
        let warnings: Vec<Warning> = Vec::new();

        if let Some(content_arr) = body["content"].as_array() {
            for block in content_arr {
                match block["type"].as_str() {
                    Some("text") => {
                        let text = block["text"].as_str().unwrap_or("");
                        content.push(ContentPart::text(text));
                    }
                    Some("tool_use") => {
                        let call_id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        let input_raw = serde_json::to_string(&block["input"]).unwrap_or_default();
                        content.push(ContentPart::tool_call(ToolCallData {
                            id: call_id,
                            name,
                            arguments: block["input"].clone(),
                            raw_arguments: Some(input_raw),
                        }));
                    }
                    Some("thinking") => {
                        let text = block["thinking"].as_str().unwrap_or("").to_string();
                        let signature = block["signature"].as_str().map(|s| s.to_string());
                        content.push(ContentPart::thinking(ThinkingData {
                            text,
                            signature,
                            redacted: false,
                        }));
                    }
                    Some("redacted_thinking") => {
                        let sig = block["data"].as_str().unwrap_or("").to_string();
                        let mut part = ContentPart {
                            kind: ContentKind::RedactedThinking,
                            text: None,
                            image: None,
                            audio: None,
                            document: None,
                            tool_call: None,
                            tool_result: None,
                            thinking: Some(ThinkingData {
                                text: String::new(),
                                signature: Some(sig),
                                redacted: true,
                            }),
                        };
                        // satisfy the kind field correctly
                        part.kind = ContentKind::RedactedThinking;
                        content.push(part);
                    }
                    _ => {}
                }
            }
        }

        let finish_reason = finish_reason_from_anthropic_stop_reason(body["stop_reason"].as_str());
        let usage = parse_anthropic_usage(&body["usage"]);

        if content.is_empty() {
            // No content parsed — emit an empty text warning? Spec doesn't require it.
        }

        let message = Message {
            role: Role::Assistant,
            content,
            name: None,
            tool_call_id: None,
        };

        Ok(Response {
            id,
            model,
            provider: "anthropic".to_string(),
            message,
            finish_reason,
            usage,
            raw: Some(body),
            warnings,
            rate_limit,
        })
    }

    /// Classify an HTTP error status into the appropriate `UnifiedLlmError`.
    ///
    /// Extracted from `handle_error_response` for unit-testability (V2-ULM-009).
    pub(crate) fn parse_error_status(
        status: u16,
        message: &str,
        error_type: Option<&str>,
        retry_after: Option<f64>,
    ) -> UnifiedLlmError {
        match status {
            401 => UnifiedLlmError::Authentication {
                provider: "anthropic".to_string(),
                message: message.to_string(),
            },
            429 => UnifiedLlmError::RateLimit {
                provider: "anthropic".to_string(),
                message: message.to_string(),
                retry_after,
            },
            400 if error_type == Some("invalid_request_error")
                && (message.contains("max_tokens") || message.contains("context")) =>
            {
                UnifiedLlmError::ContextLength {
                    message: message.to_string(),
                }
            }
            400 => UnifiedLlmError::Provider {
                provider: "anthropic".to_string(),
                message: message.to_string(),
                status_code: Some(status),
                error_code: error_type.map(|s| s.to_string()),
                retryable: false,
                retry_after: None,
                raw: None,
            },
            500..=599 => UnifiedLlmError::Provider {
                provider: "anthropic".to_string(),
                message: message.to_string(),
                status_code: Some(status),
                error_code: error_type.map(|s| s.to_string()),
                retryable: true,
                retry_after: None,
                raw: None,
            },
            _ => UnifiedLlmError::Provider {
                provider: "anthropic".to_string(),
                message: message.to_string(),
                status_code: Some(status),
                error_code: error_type.map(|s| s.to_string()),
                retryable: false,
                retry_after: None,
                raw: None,
            },
        }
    }

    async fn handle_error_response(response: reqwest::Response) -> UnifiedLlmError {
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get("Retry-After")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok());
        let body_text = response.text().await.unwrap_or_default();
        let body_json: Value =
            serde_json::from_str(&body_text).unwrap_or(Value::String(body_text.clone()));
        let message = body_json["error"]["message"]
            .as_str()
            .unwrap_or(&body_text)
            .to_string();
        let error_type = body_json["error"]["type"].as_str().map(|s| s.to_string());

        let mut err =
            Self::parse_error_status(status, &message, error_type.as_deref(), retry_after);
        // Attach raw body where applicable.
        if let UnifiedLlmError::Provider { raw, .. } = &mut err {
            *raw = Some(body_json);
        }
        err
    }
}

// ---------------------------------------------------------------------------
// Prompt caching injection (F-013)
// ---------------------------------------------------------------------------

/// Inject `cache_control: {"type": "ephemeral"}` into:
/// 1. The system field (if present) — transforms string to array of blocks.
/// 2. The last content block of the last two user messages.
///
/// Mutates `body` directly. `messages` is the already-translated messages
/// array which is also embedded in body["messages"].
fn inject_cache_control(system_text: Option<&str>, body: &mut Value, _messages: &mut Vec<Value>) {
    // --- System message caching ---
    if system_text.is_some() {
        match &body["system"] {
            Value::String(s) => {
                let s = s.clone();
                body["system"] = json!([{
                    "type": "text",
                    "text": s,
                    "cache_control": { "type": "ephemeral" }
                }]);
            }
            Value::Array(_) => {
                // Add cache_control to last block in array
                if let Some(arr) = body["system"].as_array_mut() {
                    if let Some(last) = arr.last_mut() {
                        last["cache_control"] = json!({ "type": "ephemeral" });
                    }
                }
            }
            _ => {}
        }
    }

    // --- Tool caching: last tool definition (V2-ULM-013) ---
    // Inject cache_control on the last tool so the tool list is cached.
    if let Some(tools) = body["tools"].as_array_mut() {
        if let Some(last_tool) = tools.last_mut() {
            last_tool["cache_control"] = json!({ "type": "ephemeral" });
        }
    }

    // --- User message caching: last 2 user messages ---
    if let Some(msgs) = body["messages"].as_array_mut() {
        let user_indices: Vec<usize> = msgs
            .iter()
            .enumerate()
            .filter(|(_, m)| m["role"] == "user")
            .map(|(i, _)| i)
            .collect();

        // Take the last 2 user message indices
        let targets: Vec<usize> = user_indices.iter().rev().take(2).copied().collect();

        for idx in targets {
            if let Some(content_arr) = msgs[idx]["content"].as_array_mut() {
                if let Some(last_block) = content_arr.last_mut() {
                    last_block["cache_control"] = json!({ "type": "ephemeral" });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free-standing translation helpers
// ---------------------------------------------------------------------------

fn translate_tool_choice_anthropic(tc: &crate::types::ToolChoice) -> Value {
    match tc.mode.as_str() {
        "auto" => json!({ "type": "auto" }),
        "required" => json!({ "type": "any" }),
        "named" => {
            let name = tc.tool_name.as_deref().unwrap_or("");
            json!({ "type": "tool", "name": name })
        }
        _ => json!({ "type": "auto" }),
    }
}

fn finish_reason_from_anthropic_stop_reason(stop_reason: Option<&str>) -> FinishReason {
    match stop_reason {
        Some("end_turn") => FinishReason {
            reason: "stop".to_string(),
            raw: Some("end_turn".to_string()),
        },
        Some("tool_use") => FinishReason {
            reason: "tool_calls".to_string(),
            raw: Some("tool_use".to_string()),
        },
        Some("max_tokens") => FinishReason {
            reason: "length".to_string(),
            raw: Some("max_tokens".to_string()),
        },
        Some("stop_sequence") => FinishReason {
            reason: "stop".to_string(),
            raw: Some("stop_sequence".to_string()),
        },
        Some(other) => FinishReason {
            reason: "other".to_string(),
            raw: Some(other.to_string()),
        },
        None => FinishReason {
            reason: "stop".to_string(),
            raw: None,
        },
    }
}

fn parse_anthropic_usage(usage_val: &Value) -> Usage {
    let input_tokens = usage_val["input_tokens"].as_u64().unwrap_or(0) as u32;
    let output_tokens = usage_val["output_tokens"].as_u64().unwrap_or(0) as u32;
    let total_tokens = input_tokens + output_tokens;

    // Cache read: legacy field (Claude 3/3.5) or newer nested object (Claude 4+).
    let cache_read_tokens = usage_val["cache_read_input_tokens"]
        .as_u64()
        .map(|v| v as u32);

    // Cache write: legacy flat field (Claude 3/3.5) plus newer nested object fields
    // (Claude 4+ extended-cache-ttl-2025-02-19 beta).  Sum all TTL buckets.
    let legacy_write = usage_val["cache_creation_input_tokens"]
        .as_u64()
        .unwrap_or(0) as u32;
    let extended_5m = usage_val["cache_creation"]["ephemeral_5m_input_tokens"]
        .as_u64()
        .unwrap_or(0) as u32;
    let extended_1h = usage_val["cache_creation"]["ephemeral_1h_input_tokens"]
        .as_u64()
        .unwrap_or(0) as u32;
    let total_write = legacy_write + extended_5m + extended_1h;
    // Only expose Some(n) when the field exists in the response (avoids spurious Some(0)
    // for providers that don't support caching at all).
    let cache_write_tokens = if usage_val.get("cache_creation_input_tokens").is_some()
        || usage_val.get("cache_creation").is_some()
    {
        Some(total_write)
    } else {
        None
    };

    // V2-ULM-012: populate reasoning_tokens from thinking block token count.
    // Anthropic may surface this in future API revisions; check common field names.
    let reasoning_tokens = usage_val
        .get("thinking_tokens")
        .or_else(|| usage_val.get("reasoning_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    Usage {
        input_tokens,
        output_tokens,
        total_tokens,
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens,
        raw: Some(usage_val.clone()),
    }
}

/// Translate a unified `ContentPart` to an Anthropic content block JSON value.
/// Returns `None` if the part should be skipped (with optional warning).
fn translate_content_part_anthropic(part: &ContentPart) -> Option<Value> {
    match &part.kind {
        ContentKind::Text => Some(json!({
            "type": "text",
            "text": part.text.as_deref().unwrap_or("")
        })),
        ContentKind::Image => {
            let img = part.image.as_ref()?;
            if let Some(url) = &img.url {
                Some(json!({
                    "type": "image",
                    "source": { "type": "url", "url": url }
                }))
            } else {
                // V2-ULM-005: resolve data from existing bytes or file path.
                let resolved: Option<Vec<u8>> = img
                    .data
                    .clone()
                    .or_else(|| img.path.as_ref().and_then(|p| std::fs::read(p).ok()));
                if let Some(data) = resolved {
                    let b64 = STANDARD.encode(&data);
                    let media_type = img.media_type.as_deref().unwrap_or("image/jpeg");
                    Some(json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64
                        }
                    }))
                } else {
                    None // image without source → skip
                }
            }
        }
        ContentKind::Document => {
            let doc = part.document.as_ref()?;
            let data = doc.data.as_ref()?;
            let b64 = STANDARD.encode(data);
            let media_type = doc.media_type.as_deref().unwrap_or("application/pdf");
            Some(json!({
                "type": "document",
                "source": { "type": "base64", "media_type": media_type, "data": b64 }
            }))
        }
        ContentKind::ToolCall => {
            let tc = part.tool_call.as_ref()?;
            Some(json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }))
        }
        ContentKind::ToolResult => {
            let tr = part.tool_result.as_ref()?;
            let content_str = match &tr.content {
                Value::String(s) => s.clone(),
                v => serde_json::to_string(v).unwrap_or_default(),
            };
            Some(json!({
                "type": "tool_result",
                "tool_use_id": tr.tool_call_id,
                "content": content_str,
                "is_error": tr.is_error,
            }))
        }
        ContentKind::Thinking => {
            let td = part.thinking.as_ref()?;
            Some(json!({
                "type": "thinking",
                "thinking": td.text,
                "signature": td.signature.as_deref().unwrap_or(""),
            }))
        }
        ContentKind::RedactedThinking => {
            let td = part.thinking.as_ref()?;
            Some(json!({
                "type": "redacted_thinking",
                "data": td.signature.as_deref().unwrap_or(""),
            }))
        }
        ContentKind::Audio => None,
    }
}

/// Translate a unified `Message` to a `(role_str, content_blocks)` pair for Anthropic.
/// Tool messages are converted to user role with tool_result blocks.
fn translate_message_to_anthropic_parts(msg: &Message) -> (String, Vec<Value>) {
    let role = match msg.role {
        Role::User => "user".to_string(),
        Role::Assistant => "assistant".to_string(),
        Role::Tool => "user".to_string(), // Tool messages become user messages
        Role::System | Role::Developer => "user".to_string(),
    };

    let content: Vec<Value> = msg
        .content
        .iter()
        .filter_map(translate_content_part_anthropic)
        .collect();

    (role, content)
}

/// Normalize messages to satisfy Anthropic's strict user/assistant alternation:
/// 1. Convert Role::Tool messages to user-role with tool_result content.
/// 2. Merge consecutive messages with the same role.
/// 3. If first message is assistant, prepend an empty user message.
fn normalize_messages_anthropic(messages: &[&Message]) -> Vec<Value> {
    // Build (role, content_blocks) list, merging consecutive same-role entries
    let mut merged: Vec<(String, Vec<Value>)> = Vec::new();

    for msg in messages {
        let (role, mut parts) = translate_message_to_anthropic_parts(msg);
        if let Some(last) = merged.last_mut() {
            if last.0 == role {
                last.1.append(&mut parts);
                continue;
            }
        }
        merged.push((role, parts));
    }

    // Ensure starts with user
    if let Some(first) = merged.first() {
        if first.0 == "assistant" {
            merged.insert(0, ("user".to_string(), vec![]));
        }
    }

    merged
        .into_iter()
        .map(|(role, content)| json!({ "role": role, "content": content }))
        .collect()
}

// ---------------------------------------------------------------------------
// Streaming block state (F-012)
// ---------------------------------------------------------------------------

/// Per-block state tracked during Anthropic SSE streaming.
struct BlockState {
    block_type: String, // "text", "tool_use", "thinking"
    call_id: Option<String>,
    name: Option<String>,
    partial_json: String, // accumulated for tool_use input_json_delta
}

/// Mutable state accumulated during Anthropic SSE streaming.
struct AnthropicSseState {
    input_tokens: u32,
    output_tokens: u32,
    cache_read_tokens: Option<u32>,
    cache_write_tokens: Option<u32>,
    stop_reason: Option<String>,
    blocks: HashMap<usize, BlockState>,
}

impl AnthropicSseState {
    fn new() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: None,
            cache_write_tokens: None,
            stop_reason: None,
            blocks: HashMap::new(),
        }
    }
}

/// Translate Anthropic SSE events to unified StreamEvents.
/// Returns a list of events to emit and whether the stream should terminate.
fn translate_anthropic_sse_event(
    event_type: &str,
    data: &str,
    state: &mut AnthropicSseState,
) -> (Vec<Result<StreamEvent, UnifiedLlmError>>, bool) {
    let mut out: Vec<Result<StreamEvent, UnifiedLlmError>> = Vec::new();
    let mut terminate = false;

    let payload: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            out.push(Err(UnifiedLlmError::Stream {
                message: format!("SSE JSON parse error on '{event_type}': {e}"),
            }));
            return (out, true);
        }
    };

    match event_type {
        "message_start" => {
            let usage = &payload["message"]["usage"];
            state.input_tokens = usage["input_tokens"].as_u64().unwrap_or(0) as u32;
            state.cache_read_tokens = usage["cache_read_input_tokens"].as_u64().map(|v| v as u32);
            state.cache_write_tokens = usage["cache_creation_input_tokens"]
                .as_u64()
                .map(|v| v as u32);
            out.push(Ok(StreamEvent::stream_start()));
        }
        "content_block_start" => {
            let index = payload["index"].as_u64().unwrap_or(0) as usize;
            let block = &payload["content_block"];
            match block["type"].as_str() {
                Some("text") => {
                    state.blocks.insert(
                        index,
                        BlockState {
                            block_type: "text".to_string(),
                            call_id: None,
                            name: None,
                            partial_json: String::new(),
                        },
                    );
                    out.push(Ok(StreamEvent::text_start()));
                }
                Some("tool_use") => {
                    let call_id = block["id"].as_str().unwrap_or("").to_string();
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    state.blocks.insert(
                        index,
                        BlockState {
                            block_type: "tool_use".to_string(),
                            call_id: Some(call_id.clone()),
                            name: Some(name.clone()),
                            partial_json: String::new(),
                        },
                    );
                    out.push(Ok(StreamEvent::tool_call_start(call_id, name)));
                }
                Some("thinking") => {
                    state.blocks.insert(
                        index,
                        BlockState {
                            block_type: "thinking".to_string(),
                            call_id: None,
                            name: None,
                            partial_json: String::new(),
                        },
                    );
                    let mut ev = StreamEvent::stream_start(); // blank
                    ev.event_type = StreamEventType::ReasoningStart;
                    out.push(Ok(ev));
                }
                _ => {}
            }
        }
        "content_block_delta" => {
            let index = payload["index"].as_u64().unwrap_or(0) as usize;
            let delta = &payload["delta"];

            let block_type = state.blocks.get(&index).map(|b| b.block_type.as_str());

            match (delta["type"].as_str(), block_type) {
                (Some("text_delta"), Some("text")) => {
                    let text = delta["text"].as_str().unwrap_or("");
                    out.push(Ok(StreamEvent::text_delta(text)));
                }
                (Some("input_json_delta"), Some("tool_use")) => {
                    let partial = delta["partial_json"].as_str().unwrap_or("");
                    if let Some(block) = state.blocks.get_mut(&index) {
                        block.partial_json.push_str(partial);
                        let call_id = block.call_id.clone().unwrap_or_default();
                        out.push(Ok(StreamEvent::tool_call_delta(call_id, partial)));
                    }
                }
                (Some("thinking_delta"), Some("thinking")) => {
                    let thinking = delta["thinking"].as_str().unwrap_or("");
                    out.push(Ok(StreamEvent::reasoning_delta(thinking)));
                }
                (Some("signature_delta"), _) => {
                    // Thinking block signature — emit as ProviderEvent
                    let mut ev = StreamEvent::stream_start();
                    ev.event_type = StreamEventType::ProviderEvent;
                    ev.raw = serde_json::from_str(data).ok();
                    out.push(Ok(ev));
                }
                _ => {
                    // Unknown delta type for known block or unknown block index
                    if !state.blocks.contains_key(&index) {
                        out.push(Err(UnifiedLlmError::Stream {
                            message: format!("content_block_delta for unknown block index {index}"),
                        }));
                        terminate = true;
                    }
                }
            }
        }
        "content_block_stop" => {
            let index = payload["index"].as_u64().unwrap_or(0) as usize;
            if let Some(block) = state.blocks.remove(&index) {
                match block.block_type.as_str() {
                    "text" => {
                        out.push(Ok(StreamEvent::text_end()));
                    }
                    "tool_use" => {
                        let call_id = block.call_id.unwrap_or_default();
                        let name = block.name.unwrap_or_default();
                        let args_str = block.partial_json.as_str();
                        // Check if args parse is valid
                        if !block.partial_json.is_empty() {
                            match serde_json::from_str::<Value>(args_str) {
                                Ok(parsed_args) => {
                                    out.push(Ok(StreamEvent::tool_call_end(ToolCall {
                                        id: call_id,
                                        name,
                                        arguments: parsed_args,
                                        raw_arguments: Some(block.partial_json),
                                    })));
                                }
                                Err(e) => {
                                    out.push(Err(UnifiedLlmError::Stream {
                                        message: format!(
                                            "invalid JSON in tool call arguments: {e}"
                                        ),
                                    }));
                                    terminate = true;
                                }
                            }
                        } else {
                            out.push(Ok(StreamEvent::tool_call_end(ToolCall {
                                id: call_id,
                                name,
                                arguments: Value::Object(serde_json::Map::new()),
                                raw_arguments: None,
                            })));
                        }
                    }
                    "thinking" => {
                        let mut ev = StreamEvent::stream_start();
                        ev.event_type = StreamEventType::ReasoningEnd;
                        out.push(Ok(ev));
                    }
                    _ => {}
                }
            }
        }
        "message_delta" => {
            let delta = &payload["delta"];
            state.stop_reason = delta["stop_reason"].as_str().map(|s| s.to_string());
            if let Some(out_tokens) = payload["usage"]["output_tokens"].as_u64() {
                state.output_tokens = out_tokens as u32;
            }
            // Emit nothing yet — wait for message_stop
        }
        "message_stop" => {
            // Assemble final usage
            let total_tokens = state.input_tokens + state.output_tokens;
            let usage = Usage {
                input_tokens: state.input_tokens,
                output_tokens: state.output_tokens,
                total_tokens,
                reasoning_tokens: None,
                cache_read_tokens: state.cache_read_tokens,
                cache_write_tokens: state.cache_write_tokens,
                raw: None,
            };
            let finish_reason =
                finish_reason_from_anthropic_stop_reason(state.stop_reason.as_deref());
            out.push(Ok(StreamEvent::finish(finish_reason, usage)));
            terminate = true;
        }
        "error" => {
            let msg = payload["error"]["message"]
                .as_str()
                .unwrap_or("unknown error")
                .to_string();
            out.push(Ok(StreamEvent::error(msg)));
            terminate = true;
        }
        _ => {
            // Unknown event — emit as ProviderEvent
            let mut ev = StreamEvent::stream_start();
            ev.event_type = StreamEventType::ProviderEvent;
            ev.raw = serde_json::from_str(data).ok();
            out.push(Ok(ev));
        }
    }

    (out, terminate)
}

// ---------------------------------------------------------------------------
// ProviderAdapter implementation
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ProviderAdapter for AnthropicAdapter {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn supports_tool_choice(&self, mode: &str) -> bool {
        matches!(mode, "auto" | "none" | "required" | "named")
    }

    async fn complete(&self, request: &Request) -> Result<Response, UnifiedLlmError> {
        // V2-ULM-006: reject audio content rather than silently dropping it.
        super::reject_audio_content(request, "anthropic")?;
        let body = self.build_request_body(request, false);
        let url = format!("{}/v1/messages", self.base_url);
        let betas = self.collect_beta_headers(request);
        let headers = self.build_headers(&betas);
        let client = self.http_client.clone();

        // V2-ULM-001 fix: wrap the full request → status-check → parse cycle so that
        // HTTP 429 / 5xx responses are classified as retryable errors INSIDE the retry
        // closure, giving the retry policy a chance to act on them.
        let (body_json, rate_limit) = crate::retry::RetryPolicy::default_policy()
            .execute(|| {
                let url = url.clone();
                let headers = headers.clone();
                let body = body.clone();
                let client = client.clone();
                async move {
                    let http_resp = client
                        .post(&url)
                        .headers(headers)
                        .json(&body)
                        .send()
                        .await
                        .map_err(|e| UnifiedLlmError::Network {
                            message: e.to_string(),
                            source: Some(Box::new(e)),
                        })?;

                    let rate_limit = Self::parse_rate_limit_headers(http_resp.headers());

                    if !http_resp.status().is_success() {
                        return Err(Self::handle_error_response(http_resp).await);
                    }

                    let body_text =
                        http_resp
                            .text()
                            .await
                            .map_err(|e| UnifiedLlmError::Network {
                                message: e.to_string(),
                                source: Some(Box::new(e)),
                            })?;

                    let body_json: Value = serde_json::from_str(&body_text).map_err(|e| {
                        UnifiedLlmError::Provider {
                            provider: "anthropic".to_string(),
                            message: format!("failed to parse response JSON: {e}"),
                            status_code: None,
                            error_code: None,
                            retryable: false,
                            retry_after: None,
                            raw: Some(Value::String(body_text)),
                        }
                    })?;

                    Ok((body_json, rate_limit))
                }
            })
            .await?;

        Self::parse_response_body(body_json, rate_limit)
    }

    async fn stream(&self, request: &Request) -> Result<EventStream, UnifiedLlmError> {
        // V2-ULM-006: reject audio content rather than silently dropping it.
        super::reject_audio_content(request, "anthropic")?;
        let body = self.build_request_body(request, true);
        let url = format!("{}/v1/messages", self.base_url);
        let betas = self.collect_beta_headers(request);
        let headers = self.build_headers(&betas);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| UnifiedLlmError::Network {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if !response.status().is_success() {
            return Err(Self::handle_error_response(response).await);
        }

        let (mut tx, rx) =
            futures::channel::mpsc::channel::<Result<StreamEvent, UnifiedLlmError>>(64);

        tokio::spawn(async move {
            let mut byte_stream = response.bytes_stream();
            let mut line_buf = String::new();
            let mut sse_event_type = String::new();
            let mut sse_data_lines: Vec<String> = Vec::new();
            let mut sse_last_id: Option<String> = None;
            let mut sse_retry_ms: Option<u64> = None;

            // Per-stream state (F-012)
            let mut sse_state = AnthropicSseState::new();

            while let Some(chunk_result) = byte_stream.next().await {
                let bytes = match chunk_result {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = tx
                            .try_send(Err(UnifiedLlmError::Network {
                                message: e.to_string(),
                                source: None,
                            }))
                            .ok();
                        return;
                    }
                };

                match std::str::from_utf8(&bytes) {
                    Ok(s) => line_buf.push_str(s),
                    Err(_) => {
                        let _ = tx
                            .try_send(Err(UnifiedLlmError::Stream {
                                message: "invalid UTF-8 in SSE stream".to_string(),
                            }))
                            .ok();
                        return;
                    }
                }

                while let Some(pos) = line_buf.find('\n') {
                    let line = line_buf[..pos].trim_end_matches('\r').to_string();
                    line_buf = line_buf[pos + 1..].to_string();

                    match process_sse_line(
                        &line,
                        &mut sse_event_type,
                        &mut sse_data_lines,
                        &mut sse_last_id,
                        &mut sse_retry_ms,
                    ) {
                        Ok(Some(sse_ev)) => {
                            let (events, terminate) = translate_anthropic_sse_event(
                                &sse_ev.event_type,
                                &sse_ev.data,
                                &mut sse_state,
                            );
                            for ev in events {
                                if tx.try_send(ev).is_err() {
                                    return;
                                }
                            }
                            if terminate {
                                return;
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            let _ = tx
                                .try_send(Err(UnifiedLlmError::Stream {
                                    message: e.to_string(),
                                }))
                                .ok();
                            return;
                        }
                    }
                }
            }

            // Stream closed without message_stop
            let _ = tx
                .try_send(Ok(StreamEvent::error(
                    "stream closed without message_stop event",
                )))
                .ok();
        });

        Ok(Box::pin(rx))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, Request, Tool, ToolChoice};

    fn make_request(model: &str, messages: Vec<Message>) -> Request {
        Request::new(model, messages)
    }

    // AC-1: from_env() returns Err when ANTHROPIC_API_KEY unset
    #[test]
    fn from_env_no_key_returns_config_error() {
        // SAFETY: single-threaded test context
        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
        let err = AnthropicAdapter::from_env().unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Configuration { .. }));
    }

    // AC-3: System message → "system" field, not in "messages"
    #[test]
    fn system_message_goes_to_system_field() {
        let adapter = AnthropicAdapter::new("key");
        let req = make_request(
            "claude-opus-4-5",
            vec![Message::system("Be helpful."), Message::user("Hello")],
        );
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["system"], "Be helpful.");
        let msgs = body["messages"].as_array().unwrap();
        for m in msgs {
            assert_ne!(m["role"], "system");
        }
    }

    // AC-9: max_tokens defaults to 4096 when not set
    #[test]
    fn max_tokens_defaults_to_4096() {
        let adapter = AnthropicAdapter::new("key");
        let req = make_request("claude", vec![Message::user("hi")]);
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["max_tokens"], 4096);
    }

    // AC-9: max_tokens uses request value when set
    #[test]
    fn max_tokens_uses_request_value() {
        let adapter = AnthropicAdapter::new("key");
        let req = make_request("claude", vec![Message::user("hi")]).with_max_tokens(1000);
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["max_tokens"], 1000);
    }

    // AC-4: Consecutive user messages are merged
    #[test]
    fn consecutive_user_messages_merged() {
        let adapter = AnthropicAdapter::new("key");
        let req = make_request(
            "claude",
            vec![Message::user("First"), Message::user("Second")],
        );
        let body = adapter.build_request_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        let content = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
    }

    // AC-5: Tool result becomes tool_result block in user message
    #[test]
    fn tool_result_becomes_user_tool_result_block() {
        let adapter = AnthropicAdapter::new("key");
        let tool_msg = Message::tool_result("call_1", "result_value", false);
        let req = make_request(
            "claude",
            vec![Message::user("hi"), Message::assistant("calling"), tool_msg],
        );
        let body = adapter.build_request_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        // Should have: user, assistant, user (tool_result folded into user)
        let last_user = msgs.iter().rev().find(|m| m["role"] == "user").unwrap();
        let content = last_user["content"].as_array().unwrap();
        let has_tool_result = content.iter().any(|c| c["type"] == "tool_result");
        assert!(has_tool_result);
    }

    // AC-13: ToolChoice::none() → tools omitted entirely
    #[test]
    fn tool_choice_none_omits_tools() {
        let adapter = AnthropicAdapter::new("key");
        let mut req = make_request("claude", vec![Message::user("hi")]).with_tools(vec![Tool {
            name: "fn".to_string(),
            description: "test".to_string(),
            parameters: serde_json::json!({}),
        }]);
        req.tool_choice = Some(ToolChoice::none());
        let body = adapter.build_request_body(&req, false);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    // Tool choice auto → {"type": "auto"}
    #[test]
    fn tool_choice_auto_serializes_correctly() {
        let tc = translate_tool_choice_anthropic(&ToolChoice::auto());
        assert_eq!(tc["type"], "auto");
    }

    // Tool choice required → {"type": "any"}
    #[test]
    fn tool_choice_required_serializes_as_any() {
        let tc = translate_tool_choice_anthropic(&ToolChoice::required());
        assert_eq!(tc["type"], "any");
    }

    // Tool choice named → {"type": "tool", "name": "..."}
    #[test]
    fn tool_choice_named_serializes_correctly() {
        let tc = translate_tool_choice_anthropic(&ToolChoice::named("my_fn"));
        assert_eq!(tc["type"], "tool");
        assert_eq!(tc["name"], "my_fn");
    }

    // Finish reason mapping
    #[test]
    fn finish_reason_end_turn_maps_to_stop() {
        let fr = finish_reason_from_anthropic_stop_reason(Some("end_turn"));
        assert_eq!(fr.reason, "stop");
        assert_eq!(fr.raw, Some("end_turn".to_string()));
    }

    #[test]
    fn finish_reason_tool_use_maps_to_tool_calls() {
        let fr = finish_reason_from_anthropic_stop_reason(Some("tool_use"));
        assert!(fr.is_tool_calls());
    }

    #[test]
    fn finish_reason_max_tokens_maps_to_length() {
        let fr = finish_reason_from_anthropic_stop_reason(Some("max_tokens"));
        assert_eq!(fr.reason, "length");
    }

    // Usage parsing (AC-10)
    #[test]
    fn usage_cache_tokens_parsed() {
        let usage_val = serde_json::json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 20,
        });
        let usage = parse_anthropic_usage(&usage_val);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150); // computed
        assert_eq!(usage.cache_read_tokens, Some(30));
        assert_eq!(usage.cache_write_tokens, Some(20));
        assert_eq!(usage.reasoning_tokens, None);
    }

    // Response parsing: thinking block → Thinking ContentPart (AC-7)
    #[test]
    fn response_thinking_block_parsed() {
        let body = serde_json::json!({
            "id": "msg_1",
            "model": "claude-opus-4-5",
            "stop_reason": "end_turn",
            "content": [{
                "type": "thinking",
                "thinking": "Let me reason...",
                "signature": "sig_abc",
            }],
            "usage": { "input_tokens": 10, "output_tokens": 5 },
        });
        let resp = AnthropicAdapter::parse_response_body(body, None).unwrap();
        assert_eq!(resp.reasoning(), Some("Let me reason...".to_string()));
        let thinking_parts: Vec<_> = resp
            .message
            .content
            .iter()
            .filter(|p| p.kind == ContentKind::Thinking)
            .collect();
        assert_eq!(thinking_parts.len(), 1);
        assert_eq!(
            thinking_parts[0].thinking.as_ref().unwrap().signature,
            Some("sig_abc".to_string())
        );
    }

    // Response parsing: redacted_thinking block (AC-8)
    #[test]
    fn response_redacted_thinking_parsed() {
        let body = serde_json::json!({
            "id": "msg_2",
            "model": "claude",
            "stop_reason": "end_turn",
            "content": [{
                "type": "redacted_thinking",
                "data": "redacted_sig",
            }],
            "usage": { "input_tokens": 5, "output_tokens": 3 },
        });
        let resp = AnthropicAdapter::parse_response_body(body, None).unwrap();
        let rt_parts: Vec<_> = resp
            .message
            .content
            .iter()
            .filter(|p| p.kind == ContentKind::RedactedThinking)
            .collect();
        assert_eq!(rt_parts.len(), 1);
        assert_eq!(
            rt_parts[0].thinking.as_ref().unwrap().signature,
            Some("redacted_sig".to_string())
        );
    }

    // Response parsing: tool_use → ToolCall ContentPart
    #[test]
    fn response_tool_use_block_parsed() {
        let body = serde_json::json!({
            "id": "msg_3",
            "model": "claude",
            "stop_reason": "tool_use",
            "content": [{
                "type": "tool_use",
                "id": "call_1",
                "name": "my_fn",
                "input": { "x": 1 }
            }],
            "usage": { "input_tokens": 5, "output_tokens": 3 },
        });
        let resp = AnthropicAdapter::parse_response_body(body, None).unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "my_fn");
        assert_eq!(calls[0].arguments["x"], 1);
        assert!(resp.finish_reason.is_tool_calls());
    }

    // No system message → no "system" field in body
    #[test]
    fn no_system_message_omits_system_field() {
        let adapter = AnthropicAdapter::new("key");
        let req = make_request("claude", vec![Message::user("Hello")]);
        let body = adapter.build_request_body(&req, false);
        assert!(body.get("system").is_none());
    }

    // First message is assistant → empty user message prepended
    #[test]
    fn first_assistant_message_gets_empty_user_prepended() {
        let adapter = AnthropicAdapter::new("key");
        let req = make_request("claude", vec![Message::assistant("hi")]);
        let body = adapter.build_request_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "user");
    }

    // stream body has "stream": true
    #[test]
    fn stream_body_has_stream_true() {
        let adapter = AnthropicAdapter::new("key");
        let req = make_request("claude", vec![Message::user("hi")]);
        let body = adapter.build_request_body(&req, true);
        assert_eq!(body["stream"], true);
    }

    // --- F-013: Prompt caching tests ---

    // AC-1: system message gets cache_control when prompt_caching enabled
    #[test]
    fn caching_adds_cache_control_to_system() {
        let adapter = AnthropicAdapter::new("key").with_prompt_caching(true);
        let req = make_request(
            "claude",
            vec![Message::system("system text"), Message::user("hi")],
        );
        let body = adapter.build_request_body(&req, false);
        // System should be array with cache_control
        let sys = &body["system"];
        assert!(sys.is_array());
        let blocks = sys.as_array().unwrap();
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
    }

    // AC-2 & AC-3: last two user messages get cache_control
    #[test]
    fn caching_adds_cache_control_to_last_two_user_messages() {
        let adapter = AnthropicAdapter::new("key").with_prompt_caching(true);
        let req = make_request(
            "claude",
            vec![
                Message::user("first"),
                Message::assistant("reply1"),
                Message::user("second"),
                Message::assistant("reply2"),
                Message::user("third"),
            ],
        );
        let body = adapter.build_request_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        let user_msgs: Vec<&Value> = msgs.iter().filter(|m| m["role"] == "user").collect();
        // Should be 3 user messages; last 2 should have cache_control
        assert_eq!(user_msgs.len(), 3);
        // Last user message
        let last_content = user_msgs[2]["content"].as_array().unwrap();
        assert_eq!(
            last_content.last().unwrap()["cache_control"]["type"],
            "ephemeral"
        );
        // Second to last user message
        let second_last_content = user_msgs[1]["content"].as_array().unwrap();
        assert_eq!(
            second_last_content.last().unwrap()["cache_control"]["type"],
            "ephemeral"
        );
        // First user message should NOT have cache_control
        let first_content = user_msgs[0]["content"].as_array().unwrap();
        assert!(first_content.last().unwrap().get("cache_control").is_none());
    }

    // AC-4: prompt_caching adds caching beta header
    #[test]
    fn caching_adds_prompt_caching_beta_header() {
        let adapter = AnthropicAdapter::new("key").with_prompt_caching(true);
        let req = make_request("claude", vec![Message::user("hi")]);
        let betas = adapter.collect_beta_headers(&req);
        assert!(betas.contains(&"prompt-caching-2024-07-31".to_string()));
    }

    // AC-5: caching beta merged with other betas from provider_options
    #[test]
    fn caching_beta_merged_with_provider_options_betas() {
        let adapter = AnthropicAdapter::new("key").with_prompt_caching(true);
        let mut req = make_request("claude", vec![Message::user("hi")]);
        req.provider_options = Some(serde_json::json!({
            "beta_headers": ["interleaved-thinking-2025-05-14"]
        }));
        let betas = adapter.collect_beta_headers(&req);
        assert!(betas.contains(&"prompt-caching-2024-07-31".to_string()));
        assert!(betas.contains(&"interleaved-thinking-2025-05-14".to_string()));
    }

    // AC-6: disabled caching → no cache_control, no beta header
    #[test]
    fn disabled_caching_no_cache_control_no_beta() {
        let adapter = AnthropicAdapter::new("key"); // prompt_caching defaults to false
        let req = make_request("claude", vec![Message::system("sys"), Message::user("hi")]);
        let body = adapter.build_request_body(&req, false);
        let betas = adapter.collect_beta_headers(&req);
        // System should be a plain string
        assert!(body["system"].is_string());
        assert!(!betas.contains(&"prompt-caching-2024-07-31".to_string()));
    }

    // AC-9: one user message → only that gets cache_control
    #[test]
    fn caching_one_user_message_gets_cache_control() {
        let adapter = AnthropicAdapter::new("key").with_prompt_caching(true);
        let req = make_request("claude", vec![Message::user("only one")]);
        let body = adapter.build_request_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        let user_msgs: Vec<&Value> = msgs.iter().filter(|m| m["role"] == "user").collect();
        assert_eq!(user_msgs.len(), 1);
        let content = user_msgs[0]["content"].as_array().unwrap();
        assert_eq!(
            content.last().unwrap()["cache_control"]["type"],
            "ephemeral"
        );
    }

    // AC-10: no user messages → no crash (system still gets cache_control)
    #[test]
    fn caching_no_user_messages_no_panic() {
        let adapter = AnthropicAdapter::new("key").with_prompt_caching(true);
        let req = make_request("claude", vec![Message::system("only system")]);
        // Should not panic
        let body = adapter.build_request_body(&req, false);
        // System gets cache_control
        assert!(body["system"].is_array());
    }

    // SSE: text block events
    #[test]
    fn sse_text_block_events_translate_correctly() {
        use crate::streaming::StreamEventType;
        let mut state = AnthropicSseState::new();

        // content_block_start for text
        let (evs, term) = translate_anthropic_sse_event(
            "content_block_start",
            r#"{"index":0,"content_block":{"type":"text"}}"#,
            &mut state,
        );
        assert!(!term);
        assert_eq!(evs.len(), 1);
        assert!(matches!(&evs[0], Ok(e) if e.event_type == StreamEventType::TextStart));

        // content_block_delta text_delta
        let (evs2, term2) = translate_anthropic_sse_event(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            &mut state,
        );
        assert!(!term2);
        assert!(matches!(&evs2[0], Ok(e) if e.event_type == StreamEventType::TextDelta));
        assert_eq!(evs2[0].as_ref().unwrap().delta.as_deref(), Some("Hello"));

        // content_block_stop → TextEnd
        let (evs3, term3) =
            translate_anthropic_sse_event("content_block_stop", r#"{"index":0}"#, &mut state);
        assert!(!term3);
        assert!(matches!(&evs3[0], Ok(e) if e.event_type == StreamEventType::TextEnd));
    }

    // SSE: message_stop emits Finish with correct usage
    #[test]
    fn sse_message_stop_emits_finish_with_usage() {
        use crate::streaming::StreamEventType;
        let mut state = AnthropicSseState {
            input_tokens: 10,
            ..AnthropicSseState::new()
        };

        // message_delta captures output_tokens and stop_reason
        let _ = translate_anthropic_sse_event(
            "message_delta",
            r#"{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
            &mut state,
        );
        assert_eq!(state.output_tokens, 5);
        assert_eq!(state.stop_reason.as_deref(), Some("end_turn"));

        // message_stop emits Finish
        let (evs, term) = translate_anthropic_sse_event("message_stop", r#"{}"#, &mut state);
        assert!(term);
        assert!(matches!(&evs[0], Ok(e) if e.event_type == StreamEventType::Finish));
        let finish = evs[0].as_ref().unwrap();
        assert_eq!(finish.usage.as_ref().unwrap().input_tokens, 10);
        assert_eq!(finish.usage.as_ref().unwrap().output_tokens, 5);
        assert!(finish.finish_reason.as_ref().unwrap().is_stop());
    }

    // SSE: tool_use block events
    #[test]
    fn sse_tool_use_block_events_translate_correctly() {
        use crate::streaming::StreamEventType;
        let mut state = AnthropicSseState::new();

        // content_block_start for tool_use
        let (evs, _) = translate_anthropic_sse_event(
            "content_block_start",
            r#"{"index":0,"content_block":{"type":"tool_use","id":"call_1","name":"my_fn"}}"#,
            &mut state,
        );
        assert!(matches!(&evs[0], Ok(e) if e.event_type == StreamEventType::ToolCallStart));

        // content_block_delta with input_json_delta
        let (evs2, _) = translate_anthropic_sse_event(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}"#,
            &mut state,
        );
        assert!(matches!(&evs2[0], Ok(e) if e.event_type == StreamEventType::ToolCallDelta));

        // content_block_stop → ToolCallEnd
        let (evs3, _) =
            translate_anthropic_sse_event("content_block_stop", r#"{"index":0}"#, &mut state);
        assert!(matches!(&evs3[0], Ok(e) if e.event_type == StreamEventType::ToolCallEnd));
        let end_ev = evs3[0].as_ref().unwrap();
        assert_eq!(end_ev.tool_call.as_ref().unwrap().arguments["x"], 1);
    }

    // beta_headers as single string treated as list
    #[test]
    fn single_string_beta_header_treated_as_list() {
        let adapter = AnthropicAdapter::new("key");
        let mut req = make_request("claude", vec![Message::user("hi")]);
        req.provider_options = Some(serde_json::json!({
            "beta_headers": "interleaved-thinking-2025-05-14"
        }));
        let betas = adapter.collect_beta_headers(&req);
        assert!(betas.contains(&"interleaved-thinking-2025-05-14".to_string()));
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-001: URL construction
    // ---------------------------------------------------------------------------

    // Request URL contains /v1/messages path
    #[test]
    fn complete_url_format_uses_v1_messages() {
        let adapter = AnthropicAdapter::new("key").with_base_url("http://localhost:9090");
        let url = format!("{}/v1/messages", adapter.base_url);
        assert!(
            url.contains("/v1/messages"),
            "URL should contain /v1/messages: {url}"
        );
        assert!(
            url.starts_with("http://localhost:9090"),
            "URL should use base_url: {url}"
        );
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-002: provider_options pass-through
    // ---------------------------------------------------------------------------

    // provider_options fields (excluding beta_headers) are merged into body
    #[test]
    fn provider_options_merged_into_body() {
        let adapter = AnthropicAdapter::new("key");
        let mut req = make_request("claude", vec![Message::user("hi")]);
        req.provider_options = Some(serde_json::json!({ "custom_param": "custom_val" }));
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["custom_param"], "custom_val");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-003: All 5 roles wire format
    // ---------------------------------------------------------------------------

    // Developer role treated same as System — goes to body["system"] field
    #[test]
    fn developer_role_goes_to_system_field() {
        use crate::types::{ContentPart, Role};
        let adapter = AnthropicAdapter::new("key");
        let dev_msg = Message {
            role: Role::Developer,
            content: vec![ContentPart::text("Dev instructions.")],
            name: None,
            tool_call_id: None,
        };
        let req = make_request("claude", vec![dev_msg, Message::user("hello")]);
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["system"], "Dev instructions.");
        let msgs = body["messages"].as_array().unwrap();
        for m in msgs {
            assert_ne!(m["role"], "system");
            assert_ne!(m["role"], "developer");
        }
    }

    // User role → "user" in messages
    #[test]
    fn user_role_wire_format() {
        let adapter = AnthropicAdapter::new("key");
        let req = make_request("claude", vec![Message::user("hello")]);
        let body = adapter.build_request_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["type"], "text");
        assert_eq!(msgs[0]["content"][0]["text"], "hello");
    }

    // Assistant role → "assistant" with text type content
    #[test]
    fn assistant_role_wire_format() {
        let adapter = AnthropicAdapter::new("key");
        let req = make_request(
            "claude",
            vec![Message::user("hi"), Message::assistant("response")],
        );
        let body = adapter.build_request_body(&req, false);
        let msgs = body["messages"].as_array().unwrap();
        let asst = msgs.iter().find(|m| m["role"] == "assistant").unwrap();
        assert_eq!(asst["content"][0]["type"], "text");
        assert_eq!(asst["content"][0]["text"], "response");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-005: Image content serialization
    // ---------------------------------------------------------------------------

    // Image URL → {"type":"image","source":{"type":"url","url":...}}
    #[test]
    fn image_url_translates_correctly() {
        use crate::types::{ContentPart, ImageData};
        let part = ContentPart::image(ImageData {
            url: Some("https://example.com/img.png".to_string()),
            data: None,
            media_type: None,
            detail: None,
            path: None,
        });
        let result = translate_content_part_anthropic(&part).unwrap();
        assert_eq!(result["type"], "image");
        assert_eq!(result["source"]["type"], "url");
        assert_eq!(result["source"]["url"], "https://example.com/img.png");
    }

    // Image base64 → {"type":"image","source":{"type":"base64","media_type":...,"data":...}}
    #[test]
    fn image_base64_translates_correctly() {
        use crate::types::{ContentPart, ImageData};
        let part = ContentPart::image(ImageData {
            url: None,
            data: Some(vec![0u8, 1u8, 2u8]),
            media_type: Some("image/png".to_string()),
            detail: None,
            path: None,
        });
        let result = translate_content_part_anthropic(&part).unwrap();
        assert_eq!(result["type"], "image");
        assert_eq!(result["source"]["type"], "base64");
        assert_eq!(result["source"]["media_type"], "image/png");
        assert!(result["source"]["data"].as_str().is_some());
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-006: Tool call round-trip
    // ---------------------------------------------------------------------------

    // ToolCall content part → tool_use JSON
    #[test]
    fn tool_call_content_part_wire_format() {
        use crate::types::{ContentPart, ToolCallData};
        let part = ContentPart::tool_call(ToolCallData {
            id: "call-42".to_string(),
            name: "my_tool".to_string(),
            arguments: serde_json::json!({"x": 1}),
            raw_arguments: None,
        });
        let result = translate_content_part_anthropic(&part).unwrap();
        assert_eq!(result["type"], "tool_use");
        assert_eq!(result["id"], "call-42");
        assert_eq!(result["name"], "my_tool");
        assert_eq!(result["input"]["x"], 1);
    }

    // Response body with tool_use block → ToolCall content part (GAP-ULM-006)
    #[test]
    fn tool_use_response_body_round_trip() {
        let body = serde_json::json!({
            "id": "msg_1",
            "model": "claude-opus-4-5",
            "stop_reason": "tool_use",
            "content": [{
                "type": "tool_use",
                "id": "call_1",
                "name": "my_tool",
                "input": {"x": 42}
            }],
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        });
        let resp = AnthropicAdapter::parse_response_body(body, None).unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "my_tool");
        assert_eq!(calls[0].arguments["x"], 42);
        assert!(resp.finish_reason.is_tool_calls());
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-012: All ToolChoice modes serialized correctly
    // ---------------------------------------------------------------------------

    // ToolChoice::none() with tools → tools omitted entirely (already tested)
    // Adding: tool_choice per-provider completeness check
    #[test]
    fn all_tool_choice_modes_translated() {
        // auto → {"type": "auto"}
        let tc_auto = translate_tool_choice_anthropic(&ToolChoice::auto());
        assert_eq!(tc_auto["type"], "auto");
        // required → {"type": "any"}
        let tc_req = translate_tool_choice_anthropic(&ToolChoice::required());
        assert_eq!(tc_req["type"], "any");
        // named → {"type": "tool", "name": "fn"}
        let tc_named = translate_tool_choice_anthropic(&ToolChoice::named("fn"));
        assert_eq!(tc_named["type"], "tool");
        assert_eq!(tc_named["name"], "fn");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-013: is_error=true on ToolResult flows to provider JSON
    // ---------------------------------------------------------------------------

    // ToolResult with is_error=true → {"is_error": true} in tool_result block
    #[test]
    fn tool_result_is_error_true_in_wire_format() {
        use crate::types::{ContentPart, ToolResultData};
        let part = ContentPart::tool_result(ToolResultData {
            tool_call_id: "call-err".to_string(),
            content: serde_json::Value::String("something went wrong".to_string()),
            is_error: true,
        });
        let result = translate_content_part_anthropic(&part).unwrap();
        assert_eq!(result["type"], "tool_result");
        assert_eq!(result["is_error"], true);
        assert_eq!(result["content"], "something went wrong");
    }

    // ToolResult with is_error=false → {"is_error": false}
    #[test]
    fn tool_result_is_error_false_in_wire_format() {
        use crate::types::{ContentPart, ToolResultData};
        let part = ContentPart::tool_result(ToolResultData {
            tool_call_id: "call-ok".to_string(),
            content: serde_json::Value::String("success".to_string()),
            is_error: false,
        });
        let result = translate_content_part_anthropic(&part).unwrap();
        assert_eq!(result["is_error"], false);
    }

    // ---------------------------------------------------------------------------
    // V2-ULM-001: complete() retries HTTP 429 responses
    // ---------------------------------------------------------------------------

    /// V2-ULM-001: The retry wrapper must classify 429 as retryable INSIDE the
    /// retry closure so the policy can act on it.
    #[tokio::test]
    async fn complete_retries_on_429() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // First call returns 429
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_json(serde_json::json!({"error": {"message": "rate limited"}})),
            )
            .up_to_n_times(1)
            .mount(&mock_server)
            .await;

        // Second call returns a valid 200 response
        let ok_body = serde_json::json!({
            "id": "msg_1",
            "model": "claude-opus-4-5",
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        });
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body))
            .mount(&mock_server)
            .await;

        let adapter =
            AnthropicAdapter::new("test-key").with_base_url(mock_server.uri().to_string());

        let req = make_request("claude-opus-4-5", vec![Message::user("hi")]);
        let result = adapter.complete(&req).await;

        assert!(
            result.is_ok(),
            "complete() should succeed after retrying 429, got: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().text(), "hello");
    }

    // -----------------------------------------------------------------------
    // V2-ULM-005: ImageData.path reads from local file
    // -----------------------------------------------------------------------
    #[test]
    fn image_path_reads_local_file_and_base64_encodes() {
        use crate::types::{ContentPart, ImageData};
        // Write a small PNG-like file to /tmp
        let tmp = std::env::temp_dir().join("ulm_test_image_v2_005.png");
        std::fs::write(&tmp, b"\x89PNG\r\n\x1a\n").unwrap();

        let img = ImageData {
            url: None,
            data: None,
            path: Some(tmp.to_str().unwrap().to_string()),
            media_type: Some("image/png".to_string()),
            detail: None,
        };
        let part = ContentPart::image(img);
        let result = translate_content_part_anthropic(&part).unwrap();
        assert_eq!(result["type"], "image");
        assert_eq!(result["source"]["type"], "base64");
        assert_eq!(result["source"]["media_type"], "image/png");
        assert!(
            result["source"]["data"].as_str().is_some(),
            "base64 data must be present"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    // -----------------------------------------------------------------------
    // V2-ULM-006: Audio content → Err(InvalidRequest) not silent drop
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn complete_rejects_audio_content_with_error() {
        use crate::types::{AudioData, ContentKind, ContentPart, Message, Role};

        let audio_part = ContentPart {
            kind: ContentKind::Audio,
            text: None,
            image: None,
            audio: Some(AudioData {
                url: Some("https://example.com/audio.mp3".to_string()),
                data: None,
                media_type: Some("audio/mpeg".to_string()),
            }),
            document: None,
            tool_call: None,
            tool_result: None,
            thinking: None,
        };
        let msg = Message {
            role: Role::User,
            content: vec![audio_part],
            name: None,
            tool_call_id: None,
        };
        let req = make_request("claude-opus-4-5", vec![msg]);
        let adapter = AnthropicAdapter::new("key");
        let result = adapter.complete(&req).await;
        assert!(
            result.is_err(),
            "complete() with audio content must return an error"
        );
        assert!(
            matches!(result.unwrap_err(), UnifiedLlmError::InvalidRequest { .. }),
            "expected InvalidRequest error for audio content"
        );
    }

    // -----------------------------------------------------------------------
    // V2-ULM-009: HTTP error handling tests
    // -----------------------------------------------------------------------
    #[test]
    fn handle_error_401_returns_auth() {
        use crate::error::UnifiedLlmError;
        let err = AnthropicAdapter::parse_error_status(401, "bad key", None, None);
        assert!(matches!(err, UnifiedLlmError::Authentication { .. }));
    }

    #[test]
    fn handle_error_429_returns_rate_limit() {
        let err = AnthropicAdapter::parse_error_status(429, "rate limited", None, None);
        assert!(matches!(err, UnifiedLlmError::RateLimit { .. }));
    }

    #[test]
    fn handle_error_500_returns_provider_retryable() {
        let err = AnthropicAdapter::parse_error_status(500, "server error", None, None);
        match err {
            UnifiedLlmError::Provider { retryable, .. } => assert!(retryable),
            _ => panic!("expected Provider error for 500"),
        }
    }

    #[test]
    fn handle_error_400_context_length_returns_context_length() {
        let err = AnthropicAdapter::parse_error_status(
            400,
            "max context length exceeded",
            Some("invalid_request_error"),
            None,
        );
        assert!(
            matches!(err, UnifiedLlmError::ContextLength { .. }),
            "expected ContextLength for 400 + context error"
        );
    }

    // -----------------------------------------------------------------------
    // V2-ULM-011: Thinking block outbound serialization
    // -----------------------------------------------------------------------
    #[test]
    fn thinking_block_serialized_outbound() {
        use crate::types::{ContentPart, Message, Role, ThinkingData};
        let thinking_part = ContentPart::thinking(ThinkingData {
            text: "I think about this...".to_string(),
            signature: Some("sig_abc".to_string()),
            redacted: false,
        });
        let msg = Message {
            role: Role::Assistant,
            content: vec![thinking_part, ContentPart::text("answer")],
            name: None,
            tool_call_id: None,
        };
        let adapter = AnthropicAdapter::new("key");
        let req = make_request("claude", vec![Message::user("q"), msg]);
        let body = adapter.build_request_body(&req, false);
        // Find the thinking block in the messages
        let msgs = body["messages"].as_array().unwrap();
        let asst = msgs
            .iter()
            .find(|m| m["role"] == "assistant")
            .expect("assistant message expected");
        let content = asst["content"].as_array().unwrap();
        let has_thinking = content
            .iter()
            .any(|b| b["type"] == "thinking" && b.get("thinking").is_some());
        assert!(has_thinking, "thinking block must appear in outbound body");
    }

    // -----------------------------------------------------------------------
    // V2-ULM-012: reasoning_tokens populated from thinking_tokens field
    // -----------------------------------------------------------------------
    #[test]
    fn reasoning_tokens_populated_from_thinking_tokens_field() {
        let usage_val = serde_json::json!({
            "input_tokens": 10,
            "output_tokens": 20,
            "thinking_tokens": 5,
        });
        let usage = parse_anthropic_usage(&usage_val);
        assert_eq!(usage.reasoning_tokens, Some(5));
    }

    #[test]
    fn reasoning_tokens_none_when_no_thinking_field() {
        let usage_val = serde_json::json!({
            "input_tokens": 10,
            "output_tokens": 20,
        });
        let usage = parse_anthropic_usage(&usage_val);
        assert_eq!(usage.reasoning_tokens, None);
    }

    // -----------------------------------------------------------------------
    // V2-ULM-013: cache_control injected on last tool definition
    // -----------------------------------------------------------------------
    #[test]
    fn cache_control_injected_on_last_tool_definition() {
        use crate::types::Tool;
        let adapter = AnthropicAdapter::new("key").with_prompt_caching(true);
        let tools = vec![
            Tool {
                name: "tool_a".to_string(),
                description: "Tool A".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
            Tool {
                name: "tool_b".to_string(),
                description: "Tool B".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
        ];
        let req = make_request("claude", vec![Message::user("hi")]).with_tools(tools);
        let body = adapter.build_request_body(&req, false);
        let tool_arr = body["tools"].as_array().expect("tools array expected");
        // Only the LAST tool must have cache_control
        let last_tool = tool_arr.last().unwrap();
        assert!(
            last_tool.get("cache_control").is_some(),
            "last tool must have cache_control: {last_tool}"
        );
        // First tool must NOT have cache_control
        let first_tool = &tool_arr[0];
        assert!(
            first_tool.get("cache_control").is_none(),
            "first tool must NOT have cache_control: {first_tool}"
        );
    }
}
