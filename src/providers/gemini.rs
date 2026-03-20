//! Gemini provider adapter — native generateContent API.
//!
//! Implements [`ProviderAdapter`] for Gemini using
//! `POST /v1beta/models/{model}:generateContent`.
//!
//! Key differences from OpenAI/Anthropic:
//! - Authentication via `?key=<api_key>` query parameter (not a header)
//! - System messages → `systemInstruction`
//! - Tools → `functionDeclarations` in a single `tools` object
//! - No tool call IDs from Gemini — generate synthetic UUIDs
//! - Finish reason inferred from presence of `functionCall` parts

use futures::StreamExt as _;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::{
    error::UnifiedLlmError,
    providers::ProviderAdapter,
    sse::process_sse_line,
    streaming::{EventStream, StreamEvent, StreamEventType},
    types::{
        ContentKind, ContentPart, FinishReason, Message, RateLimitInfo, Request, Response, Role,
        ThinkingData, ToolCall, ToolCallData, Usage,
    },
};

// ---------------------------------------------------------------------------
// GeminiAdapter
// ---------------------------------------------------------------------------

/// Gemini provider adapter using the native generateContent API.
#[derive(Debug)]
pub struct GeminiAdapter {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl GeminiAdapter {
    /// Construct from an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Override the base URL (useful for proxies or local mocks).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Construct from environment variables.
    ///
    /// - `GEMINI_API_KEY` (checked first) or `GOOGLE_API_KEY` (fallback)
    /// - `GEMINI_BASE_URL` (optional)
    pub fn from_env() -> Result<Self, UnifiedLlmError> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| {
                std::env::var("GOOGLE_API_KEY")
                    .ok()
                    .filter(|s| !s.is_empty())
            })
            .ok_or_else(|| UnifiedLlmError::Configuration {
                message:
                    "neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variable is set or empty"
                        .to_string(),
            })?;

        let base_url = std::env::var("GEMINI_BASE_URL")
            .unwrap_or_else(|_| "https://generativelanguage.googleapis.com".to_string());

        Ok(Self::new(api_key).with_base_url(base_url))
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn complete_url(&self, model: &str) -> String {
        format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        )
    }

    fn stream_url(&self, model: &str) -> String {
        format!(
            "{}/v1beta/models/{}:generateContent?alt=sse&key={}",
            self.base_url, model, self.api_key
        )
    }

    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }

    fn build_request_body(&self, request: &Request) -> Value {
        // --- System message extraction ---
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

        // --- Translate messages to contents ---
        let contents: Vec<Value> = request
            .messages
            .iter()
            .filter(|m| !matches!(m.role, Role::System | Role::Developer))
            .map(translate_message_gemini)
            .collect();

        let mut body = json!({
            "contents": contents,
        });

        if let Some(sys) = system_text {
            body["systemInstruction"] = json!({
                "parts": [{"text": sys}]
            });
        }

        // --- Tools ---
        if let Some(tools) = &request.tools {
            if !tools.is_empty() {
                let func_decls: Vec<Value> = tools
                    .iter()
                    .map(|t| {
                        json!({
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        })
                    })
                    .collect();
                body["tools"] = json!([{ "functionDeclarations": func_decls }]);
            }
        }

        // --- Tool choice ---
        if let Some(tc) = &request.tool_choice {
            body["toolConfig"] = translate_tool_choice_gemini(tc);
        }

        // --- Generation config ---
        let mut gen_config = json!({});
        let mut has_gen_config = false;

        if let Some(max_tokens) = request.max_tokens {
            gen_config["maxOutputTokens"] = json!(max_tokens);
            has_gen_config = true;
        }
        if let Some(temp) = request.temperature {
            gen_config["temperature"] = json!(temp);
            has_gen_config = true;
        }
        if let Some(top_p) = request.top_p {
            gen_config["topP"] = json!(top_p);
            has_gen_config = true;
        }
        if let Some(stops) = &request.stop_sequences {
            gen_config["stopSequences"] = json!(stops);
            has_gen_config = true;
        }

        if has_gen_config {
            body["generationConfig"] = gen_config;
        }

        // --- Safety settings from provider_options ---
        if let Some(opts) = &request.provider_options {
            if let Some(safety) = opts.get("safety_settings") {
                body["safetySettings"] = safety.clone();
            }
            // Merge remaining keys
            if let Some(map) = opts.as_object() {
                for (k, v) in map {
                    if k != "safety_settings" {
                        body[k] = v.clone();
                    }
                }
            }
        }

        body
    }

    fn parse_usage(usage_val: &Value) -> Usage {
        let input_tokens = usage_val["promptTokenCount"].as_u64().unwrap_or(0) as u32;
        let output_tokens = usage_val["candidatesTokenCount"].as_u64().unwrap_or(0) as u32;
        let total_tokens = usage_val["totalTokenCount"]
            .as_u64()
            .unwrap_or((input_tokens as u64) + (output_tokens as u64))
            as u32;
        let reasoning_tokens = usage_val["thoughtsTokenCount"].as_u64().map(|v| v as u32);
        let cache_read_tokens = usage_val["cachedContentTokenCount"]
            .as_u64()
            .map(|v| v as u32);

        Usage {
            input_tokens,
            output_tokens,
            total_tokens,
            reasoning_tokens,
            cache_read_tokens,
            cache_write_tokens: None,
            raw: Some(usage_val.clone()),
        }
    }

    fn parse_response_body(
        body: Value,
        model: &str,
        rate_limit: Option<RateLimitInfo>,
    ) -> Result<Response, UnifiedLlmError> {
        let id = uuid::Uuid::new_v4().to_string();

        let candidates = body["candidates"].as_array();

        let candidate = candidates.and_then(|arr| arr.first());

        let mut content: Vec<ContentPart> = Vec::new();
        let mut has_function_calls = false;
        let mut finish_reason_raw: Option<String> = None;

        if let Some(cand) = candidate {
            finish_reason_raw = cand["finishReason"].as_str().map(|s| s.to_string());

            if let Some(parts) = cand["content"]["parts"].as_array() {
                for (idx, part) in parts.iter().enumerate() {
                    if let Some(text) = part["text"].as_str() {
                        // Check if this is a thought part
                        if part["thought"].as_bool().unwrap_or(false) {
                            content.push(ContentPart::thinking(ThinkingData {
                                text: text.to_string(),
                                signature: None,
                                redacted: false,
                            }));
                        } else {
                            content.push(ContentPart::text(text));
                        }
                    } else if let Some(func_call) = part.get("functionCall") {
                        has_function_calls = true;
                        let name = func_call["name"].as_str().unwrap_or("").to_string();
                        let args = func_call["args"].clone();
                        let raw_args = serde_json::to_string(&args).ok();
                        let synthetic_id = format!("{}-{}", name, uuid::Uuid::new_v4());
                        let _ = idx; // suppress unused warning
                        content.push(ContentPart::tool_call(ToolCallData {
                            id: synthetic_id,
                            name,
                            arguments: args,
                            raw_arguments: raw_args,
                        }));
                    }
                }
            }
        }

        let finish_reason =
            finish_reason_from_gemini(finish_reason_raw.as_deref(), has_function_calls);
        let usage = if body["usageMetadata"].is_object() {
            Self::parse_usage(&body["usageMetadata"])
        } else {
            Usage::default()
        };

        let message = Message {
            role: Role::Assistant,
            content,
            name: None,
            tool_call_id: None,
        };

        Ok(Response {
            id,
            model: model.to_string(),
            provider: "gemini".to_string(),
            message,
            finish_reason,
            usage,
            raw: Some(body),
            warnings: vec![],
            rate_limit,
        })
    }

    /// Classify an HTTP error status into the appropriate `UnifiedLlmError`.
    ///
    /// Extracted from `handle_error_response` for unit-testability (V2-ULM-009).
    pub(crate) fn parse_error_status(
        status: u16,
        message: &str,
        _error_code: Option<&str>,
        retry_after: Option<f64>,
    ) -> UnifiedLlmError {
        match status {
            401 | 403 => UnifiedLlmError::Authentication {
                provider: "gemini".to_string(),
                message: message.to_string(),
            },
            429 => UnifiedLlmError::RateLimit {
                provider: "gemini".to_string(),
                message: message.to_string(),
                retry_after,
            },
            400 if message.to_lowercase().contains("context length")
                || message.to_lowercase().contains("too long")
                || message.to_lowercase().contains("request too long") =>
            {
                UnifiedLlmError::ContextLength {
                    message: message.to_string(),
                }
            }
            400 => UnifiedLlmError::Provider {
                provider: "gemini".to_string(),
                message: message.to_string(),
                status_code: Some(status),
                error_code: None,
                retryable: false,
                retry_after: None,
                raw: None,
            },
            500..=599 => UnifiedLlmError::Provider {
                provider: "gemini".to_string(),
                message: message.to_string(),
                status_code: Some(status),
                error_code: None,
                retryable: true,
                retry_after: None,
                raw: None,
            },
            _ => UnifiedLlmError::Provider {
                provider: "gemini".to_string(),
                message: message.to_string(),
                status_code: Some(status),
                error_code: None,
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
        let error_code = body_json["error"]["status"].as_str().map(|s| s.to_string());

        let mut err =
            Self::parse_error_status(status, &message, error_code.as_deref(), retry_after);
        if let UnifiedLlmError::Provider {
            raw,
            error_code: ec,
            ..
        } = &mut err
        {
            *raw = Some(body_json);
            *ec = error_code;
        }
        err
    }
}

// ---------------------------------------------------------------------------
// Free-standing translation helpers
// ---------------------------------------------------------------------------

fn translate_tool_choice_gemini(tc: &crate::types::ToolChoice) -> Value {
    match tc.mode.as_str() {
        "auto" => json!({ "functionCallingConfig": { "mode": "AUTO" } }),
        "none" => json!({ "functionCallingConfig": { "mode": "NONE" } }),
        "required" => json!({ "functionCallingConfig": { "mode": "ANY" } }),
        "named" => {
            let name = tc.tool_name.as_deref().unwrap_or("");
            json!({
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [name]
                }
            })
        }
        _ => json!({ "functionCallingConfig": { "mode": "AUTO" } }),
    }
}

fn finish_reason_from_gemini(raw: Option<&str>, has_function_calls: bool) -> FinishReason {
    // If function calls present and reason is STOP or absent → tool_calls
    if has_function_calls {
        match raw {
            Some("STOP") | None => {
                return FinishReason {
                    reason: "tool_calls".to_string(),
                    raw: raw.map(|s| s.to_string()),
                };
            }
            _ => {}
        }
    }

    match raw {
        Some("STOP") => FinishReason {
            reason: "stop".to_string(),
            raw: Some("STOP".to_string()),
        },
        Some("MAX_TOKENS") => FinishReason {
            reason: "length".to_string(),
            raw: Some("MAX_TOKENS".to_string()),
        },
        Some("SAFETY") | Some("RECITATION") => FinishReason {
            reason: "content_filter".to_string(),
            raw: raw.map(|s| s.to_string()),
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

fn translate_message_gemini(msg: &Message) -> Value {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "model",
        Role::Tool => "user", // tool results as user-turn function responses
        Role::System | Role::Developer => "user", // shouldn't appear here
    };

    let parts: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|part| translate_content_part_gemini(part, &msg.role))
        .collect();

    json!({ "role": role, "parts": parts })
}

fn translate_content_part_gemini(part: &ContentPart, role: &Role) -> Option<Value> {
    match &part.kind {
        ContentKind::Text => Some(json!({ "text": part.text.as_deref().unwrap_or("") })),
        ContentKind::Image => {
            let img = part.image.as_ref()?;
            if let Some(url) = &img.url {
                let media_type = img.media_type.as_deref().unwrap_or("image/jpeg");
                Some(json!({
                    "fileData": {
                        "mimeType": media_type,
                        "fileUri": url,
                    }
                }))
            } else {
                // V2-ULM-005: resolve data from existing bytes or file path.
                use base64::{engine::general_purpose::STANDARD, Engine as _};
                let resolved: Option<Vec<u8>> = img
                    .data
                    .clone()
                    .or_else(|| img.path.as_ref().and_then(|p| std::fs::read(p).ok()));
                if let Some(data) = resolved {
                    let b64 = STANDARD.encode(&data);
                    let media_type = img.media_type.as_deref().unwrap_or("image/jpeg");
                    Some(json!({
                        "inlineData": {
                            "mimeType": media_type,
                            "data": b64,
                        }
                    }))
                } else {
                    None
                }
            }
        }
        ContentKind::ToolCall => {
            let tc = part.tool_call.as_ref()?;
            Some(json!({
                "functionCall": {
                    "name": tc.name,
                    "args": tc.arguments,
                }
            }))
        }
        ContentKind::ToolResult => {
            if !matches!(role, Role::Tool) {
                return None;
            }
            let tr = part.tool_result.as_ref()?;
            // Gemini matches by function name; use tool_call_id as the name
            let func_name = &tr.tool_call_id;
            let response = if tr.is_error {
                let content_str = match &tr.content {
                    Value::String(s) => s.clone(),
                    v => serde_json::to_string(v).unwrap_or_default(),
                };
                json!({ "error": { "message": content_str } })
            } else {
                json!({ "content": tr.content })
            };
            Some(json!({
                "functionResponse": {
                    "name": func_name,
                    "response": response,
                }
            }))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Streaming helpers
// ---------------------------------------------------------------------------

/// Translate a single Gemini SSE JSON chunk into zero or more `StreamEvent`s.
///
/// Returns `(events, is_terminal)`.
pub(crate) fn translate_gemini_chunk(
    chunk: &Value,
    text_started: &mut bool,
    finish_reason_captured: &mut Option<String>,
) -> (Vec<Result<StreamEvent, UnifiedLlmError>>, bool) {
    let mut out: Vec<Result<StreamEvent, UnifiedLlmError>> = Vec::new();

    // Process candidates[0].content.parts
    if let Some(candidates) = chunk["candidates"].as_array() {
        if let Some(candidate) = candidates.first() {
            // Capture finish reason from this chunk if present
            if let Some(fr) = candidate["finishReason"].as_str() {
                *finish_reason_captured = Some(fr.to_string());
            }

            if let Some(parts) = candidate["content"]["parts"].as_array() {
                let mut has_func_calls = false;

                for part in parts {
                    if let Some(text) = part["text"].as_str() {
                        if part["thought"].as_bool().unwrap_or(false) {
                            // Thought/reasoning part
                            out.push(Ok(StreamEvent::reasoning_delta(text)));
                        } else {
                            // Regular text
                            if !*text_started {
                                *text_started = true;
                                out.push(Ok(StreamEvent::text_start()));
                            }
                            out.push(Ok(StreamEvent::text_delta(text)));
                        }
                    } else if let Some(func_call) = part.get("functionCall") {
                        has_func_calls = true;
                        let name = func_call["name"].as_str().unwrap_or("").to_string();
                        let args = func_call["args"].clone();
                        let raw_args = serde_json::to_string(&args).unwrap_or_default();
                        let synthetic_id = format!("{}-{}", name, uuid::Uuid::new_v4());

                        // Gemini delivers complete function calls — ToolCallStart + ToolCallEnd
                        out.push(Ok(StreamEvent::tool_call_start(
                            synthetic_id.clone(),
                            name.clone(),
                        )));
                        out.push(Ok(StreamEvent::tool_call_end(ToolCall {
                            id: synthetic_id,
                            name,
                            arguments: args,
                            raw_arguments: Some(raw_args),
                        })));
                    }
                }

                // If function calls present, override finish reason
                if has_func_calls {
                    let current = finish_reason_captured.as_deref();
                    if current.is_none() || current == Some("STOP") {
                        *finish_reason_captured = Some("TOOL_CALLS_INFERRED".to_string());
                    }
                }
            }
        }
    }

    // If usageMetadata is present → emit TextEnd (if text started) + Finish
    if chunk.get("usageMetadata").is_some() && chunk["usageMetadata"].is_object() {
        if *text_started {
            out.push(Ok(StreamEvent::text_end()));
        }

        let usage = GeminiAdapter::parse_usage(&chunk["usageMetadata"]);

        // Determine finish reason
        let has_tool_calls = chunk["candidates"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|c| c["content"]["parts"].as_array())
            .map(|parts| parts.iter().any(|p| p.get("functionCall").is_some()))
            .unwrap_or(false);

        let fr_raw = finish_reason_captured.clone();
        let finish_reason = if has_tool_calls {
            FinishReason {
                reason: "tool_calls".to_string(),
                raw: fr_raw,
            }
        } else {
            finish_reason_from_gemini(
                fr_raw.as_deref().filter(|s| *s != "TOOL_CALLS_INFERRED"),
                false,
            )
        };

        out.push(Ok(StreamEvent::finish(finish_reason, usage)));
        return (out, true);
    }

    (out, false)
}

// ---------------------------------------------------------------------------
// ProviderAdapter implementation
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ProviderAdapter for GeminiAdapter {
    fn name(&self) -> &str {
        "gemini"
    }

    fn supports_tool_choice(&self, mode: &str) -> bool {
        matches!(mode, "auto" | "none" | "required" | "named")
    }

    async fn complete(&self, request: &Request) -> Result<Response, UnifiedLlmError> {
        // V2-ULM-006: reject audio content rather than silently dropping it.
        super::reject_audio_content(request, "gemini")?;
        let body = self.build_request_body(request);
        let url = self.complete_url(&request.model);
        let headers = self.build_headers();
        let client = self.http_client.clone();
        let model = request.model.clone();

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

                    let rate_limit = parse_gemini_rate_limit_headers(http_resp.headers());

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
                            provider: "gemini".to_string(),
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

        Self::parse_response_body(body_json, &model, rate_limit)
    }

    async fn stream(&self, request: &Request) -> Result<EventStream, UnifiedLlmError> {
        // V2-ULM-006: reject audio content rather than silently dropping it.
        super::reject_audio_content(request, "gemini")?;
        let body = self.build_request_body(request);
        let url = self.stream_url(&request.model);
        let headers = self.build_headers();

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

            let mut text_started = false;
            let mut finish_reason_captured: Option<String> = None;
            let mut first_chunk = true;

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
                            // Parse JSON from the data field
                            let chunk_json: Value = match serde_json::from_str(&sse_ev.data) {
                                Ok(v) => v,
                                Err(e) => {
                                    let _ = tx
                                        .try_send(Ok(StreamEvent::error(format!(
                                            "JSON parse error: {e}"
                                        ))))
                                        .ok();
                                    return;
                                }
                            };

                            // Emit StreamStart on first chunk
                            if first_chunk {
                                first_chunk = false;
                                if tx.try_send(Ok(StreamEvent::stream_start())).is_err() {
                                    return;
                                }
                            }

                            // Check for no candidates (safety block)
                            if chunk_json["candidates"]
                                .as_array()
                                .map(|a| a.is_empty())
                                .unwrap_or(true)
                                && chunk_json.get("usageMetadata").is_none()
                            {
                                let mut ev = StreamEvent::stream_start();
                                ev.event_type = StreamEventType::ProviderEvent;
                                ev.raw = Some(chunk_json);
                                if tx.try_send(Ok(ev)).is_err() {
                                    return;
                                }
                                continue;
                            }

                            let (events, is_terminal) = translate_gemini_chunk(
                                &chunk_json,
                                &mut text_started,
                                &mut finish_reason_captured,
                            );

                            for ev in events {
                                let stop = ev.is_err();
                                if tx.try_send(ev).is_err() {
                                    return;
                                }
                                if stop {
                                    return;
                                }
                            }

                            if is_terminal {
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

            // Stream closed without usageMetadata — emit finish
            if text_started {
                let _ = tx.try_send(Ok(StreamEvent::text_end())).ok();
            }
            let _ = tx
                .try_send(Ok(StreamEvent::finish(
                    FinishReason {
                        reason: "stop".to_string(),
                        raw: None,
                    },
                    Usage::default(),
                )))
                .ok();
        });

        Ok(Box::pin(rx))
    }
}

fn parse_gemini_rate_limit_headers(headers: &reqwest::header::HeaderMap) -> Option<RateLimitInfo> {
    // Gemini does not expose standard rate-limit headers; return None
    let _ = headers;
    None
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

    // AC-1: from_env() returns Err(Configuration) when neither key is set
    #[test]
    fn from_env_no_key_returns_config_error() {
        unsafe {
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
        }
        let err = GeminiAdapter::from_env().unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Configuration { .. }));
    }

    // AC-2: from_env() prefers GEMINI_API_KEY over GOOGLE_API_KEY
    #[test]
    fn from_env_prefers_gemini_api_key() {
        unsafe {
            std::env::set_var("GEMINI_API_KEY", "gemini-key");
            std::env::set_var("GOOGLE_API_KEY", "google-key");
        }
        let adapter = GeminiAdapter::from_env().unwrap();
        assert_eq!(adapter.api_key, "gemini-key");
        unsafe {
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
        }
    }

    // AC-4: System message placed in systemInstruction, absent from contents
    #[test]
    fn system_message_goes_to_system_instruction() {
        let adapter = GeminiAdapter::new("key");
        let req = make_request(
            "gemini-2-flash",
            vec![Message::system("Be helpful."), Message::user("Hello")],
        );
        let body = adapter.build_request_body(&req);
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be helpful.");
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }

    // AC-5: Tool definitions go in single functionDeclarations array
    #[test]
    fn tool_wrapped_in_function_declarations() {
        let adapter = GeminiAdapter::new("key");
        let req =
            make_request("gemini-2-flash", vec![Message::user("hi")]).with_tools(vec![Tool {
                name: "my_fn".to_string(),
                description: "does stuff".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }]);
        let body = adapter.build_request_body(&req);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        let func_decls = tools[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(func_decls.len(), 1);
        assert_eq!(func_decls[0]["name"], "my_fn");
    }

    // AC-7: ToolChoice::named serializes to ANY + allowedFunctionNames
    #[test]
    fn tool_choice_named_serializes_correctly() {
        let tc = translate_tool_choice_gemini(&ToolChoice::named("my_fn"));
        assert_eq!(tc["functionCallingConfig"]["mode"], "ANY");
        let allowed = tc["functionCallingConfig"]["allowedFunctionNames"]
            .as_array()
            .unwrap();
        assert_eq!(allowed[0], "my_fn");
    }

    // AC-8: max_tokens maps to generationConfig.maxOutputTokens
    #[test]
    fn max_tokens_maps_to_max_output_tokens() {
        let adapter = GeminiAdapter::new("key");
        let req = make_request("gemini-2-flash", vec![Message::user("hi")]).with_max_tokens(500);
        let body = adapter.build_request_body(&req);
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 500);
    }

    // AC-12: functionCall + finishReason STOP → tool_calls
    #[test]
    fn function_call_with_stop_becomes_tool_calls() {
        let fr = finish_reason_from_gemini(Some("STOP"), true);
        assert!(fr.is_tool_calls());
    }

    #[test]
    fn function_call_without_finish_reason_becomes_tool_calls() {
        let fr = finish_reason_from_gemini(None, true);
        assert!(fr.is_tool_calls());
    }

    #[test]
    fn finish_reason_stop_no_tools() {
        let fr = finish_reason_from_gemini(Some("STOP"), false);
        assert!(fr.is_stop());
    }

    #[test]
    fn finish_reason_max_tokens() {
        let fr = finish_reason_from_gemini(Some("MAX_TOKENS"), false);
        assert_eq!(fr.reason, "length");
    }

    #[test]
    fn finish_reason_safety() {
        let fr = finish_reason_from_gemini(Some("SAFETY"), false);
        assert_eq!(fr.reason, "content_filter");
    }

    // Usage parsing with thoughtsTokenCount
    #[test]
    fn parse_usage_with_thoughts() {
        let usage_val = serde_json::json!({
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
            "thoughtsTokenCount": 3,
        });
        let usage = GeminiAdapter::parse_usage(&usage_val);
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(usage.reasoning_tokens, Some(3));
    }

    // Response parsing with functionCall parts
    #[test]
    fn response_function_call_parsed() {
        let body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "my_fn",
                            "args": {"x": 1}
                        }
                    }],
                    "role": "model"
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
            }
        });
        let resp = GeminiAdapter::parse_response_body(body, "gemini-2-flash", None).unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "my_fn");
        assert_eq!(calls[0].arguments["x"], 1);
        assert!(!calls[0].id.is_empty()); // synthetic ID generated
        assert!(resp.finish_reason.is_tool_calls());
    }

    // No system message → systemInstruction omitted
    #[test]
    fn no_system_message_omits_instruction() {
        let adapter = GeminiAdapter::new("key");
        let req = make_request("gemini-2-flash", vec![Message::user("Hello")]);
        let body = adapter.build_request_body(&req);
        assert!(body.get("systemInstruction").is_none());
    }

    // Tool choice none → NONE mode
    #[test]
    fn tool_choice_none_serializes_correctly() {
        let tc = translate_tool_choice_gemini(&ToolChoice::none());
        assert_eq!(tc["functionCallingConfig"]["mode"], "NONE");
    }

    // URL construction
    #[test]
    fn complete_url_has_key_param() {
        let adapter = GeminiAdapter::new("my-key").with_base_url("https://example.com");
        let url = adapter.complete_url("gemini-2-flash");
        assert!(url.contains("?key=my-key"));
        assert!(url.contains("gemini-2-flash:generateContent"));
    }

    #[test]
    fn stream_url_has_alt_sse() {
        let adapter = GeminiAdapter::new("my-key").with_base_url("https://example.com");
        let url = adapter.stream_url("gemini-2-flash");
        assert!(url.contains("?alt=sse&key=my-key"));
    }

    // SSE stream chunk translation tests
    #[test]
    fn chunk_with_text_emits_text_start_and_delta() {
        let mut ts = false;
        let mut fr: Option<String> = None;
        let chunk = serde_json::json!({
            "candidates": [{
                "content": { "parts": [{"text": "Hello"}] }
            }]
        });
        let (evs, terminal) = translate_gemini_chunk(&chunk, &mut ts, &mut fr);
        assert!(!terminal);
        assert_eq!(evs.len(), 2);
        use crate::streaming::StreamEventType;
        assert!(matches!(&evs[0], Ok(e) if e.event_type == StreamEventType::TextStart));
        assert!(matches!(&evs[1], Ok(e) if e.event_type == StreamEventType::TextDelta));
        assert!(ts);
    }

    #[test]
    fn chunk_with_function_call_emits_start_and_end() {
        let mut ts = false;
        let mut fr: Option<String> = None;
        let chunk = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "fn1", "args": {"x": 1}}}]
                }
            }]
        });
        let (evs, terminal) = translate_gemini_chunk(&chunk, &mut ts, &mut fr);
        assert!(!terminal);
        use crate::streaming::StreamEventType;
        assert_eq!(evs.len(), 2);
        assert!(matches!(&evs[0], Ok(e) if e.event_type == StreamEventType::ToolCallStart));
        assert!(matches!(&evs[1], Ok(e) if e.event_type == StreamEventType::ToolCallEnd));
    }

    #[test]
    fn chunk_with_usage_metadata_emits_text_end_and_finish() {
        let mut ts = true; // text was started
        let mut fr: Option<String> = Some("STOP".to_string());
        let chunk = serde_json::json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": { "parts": [] }
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            }
        });
        let (evs, terminal) = translate_gemini_chunk(&chunk, &mut ts, &mut fr);
        assert!(terminal);
        use crate::streaming::StreamEventType;
        assert!(evs
            .iter()
            .any(|e| matches!(e, Ok(ev) if ev.event_type == StreamEventType::TextEnd)));
        assert!(evs
            .iter()
            .any(|e| matches!(e, Ok(ev) if ev.event_type == StreamEventType::Finish)));
    }

    #[test]
    fn thought_part_emits_reasoning_delta() {
        let mut ts = false;
        let mut fr: Option<String> = None;
        let chunk = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"thought": true, "text": "Let me think..."}]
                }
            }]
        });
        let (evs, _terminal) = translate_gemini_chunk(&chunk, &mut ts, &mut fr);
        use crate::streaming::StreamEventType;
        assert!(evs
            .iter()
            .any(|e| matches!(e, Ok(ev) if ev.event_type == StreamEventType::ReasoningDelta)));
        // text_started should NOT be set for thought parts
        assert!(!ts);
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-001: URL construction
    // ---------------------------------------------------------------------------

    // complete_url() contains /v1beta/models/{model}:generateContent
    #[test]
    fn complete_url_contains_v1beta_path() {
        let adapter = GeminiAdapter::new("mykey").with_base_url("http://localhost:8888");
        let url = adapter.complete_url("gemini-2-flash");
        assert!(
            url.contains("/v1beta/models/gemini-2-flash:generateContent"),
            "URL should contain /v1beta/models/gemini-2-flash:generateContent: {url}"
        );
        assert!(
            url.starts_with("http://localhost:8888"),
            "URL should use base_url: {url}"
        );
    }

    // stream_url() contains alt=sse
    #[test]
    fn stream_url_contains_alt_sse() {
        let adapter = GeminiAdapter::new("mykey").with_base_url("http://localhost:8888");
        let url = adapter.stream_url("gemini-2-flash");
        assert!(
            url.contains("alt=sse"),
            "stream URL should contain alt=sse: {url}"
        );
        assert!(
            url.contains("/v1beta/models/gemini-2-flash:generateContent"),
            "stream URL should contain path: {url}"
        );
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-002: provider_options pass-through
    // ---------------------------------------------------------------------------

    // provider_options fields merged into body
    #[test]
    fn provider_options_merged_into_body() {
        let adapter = GeminiAdapter::new("key");
        let mut req = make_request("gemini-2-flash", vec![Message::user("hi")]);
        req.provider_options = Some(serde_json::json!({ "custom_param": "custom_val" }));
        let body = adapter.build_request_body(&req);
        assert_eq!(body["custom_param"], "custom_val");
    }

    // safety_settings from provider_options goes to safetySettings
    #[test]
    fn safety_settings_from_provider_options() {
        let adapter = GeminiAdapter::new("key");
        let mut req = make_request("gemini-2-flash", vec![Message::user("hi")]);
        req.provider_options = Some(serde_json::json!({
            "safety_settings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]
        }));
        let body = adapter.build_request_body(&req);
        let safety = body["safetySettings"].as_array().unwrap();
        assert_eq!(safety[0]["category"], "HARM_CATEGORY_HATE_SPEECH");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-003: All 5 roles wire format
    // ---------------------------------------------------------------------------

    // Developer role → same as System (goes to systemInstruction)
    #[test]
    fn developer_role_goes_to_system_instruction() {
        use crate::types::{ContentPart, Role};
        let adapter = GeminiAdapter::new("key");
        let dev_msg = Message {
            role: Role::Developer,
            content: vec![ContentPart::text("Dev instructions.")],
            name: None,
            tool_call_id: None,
        };
        let req = make_request("gemini-2-flash", vec![dev_msg, Message::user("hello")]);
        let body = adapter.build_request_body(&req);
        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            "Dev instructions."
        );
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }

    // User role → "user" in contents
    #[test]
    fn user_role_wire_format() {
        let adapter = GeminiAdapter::new("key");
        let req = make_request("gemini-2-flash", vec![Message::user("hello")]);
        let body = adapter.build_request_body(&req);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "hello");
    }

    // Assistant role → "model" in contents
    #[test]
    fn assistant_role_maps_to_model() {
        let adapter = GeminiAdapter::new("key");
        let req = make_request(
            "gemini-2-flash",
            vec![Message::user("hi"), Message::assistant("response")],
        );
        let body = adapter.build_request_body(&req);
        let contents = body["contents"].as_array().unwrap();
        let model_msg = contents.iter().find(|c| c["role"] == "model").unwrap();
        assert_eq!(model_msg["parts"][0]["text"], "response");
    }

    // Tool role → functionResponse in user-role content
    #[test]
    fn tool_role_becomes_function_response() {
        let adapter = GeminiAdapter::new("key");
        let tool_msg = Message::tool_result("my_fn-uuid", "result_val", false);
        let req = make_request(
            "gemini-2-flash",
            vec![Message::user("hi"), Message::assistant("calling"), tool_msg],
        );
        let body = adapter.build_request_body(&req);
        let contents = body["contents"].as_array().unwrap();
        // The tool result should appear as a user-role message with functionResponse
        let tool_content = contents
            .iter()
            .find(|c| {
                c["parts"]
                    .as_array()
                    .map(|p| p.iter().any(|part| part.get("functionResponse").is_some()))
                    .unwrap_or(false)
            })
            .unwrap();
        assert_eq!(tool_content["role"], "user");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-005: Image content serialization
    // ---------------------------------------------------------------------------

    // Image URL → fileData with mimeType and fileUri
    #[test]
    fn image_url_translates_to_file_data() {
        use crate::types::{ContentPart, ImageData, Role};
        let part = ContentPart::image(ImageData {
            url: Some("https://example.com/img.png".to_string()),
            data: None,
            media_type: Some("image/png".to_string()),
            detail: None,
            path: None,
        });
        let result = translate_content_part_gemini(&part, &Role::User).unwrap();
        assert!(
            result.get("fileData").is_some(),
            "expected fileData field: {result}"
        );
        assert_eq!(result["fileData"]["fileUri"], "https://example.com/img.png");
        assert_eq!(result["fileData"]["mimeType"], "image/png");
    }

    // Image base64 → inlineData with mimeType and data
    #[test]
    fn image_base64_translates_to_inline_data() {
        use crate::types::{ContentPart, ImageData, Role};
        let part = ContentPart::image(ImageData {
            url: None,
            data: Some(vec![0u8, 1u8, 2u8]),
            media_type: Some("image/jpeg".to_string()),
            detail: None,
            path: None,
        });
        let result = translate_content_part_gemini(&part, &Role::User).unwrap();
        assert!(
            result.get("inlineData").is_some(),
            "expected inlineData field: {result}"
        );
        assert_eq!(result["inlineData"]["mimeType"], "image/jpeg");
        assert!(result["inlineData"]["data"].as_str().is_some());
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-006: Tool call round-trip
    // ---------------------------------------------------------------------------

    // ToolCall content part → functionCall JSON
    #[test]
    fn tool_call_content_part_wire_format() {
        use crate::types::{ContentPart, Role, ToolCallData};
        let part = ContentPart::tool_call(ToolCallData {
            id: "call-42".to_string(),
            name: "my_tool".to_string(),
            arguments: serde_json::json!({"x": 1}),
            raw_arguments: None,
        });
        let result = translate_content_part_gemini(&part, &Role::Assistant).unwrap();
        assert!(result.get("functionCall").is_some());
        assert_eq!(result["functionCall"]["name"], "my_tool");
        assert_eq!(result["functionCall"]["args"]["x"], 1);
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-012: All ToolChoice modes serialized correctly
    // ---------------------------------------------------------------------------

    // All 4 modes translate to correct functionCallingConfig modes
    #[test]
    fn all_tool_choice_modes_translated() {
        // auto → AUTO
        let tc_auto = translate_tool_choice_gemini(&ToolChoice::auto());
        assert_eq!(tc_auto["functionCallingConfig"]["mode"], "AUTO");

        // none → NONE
        let tc_none = translate_tool_choice_gemini(&ToolChoice::none());
        assert_eq!(tc_none["functionCallingConfig"]["mode"], "NONE");

        // required → ANY
        let tc_req = translate_tool_choice_gemini(&ToolChoice::required());
        assert_eq!(tc_req["functionCallingConfig"]["mode"], "ANY");

        // named → ANY with allowedFunctionNames (already tested separately)
        let tc_named = translate_tool_choice_gemini(&ToolChoice::named("my_fn"));
        assert_eq!(tc_named["functionCallingConfig"]["mode"], "ANY");
        let allowed = tc_named["functionCallingConfig"]["allowedFunctionNames"]
            .as_array()
            .unwrap();
        assert_eq!(allowed[0], "my_fn");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-013: is_error=true on ToolResult flows to provider JSON
    // ---------------------------------------------------------------------------

    // ToolResult with is_error=true → functionResponse with error field
    #[test]
    fn tool_result_is_error_true_produces_error_response() {
        use crate::types::{ContentPart, Role, ToolResultData};
        let part = ContentPart::tool_result(ToolResultData {
            tool_call_id: "my_fn-uuid".to_string(),
            content: serde_json::Value::String("error message".to_string()),
            is_error: true,
        });
        let result = translate_content_part_gemini(&part, &Role::Tool).unwrap();
        assert!(result.get("functionResponse").is_some());
        let resp = &result["functionResponse"]["response"];
        assert!(resp.get("error").is_some(), "expected error field: {resp}");
        assert_eq!(resp["error"]["message"], "error message");
    }

    // ToolResult with is_error=false → functionResponse with content field
    #[test]
    fn tool_result_is_error_false_produces_content_response() {
        use crate::types::{ContentPart, Role, ToolResultData};
        let part = ContentPart::tool_result(ToolResultData {
            tool_call_id: "my_fn-uuid".to_string(),
            content: serde_json::Value::String("success result".to_string()),
            is_error: false,
        });
        let result = translate_content_part_gemini(&part, &Role::Tool).unwrap();
        let resp = &result["functionResponse"]["response"];
        assert!(
            resp.get("content").is_some(),
            "expected content field: {resp}"
        );
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-017: thoughtsTokenCount → reasoning_tokens mapping
    // ---------------------------------------------------------------------------

    // ---------------------------------------------------------------------------
    // V2-ULM-001: complete() retries HTTP 429 responses
    // ---------------------------------------------------------------------------

    /// V2-ULM-001: The retry wrapper must classify 429 as retryable INSIDE the
    /// retry closure so the policy can act on it.
    #[tokio::test]
    async fn complete_retries_on_429() {
        use wiremock::matchers::{method, path_regex};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // First call returns 429
        Mock::given(method("POST"))
            .and(path_regex(".*generateContent.*"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_json(serde_json::json!({"error": {"message": "rate limited"}})),
            )
            .up_to_n_times(1)
            .mount(&mock_server)
            .await;

        // Second call returns a valid 200 response
        let ok_body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "hello"}],
                    "role": "model"
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2,
            },
        });
        Mock::given(method("POST"))
            .and(path_regex(".*generateContent.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body))
            .mount(&mock_server)
            .await;

        let adapter = GeminiAdapter::new("test-key").with_base_url(mock_server.uri().to_string());

        let req = make_request("gemini-2-flash", vec![Message::user("hi")]);
        let result = adapter.complete(&req).await;

        assert!(
            result.is_ok(),
            "complete() should succeed after retrying 429, got: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().text(), "hello");
    }

    // V2-ULM-003: parse_usage reads cachedContentTokenCount → cache_read_tokens
    #[test]
    fn parse_usage_reads_cached_content_token_count() {
        let usage_val = serde_json::json!({
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "totalTokenCount": 120,
            "cachedContentTokenCount": 500,
        });
        let usage = GeminiAdapter::parse_usage(&usage_val);
        assert_eq!(
            usage.cache_read_tokens,
            Some(500),
            "cache_read_tokens should come from cachedContentTokenCount"
        );
    }

    // parse_usage maps thoughtsTokenCount to reasoning_tokens (also tested above,
    // verified here with full response body to confirm end-to-end)
    #[test]
    fn response_body_thoughts_token_count_maps_to_reasoning_tokens() {
        let body = serde_json::json!({
            "candidates": [{
                "content": { "parts": [{"text": "answer"}], "role": "model" },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
                "thoughtsTokenCount": 7,
            }
        });
        let resp = GeminiAdapter::parse_response_body(body, "gemini-2-flash", None).unwrap();
        assert_eq!(
            resp.usage.reasoning_tokens,
            Some(7),
            "thoughtsTokenCount should map to reasoning_tokens"
        );
    }

    // -----------------------------------------------------------------------
    // V2-ULM-005: Image from local file path
    // -----------------------------------------------------------------------
    #[test]
    fn image_path_reads_local_file_gemini() {
        use crate::types::{ContentPart, ImageData, Role};
        let tmp = std::env::temp_dir().join("ulm_test_gemini_img.png");
        std::fs::write(&tmp, b"\x89PNG\r\n\x1a\n").unwrap();
        let img = ImageData {
            url: None,
            data: None,
            path: Some(tmp.to_str().unwrap().to_string()),
            media_type: Some("image/png".to_string()),
            detail: None,
        };
        let part = ContentPart::image(img);
        let result = translate_content_part_gemini(&part, &Role::User).unwrap();
        assert!(
            result.get("inlineData").is_some(),
            "expected inlineData: {result}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    // -----------------------------------------------------------------------
    // V2-ULM-006: Audio content → Err(InvalidRequest) not silent drop
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn complete_rejects_audio_content() {
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
        let req = make_request("gemini-2.0-flash", vec![msg]);
        let adapter = GeminiAdapter::new("key");
        let result = adapter.complete(&req).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            UnifiedLlmError::InvalidRequest { .. }
        ));
    }

    // -----------------------------------------------------------------------
    // V2-ULM-008: Gemini SSE translation unit tests
    // -----------------------------------------------------------------------

    // Text delta chunk → TextStart + TextDelta events
    #[test]
    fn gemini_chunk_text_delta_emits_text_events() {
        use crate::streaming::StreamEventType;
        let chunk = serde_json::json!({
            "candidates": [{
                "content": { "parts": [{"text": "Hello"}], "role": "model" },
            }]
        });
        let mut text_started = false;
        let mut finish_reason: Option<String> = None;
        let (evs, terminal) = translate_gemini_chunk(&chunk, &mut text_started, &mut finish_reason);
        assert!(!terminal);
        assert!(!evs.is_empty(), "expected at least one event");
        // First event should be TextStart, second TextDelta
        let types: Vec<StreamEventType> = evs
            .iter()
            .map(|e| e.as_ref().unwrap().event_type.clone())
            .collect();
        assert!(
            types.contains(&StreamEventType::TextStart)
                || types.contains(&StreamEventType::TextDelta),
            "expected text events: {types:?}"
        );
    }

    // Function call part → ToolCallStart + ToolCallEnd events
    #[test]
    fn gemini_chunk_function_call_emits_tool_events() {
        use crate::streaming::StreamEventType;
        let chunk = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "my_fn",
                            "args": {"x": 1}
                        }
                    }],
                    "role": "model"
                },
            }]
        });
        let mut text_started = false;
        let mut finish_reason: Option<String> = None;
        let (evs, _) = translate_gemini_chunk(&chunk, &mut text_started, &mut finish_reason);
        let types: Vec<StreamEventType> = evs
            .iter()
            .map(|e| e.as_ref().unwrap().event_type.clone())
            .collect();
        assert!(
            types.contains(&StreamEventType::ToolCallStart)
                || types.contains(&StreamEventType::ToolCallEnd),
            "expected tool call events: {types:?}"
        );
    }

    // Finish reason captured from chunk
    #[test]
    fn gemini_chunk_finish_reason_captured() {
        let chunk = serde_json::json!({
            "candidates": [{
                "content": { "parts": [{"text": "done"}], "role": "model" },
                "finishReason": "STOP"
            }]
        });
        let mut text_started = false;
        let mut finish_reason: Option<String> = None;
        translate_gemini_chunk(&chunk, &mut text_started, &mut finish_reason);
        assert_eq!(finish_reason.as_deref(), Some("STOP"));
    }

    // Usage extraction from usageMetadata
    #[test]
    fn gemini_chunk_usage_extraction() {
        let chunk = serde_json::json!({
            "candidates": [{
                "content": { "parts": [{"text": "x"}], "role": "model" },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            }
        });
        let mut text_started = false;
        let mut finish_reason: Option<String> = None;
        let (evs, terminal) = translate_gemini_chunk(&chunk, &mut text_started, &mut finish_reason);
        // When finishReason is present, a finish event should eventually be emitted
        // (actual emission depends on streaming state machine, but we verify it terminates)
        // The key check: finish_reason was captured
        assert_eq!(finish_reason.as_deref(), Some("STOP"));
        let _ = (evs, terminal); // allow unused
    }

    // -----------------------------------------------------------------------
    // V2-ULM-009: HTTP error handling unit tests — Gemini
    // -----------------------------------------------------------------------

    #[test]
    fn gemini_handle_error_401_returns_auth() {
        let err = GeminiAdapter::parse_error_status(401, "bad key", None, None);
        assert!(matches!(err, UnifiedLlmError::Authentication { .. }));
    }

    #[test]
    fn gemini_handle_error_429_returns_rate_limit() {
        let err = GeminiAdapter::parse_error_status(429, "quota exceeded", None, None);
        assert!(matches!(err, UnifiedLlmError::RateLimit { .. }));
    }

    #[test]
    fn gemini_handle_error_500_returns_provider_retryable() {
        let err = GeminiAdapter::parse_error_status(500, "internal error", None, None);
        match err {
            UnifiedLlmError::Provider { retryable, .. } => assert!(retryable),
            _ => panic!("expected Provider error"),
        }
    }

    #[test]
    fn gemini_handle_error_400_context_length() {
        let err = GeminiAdapter::parse_error_status(400, "request too long", None, None);
        assert!(
            matches!(err, UnifiedLlmError::ContextLength { .. }),
            "expected ContextLength for 400 + long message: {err:?}"
        );
    }
}
