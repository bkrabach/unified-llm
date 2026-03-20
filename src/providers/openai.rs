//! OpenAI provider adapter — Responses API (`POST /v1/responses`).
//!
//! This adapter implements [`ProviderAdapter`] for OpenAI's forward-looking
//! Responses API, which surfaces reasoning tokens, built-in tools, and a richer
//! output schema than Chat Completions.

use std::collections::HashMap;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures::StreamExt as _;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::{
    error::UnifiedLlmError,
    providers::ProviderAdapter,
    sse::process_sse_line,
    streaming::{EventStream, StreamEvent},
    types::{
        ContentKind, ContentPart, FinishReason, Message, RateLimitInfo, Request, Response, Role,
        ThinkingData, ToolCall, ToolCallData, Usage, Warning,
    },
};

// ---------------------------------------------------------------------------
// OpenAiAdapter
// ---------------------------------------------------------------------------

/// OpenAI provider adapter using the Responses API.
#[derive(Debug)]
pub struct OpenAiAdapter {
    api_key: String,
    base_url: String,
    org_id: Option<String>,
    project_id: Option<String>,
    http_client: reqwest::Client,
}

impl OpenAiAdapter {
    /// Construct from an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com".to_string(),
            org_id: None,
            project_id: None,
            http_client: reqwest::Client::new(),
        }
    }

    /// Override the base URL (useful for proxies or local mocks).
    ///
    /// Normalizes the URL by stripping any trailing `/v1` path segment (with
    /// or without a trailing slash) so that both of these common forms work
    /// identically:
    ///
    /// ```text
    /// OPENAI_BASE_URL=https://api.openai.com        # preferred
    /// OPENAI_BASE_URL=https://api.openai.com/v1     # also accepted
    /// ```
    ///
    /// The adapter always appends `/v1/<endpoint>` itself, so including it in
    /// the base URL would produce a doubled path (`/v1/v1/responses`).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        let mut s = url.into();
        // Strip trailing slash first, then strip a trailing "/v1" if present.
        s = s.trim_end_matches('/').to_string();
        if s.ends_with("/v1") {
            s.truncate(s.len() - 3);
        }
        self.base_url = s;
        self
    }

    /// Set the OpenAI organization ID.
    pub fn with_org_id(mut self, org_id: impl Into<String>) -> Self {
        self.org_id = Some(org_id.into());
        self
    }

    /// Set the OpenAI project ID.
    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Construct from environment variables.
    ///
    /// - `OPENAI_API_KEY` (required)
    /// - `OPENAI_BASE_URL` (optional, default `"https://api.openai.com"`)
    /// - `OPENAI_ORG_ID` (optional)
    /// - `OPENAI_PROJECT_ID` (optional)
    pub fn from_env() -> Result<Self, UnifiedLlmError> {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            return Err(UnifiedLlmError::Configuration {
                message: "OPENAI_API_KEY environment variable is not set or empty".to_string(),
            });
        }
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        let org_id = std::env::var("OPENAI_ORG_ID")
            .ok()
            .filter(|s| !s.is_empty());
        let project_id = std::env::var("OPENAI_PROJECT_ID")
            .ok()
            .filter(|s| !s.is_empty());

        let mut adapter = Self::new(api_key).with_base_url(base_url);
        if let Some(org) = org_id {
            adapter = adapter.with_org_id(org);
        }
        if let Some(proj) = project_id {
            adapter = adapter.with_project_id(proj);
        }
        Ok(adapter)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );
        if let Some(org) = &self.org_id {
            if let Ok(v) = HeaderValue::from_str(org) {
                headers.insert("OpenAI-Organization", v);
            }
        }
        if let Some(proj) = &self.project_id {
            if let Ok(v) = HeaderValue::from_str(proj) {
                headers.insert("OpenAI-Project", v);
            }
        }
        headers
    }

    fn build_request_body(&self, request: &Request, stream: bool) -> Value {
        // --- Extract system/developer messages → instructions ---
        let instructions: Option<String> = {
            let sys_texts: Vec<&str> = request
                .messages
                .iter()
                .filter(|m| matches!(m.role, Role::System | Role::Developer))
                .flat_map(|m| m.content.iter())
                .filter(|p| p.kind == ContentKind::Text)
                .filter_map(|p| p.text.as_deref())
                .collect();
            if sys_texts.is_empty() {
                None
            } else {
                Some(sys_texts.join("\n\n"))
            }
        };

        // --- Translate remaining messages → input array ---
        //
        // The Responses API is strict about which types can appear where:
        // - Text content   →  wrapped in { role, content: [{type: input_text|output_text, ...}] }
        // - Tool call      →  TOP-LEVEL item { type: function_call, call_id, name, arguments }
        // - Tool result    →  TOP-LEVEL item { type: function_call_output, call_id, output }
        //
        // Wrapping function_call or function_call_output inside a message's `content` array
        // causes a 400 "invalid value" error from the API.
        let mut input: Vec<Value> = Vec::new();
        for msg in request
            .messages
            .iter()
            .filter(|m| !matches!(m.role, Role::System | Role::Developer))
        {
            let has_tool_call = msg.content.iter().any(|p| p.kind == ContentKind::ToolCall);
            let has_tool_result = msg
                .content
                .iter()
                .any(|p| p.kind == ContentKind::ToolResult);

            if has_tool_call {
                // Emit each tool call as a top-level input item (no message wrapper).
                for part in &msg.content {
                    if part.kind == ContentKind::ToolCall {
                        if let Some(tc) = &part.tool_call {
                            let args_str = tc.raw_arguments.clone().unwrap_or_else(|| {
                                serde_json::to_string(&tc.arguments).unwrap_or_default()
                            });
                            input.push(json!({
                                "type": "function_call",
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": args_str,
                            }));
                        }
                    }
                }
            } else if has_tool_result {
                // Emit each tool result as a top-level input item (no message wrapper).
                for part in &msg.content {
                    if part.kind == ContentKind::ToolResult {
                        if let Some(tr) = &part.tool_result {
                            let call_id = msg.tool_call_id.as_deref().unwrap_or(&tr.tool_call_id);
                            let output_str = match &tr.content {
                                serde_json::Value::String(s) => s.clone(),
                                v => serde_json::to_string(v).unwrap_or_default(),
                            };
                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": output_str,
                            }));
                        }
                    }
                }
            } else {
                // Regular text (or mixed) message → translate normally.
                input.push(translate_message_openai(msg));
            }
        }

        let mut body = json!({
            "model": request.model,
            "input": input,
        });

        if let Some(instr) = instructions {
            body["instructions"] = json!(instr);
        }
        if stream {
            body["stream"] = json!(true);
        }

        // --- Tools ---
        // The Responses API uses a flat tool definition (name/description/parameters at
        // the top level alongside "type"), NOT the Chat Completions nested
        // "function": { "name": … } wrapper.
        if let Some(tools) = &request.tools {
            if !tools.is_empty() {
                let tools_json: Vec<Value> = tools
                    .iter()
                    .map(|t| {
                        json!({
                            "type": "function",
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        })
                    })
                    .collect();
                body["tools"] = json!(tools_json);
            }
        }

        // --- Tool choice ---
        if let Some(tc) = &request.tool_choice {
            body["tool_choice"] = translate_tool_choice_openai(tc);
        }

        // --- Sampling parameters ---
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(max_tokens) = request.max_tokens {
            body["max_output_tokens"] = json!(max_tokens);
        }
        if let Some(stops) = &request.stop_sequences {
            body["stop"] = json!(stops);
        }
        if let Some(effort) = &request.reasoning_effort {
            body["reasoning"] = json!({ "effort": effort });
        }

        // --- Response format ---
        if let Some(rf) = &request.response_format {
            use crate::types::ResponseFormatType;
            let fmt = match rf.format_type {
                ResponseFormatType::Json => json!({ "type": "json_object" }),
                ResponseFormatType::JsonSchema => json!({
                    "type": "json_schema",
                    "json_schema": rf.json_schema,
                    "strict": rf.strict,
                }),
                ResponseFormatType::Text => json!({ "type": "text" }),
            };
            body["text"] = json!({ "format": fmt });
        }

        // --- Metadata ---
        if let Some(meta) = &request.metadata {
            body["metadata"] = json!(meta);
        }

        // --- Provider options (escape hatch, applied last so caller wins) ---
        if let Some(opts) = &request.provider_options {
            if let Some(map) = opts.as_object() {
                for (k, v) in map {
                    body[k] = v.clone();
                }
            }
        }

        body
    }

    fn parse_usage(usage_val: &Value) -> Usage {
        let input_tokens = usage_val["input_tokens"].as_u64().unwrap_or(0) as u32;
        let output_tokens = usage_val["output_tokens"].as_u64().unwrap_or(0) as u32;
        let total_tokens = usage_val["total_tokens"]
            .as_u64()
            .unwrap_or((input_tokens as u64) + (output_tokens as u64))
            as u32;
        let reasoning_tokens = usage_val["output_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .map(|v| v as u32);
        let cache_read_tokens = usage_val["input_tokens_details"]["cached_tokens"]
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

    fn parse_rate_limit_headers(headers: &reqwest::header::HeaderMap) -> Option<RateLimitInfo> {
        let requests_limit = headers
            .get("x-ratelimit-limit-requests")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());
        let requests_remaining = headers
            .get("x-ratelimit-remaining-requests")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());
        let tokens_limit = headers
            .get("x-ratelimit-limit-tokens")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());
        let tokens_remaining = headers
            .get("x-ratelimit-remaining-tokens")
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
        let mut warnings: Vec<Warning> = Vec::new();
        let mut has_tool_calls = false;

        if let Some(output) = body["output"].as_array() {
            for item in output {
                match item["type"].as_str() {
                    Some("message") => {
                        if let Some(content_arr) = item["content"].as_array() {
                            for c in content_arr {
                                match c["type"].as_str() {
                                    Some("output_text") => {
                                        if let Some(text) = c["text"].as_str() {
                                            content.push(ContentPart::text(text));
                                        }
                                    }
                                    Some("refusal") => {
                                        let msg =
                                            c["refusal"].as_str().unwrap_or("refusal").to_string();
                                        warnings.push(Warning {
                                            message: msg,
                                            code: Some("refusal".to_string()),
                                        });
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    Some("function_call") => {
                        has_tool_calls = true;
                        let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                        let name = item["name"].as_str().unwrap_or("").to_string();
                        let args_str = item["arguments"].as_str().unwrap_or("{}");
                        let arguments: Value = serde_json::from_str(args_str)
                            .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                        content.push(ContentPart::tool_call(ToolCallData {
                            id: call_id,
                            name,
                            arguments,
                            raw_arguments: Some(args_str.to_string()),
                        }));
                    }
                    Some("reasoning") => {
                        let text = if let Some(summary) = item["summary"].as_array() {
                            summary
                                .iter()
                                .filter_map(|s| s["text"].as_str())
                                .collect::<Vec<_>>()
                                .join("")
                        } else {
                            item["text"].as_str().unwrap_or("").to_string()
                        };
                        if !text.is_empty() {
                            content.push(ContentPart::thinking(ThinkingData {
                                text,
                                signature: None,
                                redacted: false,
                            }));
                        }
                    }
                    _ => {}
                }
            }
        }

        let finish_reason = finish_reason_from_openai_status(&body, has_tool_calls);
        let usage = Self::parse_usage(&body["usage"]);

        let message = Message {
            role: Role::Assistant,
            content,
            name: None,
            tool_call_id: None,
        };

        Ok(Response {
            id,
            model,
            provider: "openai".to_string(),
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
        error_code: Option<&str>,
        retry_after: Option<f64>,
    ) -> UnifiedLlmError {
        match status {
            401 => UnifiedLlmError::Authentication {
                provider: "openai".to_string(),
                message: message.to_string(),
            },
            429 => UnifiedLlmError::RateLimit {
                provider: "openai".to_string(),
                message: message.to_string(),
                retry_after,
            },
            400 if error_code == Some("context_length_exceeded") => {
                UnifiedLlmError::ContextLength {
                    message: message.to_string(),
                }
            }
            400 => UnifiedLlmError::Provider {
                provider: "openai".to_string(),
                message: message.to_string(),
                status_code: Some(status),
                error_code: error_code.map(|s| s.to_string()),
                retryable: false,
                retry_after: None,
                raw: None,
            },
            500..=599 => UnifiedLlmError::Provider {
                provider: "openai".to_string(),
                message: message.to_string(),
                status_code: Some(status),
                error_code: error_code.map(|s| s.to_string()),
                retryable: true,
                retry_after: None,
                raw: None,
            },
            _ => UnifiedLlmError::Provider {
                provider: "openai".to_string(),
                message: message.to_string(),
                status_code: Some(status),
                error_code: error_code.map(|s| s.to_string()),
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
        let error_code = body_json["error"]["code"].as_str().map(|s| s.to_string());

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

fn translate_tool_choice_openai(tc: &crate::types::ToolChoice) -> Value {
    match tc.mode.as_str() {
        "auto" => json!("auto"),
        "none" => json!("none"),
        "required" => json!("required"),
        "named" => {
            let name = tc.tool_name.as_deref().unwrap_or("");
            json!({ "type": "function", "name": name })
        }
        other => json!(other),
    }
}

fn translate_message_openai(msg: &Message) -> Value {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        Role::System | Role::Developer => "user", // should not appear here
    };

    let content: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|part| translate_content_part_openai(part, &msg.role, &msg.tool_call_id))
        .collect();

    let mut obj = json!({ "role": role, "content": content });
    if msg.role == Role::Tool {
        if let Some(id) = &msg.tool_call_id {
            obj["tool_call_id"] = json!(id);
        }
    }
    obj
}

fn translate_content_part_openai(
    part: &ContentPart,
    role: &Role,
    tool_call_id: &Option<String>,
) -> Option<Value> {
    match &part.kind {
        ContentKind::Text => {
            // The Responses API distinguishes between input and output text content types.
            // User-authored content uses "input_text"; model-generated content uses "output_text".
            let text_type = match role {
                Role::Assistant => "output_text",
                _ => "input_text",
            };
            Some(json!({ "type": text_type, "text": part.text.as_deref().unwrap_or("") }))
        }
        ContentKind::Image => {
            let img = part.image.as_ref()?;
            let url = if let Some(u) = &img.url {
                u.clone()
            } else {
                // V2-ULM-005: resolve data from existing bytes or file path.
                let resolved: Option<Vec<u8>> = img
                    .data
                    .clone()
                    .or_else(|| img.path.as_ref().and_then(|p| std::fs::read(p).ok()));
                if let Some(data) = resolved {
                    let b64 = STANDARD.encode(&data);
                    let media_type = img.media_type.as_deref().unwrap_or("image/jpeg");
                    format!("data:{media_type};base64,{b64}")
                } else {
                    return None;
                }
            };
            let detail = img.detail.as_deref().unwrap_or("auto");
            Some(json!({ "type": "image_url", "image_url": { "url": url, "detail": detail } }))
        }
        ContentKind::ToolCall => {
            let tc = part.tool_call.as_ref()?;
            let args_str = tc
                .raw_arguments
                .clone()
                .unwrap_or_else(|| serde_json::to_string(&tc.arguments).unwrap_or_default());
            Some(json!({
                "type": "function_call",
                "call_id": tc.id,
                "name": tc.name,
                "arguments": args_str,
            }))
        }
        ContentKind::ToolResult => {
            // For Role::Tool messages only
            if !matches!(role, Role::Tool) {
                return None;
            }
            let tr = part.tool_result.as_ref()?;
            let call_id = tool_call_id.as_deref().unwrap_or(&tr.tool_call_id);
            let output_str = match &tr.content {
                Value::String(s) => s.clone(),
                v => serde_json::to_string(v).unwrap_or_default(),
            };
            Some(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_str,
            }))
        }
        ContentKind::Thinking | ContentKind::RedactedThinking => None,
        ContentKind::Audio | ContentKind::Document => None,
    }
}

fn finish_reason_from_openai_status(body: &Value, has_tool_calls: bool) -> FinishReason {
    match body["status"].as_str() {
        Some("completed") => {
            if has_tool_calls {
                FinishReason {
                    reason: "tool_calls".to_string(),
                    raw: Some("completed".to_string()),
                }
            } else {
                FinishReason {
                    reason: "stop".to_string(),
                    raw: Some("completed".to_string()),
                }
            }
        }
        Some("incomplete") => {
            let detail = body["incomplete_details"]["reason"].as_str().unwrap_or("");
            match detail {
                "max_output_tokens" => FinishReason {
                    reason: "length".to_string(),
                    raw: Some("incomplete".to_string()),
                },
                "content_filter" => FinishReason {
                    reason: "content_filter".to_string(),
                    raw: Some("incomplete".to_string()),
                },
                _ => FinishReason {
                    reason: "other".to_string(),
                    raw: Some("incomplete".to_string()),
                },
            }
        }
        Some(other) => FinishReason {
            reason: "other".to_string(),
            raw: Some(other.to_string()),
        },
        None => FinishReason::stop(),
    }
}

// ---------------------------------------------------------------------------
// SSE event translation for streaming (F-010)
// ---------------------------------------------------------------------------

/// Translate a single OpenAI Responses API SSE event into zero or more
/// unified [`StreamEvent`]s.
///
/// `text_started` tracks whether a `TextStart` was already emitted for the
/// current text block. `tool_names` tracks `call_id → name` so that name is
/// only available in the first delta but `ToolCallStart` can always carry it.
fn translate_openai_sse_event(
    event_type: &str,
    data: &str,
    text_started: &mut bool,
    tool_names: &mut HashMap<String, String>,
) -> Vec<Result<StreamEvent, UnifiedLlmError>> {
    let mut out = Vec::new();

    let payload: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            out.push(Err(UnifiedLlmError::Stream {
                message: format!("SSE JSON parse error on '{event_type}': {e}"),
            }));
            return out;
        }
    };

    match event_type {
        "response.created" => {
            out.push(Ok(StreamEvent::stream_start()));
        }
        "response.in_progress" => {
            // Intentionally ignored per spec
        }
        "response.output_text.delta" => {
            let delta = payload["delta"].as_str().unwrap_or("");
            let item_id = payload["item_id"].as_str().unwrap_or("");
            if !*text_started {
                *text_started = true;
                out.push(Ok(StreamEvent::text_start()));
            }
            out.push(Ok(StreamEvent::text_delta_with_id(delta, item_id)));
        }
        "response.output_text.done" => {
            *text_started = false;
            out.push(Ok(StreamEvent::text_end()));
        }
        "response.function_call_arguments.delta" => {
            let call_id = payload["call_id"].as_str().unwrap_or("").to_string();
            let delta = payload["delta"].as_str().unwrap_or("");
            // Name is only in first delta
            if let Some(name) = payload["name"].as_str() {
                if !tool_names.contains_key(&call_id) {
                    tool_names.insert(call_id.clone(), name.to_string());
                    out.push(Ok(StreamEvent::tool_call_start(&call_id, name)));
                }
            } else if !tool_names.contains_key(&call_id) {
                // First delta without name — still start with empty name
                tool_names.insert(call_id.clone(), String::new());
                out.push(Ok(StreamEvent::tool_call_start(&call_id, "")));
            }
            out.push(Ok(StreamEvent::tool_call_delta(&call_id, delta)));
        }
        "response.function_call_arguments.done" => {
            let call_id = payload["call_id"].as_str().unwrap_or("").to_string();
            let name = payload["name"]
                .as_str()
                .map(|s| s.to_string())
                .or_else(|| tool_names.get(&call_id).cloned())
                .unwrap_or_default();
            let args_str = payload["arguments"].as_str().unwrap_or("{}");
            let arguments: Value = serde_json::from_str(args_str)
                .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
            out.push(Ok(StreamEvent::tool_call_end(ToolCall {
                id: call_id,
                name,
                arguments,
                raw_arguments: Some(args_str.to_string()),
            })));
        }
        "response.output_item.done" => {
            // Handled by text.done / arguments.done; skip here
        }
        "response.completed" => {
            let response_obj = &payload["response"];
            let has_tool_calls = response_obj["output"]
                .as_array()
                .map(|arr| arr.iter().any(|i| i["type"] == "function_call"))
                .unwrap_or(false);
            let finish_reason = finish_reason_from_openai_status(response_obj, has_tool_calls);
            let usage = OpenAiAdapter::parse_usage(&response_obj["usage"]);
            out.push(Ok(StreamEvent::finish(finish_reason, usage)));
        }
        "response.incomplete" => {
            let finish_reason = FinishReason {
                reason: "length".to_string(),
                raw: Some("incomplete".to_string()),
            };
            let usage = OpenAiAdapter::parse_usage(&payload["response"]["usage"]);
            out.push(Ok(StreamEvent::finish(finish_reason, usage)));
        }
        "response.failed" => {
            let msg = payload["response"]["error"]["message"]
                .as_str()
                .unwrap_or("response failed")
                .to_string();
            out.push(Ok(StreamEvent::error(msg)));
        }
        _ => {
            // Unknown event — emit as ProviderEvent
            use crate::streaming::StreamEventType;
            let mut ev = StreamEvent::stream_start(); // reuse blank helper
            ev.event_type = StreamEventType::ProviderEvent;
            ev.raw = serde_json::from_str(data).ok();
            out.push(Ok(ev));
        }
    }

    out
}

// ---------------------------------------------------------------------------
// ProviderAdapter implementation
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ProviderAdapter for OpenAiAdapter {
    fn name(&self) -> &str {
        "openai"
    }

    fn supports_tool_choice(&self, mode: &str) -> bool {
        matches!(mode, "auto" | "none" | "required" | "named")
    }

    async fn complete(&self, request: &Request) -> Result<Response, UnifiedLlmError> {
        // V2-ULM-006: reject audio content rather than silently dropping it.
        super::reject_audio_content(request, "openai")?;
        let body = self.build_request_body(request, false);
        let url = format!("{}/v1/responses", self.base_url);
        let headers = self.build_headers();
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
                            provider: "openai".to_string(),
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
        super::reject_audio_content(request, "openai")?;
        let body = self.build_request_body(request, true);
        let url = format!("{}/v1/responses", self.base_url);
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
            let mut tool_names: HashMap<String, String> = HashMap::new();

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
                            let is_terminal = matches!(
                                sse_ev.event_type.as_str(),
                                "response.completed" | "response.failed" | "response.incomplete"
                            );
                            let events = translate_openai_sse_event(
                                &sse_ev.event_type,
                                &sse_ev.data,
                                &mut text_started,
                                &mut tool_names,
                            );
                            let mut had_error = false;
                            for ev in events {
                                let stop = ev.is_err();
                                if tx.try_send(ev).is_err() {
                                    return;
                                }
                                if stop {
                                    had_error = true;
                                    break;
                                }
                            }
                            if is_terminal || had_error {
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

            // Stream closed without completion event
            let _ = tx
                .try_send(Ok(StreamEvent::error(
                    "stream closed without response.completed event",
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

    // AC-1: from_env() returns Err(Configuration) when OPENAI_API_KEY is unset
    #[test]
    fn from_env_no_key_returns_config_error() {
        // SAFETY: single-threaded test context
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        let err = OpenAiAdapter::from_env().unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Configuration { .. }));
    }

    // AC-3: System message goes to instructions, not input
    #[test]
    fn system_message_becomes_instructions() {
        let adapter = OpenAiAdapter::new("key");
        let req = make_request(
            "gpt-4o",
            vec![Message::system("Be helpful."), Message::user("Hello")],
        );
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["instructions"], "Be helpful.");
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
    }

    // No system message → instructions omitted
    #[test]
    fn no_system_message_omits_instructions() {
        let adapter = OpenAiAdapter::new("key");
        let req = make_request("gpt-4o", vec![Message::user("Hello")]);
        let body = adapter.build_request_body(&req, false);
        assert!(body.get("instructions").is_none());
    }

    // AC-4: Tool definitions use flat Responses API format {type, name, description, parameters}
    // (NOT the Chat Completions nested {"type":"function","function":{...}} format)
    #[test]
    fn tool_wrapped_in_function_type() {
        let adapter = OpenAiAdapter::new("key");
        let req = make_request("gpt-4o", vec![Message::user("hi")]).with_tools(vec![Tool {
            name: "my_fn".to_string(),
            description: "does stuff".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }]);
        let body = adapter.build_request_body(&req, false);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        // Responses API: name is at the top level, NOT nested under "function"
        assert_eq!(tools[0]["name"], "my_fn");
        assert_eq!(tools[0]["description"], "does stuff");
        assert!(
            tools[0].get("function").is_none(),
            "Responses API does not use nested 'function' key"
        );
    }

    // AC-11: ToolChoice::none() serializes as "none"
    #[test]
    fn tool_choice_none_serializes_as_none() {
        assert_eq!(
            translate_tool_choice_openai(&ToolChoice::none()),
            json!("none")
        );
    }

    // AC-12: ToolChoice::named("my_fn") serializes correctly
    #[test]
    fn tool_choice_named_serializes_correctly() {
        let tc = translate_tool_choice_openai(&ToolChoice::named("my_fn"));
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["name"], "my_fn");
    }

    // AC-13: max_tokens maps to max_output_tokens
    #[test]
    fn max_tokens_maps_to_max_output_tokens() {
        let adapter = OpenAiAdapter::new("key");
        let req = make_request("gpt-4o", vec![Message::user("hi")]).with_max_tokens(500);
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["max_output_tokens"], 500);
        assert!(body.get("max_tokens").is_none());
    }

    // stream: body contains "stream": true
    #[test]
    fn stream_body_has_stream_true() {
        let adapter = OpenAiAdapter::new("key");
        let req = make_request("gpt-4o", vec![Message::user("hi")]);
        let body = adapter.build_request_body(&req, true);
        assert_eq!(body["stream"], true);
    }

    // provider_options merges into body
    #[test]
    fn provider_options_merged_last() {
        let adapter = OpenAiAdapter::new("key");
        let mut req = make_request("gpt-4o", vec![Message::user("hi")]).with_max_tokens(100);
        req.provider_options = Some(serde_json::json!({ "max_output_tokens": 9999 }));
        let body = adapter.build_request_body(&req, false);
        // provider_options wins
        assert_eq!(body["max_output_tokens"], 9999);
    }

    // Response body parse: usage.reasoning_tokens (AC-5)
    #[test]
    fn parse_usage_reasoning_tokens() {
        let usage_val = serde_json::json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "output_tokens_details": { "reasoning_tokens": 3 },
            "input_tokens_details": { "cached_tokens": 2 },
        });
        let usage = OpenAiAdapter::parse_usage(&usage_val);
        assert_eq!(usage.reasoning_tokens, Some(3));
        assert_eq!(usage.cache_read_tokens, Some(2));
        assert_eq!(usage.cache_write_tokens, None);
    }

    // Response parsing: function_call → ToolCall ContentPart (AC-10)
    #[test]
    fn response_function_call_becomes_tool_call_content() {
        let body = serde_json::json!({
            "id": "resp_1",
            "model": "gpt-4o",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "my_fn",
                "arguments": "{\"x\":1}",
            }],
            "usage": { "input_tokens": 5, "output_tokens": 2, "total_tokens": 7 },
        });
        let resp = OpenAiAdapter::parse_response_body(body, None).unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "my_fn");
        assert_eq!(calls[0].arguments["x"], 1);
        assert!(resp.finish_reason.is_tool_calls());
    }

    // Response parsing: text output
    #[test]
    fn response_text_output_parsed() {
        let body = serde_json::json!({
            "id": "resp_2",
            "model": "gpt-4o",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello!"}]
            }],
            "usage": { "input_tokens": 5, "output_tokens": 2, "total_tokens": 7 },
        });
        let resp = OpenAiAdapter::parse_response_body(body, None).unwrap();
        assert_eq!(resp.text(), "Hello!");
        assert!(resp.finish_reason.is_stop());
    }

    // Error translation: 401 → Authentication
    #[test]
    fn finish_reason_incomplete_length() {
        let body = serde_json::json!({
            "status": "incomplete",
            "incomplete_details": { "reason": "max_output_tokens" }
        });
        let fr = finish_reason_from_openai_status(&body, false);
        assert_eq!(fr.reason, "length");
    }

    // SSE event translation tests
    #[test]
    fn sse_response_created_emits_stream_start() {
        let mut ts = false;
        let mut tn: HashMap<String, String> = HashMap::new();
        let events = translate_openai_sse_event("response.created", "{}", &mut ts, &mut tn);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            Ok(e) if e.event_type == crate::streaming::StreamEventType::StreamStart
        ));
    }

    #[test]
    fn sse_text_delta_first_emits_text_start_then_delta() {
        let mut ts = false;
        let mut tn: HashMap<String, String> = HashMap::new();
        let events = translate_openai_sse_event(
            "response.output_text.delta",
            r#"{"delta":"Hello","item_id":"item_1"}"#,
            &mut ts,
            &mut tn,
        );
        assert_eq!(events.len(), 2);
        use crate::streaming::StreamEventType;
        assert!(matches!(&events[0], Ok(e) if e.event_type == StreamEventType::TextStart));
        assert!(matches!(&events[1], Ok(e) if e.event_type == StreamEventType::TextDelta));
        assert!(ts); // text_started should be true now
    }

    #[test]
    fn sse_text_delta_subsequent_no_text_start() {
        let mut ts = true; // already started
        let mut tn: HashMap<String, String> = HashMap::new();
        let events = translate_openai_sse_event(
            "response.output_text.delta",
            r#"{"delta":"World","item_id":"item_1"}"#,
            &mut ts,
            &mut tn,
        );
        assert_eq!(events.len(), 1);
        use crate::streaming::StreamEventType;
        assert!(matches!(&events[0], Ok(e) if e.event_type == StreamEventType::TextDelta));
    }

    #[test]
    fn sse_function_call_delta_first_emits_start_and_delta() {
        let mut ts = false;
        let mut tn: HashMap<String, String> = HashMap::new();
        let events = translate_openai_sse_event(
            "response.function_call_arguments.delta",
            r#"{"call_id":"call_1","name":"my_fn","delta":"{\"x\":"}"#,
            &mut ts,
            &mut tn,
        );
        use crate::streaming::StreamEventType;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], Ok(e) if e.event_type == StreamEventType::ToolCallStart));
        assert!(matches!(&events[1], Ok(e) if e.event_type == StreamEventType::ToolCallDelta));
    }

    #[test]
    fn sse_completed_emits_finish_with_usage() {
        let mut ts = false;
        let mut tn: HashMap<String, String> = HashMap::new();
        let data = r#"{
            "response": {
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "output_tokens_details": {"reasoning_tokens": 2}
                }
            }
        }"#;
        let events = translate_openai_sse_event("response.completed", data, &mut ts, &mut tn);
        assert_eq!(events.len(), 1);
        use crate::streaming::StreamEventType;
        let ev = events[0].as_ref().unwrap();
        assert_eq!(ev.event_type, StreamEventType::Finish);
        assert_eq!(ev.usage.as_ref().unwrap().reasoning_tokens, Some(2));
    }

    #[test]
    fn supports_all_tool_choice_modes() {
        let adapter = OpenAiAdapter::new("key");
        assert!(adapter.supports_tool_choice("auto"));
        assert!(adapter.supports_tool_choice("none"));
        assert!(adapter.supports_tool_choice("required"));
        assert!(adapter.supports_tool_choice("named"));
        assert!(!adapter.supports_tool_choice("unknown"));
    }

    #[test]
    fn multiple_system_messages_concatenated() {
        let adapter = OpenAiAdapter::new("key");
        let req = make_request(
            "gpt-4o",
            vec![
                Message::system("First."),
                Message::system("Second."),
                Message::user("Hi"),
            ],
        );
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["instructions"], "First.\n\nSecond.");
    }

    #[test]
    fn reasoning_effort_translates_correctly() {
        let adapter = OpenAiAdapter::new("key");
        let mut req = make_request("gpt-4o", vec![Message::user("hi")]);
        req.reasoning_effort = Some("high".to_string());
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-001: URL construction
    // ---------------------------------------------------------------------------

    // Request URL contains /v1/responses path
    #[test]
    fn complete_url_format_uses_v1_responses() {
        let adapter = OpenAiAdapter::new("key").with_base_url("http://localhost:8080");
        let url = format!("{}/v1/responses", adapter.base_url);
        assert!(
            url.contains("/v1/responses"),
            "URL should contain /v1/responses: {url}"
        );
        assert!(
            url.starts_with("http://localhost:8080"),
            "URL should use base_url: {url}"
        );
    }

    // with_base_url strips trailing /v1 to prevent doubling
    #[test]
    fn with_base_url_strips_v1_suffix() {
        let adapter = OpenAiAdapter::new("key").with_base_url("https://api.openai.com/v1");
        assert_eq!(adapter.base_url, "https://api.openai.com");
        let url = format!("{}/v1/responses", adapter.base_url);
        assert_eq!(url, "https://api.openai.com/v1/responses");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-003: All 5 roles wire format
    // ---------------------------------------------------------------------------

    // Developer role is treated like System (goes to instructions, not input)
    #[test]
    fn developer_role_goes_to_instructions() {
        use crate::types::{ContentPart, Role};
        let adapter = OpenAiAdapter::new("key");
        let dev_msg = Message {
            role: Role::Developer,
            content: vec![ContentPart::text("Dev instructions.")],
            name: None,
            tool_call_id: None,
        };
        let req = make_request("gpt-4o", vec![dev_msg, Message::user("hello")]);
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["instructions"], "Dev instructions.");
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
    }

    // User role → "user" with "input_text" type in content
    #[test]
    fn user_role_wire_format() {
        let adapter = OpenAiAdapter::new("key");
        let req = make_request("gpt-4o", vec![Message::user("hello")]);
        let body = adapter.build_request_body(&req, false);
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hello");
    }

    // Assistant role → "assistant" with "output_text" type in content
    #[test]
    fn assistant_role_wire_format() {
        let adapter = OpenAiAdapter::new("key");
        let req = make_request(
            "gpt-4o",
            vec![Message::user("hi"), Message::assistant("response")],
        );
        let body = adapter.build_request_body(&req, false);
        let input = body["input"].as_array().unwrap();
        let asst = input.iter().find(|m| m["role"] == "assistant").unwrap();
        assert_eq!(asst["content"][0]["type"], "output_text");
        assert_eq!(asst["content"][0]["text"], "response");
    }

    // Tool role → "tool" with "function_call_output" content type
    // Responses API: tool results are TOP-LEVEL function_call_output items, NOT
    // wrapped in a { role: "tool", content: [...] } message object.
    #[test]
    fn tool_role_wire_format() {
        let adapter = OpenAiAdapter::new("key");
        let tool_msg = Message::tool_result("call-1", "result_val", false);
        let req = make_request(
            "gpt-4o",
            vec![Message::user("hi"), Message::assistant("ok"), tool_msg],
        );
        let body = adapter.build_request_body(&req, false);
        let input = body["input"].as_array().unwrap();
        // Tool result should appear as a top-level function_call_output item
        // (no "role" wrapper — the Responses API expects it at the top level).
        let tool_result_item = input
            .iter()
            .find(|m| m["type"] == "function_call_output")
            .expect("should have a function_call_output item in input");
        assert_eq!(tool_result_item["call_id"], "call-1");
        assert_eq!(tool_result_item["output"], "result_val");
        // Confirm no "role":"tool" wrapper exists
        assert!(
            input
                .iter()
                .all(|m| m.get("role").map(|r| r != "tool").unwrap_or(true)),
            "Responses API must not have a 'role:tool' message wrapper"
        );
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-005: Image content serialization
    // ---------------------------------------------------------------------------

    // Image URL content → image_url type with url field
    #[test]
    fn image_url_translates_to_image_url_type() {
        use crate::types::{ContentPart, ImageData};
        let part = ContentPart::image(ImageData {
            url: Some("https://example.com/img.png".to_string()),
            data: None,
            media_type: None,
            detail: None,
            path: None,
        });
        let result =
            translate_content_part_openai(&part, &crate::types::Role::User, &None).unwrap();
        assert_eq!(result["type"], "image_url");
        assert_eq!(result["image_url"]["url"], "https://example.com/img.png");
        assert_eq!(result["image_url"]["detail"], "auto");
    }

    // Image base64 content → data URI in image_url
    #[test]
    fn image_base64_translates_to_data_uri() {
        use crate::types::{ContentPart, ImageData};
        let part = ContentPart::image(ImageData {
            url: None,
            data: Some(vec![0u8, 1u8, 2u8]),
            media_type: Some("image/png".to_string()),
            detail: None,
            path: None,
        });
        let result =
            translate_content_part_openai(&part, &crate::types::Role::User, &None).unwrap();
        assert_eq!(result["type"], "image_url");
        let url = result["image_url"]["url"].as_str().unwrap();
        assert!(
            url.starts_with("data:image/png;base64,"),
            "expected data URI: {url}"
        );
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-006: Tool call round-trip
    // ---------------------------------------------------------------------------

    // ToolCall content part → function_call JSON
    #[test]
    fn tool_call_content_part_wire_format() {
        use crate::types::{ContentPart, Role, ToolCallData};
        let part = ContentPart::tool_call(ToolCallData {
            id: "call-42".to_string(),
            name: "my_tool".to_string(),
            arguments: serde_json::json!({"x": 1}),
            raw_arguments: Some("{\"x\":1}".to_string()),
        });
        let result = translate_content_part_openai(&part, &Role::Assistant, &None).unwrap();
        assert_eq!(result["type"], "function_call");
        assert_eq!(result["call_id"], "call-42");
        assert_eq!(result["name"], "my_tool");
        assert_eq!(result["arguments"], "{\"x\":1}");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-012: All ToolChoice modes serialized correctly
    // ---------------------------------------------------------------------------

    // ToolChoice::auto() → "auto"
    #[test]
    fn tool_choice_auto_serializes_as_string_auto() {
        assert_eq!(
            translate_tool_choice_openai(&ToolChoice::auto()),
            json!("auto")
        );
    }

    // ToolChoice::required() → "required"
    #[test]
    fn tool_choice_required_serializes_as_string_required() {
        assert_eq!(
            translate_tool_choice_openai(&ToolChoice::required()),
            json!("required")
        );
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-013: is_error ToolResult flows correctly
    // ---------------------------------------------------------------------------

    // OpenAI Responses API: tool result with is_error=true still produces
    // function_call_output (OpenAI doesn't have native is_error field, so the
    // error content is passed as the output string).
    #[test]
    fn tool_result_with_is_error_produces_function_call_output() {
        use crate::types::{ContentPart, Role, ToolResultData};
        let part = ContentPart::tool_result(ToolResultData {
            tool_call_id: "call-err".to_string(),
            content: serde_json::Value::String("something went wrong".to_string()),
            is_error: true,
        });
        let result =
            translate_content_part_openai(&part, &Role::Tool, &Some("call-err".to_string()))
                .unwrap();
        assert_eq!(result["type"], "function_call_output");
        assert_eq!(result["call_id"], "call-err");
        assert_eq!(result["output"], "something went wrong");
    }

    // ---------------------------------------------------------------------------
    // GAP-ULM-018: cache_read_tokens from input_tokens_details.cached_tokens
    // ---------------------------------------------------------------------------

    // OpenAI Responses API cache_read_tokens from input_tokens_details.cached_tokens
    #[test]
    fn cache_read_tokens_from_input_tokens_details() {
        let usage_val = serde_json::json!({
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": { "cached_tokens": 80 },
        });
        let usage = OpenAiAdapter::parse_usage(&usage_val);
        assert_eq!(usage.cache_read_tokens, Some(80));
        assert_eq!(usage.input_tokens, 100);
    }

    // ---------------------------------------------------------------------------
    // V2-ULM-001: complete() retries HTTP 429 responses
    // ---------------------------------------------------------------------------

    /// V2-ULM-001: The retry wrapper must see 429s as retryable errors.
    ///
    /// Serves one HTTP 429 then one HTTP 200. `complete()` must succeed,
    /// which proves the retry loop is wrapping the status-check (not just .send()).
    #[tokio::test]
    async fn complete_retries_on_429() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // First call returns 429
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_json(serde_json::json!({"error": {"message": "rate limited"}})),
            )
            .up_to_n_times(1)
            .mount(&mock_server)
            .await;

        // Second call returns a valid 200 response
        let ok_body = serde_json::json!({
            "id": "resp_1",
            "model": "gpt-4o",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}]
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body))
            .mount(&mock_server)
            .await;

        let adapter = OpenAiAdapter::new("test-key").with_base_url(mock_server.uri().to_string());

        let req = make_request("gpt-4o", vec![Message::user("hi")]);
        let result = adapter.complete(&req).await;

        assert!(
            result.is_ok(),
            "complete() should succeed after retrying 429, got: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().text(), "hello");
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
        let req = make_request("gpt-4o", vec![msg]);
        let adapter = OpenAiAdapter::new("key");
        let result = adapter.complete(&req).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            UnifiedLlmError::InvalidRequest { .. }
        ));
    }

    // -----------------------------------------------------------------------
    // V2-ULM-009: HTTP error handling unit tests — OpenAI
    // -----------------------------------------------------------------------
    #[test]
    fn openai_handle_error_401_returns_auth() {
        let err = OpenAiAdapter::parse_error_status(401, "invalid key", None, None);
        assert!(matches!(err, UnifiedLlmError::Authentication { .. }));
    }

    #[test]
    fn openai_handle_error_429_returns_rate_limit() {
        let err = OpenAiAdapter::parse_error_status(429, "rate limit exceeded", None, None);
        assert!(matches!(err, UnifiedLlmError::RateLimit { .. }));
    }

    #[test]
    fn openai_handle_error_500_returns_provider_retryable() {
        let err = OpenAiAdapter::parse_error_status(500, "internal server error", None, None);
        match err {
            UnifiedLlmError::Provider { retryable, .. } => assert!(retryable),
            _ => panic!("expected Provider error"),
        }
    }

    #[test]
    fn openai_handle_error_400_context_length_returns_context_length() {
        let err = OpenAiAdapter::parse_error_status(
            400,
            "context_length_exceeded",
            Some("context_length_exceeded"),
            None,
        );
        assert!(
            matches!(err, UnifiedLlmError::ContextLength { .. }),
            "expected ContextLength for 400 + context_length_exceeded code: {err:?}"
        );
    }
}
