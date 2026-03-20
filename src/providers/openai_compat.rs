//! OpenAI-compatible Chat Completions adapter.
//!
//! Targets `/v1/chat/completions` — the Chat Completions API used by vLLM,
//! Ollama, Together AI, Groq, and other third-party endpoints.
//!
//! Unlike [`super::openai::OpenAiAdapter`] (which uses the Responses API),
//! this adapter uses the older Chat Completions format: no reasoning tokens,
//! no built-in tools, no server-side state.

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
        ContentKind, ContentPart, FinishReason, Message, Request, Response, Role, ToolCallData,
        Usage,
    },
};

// ---------------------------------------------------------------------------
// OpenAiCompatAdapter
// ---------------------------------------------------------------------------

/// Adapter for OpenAI-compatible Chat Completions endpoints.
///
/// Targets `/v1/chat/completions`. Does NOT use the Responses API.
/// Use `OpenAiAdapter` for the real OpenAI service (Responses API).
#[derive(Debug)]
pub struct OpenAiCompatAdapter {
    api_key: Option<String>,
    base_url: String,
    provider_name: String,
    http_client: reqwest::Client,
}

impl OpenAiCompatAdapter {
    /// Construct with a base URL (no API key).
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            api_key: None,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            provider_name: "openai-compat".to_string(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Set the API key for Bearer authentication.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Override the provider name (used in error messages and `Response.provider`).
    pub fn with_provider_name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = name.into();
        self
    }

    /// Convenience: construct for Ollama (no auth, default port 11434).
    pub fn ollama() -> Self {
        Self::new("http://localhost:11434").with_provider_name("ollama")
    }

    /// Convenience: construct for Groq.
    pub fn groq(api_key: impl Into<String>) -> Self {
        Self::new("https://api.groq.com/openai")
            .with_api_key(api_key)
            .with_provider_name("groq")
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn endpoint_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(key) = &self.api_key {
            if let Ok(v) = HeaderValue::from_str(&format!("Bearer {key}")) {
                headers.insert(AUTHORIZATION, v);
            }
        }
        headers
    }

    fn build_request_body(&self, request: &Request, stream: bool) -> Value {
        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(translate_message_compat)
            .collect();

        let mut body = json!({
            "model": request.model,
            "messages": messages,
        });

        if stream {
            body["stream"] = json!(true);
        }

        // --- Tools ---
        if let Some(tools) = &request.tools {
            if !tools.is_empty() {
                let tools_json: Vec<Value> = tools
                    .iter()
                    .map(|t| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters,
                            }
                        })
                    })
                    .collect();
                body["tools"] = json!(tools_json);
            }
        }

        // --- Tool choice ---
        if let Some(tc) = &request.tool_choice {
            body["tool_choice"] = translate_tool_choice_compat(tc);
        }

        // --- Sampling parameters ---
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(max_tokens) = request.max_tokens {
            // Chat Completions uses max_tokens (not max_output_tokens)
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(stops) = &request.stop_sequences {
            body["stop"] = json!(stops);
        }
        // reasoning_effort is silently ignored — not supported by Chat Completions

        // --- Response format ---
        if let Some(rf) = &request.response_format {
            use crate::types::ResponseFormatType;
            let fmt = match rf.format_type {
                ResponseFormatType::Json => json!({ "type": "json_object" }),
                ResponseFormatType::JsonSchema => json!({
                    "type": "json_schema",
                    "json_schema": rf.json_schema,
                }),
                ResponseFormatType::Text => json!({ "type": "text" }),
            };
            body["response_format"] = fmt;
        }

        // --- Provider options (escape hatch) ---
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
        let input_tokens = usage_val["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let output_tokens = usage_val["completion_tokens"].as_u64().unwrap_or(0) as u32;
        let total_tokens = usage_val["total_tokens"]
            .as_u64()
            .unwrap_or((input_tokens as u64) + (output_tokens as u64))
            as u32;

        Usage {
            input_tokens,
            output_tokens,
            total_tokens,
            reasoning_tokens: None,
            cache_read_tokens: None,
            cache_write_tokens: None,
            raw: Some(usage_val.clone()),
        }
    }

    fn parse_response_body(body: Value, provider_name: &str) -> Result<Response, UnifiedLlmError> {
        let id = body["id"].as_str().unwrap_or("unknown").to_string();
        let model = body["model"].as_str().unwrap_or("").to_string();

        let choice = body["choices"]
            .as_array()
            .and_then(|arr| arr.first())
            .cloned()
            .unwrap_or(json!({}));

        let msg = &choice["message"];
        let mut content: Vec<ContentPart> = Vec::new();

        // Text content
        if let Some(text) = msg["content"].as_str() {
            if !text.is_empty() {
                content.push(ContentPart::text(text));
            }
        }

        // Tool calls
        if let Some(tool_calls) = msg["tool_calls"].as_array() {
            for tc in tool_calls {
                let call_id = tc["id"].as_str().unwrap_or("").to_string();
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let arguments: Value = serde_json::from_str(args_str)
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                content.push(ContentPart::tool_call(ToolCallData {
                    id: call_id,
                    name,
                    arguments,
                    raw_arguments: Some(args_str.to_string()),
                }));
            }
        }

        let finish_reason = finish_reason_from_compat(choice["finish_reason"].as_str());
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
            provider: provider_name.to_string(),
            message,
            finish_reason,
            usage,
            raw: Some(body),
            warnings: vec![],
            rate_limit: None,
        })
    }

    async fn handle_error_response(
        response: reqwest::Response,
        provider_name: &str,
    ) -> UnifiedLlmError {
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

        match status {
            401 => UnifiedLlmError::Authentication {
                provider: provider_name.to_string(),
                message,
            },
            429 => UnifiedLlmError::RateLimit {
                provider: provider_name.to_string(),
                message,
                retry_after,
            },
            400 => UnifiedLlmError::Provider {
                provider: provider_name.to_string(),
                message,
                status_code: Some(status),
                error_code,
                retryable: false,
                retry_after: None,
                raw: Some(body_json),
            },
            500..=599 => UnifiedLlmError::Provider {
                provider: provider_name.to_string(),
                message,
                status_code: Some(status),
                error_code,
                retryable: true,
                retry_after: None,
                raw: Some(body_json),
            },
            _ => UnifiedLlmError::Provider {
                provider: provider_name.to_string(),
                message,
                status_code: Some(status),
                error_code,
                retryable: false,
                retry_after: None,
                raw: Some(body_json),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Free-standing translation helpers
// ---------------------------------------------------------------------------

fn translate_tool_choice_compat(tc: &crate::types::ToolChoice) -> Value {
    match tc.mode.as_str() {
        "auto" => json!("auto"),
        "none" => json!("none"),
        "required" => json!("required"),
        "named" => {
            let name = tc.tool_name.as_deref().unwrap_or("");
            json!({ "type": "function", "function": { "name": name } })
        }
        other => json!(other),
    }
}

fn finish_reason_from_compat(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("stop") => FinishReason {
            reason: "stop".to_string(),
            raw: Some("stop".to_string()),
        },
        Some("length") => FinishReason {
            reason: "length".to_string(),
            raw: Some("length".to_string()),
        },
        Some("tool_calls") => FinishReason {
            reason: "tool_calls".to_string(),
            raw: Some("tool_calls".to_string()),
        },
        Some("content_filter") => FinishReason {
            reason: "content_filter".to_string(),
            raw: Some("content_filter".to_string()),
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

fn translate_message_compat(msg: &Message) -> Value {
    match msg.role {
        Role::System | Role::Developer => {
            let text = msg
                .content
                .iter()
                .filter(|p| p.kind == ContentKind::Text)
                .filter_map(|p| p.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n\n");
            json!({ "role": "system", "content": text })
        }
        Role::User => {
            // Check if there are image parts
            let has_images = msg.content.iter().any(|p| p.kind == ContentKind::Image);
            if has_images {
                let parts: Vec<Value> = msg
                    .content
                    .iter()
                    .filter_map(translate_user_content_compat)
                    .collect();
                json!({ "role": "user", "content": parts })
            } else {
                // String shorthand for text-only
                let text = msg
                    .content
                    .iter()
                    .filter(|p| p.kind == ContentKind::Text)
                    .filter_map(|p| p.text.as_deref())
                    .collect::<Vec<_>>()
                    .join("");
                json!({ "role": "user", "content": text })
            }
        }
        Role::Assistant => {
            let text: Option<String> = {
                let t = msg
                    .content
                    .iter()
                    .filter(|p| p.kind == ContentKind::Text)
                    .filter_map(|p| p.text.as_deref())
                    .collect::<String>();
                if t.is_empty() {
                    None
                } else {
                    Some(t)
                }
            };

            let tool_calls: Vec<Value> = msg
                .content
                .iter()
                .filter(|p| p.kind == ContentKind::ToolCall)
                .filter_map(|p| p.tool_call.as_ref())
                .map(|tc| {
                    let args_str = tc.raw_arguments.clone().unwrap_or_else(|| {
                        serde_json::to_string(&tc.arguments).unwrap_or_default()
                    });
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": args_str,
                        }
                    })
                })
                .collect();

            let mut obj = json!({ "role": "assistant" });
            obj["content"] = match text {
                Some(t) => json!(t),
                None => Value::Null,
            };
            if !tool_calls.is_empty() {
                obj["tool_calls"] = json!(tool_calls);
            }
            obj
        }
        Role::Tool => {
            // Tool result message
            let call_id = msg.tool_call_id.as_deref().unwrap_or("");
            let content_str = msg
                .content
                .iter()
                .filter(|p| p.kind == ContentKind::ToolResult)
                .filter_map(|p| p.tool_result.as_ref())
                .map(|tr| {
                    let base = match &tr.content {
                        Value::String(s) => s.clone(),
                        v => serde_json::to_string(v).unwrap_or_default(),
                    };
                    if tr.is_error {
                        format!("[Error] {base}")
                    } else {
                        base
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content_str,
            })
        }
    }
}

fn translate_user_content_compat(part: &ContentPart) -> Option<Value> {
    match &part.kind {
        ContentKind::Text => {
            Some(json!({ "type": "text", "text": part.text.as_deref().unwrap_or("") }))
        }
        ContentKind::Image => {
            let img = part.image.as_ref()?;
            let url = if let Some(u) = &img.url {
                u.clone()
            } else if let Some(data) = &img.data {
                let b64 = STANDARD.encode(data);
                let media_type = img.media_type.as_deref().unwrap_or("image/jpeg");
                format!("data:{media_type};base64,{b64}")
            } else {
                return None;
            };
            let detail = img.detail.as_deref().unwrap_or("auto");
            Some(json!({
                "type": "image_url",
                "image_url": { "url": url, "detail": detail }
            }))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// ProviderAdapter implementation
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ProviderAdapter for OpenAiCompatAdapter {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn supports_tool_choice(&self, mode: &str) -> bool {
        matches!(mode, "auto" | "none" | "required" | "named")
    }

    async fn complete(&self, request: &Request) -> Result<Response, UnifiedLlmError> {
        let body = self.build_request_body(request, false);
        let url = self.endpoint_url();
        let headers = self.build_headers();
        let client = self.http_client.clone();
        let provider_name = self.provider_name.clone();

        let response = crate::retry::RetryPolicy::default_policy()
            .execute(|| {
                let url = url.clone();
                let headers = headers.clone();
                let body = body.clone();
                let client = client.clone();
                async move {
                    client
                        .post(&url)
                        .headers(headers)
                        .json(&body)
                        .send()
                        .await
                        .map_err(|e| UnifiedLlmError::Network {
                            message: e.to_string(),
                            source: Some(Box::new(e)),
                        })
                }
            })
            .await?;

        if !response.status().is_success() {
            return Err(Self::handle_error_response(response, &provider_name).await);
        }

        let body_text = response
            .text()
            .await
            .map_err(|e| UnifiedLlmError::Network {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let body_json: Value =
            serde_json::from_str(&body_text).map_err(|e| UnifiedLlmError::Provider {
                provider: provider_name.clone(),
                message: format!("failed to parse response JSON: {e}"),
                status_code: None,
                error_code: None,
                retryable: false,
                retry_after: None,
                raw: Some(Value::String(body_text)),
            })?;

        Self::parse_response_body(body_json, &provider_name)
    }

    async fn stream(&self, request: &Request) -> Result<EventStream, UnifiedLlmError> {
        let body = self.build_request_body(request, true);
        let url = self.endpoint_url();
        let headers = self.build_headers();
        let provider_name = self.provider_name.clone();

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
            return Err(Self::handle_error_response(response, &provider_name).await);
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
            let mut finish_reason: Option<String> = None;
            // Track tool call state: index → (id, name)
            let mut tool_call_ids: HashMap<usize, String> = HashMap::new();
            let mut stream_started = false;

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
                            // [DONE] sentinel terminates the stream
                            if sse_ev.data == "[DONE]" {
                                if text_started {
                                    let _ = tx.try_send(Ok(StreamEvent::text_end())).ok();
                                }
                                let fr = finish_reason_from_compat(finish_reason.as_deref());
                                let _ = tx
                                    .try_send(Ok(StreamEvent::finish(fr, Usage::default())))
                                    .ok();
                                return;
                            }

                            let chunk: Value = match serde_json::from_str(&sse_ev.data) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };

                            if !stream_started {
                                stream_started = true;
                                if tx.try_send(Ok(StreamEvent::stream_start())).is_err() {
                                    return;
                                }
                            }

                            let choice = &chunk["choices"][0];
                            if choice.is_null() {
                                // Might be a usage-only chunk; handle usage
                                if chunk.get("usage").is_some() {
                                    let usage = OpenAiCompatAdapter::parse_usage(&chunk["usage"]);
                                    if text_started {
                                        let _ = tx.try_send(Ok(StreamEvent::text_end())).ok();
                                    }
                                    let fr = finish_reason_from_compat(finish_reason.as_deref());
                                    let _ = tx.try_send(Ok(StreamEvent::finish(fr, usage))).ok();
                                    return;
                                }
                                continue;
                            }

                            // Capture finish reason
                            if let Some(fr) = choice["finish_reason"].as_str() {
                                if !fr.is_empty() {
                                    finish_reason = Some(fr.to_string());
                                }
                            }

                            let delta = &choice["delta"];

                            // Text content
                            if let Some(content) = delta["content"].as_str() {
                                if !content.is_empty() {
                                    if !text_started {
                                        text_started = true;
                                        if tx.try_send(Ok(StreamEvent::text_start())).is_err() {
                                            return;
                                        }
                                    }
                                    if tx.try_send(Ok(StreamEvent::text_delta(content))).is_err() {
                                        return;
                                    }
                                }
                            }

                            // Tool calls delta
                            if let Some(tool_calls) = delta["tool_calls"].as_array() {
                                for tc_delta in tool_calls {
                                    let idx = tc_delta["index"].as_u64().unwrap_or(0) as usize;
                                    let id = tc_delta["id"].as_str().unwrap_or("").to_string();
                                    let name = tc_delta["function"]["name"]
                                        .as_str()
                                        .unwrap_or("")
                                        .to_string();

                                    // First occurrence for this index → ToolCallStart
                                    if let std::collections::hash_map::Entry::Vacant(e) =
                                        tool_call_ids.entry(idx)
                                    {
                                        let call_id = if !id.is_empty() {
                                            id.clone()
                                        } else {
                                            format!("call_{idx}")
                                        };
                                        e.insert(call_id.clone());
                                        if tx
                                            .try_send(Ok(StreamEvent::tool_call_start(
                                                call_id, name,
                                            )))
                                            .is_err()
                                        {
                                            return;
                                        }
                                    }

                                    // Argument delta
                                    if let Some(args_delta) =
                                        tc_delta["function"]["arguments"].as_str()
                                    {
                                        if !args_delta.is_empty() {
                                            let call_id = tool_call_ids
                                                .get(&idx)
                                                .cloned()
                                                .unwrap_or_default();
                                            if tx
                                                .try_send(Ok(StreamEvent::tool_call_delta(
                                                    call_id, args_delta,
                                                )))
                                                .is_err()
                                            {
                                                return;
                                            }
                                        }
                                    }
                                }
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

            // Stream closed without [DONE]
            if text_started {
                let _ = tx.try_send(Ok(StreamEvent::text_end())).ok();
            }
            let fr = finish_reason_from_compat(finish_reason.as_deref());
            let _ = tx
                .try_send(Ok(StreamEvent::finish(fr, Usage::default())))
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

    // AC-2: system role → {"role":"system","content":"..."}
    #[test]
    fn system_message_translates_to_system_role() {
        let msg = Message::system("Be helpful.");
        let v = translate_message_compat(&msg);
        assert_eq!(v["role"], "system");
        assert_eq!(v["content"], "Be helpful.");
    }

    // AC-3: developer role → "system"
    #[test]
    fn developer_message_translates_to_system_role() {
        let msg = Message {
            role: Role::Developer,
            content: vec![ContentPart::text("instructions")],
            name: None,
            tool_call_id: None,
        };
        let v = translate_message_compat(&msg);
        assert_eq!(v["role"], "system");
    }

    // AC-4: tool definitions wrapped with type:function
    #[test]
    fn tool_wrapped_in_function_type() {
        let adapter = OpenAiCompatAdapter::new("http://localhost:11434");
        let req = make_request("llama3", vec![Message::user("hi")]).with_tools(vec![Tool {
            name: "my_fn".to_string(),
            description: "does stuff".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }]);
        let body = adapter.build_request_body(&req, false);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "my_fn");
    }

    // AC-5: max_tokens maps to "max_tokens" (not "max_output_tokens")
    #[test]
    fn max_tokens_maps_correctly() {
        let adapter = OpenAiCompatAdapter::new("http://localhost");
        let req = make_request("model", vec![Message::user("hi")]).with_max_tokens(500);
        let body = adapter.build_request_body(&req, false);
        assert_eq!(body["max_tokens"], 500);
        assert!(body.get("max_output_tokens").is_none());
    }

    // AC-6: reasoning_effort omitted from body
    #[test]
    fn reasoning_effort_omitted() {
        let adapter = OpenAiCompatAdapter::new("http://localhost");
        let mut req = make_request("model", vec![Message::user("hi")]);
        req.reasoning_effort = Some("high".to_string());
        let body = adapter.build_request_body(&req, false);
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("reasoning").is_none());
    }

    // AC-7: api_key = None → no Authorization header
    #[test]
    fn no_api_key_no_auth_header() {
        let adapter = OpenAiCompatAdapter::new("http://localhost:11434");
        let headers = adapter.build_headers();
        assert!(headers.get(AUTHORIZATION).is_none());
    }

    // AC-8: api_key set → Authorization: Bearer header present
    #[test]
    fn with_api_key_sets_auth_header() {
        let adapter = OpenAiCompatAdapter::new("http://localhost").with_api_key("sk-test");
        let headers = adapter.build_headers();
        let auth = headers.get(AUTHORIZATION).unwrap().to_str().unwrap();
        assert_eq!(auth, "Bearer sk-test");
    }

    // AC-12: ollama() constructs with correct base_url
    #[test]
    fn ollama_constructor() {
        let adapter = OpenAiCompatAdapter::ollama();
        assert_eq!(adapter.base_url, "http://localhost:11434");
        assert!(adapter.api_key.is_none());
        assert_eq!(adapter.provider_name, "ollama");
    }

    // Response parsing: usage.prompt_tokens → input_tokens
    #[test]
    fn parse_usage_maps_prompt_tokens() {
        let usage_val = serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        });
        let usage = OpenAiCompatAdapter::parse_usage(&usage_val);
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert!(usage.reasoning_tokens.is_none());
    }

    // Response parsing: text content
    #[test]
    fn parse_response_body_text() {
        let body = serde_json::json!({
            "id": "resp-1",
            "model": "llama3",
            "choices": [{
                "message": { "role": "assistant", "content": "Hello!" },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 },
        });
        let resp = OpenAiCompatAdapter::parse_response_body(body, "ollama").unwrap();
        assert_eq!(resp.text(), "Hello!");
        assert!(resp.finish_reason.is_stop());
    }

    // Response parsing: tool_calls
    #[test]
    fn parse_response_body_tool_calls() {
        let body = serde_json::json!({
            "id": "resp-2",
            "model": "llama3",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "my_fn", "arguments": "{\"x\":1}" }
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 },
        });
        let resp = OpenAiCompatAdapter::parse_response_body(body, "groq").unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "my_fn");
        assert_eq!(calls[0].arguments["x"], 1);
        assert!(resp.finish_reason.is_tool_calls());
    }

    // finish_reason mapping
    #[test]
    fn finish_reason_stop() {
        assert!(finish_reason_from_compat(Some("stop")).is_stop());
    }

    #[test]
    fn finish_reason_tool_calls() {
        assert!(finish_reason_from_compat(Some("tool_calls")).is_tool_calls());
    }

    #[test]
    fn finish_reason_length() {
        assert_eq!(finish_reason_from_compat(Some("length")).reason, "length");
    }

    // Tool choice translation
    #[test]
    fn tool_choice_none_serializes() {
        assert_eq!(
            translate_tool_choice_compat(&ToolChoice::none()),
            json!("none")
        );
    }

    #[test]
    fn tool_choice_named_serializes() {
        let tc = translate_tool_choice_compat(&ToolChoice::named("my_fn"));
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "my_fn");
    }

    // base_url trailing slash stripped
    #[test]
    fn trailing_slash_stripped() {
        let adapter = OpenAiCompatAdapter::new("http://localhost:11434/");
        assert_eq!(adapter.base_url, "http://localhost:11434");
        let url = adapter.endpoint_url();
        assert_eq!(url, "http://localhost:11434/v1/chat/completions");
    }

    // Stream body has stream: true
    #[test]
    fn stream_body_has_stream_true() {
        let adapter = OpenAiCompatAdapter::new("http://localhost");
        let req = make_request("model", vec![Message::user("hi")]);
        let body = adapter.build_request_body(&req, true);
        assert_eq!(body["stream"], true);
    }

    // Groq convenience constructor
    #[test]
    fn groq_constructor() {
        let adapter = OpenAiCompatAdapter::groq("my-key");
        assert_eq!(adapter.base_url, "https://api.groq.com/openai");
        assert_eq!(adapter.api_key, Some("my-key".to_string()));
        assert_eq!(adapter.provider_name, "groq");
    }
}
