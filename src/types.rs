//! Core data types used throughout the `unified-llm` crate.
//!
//! These types are the lingua franca of the entire library — every provider
//! adapter, every middleware, and every consumer of the API uses these types.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// The role of a message participant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    Developer,
}

// ---------------------------------------------------------------------------
// ContentKind
// ---------------------------------------------------------------------------

/// Discriminant for the kind of content stored in a [`ContentPart`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContentKind {
    Text,
    Image,
    Audio,
    Document,
    ToolCall,
    ToolResult,
    Thinking,
    RedactedThinking,
}

// ---------------------------------------------------------------------------
// Leaf data types embedded in ContentPart
// ---------------------------------------------------------------------------

/// Base64-encoded or URL-referenced image data.
///
/// Priority for image source: `url` > `data` > `path`.
/// When `path` is set and `data` is `None`, provider adapters read the file
/// and base64-encode it on the fly before building the request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub url: Option<String>,
    pub data: Option<Vec<u8>>,
    /// Local file path to read and base64-encode.  Ignored when `data` is already set.
    pub path: Option<String>,
    pub media_type: Option<String>,
    pub detail: Option<String>,
}

/// Base64-encoded or URL-referenced audio data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    pub url: Option<String>,
    pub data: Option<Vec<u8>>,
    pub media_type: Option<String>,
}

/// Base64-encoded or URL-referenced document data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentData {
    pub url: Option<String>,
    pub data: Option<Vec<u8>>,
    pub media_type: Option<String>,
    pub file_name: Option<String>,
}

/// A tool call request embedded in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallData {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    pub raw_arguments: Option<String>,
}

/// A tool result returned to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultData {
    pub tool_call_id: String,
    pub content: serde_json::Value,
    pub is_error: bool,
}

/// Extended thinking / reasoning data from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingData {
    pub text: String,
    pub signature: Option<String>,
    pub redacted: bool,
}

// ---------------------------------------------------------------------------
// ContentPart
// ---------------------------------------------------------------------------

/// A single piece of content within a [`Message`].
///
/// Uses the "struct with kind tag" pattern so adapters can build parts
/// field-by-field without matching on a fully-tagged enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    pub kind: ContentKind,
    pub text: Option<String>,
    pub image: Option<ImageData>,
    pub audio: Option<AudioData>,
    pub document: Option<DocumentData>,
    pub tool_call: Option<ToolCallData>,
    pub tool_result: Option<ToolResultData>,
    pub thinking: Option<ThinkingData>,
}

impl ContentPart {
    /// Create a plain-text content part.
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            kind: ContentKind::Text,
            text: Some(s.into()),
            image: None,
            audio: None,
            document: None,
            tool_call: None,
            tool_result: None,
            thinking: None,
        }
    }

    /// Create an image content part.
    pub fn image(data: ImageData) -> Self {
        Self {
            kind: ContentKind::Image,
            text: None,
            image: Some(data),
            audio: None,
            document: None,
            tool_call: None,
            tool_result: None,
            thinking: None,
        }
    }

    /// Create a tool-call content part.
    pub fn tool_call(data: ToolCallData) -> Self {
        Self {
            kind: ContentKind::ToolCall,
            text: None,
            image: None,
            audio: None,
            document: None,
            tool_call: Some(data),
            tool_result: None,
            thinking: None,
        }
    }

    /// Create a tool-result content part.
    pub fn tool_result(data: ToolResultData) -> Self {
        Self {
            kind: ContentKind::ToolResult,
            text: None,
            image: None,
            audio: None,
            document: None,
            tool_call: None,
            tool_result: Some(data),
            thinking: None,
        }
    }

    /// Create a thinking/reasoning content part.
    pub fn thinking(data: ThinkingData) -> Self {
        Self {
            kind: ContentKind::Thinking,
            text: None,
            image: None,
            audio: None,
            document: None,
            tool_call: None,
            tool_result: None,
            thinking: Some(data),
        }
    }
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// A single turn in a conversation, carrying one or more content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a system message.
    pub fn system(text: &str) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::text(text)],
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a user message.
    pub fn user(text: &str) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentPart::text(text)],
            name: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message.
    pub fn assistant(text: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::text(text)],
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a tool-result message.
    pub fn tool_result(tool_call_id: &str, content: &str, is_error: bool) -> Self {
        let data = ToolResultData {
            tool_call_id: tool_call_id.to_string(),
            content: serde_json::Value::String(content.to_string()),
            is_error,
        };
        Self {
            role: Role::Tool,
            content: vec![ContentPart::tool_result(data)],
            name: None,
            tool_call_id: Some(tool_call_id.to_string()),
        }
    }

    /// Concatenates all `Text`-kind content parts into a single `String`.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter(|p| p.kind == ContentKind::Text)
            .filter_map(|p| p.text.as_deref())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tool types
// ---------------------------------------------------------------------------

/// A tool (function) the model can call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the function parameters.
    pub parameters: serde_json::Value,
}

/// A top-level representation of a tool call request from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    pub raw_arguments: Option<String>,
}

/// A tool result returned to the model after executing a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
}

/// Specifies which tool (if any) the model should call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoice {
    /// `"auto"` | `"none"` | `"required"` | `"named"`
    pub mode: String,
    pub tool_name: Option<String>,
}

impl ToolChoice {
    pub fn auto() -> Self {
        Self {
            mode: "auto".to_string(),
            tool_name: None,
        }
    }

    pub fn none() -> Self {
        Self {
            mode: "none".to_string(),
            tool_name: None,
        }
    }

    pub fn required() -> Self {
        Self {
            mode: "required".to_string(),
            tool_name: None,
        }
    }

    pub fn named(name: impl Into<String>) -> Self {
        Self {
            mode: "named".to_string(),
            tool_name: Some(name.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// ResponseFormat
// ---------------------------------------------------------------------------

/// The desired output format for a completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    pub format_type: ResponseFormatType,
    pub json_schema: Option<serde_json::Value>,
    pub strict: bool,
}

/// Discriminant for [`ResponseFormat`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResponseFormatType {
    Text,
    Json,
    JsonSchema,
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// A completion request sent to an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    pub provider: Option<String>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub response_format: Option<ResponseFormat>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub reasoning_effort: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
    pub provider_options: Option<serde_json::Value>,
}

impl Request {
    /// Create a new request with all optional fields set to `None`.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            provider: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop_sequences: None,
            reasoning_effort: None,
            metadata: None,
            provider_options: None,
        }
    }

    /// Set the tools available for this request.
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum number of output tokens.
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// Token usage statistics for a completion.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub reasoning_tokens: Option<u32>,
    pub cache_read_tokens: Option<u32>,
    pub cache_write_tokens: Option<u32>,
    pub raw: Option<serde_json::Value>,
}

/// Helper: add two `Option<u32>` values.
/// `None + None = None`; `Some(a) + None = Some(a)`; `Some(a) + Some(b) = Some(a + b)`.
fn add_opt(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some(x), Some(y)) => Some(x + y),
    }
}

impl std::ops::Add for Usage {
    type Output = Usage;

    fn add(self, rhs: Usage) -> Usage {
        Usage {
            input_tokens: self.input_tokens + rhs.input_tokens,
            output_tokens: self.output_tokens + rhs.output_tokens,
            total_tokens: self.total_tokens + rhs.total_tokens,
            reasoning_tokens: add_opt(self.reasoning_tokens, rhs.reasoning_tokens),
            cache_read_tokens: add_opt(self.cache_read_tokens, rhs.cache_read_tokens),
            cache_write_tokens: add_opt(self.cache_write_tokens, rhs.cache_write_tokens),
            raw: None, // raw is not meaningful after addition
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Usage) {
        *self = self.clone() + rhs;
    }
}

// ---------------------------------------------------------------------------
// FinishReason
// ---------------------------------------------------------------------------

/// The reason a model stopped generating tokens.
///
/// Uses a `String` discriminant rather than a closed enum so that unknown
/// provider-specific reasons can be stored without a parse error.
///
/// Normalized reasons: `"stop"`, `"length"`, `"tool_calls"`,
/// `"content_filter"`, `"error"`, `"other"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinishReason {
    pub reason: String,
    pub raw: Option<String>,
}

impl FinishReason {
    pub fn stop() -> Self {
        Self {
            reason: "stop".to_string(),
            raw: None,
        }
    }

    pub fn length() -> Self {
        Self {
            reason: "length".to_string(),
            raw: None,
        }
    }

    pub fn tool_calls() -> Self {
        Self {
            reason: "tool_calls".to_string(),
            raw: None,
        }
    }

    pub fn is_stop(&self) -> bool {
        self.reason == "stop"
    }

    pub fn is_tool_calls(&self) -> bool {
        self.reason == "tool_calls"
    }
}

// ---------------------------------------------------------------------------
// Warning / RateLimitInfo
// ---------------------------------------------------------------------------

/// A non-fatal warning attached to a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Warning {
    pub message: String,
    pub code: Option<String>,
}

/// Rate-limit metadata returned by the provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub requests_remaining: Option<u32>,
    pub requests_limit: Option<u32>,
    pub tokens_remaining: Option<u32>,
    pub tokens_limit: Option<u32>,
    pub reset_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// A completed response from an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub id: String,
    pub model: String,
    pub provider: String,
    pub message: Message,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub raw: Option<serde_json::Value>,
    pub warnings: Vec<Warning>,
    pub rate_limit: Option<RateLimitInfo>,
}

impl Response {
    /// Returns the concatenated text of all `Text`-kind content parts in `message`.
    pub fn text(&self) -> String {
        self.message.text()
    }

    /// Returns all `ToolCallData` values from `ToolCall`-kind content parts.
    pub fn tool_calls(&self) -> Vec<&ToolCallData> {
        self.message
            .content
            .iter()
            .filter(|p| p.kind == ContentKind::ToolCall)
            .filter_map(|p| p.tool_call.as_ref())
            .collect()
    }

    /// Returns the concatenated text of all `Thinking`-kind content parts,
    /// or `None` if there are none.
    pub fn reasoning(&self) -> Option<String> {
        let parts: Vec<&str> = self
            .message
            .content
            .iter()
            .filter(|p| p.kind == ContentKind::Thinking)
            .filter_map(|p| p.thinking.as_ref())
            .map(|t| t.text.as_str())
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(""))
        }
    }
}

// ---------------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------------

/// Metadata for a known LLM model in the built-in catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Canonical model identifier (e.g., `"gpt-4o"`).
    pub id: String,
    /// Provider name (e.g., `"openai"`, `"anthropic"`, `"gemini"`).
    pub provider: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Maximum input context window in tokens.
    pub context_window: u32,
    /// Maximum output tokens, if the provider publishes a limit.
    pub max_output: Option<u32>,
    /// Whether the model supports tool/function calling.
    pub supports_tools: bool,
    /// Whether the model supports image/vision inputs.
    pub supports_vision: bool,
    /// Whether the model has extended reasoning / chain-of-thought capability.
    pub supports_reasoning: bool,
    /// Input cost per one million tokens (USD), if known.
    pub input_cost_per_million: Option<f64>,
    /// Output cost per one million tokens (USD), if known.
    pub output_cost_per_million: Option<f64>,
    /// Alternative identifiers that resolve to this model (e.g., `"gpt-4o-latest"`).
    pub aliases: Vec<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // AC-1: Message::system("hello").text() returns "hello"
    #[test]
    fn message_system_text() {
        assert_eq!(Message::system("hello").text(), "hello");
    }

    // AC-2: Message::user("hi").role == Role::User
    #[test]
    fn message_user_role() {
        assert_eq!(Message::user("hi").role, Role::User);
    }

    // AC-3: tool_result has Role::Tool and tool_call_id == Some("id1")
    #[test]
    fn message_tool_result_fields() {
        let msg = Message::tool_result("id1", "ok", false);
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id, Some("id1".to_string()));
    }

    // AC-4: Usage::default() has all zero/None fields
    #[test]
    fn usage_default() {
        let u = Usage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
        assert_eq!(u.total_tokens, 0);
        assert!(u.reasoning_tokens.is_none());
        assert!(u.cache_read_tokens.is_none());
        assert!(u.cache_write_tokens.is_none());
    }

    // AC-5: Usage addition sums tokens
    #[test]
    fn usage_add_sums_tokens() {
        let a = Usage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let b = Usage {
            input_tokens: 3,
            output_tokens: 2,
            total_tokens: 5,
            ..Default::default()
        };
        let c = a + b;
        assert_eq!(c.input_tokens, 13);
        assert_eq!(c.output_tokens, 7);
        assert_eq!(c.total_tokens, 20);
    }

    // AC-6: None handling in Usage::add
    #[test]
    fn usage_add_none_handling() {
        let a = Usage {
            reasoning_tokens: Some(8),
            ..Default::default()
        };
        let b = Usage {
            reasoning_tokens: None,
            ..Default::default()
        };
        let c = a + b;
        assert_eq!(c.reasoning_tokens, Some(8));
    }

    // AC-7: ToolChoice::auto().mode == "auto"
    #[test]
    fn tool_choice_auto() {
        assert_eq!(ToolChoice::auto().mode, "auto");
    }

    // AC-8: ToolChoice::named("my_fn").tool_name == Some("my_fn")
    #[test]
    fn tool_choice_named() {
        let tc = ToolChoice::named("my_fn");
        assert_eq!(tc.tool_name, Some("my_fn".to_string()));
        assert_eq!(tc.mode, "named");
    }

    // AC-9: ContentPart::text("foo").kind == ContentKind::Text
    #[test]
    fn content_part_text_kind() {
        assert_eq!(ContentPart::text("foo").kind, ContentKind::Text);
    }

    // AC-10: Round-trip through serde_json
    #[test]
    fn serde_roundtrip_request() {
        let req = Request::new("gpt-4o", vec![Message::user("hello")])
            .with_temperature(0.7)
            .with_max_tokens(100);
        let json = serde_json::to_string(&req).unwrap();
        let decoded: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.model, "gpt-4o");
        assert_eq!(decoded.temperature, Some(0.7));
        assert_eq!(decoded.max_tokens, Some(100));
    }

    #[test]
    fn serde_roundtrip_usage() {
        let u = Usage {
            input_tokens: 5,
            output_tokens: 10,
            total_tokens: 15,
            ..Default::default()
        };
        let json = serde_json::to_string(&u).unwrap();
        let decoded: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.input_tokens, 5);
        assert_eq!(decoded.total_tokens, 15);
    }

    #[test]
    fn serde_roundtrip_message() {
        let msg = Message::assistant("world");
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.text(), "world");
        assert_eq!(decoded.role, Role::Assistant);
    }

    // AC-11: tested by clippy pass

    // AC-12: FinishReason::is_stop()
    #[test]
    fn finish_reason_is_stop() {
        assert!(FinishReason::stop().is_stop());
        assert!(!FinishReason::length().is_stop());
        assert!(!FinishReason::tool_calls().is_stop());
    }

    // AC-13: Response::tool_calls() returns empty vec when no ToolCall parts
    #[test]
    fn response_tool_calls_empty() {
        let resp = Response {
            id: "r1".to_string(),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            message: Message::assistant("hi"),
            finish_reason: FinishReason::stop(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        assert!(resp.tool_calls().is_empty());
    }

    // Edge: Message::text() on no-text message returns empty string
    #[test]
    fn message_text_empty_when_no_text_parts() {
        let msg = Message {
            role: Role::User,
            content: vec![],
            name: None,
            tool_call_id: None,
        };
        assert_eq!(msg.text(), "");
    }

    // Edge: Usage::add both None → result None
    #[test]
    fn usage_add_both_none() {
        let a = Usage {
            cache_read_tokens: None,
            ..Default::default()
        };
        let b = Usage {
            cache_read_tokens: None,
            ..Default::default()
        };
        let c = a + b;
        assert!(c.cache_read_tokens.is_none());
    }

    // Edge: Request::with_tools([]) sets Some(vec![])
    #[test]
    fn request_with_empty_tools() {
        let req = Request::new("model", vec![]).with_tools(vec![]);
        assert!(req.tools.is_some());
        assert!(req.tools.unwrap().is_empty());
    }

    // V2-ULM-005: ImageData.path field exists and round-trips through serde
    #[test]
    fn image_data_path_field() {
        let img = crate::types::ImageData {
            url: None,
            data: None,
            path: Some("/tmp/test.png".to_string()),
            media_type: Some("image/png".to_string()),
            detail: None,
        };
        assert_eq!(img.path, Some("/tmp/test.png".to_string()));
        // Serde round-trip
        let json = serde_json::to_string(&img).unwrap();
        let decoded: crate::types::ImageData = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.path, Some("/tmp/test.png".to_string()));
    }

    // ContentKind::RedactedThinking round-trips through serde
    #[test]
    fn content_kind_redacted_thinking_serde() {
        let kind = ContentKind::RedactedThinking;
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"RedactedThinking\"");
        let decoded: ContentKind = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ContentKind::RedactedThinking);
    }

    // Response::reasoning() returns Some when Thinking parts present
    #[test]
    fn response_reasoning_some() {
        let thinking_part = ContentPart::thinking(ThinkingData {
            text: "I think...".to_string(),
            signature: None,
            redacted: false,
        });
        let msg = Message {
            role: Role::Assistant,
            content: vec![thinking_part, ContentPart::text("answer")],
            name: None,
            tool_call_id: None,
        };
        let resp = Response {
            id: "r2".to_string(),
            model: "claude".to_string(),
            provider: "anthropic".to_string(),
            message: msg,
            finish_reason: FinishReason::stop(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        assert_eq!(resp.reasoning(), Some("I think...".to_string()));
        assert_eq!(resp.text(), "answer");
    }

    // Usage::add_assign works
    #[test]
    fn usage_add_assign() {
        let mut a = Usage {
            input_tokens: 5,
            output_tokens: 3,
            total_tokens: 8,
            ..Default::default()
        };
        let b = Usage {
            input_tokens: 2,
            output_tokens: 1,
            total_tokens: 3,
            ..Default::default()
        };
        a += b;
        assert_eq!(a.input_tokens, 7);
        assert_eq!(a.total_tokens, 11);
    }
}
