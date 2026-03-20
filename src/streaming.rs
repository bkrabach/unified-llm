//! Stream event types and accumulator for unified LLM streaming.
//!
//! Each provider adapter emits a sequence of [`StreamEvent`] values. The
//! [`StreamAccumulator`] collects those events and assembles a final
//! [`crate::types::Response`].

use std::collections::HashMap;
use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::UnifiedLlmError;
use crate::types::{
    ContentPart, FinishReason, Message, Response, Role, ThinkingData, ToolCall, ToolCallData,
    Usage, Warning,
};

// ---------------------------------------------------------------------------
// StreamEventType
// ---------------------------------------------------------------------------

/// Discriminant for the kind of a [`StreamEvent`].
///
/// This enum is `#[non_exhaustive]` — downstream `match` arms must include
/// a wildcard (`_ => {}`) to compile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamEventType {
    /// Stream opened; no content yet.
    StreamStart,
    /// A text generation block is beginning.
    TextStart,
    /// Incremental text content (the most frequent event).
    TextDelta,
    /// A text generation block has ended.
    TextEnd,
    /// Reasoning/thinking block is beginning.
    ReasoningStart,
    /// Incremental reasoning content.
    ReasoningDelta,
    /// Reasoning block ended.
    ReasoningEnd,
    /// A tool call is beginning (id and name available).
    ToolCallStart,
    /// Incremental JSON arguments for an in-progress tool call.
    ToolCallDelta,
    /// Tool call arguments are complete.
    ToolCallEnd,
    /// Stream is ending; usage and finish_reason available.
    Finish,
    /// A non-fatal provider-level event (passthrough).
    ProviderEvent,
    /// An error occurred during streaming.
    Error,
}

// ---------------------------------------------------------------------------
// StreamEvent
// ---------------------------------------------------------------------------

/// A single event emitted by a streaming LLM response.
///
/// Most fields are `None`; the populated fields depend on [`StreamEventType`].
/// Use the constructor helpers (`text_delta`, `finish`, etc.) to build events
/// with only the relevant fields set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    pub event_type: StreamEventType,
    /// Incremental text (TextDelta).
    pub delta: Option<String>,
    /// Provider-specific block identifier for disambiguation.
    pub text_id: Option<String>,
    /// Incremental reasoning text (ReasoningDelta).
    pub reasoning_delta: Option<String>,
    /// Tool call data (ToolCallStart / ToolCallEnd).
    pub tool_call: Option<ToolCall>,
    /// Partial JSON argument string (ToolCallDelta).
    pub tool_call_delta: Option<String>,
    /// Tool call identifier used for delta association.
    pub tool_call_id: Option<String>,
    /// Finish reason (Finish event only).
    pub finish_reason: Option<FinishReason>,
    /// Token usage (Finish event only).
    pub usage: Option<Usage>,
    /// Fully assembled response attached by the high-level API at Finish time.
    pub response: Option<Box<Response>>,
    /// Error message string (Error event only).
    pub error: Option<String>,
    /// Raw provider event payload for passthrough / debugging.
    pub raw: Option<serde_json::Value>,
}

impl StreamEvent {
    /// Internal helper: allocate an event with all optional fields `None`.
    fn blank(event_type: StreamEventType) -> Self {
        Self {
            event_type,
            delta: None,
            text_id: None,
            reasoning_delta: None,
            tool_call: None,
            tool_call_delta: None,
            tool_call_id: None,
            finish_reason: None,
            usage: None,
            response: None,
            error: None,
            raw: None,
        }
    }

    /// Construct a `StreamStart` event.
    pub fn stream_start() -> Self {
        Self::blank(StreamEventType::StreamStart)
    }

    /// Construct a `TextStart` event.
    pub fn text_start() -> Self {
        Self::blank(StreamEventType::TextStart)
    }

    /// Construct a `TextEnd` event.
    pub fn text_end() -> Self {
        Self::blank(StreamEventType::TextEnd)
    }

    /// Construct a `TextDelta` event.
    pub fn text_delta(delta: impl Into<String>) -> Self {
        let mut ev = Self::blank(StreamEventType::TextDelta);
        ev.delta = Some(delta.into());
        ev
    }

    /// Construct a `TextDelta` event with an associated text block ID.
    pub fn text_delta_with_id(delta: impl Into<String>, text_id: impl Into<String>) -> Self {
        let mut ev = Self::blank(StreamEventType::TextDelta);
        ev.delta = Some(delta.into());
        ev.text_id = Some(text_id.into());
        ev
    }

    /// Construct a `ReasoningDelta` event.
    pub fn reasoning_delta(delta: impl Into<String>) -> Self {
        let mut ev = Self::blank(StreamEventType::ReasoningDelta);
        ev.reasoning_delta = Some(delta.into());
        ev
    }

    /// Construct a `Finish` event.
    pub fn finish(finish_reason: FinishReason, usage: Usage) -> Self {
        let mut ev = Self::blank(StreamEventType::Finish);
        ev.finish_reason = Some(finish_reason);
        ev.usage = Some(usage);
        ev
    }

    /// Construct an `Error` event.
    pub fn error(message: impl Into<String>) -> Self {
        let mut ev = Self::blank(StreamEventType::Error);
        ev.error = Some(message.into());
        ev
    }

    /// Construct a `ToolCallStart` event.
    ///
    /// Both `id` and `name` are set; arguments are empty at this point.
    pub fn tool_call_start(id: impl Into<String>, name: impl Into<String>) -> Self {
        let mut ev = Self::blank(StreamEventType::ToolCallStart);
        let id_str = id.into();
        let name_str = name.into();
        ev.tool_call_id = Some(id_str.clone());
        ev.tool_call = Some(ToolCall {
            id: id_str,
            name: name_str,
            arguments: serde_json::Value::Null,
            raw_arguments: None,
        });
        ev
    }

    /// Construct a `ToolCallDelta` event carrying a partial JSON argument string.
    pub fn tool_call_delta(id: impl Into<String>, delta: impl Into<String>) -> Self {
        let mut ev = Self::blank(StreamEventType::ToolCallDelta);
        ev.tool_call_id = Some(id.into());
        ev.tool_call_delta = Some(delta.into());
        ev
    }

    /// Construct a `ToolCallEnd` event with the complete tool call data.
    pub fn tool_call_end(call: ToolCall) -> Self {
        let mut ev = Self::blank(StreamEventType::ToolCallEnd);
        ev.tool_call_id = Some(call.id.clone());
        ev.tool_call = Some(call);
        ev
    }

    /// Returns `true` if this is a terminal event (`Finish` or `Error`).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.event_type,
            StreamEventType::Finish | StreamEventType::Error
        )
    }
}

// ---------------------------------------------------------------------------
// EventStream type alias
// ---------------------------------------------------------------------------

/// Type alias for the async streaming return type used throughout the crate.
///
/// This is a pinned, boxed, `Send`-able stream of `Result<StreamEvent, …>`.
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, UnifiedLlmError>> + Send>>;

// ---------------------------------------------------------------------------
// StreamAccumulator internals
// ---------------------------------------------------------------------------

/// Per-tool-call state built up from ToolCallStart + ToolCallDelta events.
struct ToolCallBuffer {
    name: String,
    arguments: String,
}

// ---------------------------------------------------------------------------
// StreamAccumulator
// ---------------------------------------------------------------------------

/// Collects [`StreamEvent`] values and assembles a complete [`Response`].
///
/// Usage pattern:
/// 1. Create with [`StreamAccumulator::new`].
/// 2. Call [`process`][StreamAccumulator::process] for each incoming event.
/// 3. Call [`finalize`][StreamAccumulator::finalize] after the stream ends
///    to obtain the assembled `Response`.
///
/// Not [`Clone`]: the accumulator owns mutable buffering state.
pub struct StreamAccumulator {
    provider: String,
    model: String,
    response_id: String,
    text_buffer: String,
    reasoning_buffer: String,
    /// Keyed by tool_call_id.
    tool_call_buffers: HashMap<String, ToolCallBuffer>,
    /// Ordered list of tool call IDs so finalize output is deterministic.
    tool_call_order: Vec<String>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    warnings: Vec<Warning>,
    raw: Option<serde_json::Value>,
}

impl StreamAccumulator {
    /// Create a new accumulator. `provider` and `model` are written into the
    /// final [`Response`].
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            response_id: uuid::Uuid::new_v4().to_string(),
            text_buffer: String::new(),
            reasoning_buffer: String::new(),
            tool_call_buffers: HashMap::new(),
            tool_call_order: Vec::new(),
            finish_reason: None,
            usage: None,
            warnings: Vec::new(),
            raw: None,
        }
    }

    /// Process one event from the stream, updating internal buffers.
    ///
    /// Returns `Err` if:
    /// - A `ToolCallDelta` or `ToolCallEnd` arrives for an unknown
    ///   `tool_call_id` (missing `ToolCallStart`).
    ///
    /// JSON parsing of tool call arguments is deferred to [`finalize`][Self::finalize].
    pub fn process(&mut self, event: &StreamEvent) -> Result<(), UnifiedLlmError> {
        match &event.event_type {
            StreamEventType::TextDelta => {
                if let Some(delta) = &event.delta {
                    self.text_buffer.push_str(delta);
                }
                // delta == None → treat as empty delta; no error, no content
            }
            StreamEventType::ReasoningDelta => {
                // Accept either `reasoning_delta` or `delta` field.
                let delta = event
                    .reasoning_delta
                    .as_deref()
                    .or(event.delta.as_deref())
                    .unwrap_or("");
                self.reasoning_buffer.push_str(delta);
            }
            StreamEventType::ToolCallStart => {
                if let Some(id) = &event.tool_call_id {
                    if !self.tool_call_buffers.contains_key(id) {
                        self.tool_call_order.push(id.clone());
                    }
                    let name = event
                        .tool_call
                        .as_ref()
                        .map(|tc| tc.name.clone())
                        .unwrap_or_default();
                    self.tool_call_buffers.insert(
                        id.clone(),
                        ToolCallBuffer {
                            name,
                            arguments: String::new(),
                        },
                    );
                }
            }
            StreamEventType::ToolCallDelta => {
                if let Some(id) = &event.tool_call_id {
                    let buf = self.tool_call_buffers.get_mut(id).ok_or_else(|| {
                        UnifiedLlmError::Stream {
                            message: format!("no buffer for tool_call_id: {id}"),
                        }
                    })?;
                    if let Some(delta) = &event.tool_call_delta {
                        buf.arguments.push_str(delta);
                    }
                }
            }
            StreamEventType::ToolCallEnd => {
                if let Some(id) = &event.tool_call_id {
                    if !self.tool_call_buffers.contains_key(id) {
                        return Err(UnifiedLlmError::Stream {
                            message: format!("no buffer for tool_call_id: {id}"),
                        });
                    }
                    // JSON parsing is deferred to finalize() per AC-5.
                }
            }
            StreamEventType::Finish => {
                self.finish_reason = event.finish_reason.clone();
                self.usage = event.usage.clone();
            }
            // ProviderEvent, StreamStart, TextStart, TextEnd, ReasoningStart,
            // ReasoningEnd, Error, and any future variants are silently ignored.
            _ => {}
        }
        Ok(())
    }

    /// Finalize accumulation and return the assembled [`Response`].
    ///
    /// # Errors
    /// - [`UnifiedLlmError::Stream`] if no `Finish` event was processed.
    /// - [`UnifiedLlmError::InvalidToolCall`] if any tool call's buffered
    ///   arguments are not valid JSON.
    pub fn finalize(self) -> Result<Response, UnifiedLlmError> {
        if !self.is_complete() {
            return Err(UnifiedLlmError::Stream {
                message: "stream ended without Finish event".to_string(),
            });
        }

        let mut content: Vec<ContentPart> = Vec::new();

        // Text content
        if !self.text_buffer.is_empty() {
            content.push(ContentPart::text(self.text_buffer));
        }

        // Reasoning / thinking content
        if !self.reasoning_buffer.is_empty() {
            content.push(ContentPart::thinking(ThinkingData {
                text: self.reasoning_buffer,
                signature: None,
                redacted: false,
            }));
        }

        // Tool calls — parsed in deterministic order
        for id in &self.tool_call_order {
            if let Some(buf) = self.tool_call_buffers.get(id) {
                let arguments: serde_json::Value = if buf.arguments.is_empty() {
                    serde_json::Value::Object(serde_json::Map::new())
                } else {
                    serde_json::from_str(&buf.arguments).map_err(|e| {
                        UnifiedLlmError::InvalidToolCall {
                            message: format!(
                                "invalid JSON in tool call arguments for id={id}: {e}"
                            ),
                        }
                    })?
                };
                content.push(ContentPart::tool_call(ToolCallData {
                    id: id.clone(),
                    name: buf.name.clone(),
                    arguments,
                    raw_arguments: if buf.arguments.is_empty() {
                        None
                    } else {
                        Some(buf.arguments.clone())
                    },
                }));
            }
        }

        let message = Message {
            role: Role::Assistant,
            content,
            name: None,
            tool_call_id: None,
        };

        Ok(Response {
            id: self.response_id,
            model: self.model,
            provider: self.provider,
            message,
            finish_reason: self.finish_reason.unwrap_or_else(FinishReason::stop),
            usage: self.usage.unwrap_or_default(),
            raw: self.raw,
            warnings: self.warnings,
            rate_limit: None,
        })
    }

    /// Returns `true` if a `Finish` event has been processed.
    pub fn is_complete(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Returns the accumulated text so far (for progress display).
    pub fn current_text(&self) -> &str {
        &self.text_buffer
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Usage;

    fn make_finish() -> StreamEvent {
        StreamEvent::finish(
            FinishReason::stop(),
            Usage {
                input_tokens: 5,
                output_tokens: 3,
                total_tokens: 8,
                ..Default::default()
            },
        )
    }

    // AC-1: 3 TextDelta + Finish → Response.text() == concatenation
    #[test]
    fn text_deltas_accumulate() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        acc.process(&StreamEvent::text_delta("Hello")).unwrap();
        acc.process(&StreamEvent::text_delta(", ")).unwrap();
        acc.process(&StreamEvent::text_delta("world")).unwrap();
        acc.process(&make_finish()).unwrap();

        let resp = acc.finalize().unwrap();
        assert_eq!(resp.text(), "Hello, world");
    }

    // AC-2: finalize without Finish → Err(Stream)
    #[test]
    fn finalize_without_finish_is_error() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        acc.process(&StreamEvent::text_delta("partial")).unwrap();
        let err = acc.finalize().unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Stream { .. }));
    }

    // AC-3: ToolCallStart + 3 ToolCallDelta + ToolCallEnd → ToolCall in Response
    #[test]
    fn tool_call_accumulates() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        acc.process(&StreamEvent::tool_call_start("call-1", "my_func"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_delta("call-1", "{\"x\":"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_delta("call-1", "1,\"y\":"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_delta("call-1", "2}"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_end(ToolCall {
            id: "call-1".to_string(),
            name: "my_func".to_string(),
            arguments: serde_json::json!({"x": 1, "y": 2}),
            raw_arguments: None,
        }))
        .unwrap();
        acc.process(&make_finish()).unwrap();

        let resp = acc.finalize().unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "my_func");
        assert_eq!(calls[0].arguments["x"], 1);
    }

    // AC-4: is_terminal() true for Finish and Error, false for TextDelta
    #[test]
    fn is_terminal_variants() {
        assert!(StreamEvent::finish(FinishReason::stop(), Usage::default()).is_terminal());
        assert!(StreamEvent::error("oops").is_terminal());
        assert!(!StreamEvent::text_delta("hello").is_terminal());
        assert!(!StreamEvent::stream_start().is_terminal());
    }

    // AC-5: invalid JSON in ToolCallDelta → finalize returns Err(InvalidToolCall)
    #[test]
    fn invalid_tool_call_json_error_at_finalize() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        acc.process(&StreamEvent::tool_call_start("call-bad", "fn"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_delta("call-bad", "{bad json"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_end(ToolCall {
            id: "call-bad".to_string(),
            name: "fn".to_string(),
            arguments: serde_json::Value::Null,
            raw_arguments: None,
        }))
        .unwrap();
        acc.process(&make_finish()).unwrap();

        let err = acc.finalize().unwrap_err();
        assert!(
            matches!(err, UnifiedLlmError::InvalidToolCall { .. }),
            "expected InvalidToolCall, got: {err:?}"
        );
    }

    // AC-6: current_text() returns accumulated text mid-stream
    #[test]
    fn current_text_mid_stream() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        acc.process(&StreamEvent::text_delta("abc")).unwrap();
        assert_eq!(acc.current_text(), "abc");
        acc.process(&StreamEvent::text_delta("def")).unwrap();
        assert_eq!(acc.current_text(), "abcdef");
    }

    // AC-7: is_complete() false before Finish, true after
    #[test]
    fn is_complete_lifecycle() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        assert!(!acc.is_complete());
        acc.process(&make_finish()).unwrap();
        assert!(acc.is_complete());
    }

    // AC-8: ReasoningDelta → Thinking ContentPart in Response
    #[test]
    fn reasoning_delta_produces_thinking_part() {
        let mut acc = StreamAccumulator::new("anthropic", "claude-opus-4-5");
        acc.process(&StreamEvent::reasoning_delta("think1"))
            .unwrap();
        acc.process(&StreamEvent::reasoning_delta("think2"))
            .unwrap();
        acc.process(&StreamEvent::text_delta("answer")).unwrap();
        acc.process(&make_finish()).unwrap();

        let resp = acc.finalize().unwrap();
        assert_eq!(resp.text(), "answer");
        assert_eq!(resp.reasoning(), Some("think1think2".to_string()));
    }

    // AC-9: StreamEventType is non_exhaustive — compile test via wildcard
    #[test]
    fn stream_event_type_non_exhaustive_wildcard() {
        let ev_type = StreamEventType::TextDelta;
        // This must compile, proving non_exhaustive requires wildcard.
        let _ = match ev_type {
            StreamEventType::TextDelta => "text",
            StreamEventType::Finish => "finish",
            _ => "other",
        };
    }

    // AC-10: EventStream compiles as a function return type
    #[test]
    fn event_stream_type_alias_compiles() {
        fn _make_stream() -> EventStream {
            use futures::stream;
            Box::pin(stream::empty())
        }
        let _: EventStream = _make_stream();
    }

    // AC-11: Two parallel tool calls accumulate independently
    #[test]
    fn parallel_tool_calls() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        acc.process(&StreamEvent::tool_call_start("id-A", "fn_a"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_start("id-B", "fn_b"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_delta("id-A", "{\"a\":1}"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_delta("id-B", "{\"b\":2}"))
            .unwrap();
        acc.process(&StreamEvent::tool_call_end(ToolCall {
            id: "id-A".to_string(),
            name: "fn_a".to_string(),
            arguments: serde_json::json!({"a": 1}),
            raw_arguments: None,
        }))
        .unwrap();
        acc.process(&StreamEvent::tool_call_end(ToolCall {
            id: "id-B".to_string(),
            name: "fn_b".to_string(),
            arguments: serde_json::json!({"b": 2}),
            raw_arguments: None,
        }))
        .unwrap();
        acc.process(&make_finish()).unwrap();

        let resp = acc.finalize().unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 2);
        // IDs are distinct
        let ids: Vec<&str> = calls.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"id-A"));
        assert!(ids.contains(&"id-B"));
    }

    // Edge: TextDelta with delta == None → treated as empty, no error
    #[test]
    fn text_delta_none_is_noop() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        let ev = StreamEvent {
            event_type: StreamEventType::TextDelta,
            delta: None,
            text_id: None,
            reasoning_delta: None,
            tool_call: None,
            tool_call_delta: None,
            tool_call_id: None,
            finish_reason: None,
            usage: None,
            response: None,
            error: None,
            raw: None,
        };
        acc.process(&ev).unwrap();
        acc.process(&make_finish()).unwrap();
        let resp = acc.finalize().unwrap();
        // Empty text → no text content part
        assert_eq!(resp.text(), "");
    }

    // Edge: ToolCallEnd before ToolCallStart → Stream error
    #[test]
    fn tool_call_end_without_start_is_error() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        let ev = StreamEvent::tool_call_end(ToolCall {
            id: "no-such-id".to_string(),
            name: "fn".to_string(),
            arguments: serde_json::Value::Null,
            raw_arguments: None,
        });
        let err = acc.process(&ev).unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Stream { .. }));
    }

    // Edge: Finish processed twice → second overwrites first (idempotent)
    #[test]
    fn double_finish_overwrites() {
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        acc.process(&StreamEvent::finish(
            FinishReason::length(),
            Usage::default(),
        ))
        .unwrap();
        acc.process(&StreamEvent::finish(FinishReason::stop(), Usage::default()))
            .unwrap();
        let resp = acc.finalize().unwrap();
        assert!(resp.finish_reason.is_stop());
    }

    // Edge: stream with only reasoning and no text
    #[test]
    fn only_reasoning_no_text() {
        let mut acc = StreamAccumulator::new("anthropic", "claude");
        acc.process(&StreamEvent::reasoning_delta("inner")).unwrap();
        acc.process(&make_finish()).unwrap();
        let resp = acc.finalize().unwrap();
        assert_eq!(resp.text(), "");
        assert_eq!(resp.reasoning(), Some("inner".to_string()));
    }

    // Edge: ProviderEvent → ignored
    #[test]
    fn provider_event_ignored() {
        let ev = StreamEvent::blank(StreamEventType::ProviderEvent);
        let mut acc = StreamAccumulator::new("openai", "gpt-4o");
        acc.process(&ev).unwrap();
        acc.process(&make_finish()).unwrap();
        acc.finalize().unwrap();
    }
}
