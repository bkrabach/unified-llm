//! Test utilities for `unified-llm`.
//!
//! Provides [`MockProviderAdapter`] — a configurable implementation of
//! [`ProviderAdapter`] that returns canned responses without making HTTP calls.
//! Used by all integration tests within this crate and by downstream crates
//! that need deterministic LLM responses in their test suites.
//!
//! # Why unconditional compilation?
//!
//! This module is compiled in both test and non-test builds so that downstream
//! crates can list `unified-llm` in their `[dev-dependencies]` and access
//! `MockProviderAdapter` in their own test code.
//!
//! # Usage
//!
//! ```rust
//! use unified_llm::testing::MockProviderAdapter;
//!
//! let mock = MockProviderAdapter::default()
//!     .push_text_response("Hello!")
//!     .push_text_response("Goodbye!");
//!
//! assert!(!mock.is_exhausted());
//! ```

use std::sync::{Arc, Mutex};

use futures::stream;

use crate::error::UnifiedLlmError;
use crate::providers::ProviderAdapter;
use crate::streaming::{EventStream, StreamEvent};
use crate::types::{
    ContentPart, FinishReason, Message, Request, Response, Role, ToolCallData, Usage,
};

// ---------------------------------------------------------------------------
// MockResponse
// ---------------------------------------------------------------------------

/// A canned response for `complete()`.
// Both variants can be large depending on the payload.  We keep the spec API
// as-is and suppress the size-difference warning.
#[allow(clippy::large_enum_variant)]
pub enum MockResponse {
    /// Return this `Response`.
    Ok(Response),
    /// Return this error.
    Err(UnifiedLlmError),
}

// ---------------------------------------------------------------------------
// MockStreamResponse
// ---------------------------------------------------------------------------

/// A canned stream for `stream()`.
// The Events variant holds a Vec (heap-allocated), but its fat pointer is
// still smaller than the largest UnifiedLlmError variant.  Boxing the error
// would change the public API without improving usability, so we suppress the
// lint here.
#[allow(clippy::large_enum_variant)]
pub enum MockStreamResponse {
    /// Emit these events as an in-memory stream.
    Events(Vec<StreamEvent>),
    /// Return this error immediately (before any events).
    Err(UnifiedLlmError),
}

// ---------------------------------------------------------------------------
// MockCounterHandle
// ---------------------------------------------------------------------------

/// A shareable reference to [`MockProviderAdapter`]'s counter state.
///
/// Obtain via [`MockProviderAdapter::counter_handle()`]. The handle remains
/// valid (and reflects live counter values) even after the adapter is moved
/// into a builder or client.
pub struct MockCounterHandle {
    call_count: Arc<Mutex<u32>>,
    stream_call_count: Arc<Mutex<u32>>,
    initialize_count: Arc<Mutex<u32>>,
    close_count: Arc<Mutex<u32>>,
}

impl MockCounterHandle {
    /// Returns the total number of times `complete()` was called.
    pub fn call_count(&self) -> u32 {
        *self.call_count.lock().unwrap()
    }
    /// Returns the total number of times `stream()` was called.
    pub fn stream_call_count(&self) -> u32 {
        *self.stream_call_count.lock().unwrap()
    }
    /// Returns the total number of times `initialize()` was called.
    pub fn initialize_count(&self) -> u32 {
        *self.initialize_count.lock().unwrap()
    }
    /// Returns the total number of times `close()` was called.
    pub fn close_count(&self) -> u32 {
        *self.close_count.lock().unwrap()
    }
}

// ---------------------------------------------------------------------------
// MockProviderAdapter
// ---------------------------------------------------------------------------

/// A configurable mock implementation of [`ProviderAdapter`] for testing.
///
/// Responses are consumed in FIFO order. When the queue is exhausted,
/// returns a `Configuration` error by default.
///
/// The builder methods (`push_response`, `push_text_response`, etc.) consume
/// and return `self` for chaining:
/// ```rust
/// use unified_llm::testing::MockProviderAdapter;
///
/// let mock = MockProviderAdapter::default()
///     .push_text_response("first")
///     .push_text_response("second");
/// ```
pub struct MockProviderAdapter {
    name: String,
    responses: Arc<Mutex<Vec<MockResponse>>>,
    stream_responses: Arc<Mutex<Vec<MockStreamResponse>>>,
    call_count: Arc<Mutex<u32>>,
    stream_call_count: Arc<Mutex<u32>>,
    initialize_count: Arc<Mutex<u32>>,
    close_count: Arc<Mutex<u32>>,
    recorded_requests: Arc<Mutex<Vec<Request>>>,
    /// When the response queue is empty, call this function to produce an error.
    exhausted_error: Arc<dyn Fn() -> UnifiedLlmError + Send + Sync>,
}

impl MockProviderAdapter {
    /// Create a new mock with the given provider name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            responses: Arc::new(Mutex::new(Vec::new())),
            stream_responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
            stream_call_count: Arc::new(Mutex::new(0)),
            initialize_count: Arc::new(Mutex::new(0)),
            close_count: Arc::new(Mutex::new(0)),
            recorded_requests: Arc::new(Mutex::new(Vec::new())),
            exhausted_error: Arc::new(|| UnifiedLlmError::Configuration {
                message: "mock response queue exhausted".to_string(),
            }),
        }
    }

    /// Returns a [`MockCounterHandle`] that shares live counter state with
    /// this adapter. The handle remains valid after the adapter is moved.
    pub fn counter_handle(&self) -> MockCounterHandle {
        MockCounterHandle {
            call_count: Arc::clone(&self.call_count),
            stream_call_count: Arc::clone(&self.stream_call_count),
            initialize_count: Arc::clone(&self.initialize_count),
            close_count: Arc::clone(&self.close_count),
        }
    }

    /// Internal: a handle that also exposes counters.
    /// Used in client.rs tests to satisfy AC-9 requirement.
    pub fn clone_counters(&self) -> MockCounterHandle {
        self.counter_handle()
    }

    /// Returns a shared handle to the recorded-request log.
    ///
    /// The handle remains valid (and reflects live recorded requests) even
    /// after the adapter is moved into a builder or client. Useful in tests
    /// that need to inspect the outgoing [`Request`] after the adapter has
    /// been moved.
    pub fn request_log_handle(&self) -> Arc<Mutex<Vec<Request>>> {
        Arc::clone(&self.recorded_requests)
    }

    /// Queue a successful `Response` to be returned by the next `complete()`.
    pub fn push_response(self, response: Response) -> Self {
        self.responses
            .lock()
            .unwrap()
            .push(MockResponse::Ok(response));
        self
    }

    /// Queue a text `Response` built from `text`.
    ///
    /// Sets `model = "mock-model"`, `provider = self.name`,
    /// `id = "mock-response-{n}"`, `finish_reason = stop()`, and usage based
    /// on word count.
    pub fn push_text_response(self, text: &str) -> Self {
        let n = self.responses.lock().unwrap().len() + 1;
        let output_tokens = text.split_whitespace().count() as u32;
        let response = Response {
            id: format!("mock-response-{n}"),
            model: "mock-model".to_string(),
            provider: self.name.clone(),
            message: Message::assistant(text),
            finish_reason: FinishReason::stop(),
            usage: Usage {
                input_tokens: 10,
                output_tokens,
                total_tokens: 10 + output_tokens,
                ..Default::default()
            },
            raw: None,
            warnings: vec![],
            rate_limit: None,
        };
        self.push_response(response)
    }

    /// Queue an error to be returned by the next `complete()`.
    pub fn push_error(self, error: UnifiedLlmError) -> Self {
        self.responses
            .lock()
            .unwrap()
            .push(MockResponse::Err(error));
        self
    }

    /// Queue a stream that will emit the given events.
    pub fn push_stream_events(self, events: Vec<StreamEvent>) -> Self {
        self.stream_responses
            .lock()
            .unwrap()
            .push(MockStreamResponse::Events(events));
        self
    }

    /// Queue a stream that emits `text` as a series of up-to-5-character chunks.
    ///
    /// Event sequence: `StreamStart`, `TextStart`, N×`TextDelta`, `TextEnd`, `Finish`.
    ///
    /// Chunks `text` into pieces of at most 5 characters so that accumulator
    /// logic is exercised across multiple deltas.
    pub fn push_text_stream(self, text: &str) -> Self {
        let mut events = vec![StreamEvent::stream_start()];

        if !text.is_empty() {
            events.push(StreamEvent::text_start());

            // Chunk into pieces of ≤ 5 characters.
            let mut chars = text.chars().peekable();
            while chars.peek().is_some() {
                let chunk: String = chars.by_ref().take(5).collect();
                events.push(StreamEvent::text_delta(chunk));
            }

            events.push(StreamEvent::text_end());
        }

        let word_count = text.split_whitespace().count() as u32;
        let usage = Usage {
            input_tokens: 10,
            output_tokens: word_count,
            total_tokens: 10 + word_count,
            ..Default::default()
        };
        events.push(StreamEvent::finish(FinishReason::stop(), usage));

        self.push_stream_events(events)
    }

    /// Queue a stream error (returned before any events).
    pub fn push_stream_error(self, error: UnifiedLlmError) -> Self {
        self.stream_responses
            .lock()
            .unwrap()
            .push(MockStreamResponse::Err(error));
        self
    }

    /// Returns the total number of times `complete()` has been called.
    // Not impl Default directly in the inherent impl to avoid the
    // should_implement_trait clippy warning.
    pub fn call_count(&self) -> u32 {
        *self.call_count.lock().unwrap()
    }

    /// Returns the total number of times `stream()` has been called.
    pub fn stream_call_count(&self) -> u32 {
        *self.stream_call_count.lock().unwrap()
    }

    /// Returns the total number of times `initialize()` has been called.
    pub fn initialize_count(&self) -> u32 {
        *self.initialize_count.lock().unwrap()
    }

    /// Returns the total number of times `close()` has been called.
    pub fn close_count(&self) -> u32 {
        *self.close_count.lock().unwrap()
    }

    /// Returns all requests received by `complete()` in call order.
    pub fn recorded_requests(&self) -> Vec<Request> {
        self.recorded_requests.lock().unwrap().clone()
    }

    /// Clears both response queues and resets all counters to zero.
    pub fn reset(&mut self) {
        *self.responses.lock().unwrap() = Vec::new();
        *self.stream_responses.lock().unwrap() = Vec::new();
        *self.call_count.lock().unwrap() = 0;
        *self.stream_call_count.lock().unwrap() = 0;
        *self.initialize_count.lock().unwrap() = 0;
        *self.close_count.lock().unwrap() = 0;
        *self.recorded_requests.lock().unwrap() = Vec::new();
    }

    /// Returns `true` if the `complete()` response queue is empty.
    pub fn is_exhausted(&self) -> bool {
        self.responses.lock().unwrap().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Default impl (satisfies the should_implement_trait lint)
// ---------------------------------------------------------------------------

impl Default for MockProviderAdapter {
    /// Create a mock with the default name `"mock"`.
    fn default() -> Self {
        Self::new("mock")
    }
}

// ---------------------------------------------------------------------------
// ProviderAdapter implementation
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ProviderAdapter for MockProviderAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, request: &Request) -> Result<Response, UnifiedLlmError> {
        // Always increment call_count.
        *self.call_count.lock().unwrap() += 1;
        // Record the request.
        self.recorded_requests.lock().unwrap().push(request.clone());

        // Pop the front of the queue (FIFO).
        let front = {
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                None
            } else {
                Some(q.remove(0))
            }
        };
        match front {
            Some(MockResponse::Ok(resp)) => Ok(resp),
            Some(MockResponse::Err(e)) => Err(e),
            None => Err((self.exhausted_error)()),
        }
    }

    async fn stream(&self, _request: &Request) -> Result<EventStream, UnifiedLlmError> {
        *self.stream_call_count.lock().unwrap() += 1;

        let front = self.stream_responses.lock().unwrap().drain(0..1).next();
        match front {
            Some(MockStreamResponse::Events(events)) => {
                let mapped: Vec<Result<StreamEvent, UnifiedLlmError>> =
                    events.into_iter().map(Ok).collect();
                Ok(Box::pin(stream::iter(mapped)))
            }
            Some(MockStreamResponse::Err(e)) => Err(e),
            None => Err((self.exhausted_error)()),
        }
    }

    async fn initialize(&self) -> Result<(), UnifiedLlmError> {
        *self.initialize_count.lock().unwrap() += 1;
        Ok(())
    }

    async fn close(&self) -> Result<(), UnifiedLlmError> {
        *self.close_count.lock().unwrap() += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free-function helpers
// ---------------------------------------------------------------------------

/// Build a minimal valid `Response` with the given text.
///
/// `provider = "mock"`, `model = "mock-model"`, `id = "mock-{uuid}"`.
pub fn make_text_response(text: &str) -> Response {
    Response {
        id: format!("mock-{}", uuid::Uuid::new_v4()),
        model: "mock-model".to_string(),
        provider: "mock".to_string(),
        message: Message::assistant(text),
        finish_reason: FinishReason::stop(),
        usage: Usage::default(),
        raw: None,
        warnings: vec![],
        rate_limit: None,
    }
}

/// Build a minimal valid `Response` carrying tool calls.
///
/// `calls` is a vec of `(call_id, function_name, arguments)` tuples.
pub fn make_tool_call_response(calls: Vec<(String, String, serde_json::Value)>) -> Response {
    let content: Vec<ContentPart> = calls
        .into_iter()
        .map(|(id, name, arguments)| {
            ContentPart::tool_call(ToolCallData {
                id,
                name,
                arguments,
                raw_arguments: None,
            })
        })
        .collect();

    Response {
        id: format!("mock-{}", uuid::Uuid::new_v4()),
        model: "mock-model".to_string(),
        provider: "mock".to_string(),
        message: Message {
            role: Role::Assistant,
            content,
            name: None,
            tool_call_id: None,
        },
        finish_reason: FinishReason::tool_calls(),
        usage: Usage::default(),
        raw: None,
        warnings: vec![],
        rate_limit: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::StreamAccumulator;
    use futures::StreamExt as _;

    fn make_req() -> Request {
        Request::new("mock-model", vec![Message::user("hi")])
    }

    // AC-1: push_text_response + complete returns Response with correct text
    #[tokio::test]
    async fn push_text_response_complete() {
        let mock = MockProviderAdapter::default().push_text_response("hello");
        let resp = mock.complete(&make_req()).await.unwrap();
        assert_eq!(resp.text(), "hello");
    }

    // AC-2: call_count increments after each complete()
    #[tokio::test]
    async fn call_count_increments() {
        let mock = MockProviderAdapter::default()
            .push_text_response("a")
            .push_text_response("b");
        mock.complete(&make_req()).await.unwrap();
        mock.complete(&make_req()).await.unwrap();
        assert_eq!(mock.call_count(), 2);
    }

    // AC-3: exhausted mock returns Err(Configuration)
    #[tokio::test]
    async fn exhausted_returns_configuration_error() {
        let mock = MockProviderAdapter::default();
        let err = mock.complete(&make_req()).await.unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Configuration { .. }));
        assert_eq!(mock.call_count(), 1, "call_count still increments");
    }

    // AC-4: push_error causes complete() to return that exact error type
    #[tokio::test]
    async fn push_error_returned() {
        let mock = MockProviderAdapter::default().push_error(UnifiedLlmError::RateLimit {
            provider: "test".to_string(),
            message: "slow down".to_string(),
            retry_after: Some(5.0),
        });
        let err = mock.complete(&make_req()).await.unwrap_err();
        assert!(matches!(err, UnifiedLlmError::RateLimit { .. }));
    }

    // AC-5: push_text_stream + StreamAccumulator finalize yields correct text
    #[tokio::test]
    async fn push_text_stream_accumulates() {
        let mock = MockProviderAdapter::default().push_text_stream("hello world");
        let mut event_stream = mock.stream(&make_req()).await.unwrap();

        let mut acc = StreamAccumulator::new("mock", "mock-model");
        while let Some(item) = event_stream.next().await {
            let ev = item.unwrap();
            acc.process(&ev).unwrap();
        }
        let resp = acc.finalize().unwrap();
        assert_eq!(resp.text(), "hello world");
    }

    // AC-6: push_stream_error causes stream() to return Err
    #[tokio::test]
    async fn push_stream_error_returned() {
        let mock = MockProviderAdapter::default().push_stream_error(UnifiedLlmError::Network {
            message: "gone".to_string(),
            source: None,
        });
        let result = mock.stream(&make_req()).await;
        let err = result.err().expect("expected Err from stream()");
        assert!(matches!(err, UnifiedLlmError::Network { .. }));
        assert_eq!(mock.stream_call_count(), 1);
    }

    // AC-7: two queued responses returned in FIFO order
    #[tokio::test]
    async fn fifo_order() {
        let mock = MockProviderAdapter::default()
            .push_text_response("first")
            .push_text_response("second");
        let r1 = mock.complete(&make_req()).await.unwrap();
        let r2 = mock.complete(&make_req()).await.unwrap();
        assert_eq!(r1.text(), "first");
        assert_eq!(r2.text(), "second");
    }

    // AC-8: recorded_requests returns requests in call order
    #[tokio::test]
    async fn recorded_requests_in_order() {
        let mock = MockProviderAdapter::default()
            .push_text_response("a")
            .push_text_response("b");

        let req1 = Request::new("gpt-4o", vec![Message::user("first")]);
        let req2 = Request::new("claude", vec![Message::user("second")]);
        mock.complete(&req1).await.unwrap();
        mock.complete(&req2).await.unwrap();

        let recorded = mock.recorded_requests();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].model, "gpt-4o");
        assert_eq!(recorded[1].model, "claude");
    }

    // AC-9: initialize_count is 1 after initialize()
    #[tokio::test]
    async fn initialize_count_increments() {
        let mock = MockProviderAdapter::default();
        assert_eq!(mock.initialize_count(), 0);
        mock.initialize().await.unwrap();
        assert_eq!(mock.initialize_count(), 1);
    }

    // AC-10: MockProviderAdapter is Send + Sync
    #[test]
    fn mock_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockProviderAdapter>();
    }

    // AC-11: make_text_response("hi").text() == "hi"
    #[test]
    fn make_text_response_text() {
        assert_eq!(make_text_response("hi").text(), "hi");
    }

    // Edge: recorded_requests before any calls → empty
    #[test]
    fn recorded_requests_empty_initially() {
        let mock = MockProviderAdapter::default();
        assert!(mock.recorded_requests().is_empty());
    }

    // Edge: reset() clears queues and counters
    #[tokio::test]
    async fn reset_clears_everything() {
        let mut mock = MockProviderAdapter::default()
            .push_text_response("x")
            .push_text_response("y");
        mock.complete(&make_req()).await.unwrap();
        mock.reset();
        assert_eq!(mock.call_count(), 0);
        assert!(mock.is_exhausted());
        assert!(mock.recorded_requests().is_empty());
    }

    // Edge: is_exhausted() false when queue has items, true when empty
    #[test]
    fn is_exhausted_logic() {
        let mock = MockProviderAdapter::default();
        assert!(mock.is_exhausted());
        let mock = mock.push_text_response("x");
        assert!(!mock.is_exhausted());
    }

    // Edge: push_text_stream("") → StreamStart + Finish only, no TextDelta
    #[tokio::test]
    async fn empty_text_stream_no_delta() {
        let mock = MockProviderAdapter::default().push_text_stream("");
        let mut event_stream = mock.stream(&make_req()).await.unwrap();

        let mut acc = StreamAccumulator::new("mock", "mock-model");
        while let Some(item) = event_stream.next().await {
            let ev = item.unwrap();
            acc.process(&ev).unwrap();
        }
        let resp = acc.finalize().unwrap();
        assert_eq!(resp.text(), "");
    }

    // Mixed queue: Ok, Err, Ok
    #[tokio::test]
    async fn mixed_queue_ok_err_ok() {
        let mock = MockProviderAdapter::default()
            .push_text_response("ok1")
            .push_error(UnifiedLlmError::RateLimit {
                provider: "mock".to_string(),
                message: "throttled".to_string(),
                retry_after: None,
            })
            .push_text_response("ok2");

        let r1 = mock.complete(&make_req()).await.unwrap();
        let e = mock.complete(&make_req()).await.unwrap_err();
        let r2 = mock.complete(&make_req()).await.unwrap();

        assert_eq!(r1.text(), "ok1");
        assert!(matches!(e, UnifiedLlmError::RateLimit { .. }));
        assert_eq!(r2.text(), "ok2");
    }

    // make_tool_call_response
    #[test]
    fn make_tool_call_response_has_tool_calls() {
        let calls = vec![(
            "call-1".to_string(),
            "my_fn".to_string(),
            serde_json::json!({"key": "val"}),
        )];
        let resp = make_tool_call_response(calls);
        assert_eq!(resp.tool_calls().len(), 1);
        assert_eq!(resp.tool_calls()[0].name, "my_fn");
    }

    // close_count increments
    #[tokio::test]
    async fn close_count_increments() {
        let mock = MockProviderAdapter::default();
        mock.close().await.unwrap();
        assert_eq!(mock.close_count(), 1);
    }

    // counter_handle stays valid after mock is used
    #[tokio::test]
    async fn counter_handle_reflects_live_state() {
        let mock = MockProviderAdapter::default().push_text_response("x");
        let handle = mock.counter_handle();
        assert_eq!(handle.call_count(), 0);
        mock.complete(&make_req()).await.unwrap();
        assert_eq!(handle.call_count(), 1);
    }
}
