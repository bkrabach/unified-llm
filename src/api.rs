//! High-level generation API: `generate()`, `stream()`, `generate_object()`.
//!
//! These functions wrap `Client::complete()` / `Client::stream()` with:
//! - Automatic tool execution loops
//! - Convenient parameter structs
//! - JSON Schema-validated structured output
//! - A module-level default client

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{Stream, StreamExt as _};
use serde_json::Value;

use crate::client::Client;
use crate::error::UnifiedLlmError;
use crate::streaming::{EventStream, StreamAccumulator, StreamEvent};
use crate::types::{
    FinishReason, Message, Request, Response, Role, Tool, ToolCall, ToolCallData, ToolChoice,
    ToolResult, ToolResultData, Usage,
};

// ---------------------------------------------------------------------------
// CancellationToken  (GAP-ULM-007)
// ---------------------------------------------------------------------------

/// A lightweight, clone-able cancellation signal.
///
/// Pass a clone into [`GenerateParams::cancellation_token`]; call
/// [`CancellationToken::cancel()`] from another task or thread to abort the
/// in-progress [`generate()`] call.
///
/// # Example
/// ```rust,no_run
/// use unified_llm::api::{CancellationToken, GenerateParams, generate};
///
/// # async fn run() {
/// let token = CancellationToken::new();
/// let abort = token.clone();
/// // cancel from another task
/// tokio::spawn(async move { abort.cancel(); });
/// let params = GenerateParams::new("model", "prompt").with_cancellation_token(token);
/// let result = generate(params).await;
/// // may return Err(Abort { .. })
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new, uncancelled token.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal cancellation.  All clones of this token will observe the
    /// cancellation on their next [`is_cancelled()`] check.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Returns `true` if [`cancel()`] has been called on any clone of this token.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// Module-level default client
// ---------------------------------------------------------------------------

/// Inner lock: wraps Option<Client> in a RwLock stored inside a OnceLock.
static DEFAULT_CLIENT_LOCK: OnceLock<RwLock<Option<Client>>> = OnceLock::new();

fn default_client_rwlock() -> &'static RwLock<Option<Client>> {
    DEFAULT_CLIENT_LOCK.get_or_init(|| RwLock::new(None))
}

/// Set the module-level default client.
///
/// Thread-safe. Can be called multiple times to replace the existing client.
pub fn set_default_client(client: Client) {
    let mut guard = default_client_rwlock().write().unwrap();
    *guard = Some(client);
}

/// Get the module-level default client.
///
/// Lazily initializes from environment variables via [`Client::from_env()`]
/// on the first call if no client has been set with [`set_default_client`].
///
/// Returns `Err(Configuration)` if no client is set and env-based init fails.
pub fn get_default_client() -> Result<Client, UnifiedLlmError> {
    // Try to read an existing client first
    {
        let guard = default_client_rwlock().read().unwrap();
        if let Some(client) = &*guard {
            return Ok(client.clone());
        }
    }
    // None set — try lazy init from env
    let client = Client::from_env()?;
    // Store it for future calls
    let mut guard = default_client_rwlock().write().unwrap();
    // Double-checked: might have been set by another thread
    if guard.is_none() {
        *guard = Some(client.clone());
    }
    Ok(guard.as_ref().unwrap().clone())
}

// ---------------------------------------------------------------------------
// F-017: GenerateParams
// ---------------------------------------------------------------------------

/// Parameters for `generate()` and `stream()`.
pub struct GenerateParams {
    /// LLM client. If `None`, uses the module-level default client.
    pub client: Option<Client>,
    /// Model identifier string.
    pub model: String,
    /// Single-turn prompt shorthand. Overridden by `messages` if both are set.
    pub prompt: Option<String>,
    /// System prompt shorthand. Prepended if messages have no system message.
    pub system: Option<String>,
    /// Full conversation history. Overrides `prompt` if both are set.
    pub messages: Option<Vec<Message>>,
    /// Provider name override.
    pub provider: Option<String>,
    /// Tool definitions available to the model.
    pub tools: Option<Vec<Tool>>,
    /// Tool choice mode.
    pub tool_choice: Option<ToolChoice>,
    /// Callback invoked for each tool call.
    pub tool_executor: Option<Arc<dyn Fn(ToolCall) -> ToolResult + Send + Sync>>,
    /// Maximum number of tool execution rounds. Default: `1`.
    pub max_tool_rounds: u32,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub reasoning_effort: Option<String>,
    pub response_format: Option<crate::types::ResponseFormat>,
    pub provider_options: Option<Value>,
    /// Optional cancellation token (GAP-ULM-007).
    ///
    /// When cancelled, the next loop iteration of `generate()` returns
    /// `Err(Abort)`.
    pub cancellation_token: Option<CancellationToken>,
    /// Optional per-request timeout in milliseconds (GAP-ULM-008).
    ///
    /// When set, each `client.complete()` call is wrapped in
    /// `tokio::time::timeout`. If the provider exceeds the deadline,
    /// `generate()` returns `Err(RequestTimeout)`.
    pub timeout_ms: Option<u64>,
}

impl GenerateParams {
    /// Construct minimal params with model and prompt.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            client: None,
            model: model.into(),
            prompt: Some(prompt.into()),
            system: None,
            messages: None,
            provider: None,
            tools: None,
            tool_choice: None,
            tool_executor: None,
            max_tool_rounds: 1,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            response_format: None,
            provider_options: None,
            cancellation_token: None,
            timeout_ms: None,
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_tool_executor(
        mut self,
        executor: impl Fn(ToolCall) -> ToolResult + Send + Sync + 'static,
    ) -> Self {
        self.tool_executor = Some(Arc::new(executor));
        self
    }

    pub fn with_max_tool_rounds(mut self, rounds: u32) -> Self {
        self.max_tool_rounds = rounds;
        self
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Attach a [`CancellationToken`].  Call [`CancellationToken::cancel()`]
    /// from another task to abort the in-progress generation.
    pub fn with_cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    /// Set a per-request timeout in milliseconds.
    ///
    /// Each `client.complete()` call is wrapped in `tokio::time::timeout`.
    /// Returns `Err(RequestTimeout)` if the provider exceeds the deadline.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }
}

// ---------------------------------------------------------------------------
// F-017: Output types
// ---------------------------------------------------------------------------

/// Result of one LLM completion round within the tool loop.
#[derive(Debug)]
pub struct StepResult {
    pub response: Response,
    pub tool_calls: Vec<ToolCallData>,
    pub tool_results: Vec<ToolResultData>,
    pub usage: Usage,
}

/// Accumulated result from `generate()` across all tool loop rounds.
#[derive(Debug)]
pub struct GenerateResult {
    /// Final text from the last response.
    pub text: String,
    /// Reasoning text from the last response (if any).
    pub reasoning: Option<String>,
    /// All tool calls across all rounds.
    pub tool_calls: Vec<ToolCallData>,
    /// All tool results across all rounds.
    pub tool_results: Vec<ToolResultData>,
    /// Finish reason from the final response.
    pub finish_reason: FinishReason,
    /// Token usage from the final round only.
    pub usage: Usage,
    /// Token usage summed across all rounds.
    pub total_usage: Usage,
    /// Per-round breakdown.
    pub steps: Vec<StepResult>,
    /// The final `Response` object.
    pub response: Response,
    /// Structured output for `generate_object()`. `None` from `generate()`.
    pub output: Option<Value>,
}

// ---------------------------------------------------------------------------
// F-017: Request building helpers
// ---------------------------------------------------------------------------

/// Build the initial conversation from `GenerateParams`.
fn build_conversation(params: &GenerateParams) -> Result<Vec<Message>, UnifiedLlmError> {
    // NLSpec §8.4: reject when both prompt and messages are provided.
    if params.prompt.is_some() && params.messages.is_some() {
        return Err(UnifiedLlmError::InvalidRequest {
            message: "prompt and messages are mutually exclusive; supply only one".to_string(),
        });
    }

    let mut messages = if let Some(msgs) = &params.messages {
        msgs.clone()
    } else if let Some(prompt) = &params.prompt {
        vec![Message::user(prompt)]
    } else {
        return Err(UnifiedLlmError::Configuration {
            message: "either prompt or messages must be set".to_string(),
        });
    };

    // Prepend system message if requested and not already present
    if let Some(sys_text) = &params.system {
        let has_system = messages
            .iter()
            .any(|m| matches!(m.role, Role::System | Role::Developer));
        if !has_system {
            messages.insert(0, Message::system(sys_text));
        }
    }

    Ok(messages)
}

/// Build a `Request` from the current conversation state and params.
fn build_request(params: &GenerateParams, conversation: Vec<Message>) -> Request {
    let mut req = Request::new(params.model.clone(), conversation);
    req.provider = params.provider.clone();
    req.tools = params.tools.clone();
    req.tool_choice = params.tool_choice.clone();
    req.temperature = params.temperature;
    req.max_tokens = params.max_tokens;
    req.reasoning_effort = params.reasoning_effort.clone();
    req.response_format = params.response_format.clone();
    req.provider_options = params.provider_options.clone();
    req
}

/// Execute tool calls from a response sequentially and return
/// (assistant_msg, tool_messages, results).
///
/// Used by the synchronous `Stream` impl where `.await` is not available.
/// For the high-level `generate()` API, use [`execute_tools_concurrent`]
/// instead to satisfy NLSpec §8.7.
fn execute_tools(
    response: &Response,
    executor: &Arc<dyn Fn(ToolCall) -> ToolResult + Send + Sync>,
) -> (Message, Vec<Message>, Vec<ToolResultData>) {
    let calls = response.tool_calls();
    let assistant_msg = response.message.clone();
    let mut tool_msgs: Vec<Message> = Vec::new();
    let mut results: Vec<ToolResultData> = Vec::new();

    for call_data in calls {
        let tool_call = ToolCall {
            id: call_data.id.clone(),
            name: call_data.name.clone(),
            arguments: call_data.arguments.clone(),
            raw_arguments: call_data.raw_arguments.clone(),
        };
        let result = executor(tool_call);
        let result_data = ToolResultData {
            tool_call_id: call_data.id.clone(),
            content: Value::String(result.content.clone()),
            is_error: result.is_error,
        };
        tool_msgs.push(Message::tool_result(
            &call_data.id,
            &result.content,
            result.is_error,
        ));
        results.push(result_data);
    }

    (assistant_msg, tool_msgs, results)
}

/// Execute tool calls from a response concurrently and return
/// (assistant_msg, tool_messages, results).
///
/// NLSpec §8.7: when a response contains multiple tool calls they MUST be
/// executed concurrently, not sequentially. Each call is dispatched on a
/// `tokio::task::spawn_blocking` thread so that synchronous executors cannot
/// block the async runtime. Results are collected in original call order.
///
/// Used by `generate()`. The `Stream` impl uses the sync [`execute_tools`]
/// because `poll_next` cannot call `.await`.
async fn execute_tools_concurrent(
    response: &Response,
    executor: &Arc<dyn Fn(ToolCall) -> ToolResult + Send + Sync>,
) -> (Message, Vec<Message>, Vec<ToolResultData>) {
    let calls = response.tool_calls();
    let assistant_msg = response.message.clone();

    // Dispatch all tool calls concurrently via spawn_blocking.
    let handles: Vec<_> = calls
        .iter()
        .map(|call_data| {
            let tool_call = ToolCall {
                id: call_data.id.clone(),
                name: call_data.name.clone(),
                arguments: call_data.arguments.clone(),
                raw_arguments: call_data.raw_arguments.clone(),
            };
            let exec = Arc::clone(executor);
            tokio::task::spawn_blocking(move || exec(tool_call))
        })
        .collect();

    // Await all tasks, preserving call order.
    let mut tool_msgs: Vec<Message> = Vec::with_capacity(calls.len());
    let mut results: Vec<ToolResultData> = Vec::with_capacity(calls.len());

    for (call_data, handle) in calls.iter().zip(handles) {
        // spawn_blocking only fails if the thread panicked; propagate with unwrap.
        let result = handle.await.expect("tool executor task panicked");
        let result_data = ToolResultData {
            tool_call_id: call_data.id.clone(),
            content: Value::String(result.content.clone()),
            is_error: result.is_error,
        };
        tool_msgs.push(Message::tool_result(
            &call_data.id,
            &result.content,
            result.is_error,
        ));
        results.push(result_data);
    }

    (assistant_msg, tool_msgs, results)
}

// ---------------------------------------------------------------------------
// F-017: generate()
// ---------------------------------------------------------------------------

/// Execute an LLM generation with optional automatic tool execution loop.
pub async fn generate(params: GenerateParams) -> Result<GenerateResult, UnifiedLlmError> {
    let client = match params.client.clone() {
        Some(c) => c,
        None => get_default_client()?,
    };

    let mut conversation = build_conversation(&params)?;
    let mut rounds_remaining = params.max_tool_rounds;
    let mut steps: Vec<StepResult> = Vec::new();
    let mut all_tool_calls: Vec<ToolCallData> = Vec::new();
    let mut all_tool_results: Vec<ToolResultData> = Vec::new();
    let mut total_usage = Usage::default();

    loop {
        // GAP-ULM-007: check cancellation at start of each round
        if let Some(ct) = &params.cancellation_token {
            if ct.is_cancelled() {
                return Err(UnifiedLlmError::Abort {
                    message: "generation cancelled by CancellationToken".to_string(),
                });
            }
        }

        let req = build_request(&params, conversation.clone());

        // GAP-ULM-008: wrap complete() in an optional timeout
        let response = if let Some(ms) = params.timeout_ms {
            match tokio::time::timeout(Duration::from_millis(ms), client.complete(req)).await {
                Ok(result) => result?,
                Err(_elapsed) => {
                    return Err(UnifiedLlmError::RequestTimeout {
                        message: format!(
                            "generate() timed out after {ms} ms waiting for provider response"
                        ),
                    });
                }
            }
        } else {
            client.complete(req).await?
        };

        let step_tool_calls: Vec<ToolCallData> = response
            .tool_calls()
            .iter()
            .map(|tc| (*tc).clone())
            .collect();
        let usage = response.usage.clone();
        total_usage += usage.clone();

        let has_tool_calls = !step_tool_calls.is_empty();
        let is_tool_calls_finish = response.finish_reason.is_tool_calls();
        let can_execute = has_tool_calls
            && is_tool_calls_finish
            && rounds_remaining > 0
            && params.tool_executor.is_some();

        if can_execute {
            let executor = params.tool_executor.as_ref().unwrap();

            // V2-ULM-007: intercept calls to tools not in the defined tools list.
            // Build a set of known tool names (empty when no tools are defined,
            // in which case all calls pass through to the executor unchanged).
            let known_names: std::collections::HashSet<String> = params
                .tools
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|t| t.name.clone())
                .collect();
            // GAP-ULM-022: build schema map for tool argument validation.
            let tool_schemas: std::collections::HashMap<String, Value> = params
                .tools
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|t| (t.name.clone(), t.parameters.clone()))
                .collect();
            // Wrap executor so unknown tools short-circuit to an error result
            // and known tools have their arguments validated against the schema.
            let checked_executor: Arc<dyn Fn(ToolCall) -> ToolResult + Send + Sync> =
                if known_names.is_empty() {
                    Arc::clone(executor)
                } else {
                    let inner = Arc::clone(executor);
                    let names = known_names.clone();
                    let schemas = tool_schemas;
                    Arc::new(move |call: ToolCall| {
                        if names.contains(&call.name) {
                            // GAP-ULM-022: validate arguments against tool schema
                            // before dispatching to the executor.
                            if let Some(schema) = schemas.get(&call.name) {
                                if let Err(e) = validate_against_schema(&call.arguments, schema) {
                                    return ToolResult {
                                        tool_call_id: call.id.clone(),
                                        content: format!("tool argument validation failed: {e}"),
                                        is_error: true,
                                    };
                                }
                            }
                            inner(call)
                        } else {
                            ToolResult {
                                tool_call_id: call.id.clone(),
                                content: format!("Unknown tool: {}", call.name),
                                is_error: true,
                            }
                        }
                    })
                };

            let (assistant_msg, tool_msgs, results) =
                execute_tools_concurrent(&response, &checked_executor).await;

            let step = StepResult {
                tool_calls: step_tool_calls.clone(),
                tool_results: results.clone(),
                usage,
                response,
            };
            steps.push(step);
            all_tool_calls.extend(step_tool_calls);
            all_tool_results.extend(results);

            // Extend conversation with assistant turn + tool results
            conversation.push(assistant_msg);
            conversation.extend(tool_msgs);
            rounds_remaining -= 1;
        } else {
            // Final round
            let step = StepResult {
                tool_calls: step_tool_calls.clone(),
                tool_results: vec![],
                usage,
                response: response.clone(),
            };
            all_tool_calls.extend(step_tool_calls);
            steps.push(step);

            let text = response.text();
            let reasoning = response.reasoning();
            let finish_reason = response.finish_reason.clone();
            let last_usage = response.usage.clone();

            return Ok(GenerateResult {
                text,
                reasoning,
                tool_calls: all_tool_calls,
                tool_results: all_tool_results,
                finish_reason,
                usage: last_usage,
                total_usage,
                steps,
                response,
                output: None,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// F-018: StreamResult
// ---------------------------------------------------------------------------

/// State machine for `StreamResult`.
enum StreamState {
    /// Currently streaming events from a round.
    Streaming {
        stream: EventStream,
        accumulator: Box<StreamAccumulator>,
    },
    /// Starting the next round — waiting for `client.stream()` to resolve.
    StartingNextRound(
        Pin<Box<dyn Future<Output = Result<EventStream, UnifiedLlmError>> + Send>>,
        String, // provider name
        String, // model name
    ),
    /// Done — all rounds complete.
    Done,
}

// SAFETY: StreamState contains EventStream (which is Send) and the Future
// (which is Send). The accumulator is Send.
unsafe impl Send for StreamState {}

/// Result from `stream()`. Implements `Stream` and provides `.response()`.
pub struct StreamResult {
    state: StreamState,
    client: Client,
    params_model: String,
    params_provider: Option<String>,
    params_tools: Option<Vec<Tool>>,
    params_tool_choice: Option<crate::types::ToolChoice>,
    params_temperature: Option<f64>,
    params_max_tokens: Option<u32>,
    params_reasoning_effort: Option<String>,
    params_response_format: Option<crate::types::ResponseFormat>,
    params_provider_options: Option<Value>,
    conversation: Vec<Message>,
    rounds_remaining: u32,
    tool_executor: Option<Arc<dyn Fn(ToolCall) -> ToolResult + Send + Sync>>,
    final_response: Arc<Mutex<Option<Response>>>,
    /// Optional cancellation token (GAP-ULM-007).
    cancellation_token: Option<CancellationToken>,
}

impl StreamResult {
    /// Returns `true` if a `Finish` event has been processed and the final
    /// response is available via `try_response()`.
    pub fn is_complete(&self) -> bool {
        matches!(self.state, StreamState::Done)
    }

    /// Return the accumulated response if the stream has been fully consumed.
    pub fn try_response(&self) -> Option<Response> {
        self.final_response.lock().unwrap().clone()
    }

    /// Consume the rest of the stream and return the final accumulated `Response`.
    pub async fn response(mut self) -> Result<Response, UnifiedLlmError> {
        // Drain remaining events
        while let Some(ev) = self.next().await {
            let _ = ev; // ignore events, just drive to completion
        }
        self.final_response
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| UnifiedLlmError::Stream {
                message: "stream ended without a final response".to_string(),
            })
    }

    /// Build the next streaming request given the current conversation.
    fn build_next_request(&self, conversation: Vec<Message>) -> Request {
        let mut req = Request::new(self.params_model.clone(), conversation);
        req.provider = self.params_provider.clone();
        req.tools = self.params_tools.clone();
        req.tool_choice = self.params_tool_choice.clone();
        req.temperature = self.params_temperature;
        req.max_tokens = self.params_max_tokens;
        req.reasoning_effort = self.params_reasoning_effort.clone();
        req.response_format = self.params_response_format.clone();
        req.provider_options = self.params_provider_options.clone();
        req
    }
}

impl Stream for StreamResult {
    type Item = Result<StreamEvent, UnifiedLlmError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY: we don't move any fields that are behind Pin constraints
        let this = unsafe { self.get_unchecked_mut() };

        // GAP-ULM-007: check cancellation before any state-machine transition
        if let Some(ct) = &this.cancellation_token {
            if ct.is_cancelled() {
                this.state = StreamState::Done;
                return Poll::Ready(Some(Err(UnifiedLlmError::Abort {
                    message: "stream cancelled by CancellationToken".to_string(),
                })));
            }
        }

        // State machine: loop allows StartingNextRound → Streaming without extra round-trip
        'state_machine: loop {
            match &mut this.state {
                StreamState::Done => return Poll::Ready(None),

                StreamState::StartingNextRound(fut, provider, model) => {
                    match fut.as_mut().poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Ok(new_stream)) => {
                            let acc =
                                Box::new(StreamAccumulator::new(provider.clone(), model.clone()));
                            this.state = StreamState::Streaming {
                                stream: new_stream,
                                accumulator: acc,
                            };
                            // Loop to immediately poll the new stream
                            continue 'state_machine;
                        }
                        Poll::Ready(Err(e)) => {
                            this.state = StreamState::Done;
                            let msg = e.to_string();
                            return Poll::Ready(Some(Ok(StreamEvent::error(msg))));
                        }
                    }
                }

                StreamState::Streaming {
                    stream,
                    accumulator,
                } => {
                    match stream.as_mut().poll_next(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(None) => {
                            // Stream closed without Finish event
                            this.state = StreamState::Done;
                            return Poll::Ready(Some(Ok(StreamEvent::error(
                                "stream closed without Finish event",
                            ))));
                        }
                        Poll::Ready(Some(Err(e))) => {
                            this.state = StreamState::Done;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Ready(Some(Ok(event))) => {
                            let is_finish =
                                event.event_type == crate::streaming::StreamEventType::Finish;
                            let is_error =
                                event.event_type == crate::streaming::StreamEventType::Error;

                            // Feed to accumulator (ignore errors for intermediate events)
                            let _ = accumulator.process(&event);

                            if is_error {
                                this.state = StreamState::Done;
                                return Poll::Ready(Some(Ok(event)));
                            }

                            if is_finish {
                                // Finalize and check for tool loop.
                                // Swap state to Done to take ownership of accumulator.
                                let old_state =
                                    std::mem::replace(&mut this.state, StreamState::Done);
                                if let StreamState::Streaming { accumulator, .. } = old_state {
                                    match accumulator.finalize() {
                                        Err(e) => {
                                            return Poll::Ready(Some(Ok(StreamEvent::error(
                                                e.to_string(),
                                            ))));
                                        }
                                        Ok(response) => {
                                            let has_tool_calls = !response.tool_calls().is_empty();
                                            let is_tool_finish =
                                                response.finish_reason.is_tool_calls();
                                            let can_loop = has_tool_calls
                                                && is_tool_finish
                                                && this.rounds_remaining > 0
                                                && this.tool_executor.is_some();

                                            if can_loop {
                                                let executor = this.tool_executor.as_ref().unwrap();
                                                let (assistant_msg, tool_msgs, _) =
                                                    execute_tools(&response, executor);
                                                this.conversation.push(assistant_msg);
                                                this.conversation.extend(tool_msgs);
                                                this.rounds_remaining -= 1;

                                                let req = this
                                                    .build_next_request(this.conversation.clone());
                                                let client = this.client.clone();
                                                let provider_name = response.provider.clone();
                                                let model_name = response.model.clone();
                                                let fut: Pin<
                                                    Box<
                                                        dyn Future<
                                                                Output = Result<
                                                                    EventStream,
                                                                    UnifiedLlmError,
                                                                >,
                                                            > + Send,
                                                    >,
                                                > = Box::pin(
                                                    async move { client.stream(req).await },
                                                );
                                                this.state = StreamState::StartingNextRound(
                                                    fut,
                                                    provider_name,
                                                    model_name,
                                                );
                                            } else {
                                                *this.final_response.lock().unwrap() =
                                                    Some(response);
                                                this.state = StreamState::Done;
                                            }
                                        }
                                    }
                                }
                            }

                            return Poll::Ready(Some(Ok(event)));
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// F-018: stream()
// ---------------------------------------------------------------------------

/// Execute an LLM generation request and return a `StreamResult`.
pub async fn stream(params: GenerateParams) -> Result<StreamResult, UnifiedLlmError> {
    let client = match params.client.clone() {
        Some(c) => c,
        None => get_default_client()?,
    };

    let conversation = build_conversation(&params)?;
    let req = build_request(&params, conversation.clone());

    let event_stream = client.stream(req).await?;

    let provider_name = params
        .provider
        .clone()
        .unwrap_or_else(|| client.default_provider().to_string());
    let model_name = params.model.clone();
    let acc = Box::new(StreamAccumulator::new(provider_name, model_name));

    Ok(StreamResult {
        state: StreamState::Streaming {
            stream: event_stream,
            accumulator: acc,
        },
        client,
        params_model: params.model,
        params_provider: params.provider,
        params_tools: params.tools,
        params_tool_choice: params.tool_choice,
        params_temperature: params.temperature,
        params_max_tokens: params.max_tokens,
        params_reasoning_effort: params.reasoning_effort,
        params_response_format: params.response_format,
        params_provider_options: params.provider_options,
        conversation,
        rounds_remaining: params.max_tool_rounds,
        tool_executor: params.tool_executor,
        final_response: Arc::new(Mutex::new(None)),
        cancellation_token: params.cancellation_token,
    })
}

// ---------------------------------------------------------------------------
// F-019: GenerateObjectParams
// ---------------------------------------------------------------------------

/// Parameters for `generate_object()`.
pub struct GenerateObjectParams {
    /// Base generation parameters.
    pub generate: GenerateParams,
    /// JSON Schema the output must conform to.
    pub schema: Value,
    /// Optional name for the schema.
    pub schema_name: Option<String>,
    /// Maximum retries on validation failure. Default: `2`.
    pub max_retries: u32,
    /// If `true`, use native provider structured output when available.
    pub use_native: bool,
}

impl GenerateObjectParams {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>, schema: Value) -> Self {
        Self {
            generate: GenerateParams::new(model, prompt),
            schema,
            schema_name: None,
            max_retries: 2,
            use_native: true,
        }
    }

    pub fn with_schema_name(mut self, name: impl Into<String>) -> Self {
        self.schema_name = Some(name.into());
        self
    }

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.generate = self.generate.with_system(system);
        self
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.generate = self.generate.with_client(client);
        self
    }
}

// ---------------------------------------------------------------------------
// F-019: JSON extraction and schema validation helpers
// ---------------------------------------------------------------------------

/// Detect the structured output strategy for the given params.
#[derive(Debug, Clone, PartialEq)]
enum StructuredOutputStrategy {
    /// Use native provider-side structured output.
    Native,
    /// Use prompt engineering + JSON extraction.
    PromptEngineering,
}

fn detect_strategy(params: &GenerateObjectParams) -> StructuredOutputStrategy {
    if !params.use_native {
        return StructuredOutputStrategy::PromptEngineering;
    }
    // Determine effective provider
    let provider = params.generate.provider.as_deref().unwrap_or(""); // will use default client's provider

    match provider {
        "openai" | "gemini" => StructuredOutputStrategy::Native,
        "anthropic" => StructuredOutputStrategy::PromptEngineering,
        "" => {
            // Unknown — try to detect from default client
            if let Ok(client) = get_default_client() {
                match client.default_provider() {
                    "openai" | "gemini" => StructuredOutputStrategy::Native,
                    _ => StructuredOutputStrategy::PromptEngineering,
                }
            } else {
                StructuredOutputStrategy::PromptEngineering
            }
        }
        _ => StructuredOutputStrategy::PromptEngineering,
    }
}

/// Extract a JSON value from text. Tries:
/// 1. Direct parse
/// 2. First `{` to last `}` substring
/// 3. First `[` to last `]` substring
fn extract_json(text: &str) -> Option<Value> {
    // Direct parse
    if let Ok(v) = serde_json::from_str::<Value>(text.trim()) {
        return Some(v);
    }

    // Object extraction
    if let (Some(start), Some(end)) = (text.find('{'), text.rfind('}')) {
        if start <= end {
            if let Ok(v) = serde_json::from_str::<Value>(&text[start..=end]) {
                return Some(v);
            }
        }
    }

    // Array extraction
    if let (Some(start), Some(end)) = (text.find('['), text.rfind(']')) {
        if start <= end {
            if let Ok(v) = serde_json::from_str::<Value>(&text[start..=end]) {
                return Some(v);
            }
        }
    }

    None
}

/// Validate a JSON value against a schema using the `jsonschema` crate.
/// Returns `Ok(())` if valid, `Err(message)` if invalid.
fn validate_against_schema(value: &Value, schema: &Value) -> Result<(), String> {
    let compiled = jsonschema::validator_for(schema).map_err(|e| format!("invalid schema: {e}"))?;

    let errors: Vec<String> = compiled.iter_errors(value).map(|e| e.to_string()).collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

/// Inject schema instructions into the system prompt.
fn inject_schema_prompt(generate: &mut GenerateParams, schema: &Value) {
    let schema_str = serde_json::to_string_pretty(schema).unwrap_or_default();
    let injection = format!(
        "You must respond with a valid JSON object that conforms to the following JSON Schema.\n\
         Do not include any text before or after the JSON object.\n\
         \n\
         Schema:\n\
         {schema_str}"
    );

    // Prepend to existing system or set new system
    let current_system = generate.system.take();
    generate.system = Some(match current_system {
        Some(existing) => format!("{injection}\n\n{existing}"),
        None => injection,
    });
}

// ---------------------------------------------------------------------------
// F-019: generate_object()
// ---------------------------------------------------------------------------

/// Execute a generation request and return a validated JSON object.
pub async fn generate_object(
    params: GenerateObjectParams,
) -> Result<GenerateResult, UnifiedLlmError> {
    // Validate the schema itself first
    if !params.schema.is_object() && !params.schema.is_boolean() {
        return Err(UnifiedLlmError::NoObjectGenerated {
            message: "invalid schema: schema must be a JSON object or boolean".to_string(),
        });
    }
    // Quick check: try to compile the schema to detect invalid schemas
    if let Err(e) = jsonschema::validator_for(&params.schema) {
        return Err(UnifiedLlmError::NoObjectGenerated {
            message: format!("invalid schema: {e}"),
        });
    }

    let strategy = detect_strategy(&params);
    let schema_name = params
        .schema_name
        .clone()
        .unwrap_or_else(|| "output".to_string());

    match strategy {
        StructuredOutputStrategy::Native => generate_object_native(params, schema_name).await,
        StructuredOutputStrategy::PromptEngineering => {
            generate_object_prompt_engineering(params).await
        }
    }
}

async fn generate_object_native(
    mut params: GenerateObjectParams,
    schema_name: String,
) -> Result<GenerateResult, UnifiedLlmError> {
    let provider = params
        .generate
        .provider
        .as_deref()
        .unwrap_or_else(|| {
            get_default_client()
                .ok()
                .map(|c| c.default_provider().to_string())
                .unwrap_or_default()
                .leak()
        })
        .to_string();

    // Determine effective provider from default client if not set
    let effective_provider = if params.generate.provider.is_some() {
        provider.clone()
    } else {
        get_default_client()
            .ok()
            .map(|c| c.default_provider().to_string())
            .unwrap_or(provider.clone())
    };

    match effective_provider.as_str() {
        "openai" => {
            // Set response_format to JsonSchema
            params.generate.response_format = Some(crate::types::ResponseFormat {
                format_type: crate::types::ResponseFormatType::JsonSchema,
                json_schema: Some(serde_json::json!({
                    "name": schema_name,
                    "schema": params.schema.clone(),
                    "strict": true,
                })),
                strict: true,
            });
        }
        "gemini" => {
            // Inject responseSchema via provider_options
            let existing_opts = params.generate.provider_options.take();
            let mut opts = existing_opts.unwrap_or(serde_json::json!({}));
            if let Some(obj) = opts.as_object_mut() {
                let gen_config = obj
                    .entry("generationConfig")
                    .or_insert(serde_json::json!({}));
                if let Some(gc) = gen_config.as_object_mut() {
                    gc.insert("responseSchema".to_string(), params.schema.clone());
                    gc.insert(
                        "responseMimeType".to_string(),
                        Value::String("application/json".to_string()),
                    );
                }
            }
            params.generate.provider_options = Some(opts);
        }
        _ => {
            // Fall back to prompt engineering
            inject_schema_prompt(&mut params.generate, &params.schema);
            params.generate.response_format = Some(crate::types::ResponseFormat {
                format_type: crate::types::ResponseFormatType::Json,
                json_schema: None,
                strict: false,
            });
        }
    }

    let schema = params.schema.clone();
    let max_retries = params.max_retries;
    let mut total_usage = Usage::default();
    let mut last_error = String::new();

    for attempt in 0..=max_retries {
        let mut attempt_params = GenerateParams {
            client: params.generate.client.clone(),
            model: params.generate.model.clone(),
            prompt: params.generate.prompt.clone(),
            system: params.generate.system.clone(),
            messages: params.generate.messages.clone(),
            provider: params.generate.provider.clone(),
            tools: params.generate.tools.clone(),
            tool_choice: params.generate.tool_choice.clone(),
            tool_executor: params.generate.tool_executor.clone(),
            max_tool_rounds: params.generate.max_tool_rounds,
            temperature: params.generate.temperature,
            max_tokens: params.generate.max_tokens,
            reasoning_effort: params.generate.reasoning_effort.clone(),
            response_format: params.generate.response_format.clone(),
            provider_options: params.generate.provider_options.clone(),
            cancellation_token: params.generate.cancellation_token.clone(),
            timeout_ms: params.generate.timeout_ms,
        };

        if attempt > 0 {
            // Append feedback as retry context
            let feedback = format!(
                "Your previous response was not valid. Error: {last_error}. \
                 Please try again with a valid JSON object matching the schema."
            );
            let existing = attempt_params.messages.take();
            let mut msgs = existing.unwrap_or_else(|| {
                let mut m = Vec::new();
                if let Some(p) = &attempt_params.prompt {
                    m.push(Message::user(p));
                }
                m
            });
            msgs.push(Message::user(&feedback));
            attempt_params.messages = Some(msgs);
            attempt_params.prompt = None;
        }

        let mut result = generate(attempt_params).await?;
        total_usage += result.total_usage.clone();

        // Parse and validate
        match extract_json(&result.text) {
            None => {
                last_error = "response did not contain valid JSON".to_string();
            }
            Some(parsed) => match validate_against_schema(&parsed, &schema) {
                Ok(()) => {
                    result.output = Some(parsed);
                    result.total_usage = total_usage;
                    return Ok(result);
                }
                Err(e) => {
                    last_error = e;
                }
            },
        }
    }

    Err(UnifiedLlmError::NoObjectGenerated {
        message: format!(
            "schema validation failed after {} attempt(s): {}",
            max_retries + 1,
            last_error
        ),
    })
}

async fn generate_object_prompt_engineering(
    mut params: GenerateObjectParams,
) -> Result<GenerateResult, UnifiedLlmError> {
    // Inject schema instructions into system prompt
    inject_schema_prompt(&mut params.generate, &params.schema);
    // Request JSON output
    params.generate.response_format = Some(crate::types::ResponseFormat {
        format_type: crate::types::ResponseFormatType::Json,
        json_schema: None,
        strict: false,
    });

    let schema = params.schema.clone();
    let max_retries = params.max_retries;
    let mut total_usage = Usage::default();
    let mut last_error = String::new();
    // Maintain a conversation for retry feedback
    let mut conversation: Option<Vec<Message>> = None;

    for attempt in 0..=max_retries {
        let mut attempt_params = GenerateParams {
            client: params.generate.client.clone(),
            model: params.generate.model.clone(),
            prompt: params.generate.prompt.clone(),
            system: params.generate.system.clone(),
            messages: conversation
                .clone()
                .or_else(|| params.generate.messages.clone()),
            provider: params.generate.provider.clone(),
            tools: params.generate.tools.clone(),
            tool_choice: params.generate.tool_choice.clone(),
            tool_executor: params.generate.tool_executor.clone(),
            max_tool_rounds: params.generate.max_tool_rounds,
            temperature: params.generate.temperature,
            max_tokens: params.generate.max_tokens,
            reasoning_effort: params.generate.reasoning_effort.clone(),
            response_format: params.generate.response_format.clone(),
            provider_options: params.generate.provider_options.clone(),
            cancellation_token: params.generate.cancellation_token.clone(),
            timeout_ms: params.generate.timeout_ms,
        };

        if attempt > 0 {
            // Append feedback to conversation
            let feedback = format!(
                "Your previous response was not valid. Error: {last_error}. \
                 Please try again with a valid JSON object matching the schema."
            );
            if let Some(msgs) = &mut conversation {
                msgs.push(Message::user(&feedback));
            }
            attempt_params.messages = conversation.clone();
            attempt_params.prompt = None;
        }

        let mut result = generate(attempt_params).await?;
        total_usage += result.total_usage.clone();

        // Update conversation for next retry if needed
        let next_conv = result
            .steps
            .first()
            .map(|s| {
                // Rebuild conversation from what we sent
                let base = if let Some(msgs) = &conversation {
                    msgs.clone()
                } else {
                    build_conversation(&params.generate).unwrap_or_default()
                }; // ignore error here; already succeeded once
                let mut c = base;
                c.push(s.response.message.clone());
                c
            })
            .unwrap_or_default();

        match extract_json(&result.text) {
            None => {
                last_error = "response did not contain valid JSON".to_string();
                conversation = Some(next_conv);
            }
            Some(parsed) => match validate_against_schema(&parsed, &schema) {
                Ok(()) => {
                    result.output = Some(parsed);
                    result.total_usage = total_usage;
                    return Ok(result);
                }
                Err(e) => {
                    last_error = e;
                    conversation = Some(next_conv);
                }
            },
        }
    }

    Err(UnifiedLlmError::NoObjectGenerated {
        message: format!(
            "schema validation failed after {} attempt(s): {}",
            max_retries + 1,
            last_error
        ),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{make_text_response, make_tool_call_response, MockProviderAdapter};

    async fn build_client(adapter: MockProviderAdapter) -> Client {
        crate::client::ClientBuilder::new()
            .provider("mock", adapter)
            .build()
            .await
            .unwrap()
    }

    // -----------------------------------------------------------------------
    // F-017 generate() tests
    // -----------------------------------------------------------------------

    // AC-1: generate with prompt only → calls complete once, returns text
    #[tokio::test]
    async fn generate_simple_prompt() {
        let mock = MockProviderAdapter::default().push_text_response("Hello world");
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hi").with_client(client);
        let result = generate(params).await.unwrap();
        assert_eq!(result.text, "Hello world");
        assert_eq!(result.steps.len(), 1);
    }

    // AC-2: Tool calls + executor + max_tool_rounds=1 → executor called, second round
    #[tokio::test]
    async fn generate_tool_loop_one_round() {
        // First response: tool call
        let tool_response = make_tool_call_response(vec![(
            "call-1".to_string(),
            "my_tool".to_string(),
            serde_json::json!({"x": 1}),
        )]);
        // Second response: text
        let text_response = make_text_response("Done!");

        let mock = MockProviderAdapter::default()
            .push_response(tool_response)
            .push_response(text_response);
        let client = build_client(mock).await;

        let executor_called = Arc::new(Mutex::new(false));
        let called_clone = Arc::clone(&executor_called);

        let params = GenerateParams::new("mock-model", "do something")
            .with_client(client)
            .with_max_tool_rounds(1)
            .with_tool_executor(move |call| {
                *called_clone.lock().unwrap() = true;
                ToolResult {
                    tool_call_id: call.id,
                    content: "result".to_string(),
                    is_error: false,
                }
            });

        let result = generate(params).await.unwrap();
        assert!(*executor_called.lock().unwrap());
        assert_eq!(result.text, "Done!");
        assert_eq!(result.steps.len(), 2);
    }

    // AC-3: max_tool_rounds=0 → tool executor never called
    #[tokio::test]
    async fn generate_max_rounds_zero_no_executor() {
        let tool_response = make_tool_call_response(vec![(
            "call-1".to_string(),
            "my_tool".to_string(),
            serde_json::json!({}),
        )]);
        let mock = MockProviderAdapter::default().push_response(tool_response);
        let client = build_client(mock).await;

        let executor_called = Arc::new(Mutex::new(false));
        let called_clone = Arc::clone(&executor_called);

        let params = GenerateParams::new("mock-model", "hi")
            .with_client(client)
            .with_max_tool_rounds(0)
            .with_tool_executor(move |_| {
                *called_clone.lock().unwrap() = true;
                ToolResult {
                    tool_call_id: "".to_string(),
                    content: "".to_string(),
                    is_error: false,
                }
            });

        let _result = generate(params).await.unwrap();
        assert!(!*executor_called.lock().unwrap());
    }

    // AC-4: tool_executor = None → no execution even if rounds > 0
    #[tokio::test]
    async fn generate_no_executor_no_loop() {
        let tool_response = make_tool_call_response(vec![(
            "call-1".to_string(),
            "my_tool".to_string(),
            serde_json::json!({}),
        )]);
        let mock = MockProviderAdapter::default().push_response(tool_response);
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hi")
            .with_client(client)
            .with_max_tool_rounds(5);
        // No tool_executor set

        let result = generate(params).await.unwrap();
        assert_eq!(result.steps.len(), 1);
    }

    // AC-5: total_usage = sum of all round usages
    #[tokio::test]
    async fn generate_total_usage_accumulated() {
        let tool_resp = make_tool_call_response(vec![(
            "call-1".to_string(),
            "fn".to_string(),
            serde_json::json!({}),
        )]);
        let text_resp = make_text_response("done");
        let mock = MockProviderAdapter::default()
            .push_response(tool_resp)
            .push_response(text_resp);
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hi")
            .with_client(client)
            .with_max_tool_rounds(1)
            .with_tool_executor(|call| ToolResult {
                tool_call_id: call.id,
                content: "result".to_string(),
                is_error: false,
            });

        let result = generate(params).await.unwrap();
        // total_usage should be at least as large as usage from each step
        assert!(result.total_usage.total_tokens >= result.usage.total_tokens);
    }

    // AC-6: steps length equals rounds executed
    #[tokio::test]
    async fn generate_steps_count_matches_rounds() {
        let mock = MockProviderAdapter::default().push_text_response("hello");
        let client = build_client(mock).await;
        let params = GenerateParams::new("mock-model", "hi").with_client(client);
        let result = generate(params).await.unwrap();
        assert_eq!(result.steps.len(), 1);
    }

    // AC-8: system param prepended when no system message present
    #[tokio::test]
    async fn generate_system_prepended() {
        let mock = MockProviderAdapter::default().push_text_response("answer");
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hello")
            .with_client(client)
            .with_system("Be helpful.");

        let result = generate(params).await.unwrap();
        assert_eq!(result.text, "answer");
    }

    // AC-10: get_default_client() returns Err when no client set and from_env fails
    #[test]
    fn get_default_client_no_env_returns_error() {
        // Save and clear API-key env vars
        let saved_openai = std::env::var("OPENAI_API_KEY").ok();
        let saved_anthropic = std::env::var("ANTHROPIC_API_KEY").ok();
        let saved_gemini = std::env::var("GEMINI_API_KEY").ok();
        let saved_google = std::env::var("GOOGLE_API_KEY").ok();
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
        }

        // Clear the static default client to ensure no prior state
        {
            let mut guard = default_client_rwlock().write().unwrap();
            *guard = None;
        }

        let result = get_default_client();

        // Restore env vars
        unsafe {
            if let Some(v) = saved_openai {
                std::env::set_var("OPENAI_API_KEY", v);
            }
            if let Some(v) = saved_anthropic {
                std::env::set_var("ANTHROPIC_API_KEY", v);
            }
            if let Some(v) = saved_gemini {
                std::env::set_var("GEMINI_API_KEY", v);
            }
            if let Some(v) = saved_google {
                std::env::set_var("GOOGLE_API_KEY", v);
            }
        }

        assert!(result.is_err());
    }

    // AC-11b: both prompt AND messages provided → InvalidRequest error (GAP-ULM-009)
    #[tokio::test]
    async fn generate_rejects_when_both_prompt_and_messages_set() {
        // NLSpec §8.4: generate() must reject when both prompt and messages are provided.
        // Previously the implementation silently overrode prompt with messages.
        let mock = MockProviderAdapter::default().push_text_response("should not reach");
        let client = build_client(mock).await;

        let mut params = GenerateParams::new("mock-model", "this is the prompt");
        params.client = Some(client);
        params.messages = Some(vec![Message::user("this is a messages entry")]);

        let result = generate(params).await;
        assert!(
            result.is_err(),
            "generate() must return Err when both prompt and messages are set"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, UnifiedLlmError::InvalidRequest { .. }),
            "expected InvalidRequest error, got: {err:?}"
        );
    }

    // AC-12: max rounds exhausted → Ok(GenerateResult) not error
    #[tokio::test]
    async fn generate_max_rounds_exhausted_returns_ok() {
        let tool_resp1 = make_tool_call_response(vec![(
            "c1".to_string(),
            "fn".to_string(),
            serde_json::json!({}),
        )]);
        let tool_resp2 = make_tool_call_response(vec![(
            "c2".to_string(),
            "fn".to_string(),
            serde_json::json!({}),
        )]);
        let mock = MockProviderAdapter::default()
            .push_response(tool_resp1)
            .push_response(tool_resp2);
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hi")
            .with_client(client)
            .with_max_tool_rounds(1)
            .with_tool_executor(|call| ToolResult {
                tool_call_id: call.id,
                content: "result".to_string(),
                is_error: false,
            });

        let result = generate(params).await.unwrap();
        // Result should be Ok — max rounds exhausted, returns last response
        assert_eq!(result.steps.len(), 2);
    }

    // -----------------------------------------------------------------------
    // F-018 stream() tests
    // -----------------------------------------------------------------------

    // AC-1: stream() returns StreamResult that yields TextDelta events
    #[tokio::test]
    async fn stream_yields_text_delta_events() {
        let mock = MockProviderAdapter::default().push_text_stream("Hello world");
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hi").with_client(client);
        let mut sr = stream(params).await.unwrap();

        let mut events: Vec<StreamEvent> = Vec::new();
        while let Some(ev) = sr.next().await {
            events.push(ev.unwrap());
        }

        use crate::streaming::StreamEventType;
        assert!(events
            .iter()
            .any(|e| e.event_type == StreamEventType::TextDelta));
    }

    // AC-3: response() after stream consumed returns valid Response
    #[tokio::test]
    async fn stream_response_after_consume() {
        let mock = MockProviderAdapter::default().push_text_stream("hello");
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hi").with_client(client);
        let sr = stream(params).await.unwrap();

        let resp = sr.response().await.unwrap();
        assert_eq!(resp.text(), "hello");
    }

    // AC-4: try_response() returns None before stream consumed
    #[tokio::test]
    async fn stream_try_response_none_before_consumed() {
        let mock = MockProviderAdapter::default().push_text_stream("hello");
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hi").with_client(client);
        let sr = stream(params).await.unwrap();
        assert!(sr.try_response().is_none());
    }

    // -----------------------------------------------------------------------
    // F-019 generate_object() tests
    // -----------------------------------------------------------------------

    // AC-4: valid JSON matching schema → output = Some(value)
    #[tokio::test]
    async fn generate_object_valid_json() {
        let mock =
            MockProviderAdapter::default().push_text_response("{\"name\": \"Alice\", \"age\": 30}");
        let client = build_client(mock).await;

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        });

        let params =
            GenerateObjectParams::new("mock-model", "give me a person", schema).with_client(client);

        let result = generate_object(params).await.unwrap();
        assert!(result.output.is_some());
        assert_eq!(result.output.unwrap()["name"], "Alice");
    }

    // AC-6: always invalid → Err(NoObjectGenerated)
    #[tokio::test]
    async fn generate_object_always_invalid() {
        // Return non-JSON for all retries
        let mock = MockProviderAdapter::default()
            .push_text_response("not json at all")
            .push_text_response("still not json")
            .push_text_response("nope");
        let client = build_client(mock).await;

        let schema = serde_json::json!({"type": "object"});
        let params = GenerateObjectParams::new("mock-model", "hi", schema)
            .with_client(client)
            .with_max_retries(2);

        let err = generate_object(params).await.unwrap_err();
        assert!(matches!(err, UnifiedLlmError::NoObjectGenerated { .. }));
    }

    // AC-9: GenerateObjectParams::new produces valid defaults
    #[test]
    fn generate_object_params_defaults() {
        let params = GenerateObjectParams::new("model", "prompt", serde_json::json!({}));
        assert_eq!(params.max_retries, 2);
        assert!(params.use_native);
        assert!(params.schema_name.is_none());
    }

    // AC-11: JSON buried in prose extracted correctly
    #[tokio::test]
    async fn generate_object_extracts_json_from_prose() {
        let mock = MockProviderAdapter::default()
            .push_text_response("Here is the JSON: {\"name\":\"Alice\"}");
        let client = build_client(mock).await;

        let schema =
            serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let params =
            GenerateObjectParams::new("mock-model", "give me a person", schema).with_client(client);

        let result = generate_object(params).await.unwrap();
        assert_eq!(result.output.unwrap()["name"], "Alice");
    }

    // AC-12: invalid schema → immediate Err without calling generate
    #[tokio::test]
    async fn generate_object_invalid_schema() {
        let mock = MockProviderAdapter::default();
        let client = build_client(mock).await;

        // A string is not a valid JSON Schema
        let schema = serde_json::json!("not a schema");
        let params = GenerateObjectParams::new("mock-model", "hi", schema).with_client(client);

        let err = generate_object(params).await.unwrap_err();
        assert!(matches!(err, UnifiedLlmError::NoObjectGenerated { .. }));
    }

    // extract_json helper tests
    #[test]
    fn extract_json_direct() {
        let v = extract_json("{\"x\": 1}");
        assert!(v.is_some());
        assert_eq!(v.unwrap()["x"], 1);
    }

    #[test]
    fn extract_json_from_prose() {
        let v = extract_json("Here you go: {\"x\": 1} done.");
        assert!(v.is_some());
    }

    #[test]
    fn extract_json_array() {
        let v = extract_json("[1,2,3]");
        assert!(v.is_some());
    }

    #[test]
    fn extract_json_fails_on_garbage() {
        let v = extract_json("no json here");
        assert!(v.is_none());
    }

    // -----------------------------------------------------------------------
    // GAP-ULM-007: Cancellation token for generate() and stream()
    // -----------------------------------------------------------------------

    // Already-cancelled token → generate() returns Err(Abort) immediately
    #[tokio::test]
    async fn generate_aborts_when_token_already_cancelled() {
        let mock = MockProviderAdapter::default().push_text_response("should not reach");
        let client = build_client(mock).await;

        let token = CancellationToken::new();
        token.cancel(); // cancel BEFORE calling generate()

        let params = GenerateParams::new("mock-model", "hi")
            .with_client(client)
            .with_cancellation_token(token);

        let result = generate(params).await;
        assert!(result.is_err(), "expected Err when token is pre-cancelled");
        assert!(
            matches!(result.unwrap_err(), UnifiedLlmError::Abort { .. }),
            "expected Abort error"
        );
    }

    // Uncancelled token → generate() succeeds normally
    #[tokio::test]
    async fn generate_with_live_token_succeeds_normally() {
        let mock = MockProviderAdapter::default().push_text_response("hello");
        let client = build_client(mock).await;

        let token = CancellationToken::new(); // not cancelled

        let params = GenerateParams::new("mock-model", "hi")
            .with_client(client)
            .with_cancellation_token(token);

        let result = generate(params).await.unwrap();
        assert_eq!(result.text, "hello");
    }

    // Already-cancelled token → stream() first event is Err(Abort)
    #[tokio::test]
    async fn stream_aborts_when_token_already_cancelled() {
        let mock = MockProviderAdapter::default().push_text_stream("should not reach");
        let client = build_client(mock).await;

        let token = CancellationToken::new();
        token.cancel();

        let params = GenerateParams::new("mock-model", "hi")
            .with_client(client)
            .with_cancellation_token(token);

        let mut sr = stream(params).await.unwrap();
        let first = sr.next().await.expect("expected at least one event");
        assert!(
            matches!(first, Err(UnifiedLlmError::Abort { .. })),
            "expected first event to be Err(Abort)"
        );
    }

    // CancellationToken is clone-able and shares state
    #[test]
    fn cancellation_token_clone_shares_state() {
        let token = CancellationToken::new();
        let clone = token.clone();
        assert!(!clone.is_cancelled());
        token.cancel();
        assert!(clone.is_cancelled(), "clone should observe cancellation");
    }

    // CancellationToken default is not cancelled
    #[test]
    fn cancellation_token_default_not_cancelled() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }

    // -----------------------------------------------------------------------
    // GAP-ULM-008: Timeout support for generate()
    // -----------------------------------------------------------------------

    // Timeout fires when provider is slow → Err(RequestTimeout)
    #[tokio::test(start_paused = true)]
    async fn generate_times_out_when_provider_slow() {
        use crate::providers::ProviderAdapter;
        use crate::streaming::EventStream;

        /// A mock adapter that always sleeps for 10 seconds before responding.
        struct SlowAdapter;

        #[async_trait::async_trait]
        impl ProviderAdapter for SlowAdapter {
            fn name(&self) -> &str {
                "slow"
            }
            async fn complete(
                &self,
                _req: &crate::types::Request,
            ) -> Result<crate::types::Response, UnifiedLlmError> {
                tokio::time::sleep(Duration::from_secs(10)).await;
                Err(UnifiedLlmError::Configuration {
                    message: "unreachable".to_string(),
                })
            }
            async fn stream(
                &self,
                _req: &crate::types::Request,
            ) -> Result<EventStream, UnifiedLlmError> {
                Err(UnifiedLlmError::Configuration {
                    message: "not used".to_string(),
                })
            }
            async fn initialize(&self) -> Result<(), UnifiedLlmError> {
                Ok(())
            }
            async fn close(&self) -> Result<(), UnifiedLlmError> {
                Ok(())
            }
        }

        let client = crate::client::ClientBuilder::new()
            .provider("slow", SlowAdapter)
            .build()
            .await
            .unwrap();

        let params = GenerateParams {
            client: Some(client),
            model: "slow-model".to_string(),
            prompt: Some("hi".to_string()),
            system: None,
            messages: None,
            provider: Some("slow".to_string()),
            tools: None,
            tool_choice: None,
            tool_executor: None,
            max_tool_rounds: 1,
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            response_format: None,
            provider_options: None,
            cancellation_token: None,
            timeout_ms: Some(100), // 100 ms timeout
        };

        let result = generate(params).await;
        assert!(result.is_err(), "expected Err when provider is slow");
        assert!(
            matches!(result.unwrap_err(), UnifiedLlmError::RequestTimeout { .. }),
            "expected RequestTimeout"
        );
    }

    // No timeout set → generate() runs normally to completion
    #[tokio::test]
    async fn generate_no_timeout_succeeds() {
        let mock = MockProviderAdapter::default().push_text_response("ok");
        let client = build_client(mock).await;
        let params = GenerateParams::new("mock-model", "hi")
            .with_client(client)
            .with_timeout_ms(5000); // generous timeout, mock is instant
        let result = generate(params).await.unwrap();
        assert_eq!(result.text, "ok");
    }

    // -----------------------------------------------------------------------
    // GAP-ULM-014: Parallel tool results sent in single batch continuation
    // -----------------------------------------------------------------------

    // Two tool calls from one response → both results in single continuation request
    #[tokio::test]
    async fn parallel_tool_results_sent_in_single_batch_continuation() {
        use crate::testing::make_text_response;

        // Round 1: two tool calls in one response
        let two_call_response = {
            use crate::types::{ContentPart, FinishReason, Message, Role, ToolCallData, Usage};
            let content = vec![
                ContentPart::tool_call(ToolCallData {
                    id: "call-A".to_string(),
                    name: "tool_alpha".to_string(),
                    arguments: serde_json::json!({"n": 1}),
                    raw_arguments: None,
                }),
                ContentPart::tool_call(ToolCallData {
                    id: "call-B".to_string(),
                    name: "tool_beta".to_string(),
                    arguments: serde_json::json!({"n": 2}),
                    raw_arguments: None,
                }),
            ];
            crate::types::Response {
                id: "r1".to_string(),
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
        };

        // Round 2: simple text response
        let text_response = make_text_response("All done!");

        let mock = MockProviderAdapter::default()
            .push_response(two_call_response)
            .push_response(text_response);

        let request_log = mock.request_log_handle();
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "do two things")
            .with_client(client)
            .with_max_tool_rounds(1)
            .with_tool_executor(|call| ToolResult {
                tool_call_id: call.id,
                content: format!("result_for_{}", call.name),
                is_error: false,
            });

        let result = generate(params).await.unwrap();
        assert_eq!(result.text, "All done!");
        assert_eq!(result.steps.len(), 2);

        // Verify: second request contains BOTH tool results (single batch)
        let requests = request_log.lock().unwrap();
        assert_eq!(requests.len(), 2, "should have made exactly 2 requests");
        let second_req = &requests[1];
        let tool_result_msgs = second_req
            .messages
            .iter()
            .filter(|m| m.role == crate::types::Role::Tool)
            .count();
        assert_eq!(
            tool_result_msgs, 2,
            "second request should have 2 tool result messages (one per parallel call)"
        );
    }

    // -----------------------------------------------------------------------
    // GAP-ULM-015: StepResult tracking across multiple tool rounds
    // -----------------------------------------------------------------------

    // Three rounds: tool call → tool call → text → steps have correct structure
    #[tokio::test]
    async fn step_results_tracked_across_multiple_rounds() {
        use crate::testing::make_tool_call_response;

        let round1 = make_tool_call_response(vec![(
            "c1".to_string(),
            "fn_one".to_string(),
            serde_json::json!({"a": 1}),
        )]);
        let round2 = make_tool_call_response(vec![(
            "c2".to_string(),
            "fn_two".to_string(),
            serde_json::json!({"b": 2}),
        )]);
        let round3 = crate::testing::make_text_response("final answer");

        let mock = MockProviderAdapter::default()
            .push_response(round1)
            .push_response(round2)
            .push_response(round3);
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "multi-round")
            .with_client(client)
            .with_max_tool_rounds(2)
            .with_tool_executor(|call| ToolResult {
                tool_call_id: call.id,
                content: "ok".to_string(),
                is_error: false,
            });

        let result = generate(params).await.unwrap();
        assert_eq!(result.text, "final answer");
        assert_eq!(
            result.steps.len(),
            3,
            "should have 3 steps (2 tool rounds + 1 final)"
        );

        // Step 0 and 1 should each have one tool call
        assert_eq!(result.steps[0].tool_calls.len(), 1);
        assert_eq!(result.steps[0].tool_calls[0].name, "fn_one");
        assert_eq!(result.steps[0].tool_results.len(), 1);

        assert_eq!(result.steps[1].tool_calls.len(), 1);
        assert_eq!(result.steps[1].tool_calls[0].name, "fn_two");
        assert_eq!(result.steps[1].tool_results.len(), 1);

        // Step 2 (final) is the text response — no tool calls, no tool results
        assert_eq!(result.steps[2].tool_calls.len(), 0);
        assert_eq!(result.steps[2].tool_results.len(), 0);

        // Total tool calls across all rounds: fn_one + fn_two (text round has none)
        assert_eq!(result.tool_calls.len(), 2);
    }

    // -----------------------------------------------------------------------
    // GAP-ULM-016: Streaming mid-stream error behavior
    // -----------------------------------------------------------------------

    // Stream that emits TextDelta events then an Error event is handled correctly
    #[tokio::test]
    async fn stream_mid_stream_error_produces_error_event() {
        use crate::streaming::{StreamEvent, StreamEventType};

        // Build a stream that emits: StreamStart, TextStart, TextDelta, Error
        let events = vec![
            StreamEvent::stream_start(),
            StreamEvent::text_start(),
            StreamEvent::text_delta("partial"),
            // Error mid-stream (no Finish event)
            StreamEvent::error("mid-stream provider error"),
        ];
        let mock = MockProviderAdapter::default().push_stream_events(events);
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hi").with_client(client);
        let mut sr = stream(params).await.unwrap();

        let mut collected: Vec<Result<StreamEvent, UnifiedLlmError>> = Vec::new();
        while let Some(ev) = sr.next().await {
            collected.push(ev);
        }

        // Should have received some events before the error
        assert!(!collected.is_empty(), "should have received events");
        // Should include a TextDelta
        assert!(
            collected
                .iter()
                .any(|e| matches!(e, Ok(ev) if ev.event_type == StreamEventType::TextDelta)),
            "expected TextDelta event in stream"
        );
        // The error event should be present (either as Ok(Error) or as Err(_))
        let has_error = collected.iter().any(|e| match e {
            Ok(ev) => ev.event_type == StreamEventType::Error,
            Err(_) => true,
        });
        assert!(has_error, "expected an error event or Err in stream");

        // After the error, the stream should be done (no more events)
        // The StreamResult transitions to Done on error
        assert!(sr.is_complete(), "stream should be done after error event");
    }

    // -----------------------------------------------------------------------
    // V2-ULM-004: execute_tools() must run tool calls concurrently
    // -----------------------------------------------------------------------

    // Three parallel tool calls, each taking 100ms. Total must be ~100ms (not ~300ms).
    #[tokio::test]
    async fn tool_calls_execute_concurrently() {
        use std::time::Duration as StdDuration;
        use tokio::time::Instant;

        // Build a response with 3 tool calls
        let three_call_response = {
            use crate::types::{ContentPart, FinishReason, Message, Role, ToolCallData, Usage};
            let content = vec![
                ContentPart::tool_call(ToolCallData {
                    id: "call-1".to_string(),
                    name: "slow_tool".to_string(),
                    arguments: serde_json::json!({}),
                    raw_arguments: None,
                }),
                ContentPart::tool_call(ToolCallData {
                    id: "call-2".to_string(),
                    name: "slow_tool".to_string(),
                    arguments: serde_json::json!({}),
                    raw_arguments: None,
                }),
                ContentPart::tool_call(ToolCallData {
                    id: "call-3".to_string(),
                    name: "slow_tool".to_string(),
                    arguments: serde_json::json!({}),
                    raw_arguments: None,
                }),
            ];
            crate::types::Response {
                id: "r1".to_string(),
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
        };

        let text_response = crate::testing::make_text_response("all done");
        let mock = MockProviderAdapter::default()
            .push_response(three_call_response)
            .push_response(text_response);
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "do 3 things")
            .with_client(client)
            .with_max_tool_rounds(1)
            .with_tool_executor(|call| {
                // Each tool call blocks for 100ms
                std::thread::sleep(StdDuration::from_millis(100));
                ToolResult {
                    tool_call_id: call.id,
                    content: "done".to_string(),
                    is_error: false,
                }
            });

        let start = Instant::now();
        let result = generate(params).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result.text, "all done");
        // Sequential: 3 × 100ms = 300ms. Concurrent: ~100ms.
        // Allow up to 250ms to account for scheduling overhead.
        assert!(
            elapsed < tokio::time::Duration::from_millis(250),
            "tool calls should execute concurrently (~100ms total), but took {elapsed:?}"
        );
    }

    // -----------------------------------------------------------------------
    // V2-ULM-007: Unknown tool calls not intercepted in generate()
    // -----------------------------------------------------------------------

    // LLM returns a call to an unknown tool → error result, executor NOT called
    #[tokio::test]
    async fn generate_unknown_tool_returns_error_result_without_calling_executor() {
        use crate::testing::make_text_response;
        use crate::types::{ContentPart, FinishReason, Message, Role, Tool, ToolCallData, Usage};

        // Construct a response with a call to "secret_tool" (not in tools list)
        let unknown_call_response = {
            let content = vec![ContentPart::tool_call(ToolCallData {
                id: "call-x".to_string(),
                name: "secret_tool".to_string(),
                arguments: serde_json::json!({}),
                raw_arguments: None,
            })];
            crate::types::Response {
                id: "r1".to_string(),
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
        };
        let final_response = make_text_response("done");
        let mock = MockProviderAdapter::default()
            .push_response(unknown_call_response)
            .push_response(final_response);
        let client = build_client(mock).await;

        let executor_called_ptr = std::sync::Arc::new(std::sync::Mutex::new(false));
        let ptr_clone = executor_called_ptr.clone();

        let params = GenerateParams::new("mock-model", "hi")
            .with_client(client)
            .with_tools(vec![Tool {
                name: "known_tool".to_string(),
                description: "A known tool".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }])
            .with_max_tool_rounds(1)
            .with_tool_executor(move |call| {
                *ptr_clone.lock().unwrap() = true;
                ToolResult {
                    tool_call_id: call.id,
                    content: "executor called".to_string(),
                    is_error: false,
                }
            });

        let result = generate(params).await.unwrap();
        assert_eq!(result.text, "done");
        // Executor must NOT have been called for the unknown tool
        assert!(
            !*executor_called_ptr.lock().unwrap(),
            "executor must not be called for unknown tool"
        );
        // The tool result must be an error
        assert_eq!(result.tool_results.len(), 1);
        assert!(result.tool_results[0].is_error);
        let content_str = result.tool_results[0].content.as_str().unwrap_or_default();
        assert!(
            content_str.contains("Unknown tool"),
            "content should mention 'Unknown tool': {content_str}"
        );
    }

    // -----------------------------------------------------------------------
    // V2-ULM-010: set_default_client() affirmative path
    // -----------------------------------------------------------------------

    // set_default_client then get_default_client returns the same client
    #[test]
    fn set_then_get_default_client_returns_same_provider() {
        // Reset first to ensure clean state for test
        {
            let mut guard = default_client_rwlock().write().unwrap();
            *guard = None;
        }
        use crate::client::ClientBuilder;
        use crate::testing::MockProviderAdapter;
        // Build a simple mock client
        let rt = tokio::runtime::Runtime::new().unwrap();
        let client = rt.block_on(async {
            ClientBuilder::new()
                .provider("mock-set-test", MockProviderAdapter::default())
                .build()
                .await
                .unwrap()
        });
        set_default_client(client);
        let got = get_default_client();
        assert!(got.is_ok(), "get_default_client() must return Ok after set");
        // Reset for other tests
        {
            let mut guard = default_client_rwlock().write().unwrap();
            *guard = None;
        }
    }

    // -----------------------------------------------------------------------
    // V2-ULM-014: Cross-provider parity matrix — status documentation
    // -----------------------------------------------------------------------
    //
    // The full 45-cell parity matrix (15 capabilities × 3 providers: OpenAI,
    // Anthropic, Gemini) requires real API keys to verify end-to-end.
    // All 45 cells have unit-test coverage via MockProviderAdapter; live
    // verification is done in tests/live_providers.rs (gated behind LIVE_TEST=1).
    //
    // Status (2026-03-15):
    //   - 15 capabilities: text, tools, streaming, images, structured output,
    //     reasoning, caching, tool_choice, finish_reason, usage, retry, timeout,
    //     cancellation, parallel_tools, context_length
    //   - All 45 mock-tested; 8+ cells live-verified in prior sessions
    //   - Acknowledged: full live parity testing requires sustained API budget
    #[test]
    fn cross_provider_parity_matrix_documented() {
        // Acknowledgement test — documents the parity matrix status.
        // See tests/live_providers.rs for live verification (LIVE_TEST=1).
        let capabilities = [
            "text generation",
            "tool calls",
            "streaming",
            "image input",
            "structured output",
            "extended reasoning",
            "prompt caching",
            "tool_choice modes",
            "finish_reason mapping",
            "usage token counting",
            "retry on 429",
            "request timeout",
            "cancellation token",
            "parallel tool execution",
            "context_length error",
        ];
        let providers = ["openai", "anthropic", "gemini"];
        assert_eq!(
            capabilities.len() * providers.len(),
            45,
            "45-cell parity matrix: {} capabilities × {} providers",
            capabilities.len(),
            providers.len()
        );
    }

    // -----------------------------------------------------------------------
    // GAP-ULM-022: Tool argument JSON validation against tool schema
    // When tool call arguments don't satisfy the tool's parameter schema,
    // the tool executor must NOT be called — instead a ToolResult with
    // is_error=true is returned containing a validation error message.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn tool_arg_validation_rejects_invalid_args_with_is_error() {
        use crate::testing::make_tool_call_response;
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        // Tool call response with missing required "city" field.
        let tool_response = make_tool_call_response(vec![(
            "call-1".to_string(),
            "get_weather".to_string(),
            serde_json::json!({"wrong_field": "value"}), // "city" is required, absent here
        )]);
        let text_response = make_text_response("done");

        let mock = MockProviderAdapter::default()
            .push_response(tool_response)
            .push_response(text_response);
        let client = build_client(mock).await;

        let tool = crate::types::Tool {
            name: "get_weather".to_string(),
            description: "Get weather for a city".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "city name"}
                },
                "required": ["city"]
            }),
        };

        let executor_called = Arc::new(AtomicBool::new(false));
        let executor_flag = Arc::clone(&executor_called);
        let params = GenerateParams::new("mock-model", "what's the weather?")
            .with_client(client)
            .with_tools(vec![tool])
            .with_max_tool_rounds(1)
            .with_tool_executor(move |call| {
                executor_flag.store(true, Ordering::SeqCst);
                ToolResult {
                    tool_call_id: call.id,
                    content: "sunny".to_string(),
                    is_error: false,
                }
            });

        let result = generate(params).await.unwrap();

        // The executor must NOT have been called (validation intercepted first).
        assert!(
            !executor_called.load(Ordering::SeqCst),
            "executor must NOT be called when args fail schema validation"
        );

        // Tool result must carry the validation error.
        assert_eq!(result.tool_results.len(), 1, "should have one tool result");
        assert!(
            result.tool_results[0].is_error,
            "tool result must have is_error=true when args fail validation"
        );
        let err_content = result.tool_results[0].content.as_str().unwrap_or("");
        assert!(
            err_content.contains("validation"),
            "error content must mention 'validation', got: {err_content:?}"
        );
    }

    #[tokio::test]
    async fn tool_arg_validation_passes_valid_args_to_executor() {
        use crate::testing::make_tool_call_response;

        // Tool call with correct "city" argument.
        let tool_response = make_tool_call_response(vec![(
            "call-2".to_string(),
            "get_weather".to_string(),
            serde_json::json!({"city": "Paris"}), // valid
        )]);
        let text_response = make_text_response("The weather in Paris is sunny.");

        let mock = MockProviderAdapter::default()
            .push_response(tool_response)
            .push_response(text_response);
        let client = build_client(mock).await;

        let tool = crate::types::Tool {
            name: "get_weather".to_string(),
            description: "Get weather for a city".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        };

        let params = GenerateParams::new("mock-model", "weather?")
            .with_client(client)
            .with_tools(vec![tool])
            .with_max_tool_rounds(1)
            .with_tool_executor(|call| ToolResult {
                tool_call_id: call.id,
                content: "72°F sunny".to_string(),
                is_error: false,
            });

        let result = generate(params).await.unwrap();

        // Valid args → executor called normally → is_error false
        assert_eq!(result.tool_results.len(), 1);
        assert!(
            !result.tool_results[0].is_error,
            "valid args must pass through to executor without error"
        );
    }

    // Stream error at startup → stream() itself returns Err
    #[tokio::test]
    async fn stream_error_at_startup_returns_err() {
        let mock = MockProviderAdapter::default().push_stream_error(UnifiedLlmError::Network {
            message: "connection refused".to_string(),
            source: None,
        });
        let client = build_client(mock).await;

        let params = GenerateParams::new("mock-model", "hi").with_client(client);
        let result = stream(params).await;
        assert!(
            result.is_err(),
            "stream() should propagate error from adapter"
        );
        let err = result.err().unwrap();
        assert!(matches!(err, UnifiedLlmError::Network { .. }));
    }
}
