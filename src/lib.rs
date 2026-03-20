//! Multi-provider LLM client library.
//!
//! Provides a unified interface for OpenAI, Anthropic, Gemini, and
//! OpenAI-compatible APIs.

pub mod api;
pub mod catalog;
pub mod client;
pub mod error;
pub mod middleware;
pub mod providers;
pub mod retry;
pub(crate) mod sse;
pub mod streaming;
pub mod testing;
pub mod types;

pub use api::{
    generate, generate_object, set_default_client, stream as stream_generate, GenerateObjectParams,
    GenerateParams, GenerateResult, StepResult, StreamResult,
};
pub use catalog::{get_latest_model, get_model_info, list_models};
pub use client::{Client, ClientBuilder};
pub use error::UnifiedLlmError;
pub use middleware::{Middleware, MiddlewareChain, MiddlewareNext, MiddlewareStreamNext};
pub use providers::anthropic::AnthropicAdapter;
pub use providers::gemini::GeminiAdapter;
pub use providers::openai::OpenAiAdapter;
pub use providers::openai_compat::OpenAiCompatAdapter;
pub use providers::ProviderAdapter;
pub use retry::{RetryConfig, RetryPolicy};
pub use streaming::{EventStream, StreamAccumulator, StreamEvent, StreamEventType};
pub use types::{
    AudioData, ContentKind, ContentPart, DocumentData, FinishReason, ImageData, Message, ModelInfo,
    RateLimitInfo, Request, Response, ResponseFormat, ResponseFormatType, Role, ThinkingData, Tool,
    ToolCall, ToolCallData, ToolChoice, ToolResult, ToolResultData, Usage, Warning,
};
