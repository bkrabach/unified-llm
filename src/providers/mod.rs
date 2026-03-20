//! Provider adapter trait and concrete provider implementations.
//!
//! Each provider implementation (OpenAI, Anthropic, Gemini) must implement
//! [`ProviderAdapter`]. The trait is object-safe — it can be used as
//! `Box<dyn ProviderAdapter>` or `Arc<dyn ProviderAdapter>`.

pub mod anthropic;
pub mod gemini;
pub mod openai;
pub mod openai_compat;

use crate::error::UnifiedLlmError;
use crate::streaming::EventStream;
use crate::types::{ContentKind, Request, Response};

// ---------------------------------------------------------------------------
// Shared pre-flight helpers (V2-ULM-006)
// ---------------------------------------------------------------------------

/// Return `Err(InvalidRequest)` if any message in `request` contains an Audio
/// content part.  Call this at the top of every provider's `complete()` and
/// `stream()` so audio is never silently dropped.
pub(crate) fn reject_audio_content(
    request: &Request,
    provider: &str,
) -> Result<(), UnifiedLlmError> {
    for msg in &request.messages {
        for part in &msg.content {
            if part.kind == ContentKind::Audio {
                return Err(UnifiedLlmError::InvalidRequest {
                    message: format!("audio content not supported by provider {provider}"),
                });
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ProviderAdapter
// ---------------------------------------------------------------------------

/// Trait that all provider adapters must implement.
///
/// Implementors translate between the unified [`Request`]/[`Response`] types
/// and a specific provider's native API format.
///
/// The `#[async_trait]` macro is required for async methods on `dyn Trait`.
/// Both the trait definition and all implementations must use it.
///
/// # Object Safety
///
/// `ProviderAdapter` is object-safe:
/// - No generic methods
/// - No `Sized` bound
/// - All async methods are erased to `Pin<Box<dyn Future>>` by `async_trait`
#[async_trait::async_trait]
pub trait ProviderAdapter: Send + Sync {
    /// The provider's identifier string (e.g., `"openai"`, `"anthropic"`, `"gemini"`).
    fn name(&self) -> &str;

    /// Execute a non-streaming completion request and return the full response.
    async fn complete(&self, request: &Request) -> Result<Response, UnifiedLlmError>;

    /// Execute a streaming completion request and return an async event stream.
    async fn stream(&self, request: &Request) -> Result<EventStream, UnifiedLlmError>;

    /// Called once when the adapter is registered with the [`crate::client::Client`].
    ///
    /// Use for one-time setup such as validating the API key format or opening
    /// a connection pool. The default implementation does nothing.
    async fn initialize(&self) -> Result<(), UnifiedLlmError> {
        Ok(())
    }

    /// Called when the [`crate::client::Client`] is shut down.
    ///
    /// Use for cleanup such as closing connection pools. The default
    /// implementation does nothing.
    async fn close(&self) -> Result<(), UnifiedLlmError> {
        Ok(())
    }

    /// Returns `true` if this adapter supports the given tool-choice mode.
    ///
    /// Default: only `"auto"` is reported as supported. Adapters override this
    /// to advertise additional modes (e.g. `"required"`, `"named"`).
    fn supports_tool_choice(&self, mode: &str) -> bool {
        mode == "auto"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::StreamEvent;
    use futures::stream;

    /// Minimal concrete implementation — only `name`, `complete`, `stream` are
    /// required; default impls handle `initialize`, `close`, `supports_tool_choice`.
    struct MinimalAdapter;

    #[async_trait::async_trait]
    impl ProviderAdapter for MinimalAdapter {
        fn name(&self) -> &str {
            "minimal"
        }

        async fn complete(&self, _request: &Request) -> Result<Response, UnifiedLlmError> {
            Err(UnifiedLlmError::Configuration {
                message: "not implemented".to_string(),
            })
        }

        async fn stream(&self, _request: &Request) -> Result<EventStream, UnifiedLlmError> {
            let s = stream::iter::<Vec<Result<StreamEvent, UnifiedLlmError>>>(vec![]);
            Ok(Box::pin(s))
        }
    }

    // AC-1: struct with only name/complete/stream compiles without overriding defaults
    #[tokio::test]
    async fn minimal_adapter_uses_defaults() {
        let adapter = MinimalAdapter;
        assert_eq!(adapter.name(), "minimal");
        // initialize() default returns Ok(())
        adapter.initialize().await.unwrap();
        // close() default returns Ok(())
        adapter.close().await.unwrap();
    }

    // AC-5: Box<dyn ProviderAdapter> compiles (trait is object-safe)
    #[test]
    fn boxed_provider_adapter_compiles() {
        fn _accept(_: Box<dyn ProviderAdapter>) {}
        // Just needs to compile
    }

    // AC-7: default supports_tool_choice("auto") returns true
    #[test]
    fn default_supports_auto_tool_choice() {
        let a = MinimalAdapter;
        assert!(a.supports_tool_choice("auto"));
    }

    // AC-8: default supports_tool_choice("named") returns false
    #[test]
    fn default_does_not_support_named_tool_choice() {
        let a = MinimalAdapter;
        assert!(!a.supports_tool_choice("named"));
    }
}
