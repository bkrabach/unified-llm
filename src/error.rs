//! Error hierarchy for the `unified-llm` crate.
//!
//! Every fallible function in the crate returns `Result<_, UnifiedLlmError>`.

/// The single error type used throughout `unified-llm`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum UnifiedLlmError {
    /// Generic error returned by a provider (non-specific HTTP error, etc.)
    #[error("provider error ({provider}): {message}")]
    Provider {
        provider: String,
        message: String,
        status_code: Option<u16>,
        error_code: Option<String>,
        retryable: bool,
        /// Seconds until the caller may retry.
        retry_after: Option<f64>,
        raw: Option<serde_json::Value>,
    },

    /// 401 / invalid API key.
    #[error("authentication error ({provider}): {message}")]
    Authentication { provider: String, message: String },

    /// 429 / quota exceeded.
    #[error("rate limit exceeded ({provider}): {message}")]
    RateLimit {
        provider: String,
        message: String,
        /// Seconds until the rate limit resets.
        retry_after: Option<f64>,
    },

    /// Input exceeds the model's context window.
    #[error("context length exceeded: {message}")]
    ContextLength { message: String },

    /// Request timed out waiting for the provider.
    #[error("request timeout: {message}")]
    RequestTimeout { message: String },

    /// TCP / TLS / DNS failure.
    #[error("network error: {message}")]
    Network {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Error while reading or parsing the SSE / streaming response.
    #[error("stream error: {message}")]
    Stream { message: String },

    /// Missing env var, invalid config, no providers registered, etc.
    #[error("configuration error: {message}")]
    Configuration { message: String },

    /// LLM returned a tool call with unparseable arguments.
    #[error("invalid tool call: {message}")]
    InvalidToolCall { message: String },

    /// `generate_object()` called but model returned no JSON object.
    #[error("no object generated: {message}")]
    NoObjectGenerated { message: String },

    /// Caller-initiated abort (e.g., `CancellationToken` fired).
    #[error("abort: {message}")]
    Abort { message: String },

    /// The caller supplied mutually-exclusive parameters.
    ///
    /// For example: both `prompt` and `messages` set in `GenerateParams`.
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
}

impl UnifiedLlmError {
    /// Returns `true` if the error is safe to retry (transient failure).
    ///
    /// - [`UnifiedLlmError::RateLimit`] — always retryable.
    /// - [`UnifiedLlmError::RequestTimeout`] — always retryable.
    /// - [`UnifiedLlmError::Provider`] — retryable only when `retryable == true`.
    /// - All other variants — **not** retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimit { .. } | Self::RequestTimeout { .. } => true,
            Self::Provider { retryable, .. } => *retryable,
            _ => false,
        }
    }

    /// Returns the HTTP status code, if available.
    ///
    /// Only [`UnifiedLlmError::Provider`] with `status_code: Some(_)` returns a value.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::Provider { status_code, .. } => *status_code,
            _ => None,
        }
    }

    /// Returns the retry-after delay in seconds, if the provider supplied one.
    pub fn retry_after(&self) -> Option<f64> {
        match self {
            Self::RateLimit { retry_after, .. } | Self::Provider { retry_after, .. } => {
                *retry_after
            }
            _ => None,
        }
    }

    /// Returns the provider name for variants that carry one, or `None`.
    pub fn provider(&self) -> Option<&str> {
        match self {
            Self::Provider { provider, .. }
            | Self::Authentication { provider, .. }
            | Self::RateLimit { provider, .. } => Some(provider.as_str()),
            _ => None,
        }
    }
}

/// Crate-local result type alias.
///
/// **Not** re-exported from `lib.rs` — downstream crates use
/// `Result<_, UnifiedLlmError>` explicitly to avoid conflicts with
/// `std::result::Result`.
pub type Result<T> = std::result::Result<T, UnifiedLlmError>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider(
        retryable: bool,
        status_code: Option<u16>,
        retry_after: Option<f64>,
    ) -> UnifiedLlmError {
        UnifiedLlmError::Provider {
            provider: "openai".to_string(),
            message: "oops".to_string(),
            status_code,
            error_code: None,
            retryable,
            retry_after,
            raw: None,
        }
    }

    // AC-1
    #[test]
    fn rate_limit_is_retryable() {
        let e = UnifiedLlmError::RateLimit {
            provider: "openai".to_string(),
            message: "too many".to_string(),
            retry_after: None,
        };
        assert!(e.is_retryable());
    }

    // AC-2
    #[test]
    fn timeout_is_retryable() {
        let e = UnifiedLlmError::RequestTimeout {
            message: "timed out".to_string(),
        };
        assert!(e.is_retryable());
    }

    // AC-3
    #[test]
    fn provider_retryable_true() {
        assert!(make_provider(true, None, None).is_retryable());
    }

    // AC-4
    #[test]
    fn provider_retryable_false() {
        assert!(!make_provider(false, None, None).is_retryable());
    }

    // AC-5
    #[test]
    fn authentication_not_retryable() {
        let e = UnifiedLlmError::Authentication {
            provider: "openai".to_string(),
            message: "bad key".to_string(),
        };
        assert!(!e.is_retryable());
    }

    // AC-6
    #[test]
    fn context_length_not_retryable() {
        let e = UnifiedLlmError::ContextLength {
            message: "too long".to_string(),
        };
        assert!(!e.is_retryable());
    }

    // AC-7
    #[test]
    fn abort_not_retryable() {
        let e = UnifiedLlmError::Abort {
            message: "cancelled".to_string(),
        };
        assert!(!e.is_retryable());
    }

    // AC-8
    #[test]
    fn provider_status_code_some() {
        let e = make_provider(false, Some(429), None);
        assert_eq!(e.status_code(), Some(429));
    }

    // AC-9
    #[test]
    fn rate_limit_status_code_none() {
        let e = UnifiedLlmError::RateLimit {
            provider: "openai".to_string(),
            message: "rate limit".to_string(),
            retry_after: None,
        };
        assert_eq!(e.status_code(), None);
    }

    // AC-10: Display format tests for each variant
    #[test]
    fn display_provider() {
        let e = make_provider(false, None, None);
        let s = format!("{}", e);
        assert!(s.contains("provider error"));
        assert!(s.contains("openai"));
        assert!(s.contains("oops"));
    }

    #[test]
    fn display_authentication() {
        let e = UnifiedLlmError::Authentication {
            provider: "anthropic".to_string(),
            message: "invalid key".to_string(),
        };
        assert!(format!("{}", e).contains("authentication error"));
    }

    #[test]
    fn display_rate_limit() {
        let e = UnifiedLlmError::RateLimit {
            provider: "gemini".to_string(),
            message: "quota".to_string(),
            retry_after: None,
        };
        assert!(format!("{}", e).contains("rate limit exceeded"));
    }

    #[test]
    fn display_context_length() {
        let e = UnifiedLlmError::ContextLength {
            message: "too long".to_string(),
        };
        assert!(format!("{}", e).contains("context length exceeded"));
    }

    #[test]
    fn display_timeout() {
        let e = UnifiedLlmError::RequestTimeout {
            message: "timed out".to_string(),
        };
        assert!(format!("{}", e).contains("request timeout"));
    }

    #[test]
    fn display_network() {
        let e = UnifiedLlmError::Network {
            message: "dns failure".to_string(),
            source: None,
        };
        assert!(format!("{}", e).contains("network error"));
    }

    #[test]
    fn display_stream() {
        let e = UnifiedLlmError::Stream {
            message: "broken pipe".to_string(),
        };
        assert!(format!("{}", e).contains("stream error"));
    }

    #[test]
    fn display_configuration() {
        let e = UnifiedLlmError::Configuration {
            message: "missing key".to_string(),
        };
        assert!(format!("{}", e).contains("configuration error"));
    }

    #[test]
    fn display_invalid_tool_call() {
        let e = UnifiedLlmError::InvalidToolCall {
            message: "bad json".to_string(),
        };
        assert!(format!("{}", e).contains("invalid tool call"));
    }

    #[test]
    fn display_no_object_generated() {
        let e = UnifiedLlmError::NoObjectGenerated {
            message: "no json".to_string(),
        };
        assert!(format!("{}", e).contains("no object generated"));
    }

    #[test]
    fn display_abort() {
        let e = UnifiedLlmError::Abort {
            message: "cancelled".to_string(),
        };
        assert!(format!("{}", e).contains("abort"));
    }

    // AC-11: Send + Sync compile-time check
    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<UnifiedLlmError>();
    }

    // AC-12: retry_after round-trips
    #[test]
    fn rate_limit_retry_after() {
        let e = UnifiedLlmError::RateLimit {
            provider: "openai".to_string(),
            message: "slow down".to_string(),
            retry_after: Some(60.0),
        };
        assert_eq!(e.retry_after(), Some(60.0));
    }

    // provider() method
    #[test]
    fn provider_method() {
        let e = make_provider(false, None, None);
        assert_eq!(e.provider(), Some("openai"));

        let e2 = UnifiedLlmError::ContextLength {
            message: "x".to_string(),
        };
        assert_eq!(e2.provider(), None);
    }

    // Network with source
    #[test]
    fn network_with_source() {
        let inner: Box<dyn std::error::Error + Send + Sync> = Box::new(UnifiedLlmError::Stream {
            message: "inner".to_string(),
        });
        let e = UnifiedLlmError::Network {
            message: "outer".to_string(),
            source: Some(inner),
        };
        assert!(!e.is_retryable());
        assert!(format!("{}", e).contains("network error"));
    }

    // retry_after on Provider
    #[test]
    fn provider_retry_after() {
        let e = make_provider(true, None, Some(30.0));
        assert_eq!(e.retry_after(), Some(30.0));
    }

    // retry_after: Some(0.0) is valid
    #[test]
    fn retry_after_zero() {
        let e = UnifiedLlmError::RateLimit {
            provider: "openai".to_string(),
            message: "hit limit".to_string(),
            retry_after: Some(0.0),
        };
        assert_eq!(e.retry_after(), Some(0.0));
    }
}
