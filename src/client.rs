//! The main `Client` entry point for all LLM calls.
//!
//! `Client` holds a registry of named [`ProviderAdapter`] instances, an
//! ordered middleware stack, and a default provider name. It routes each
//! [`Request`] to the correct adapter based on `request.provider` (falling
//! back to the default), runs it through the middleware chain, and returns
//! the result.
//!
//! # Usage
//!
//! ```rust,no_run
//! # async fn run() -> Result<(), unified_llm::error::UnifiedLlmError> {
//! use unified_llm::client::{Client, ClientBuilder};
//! use unified_llm::testing::MockProviderAdapter;
//! use unified_llm::types::{Message, Request};
//!
//! let client = ClientBuilder::new()
//!     .provider("mock", MockProviderAdapter::default().push_text_response("hello"))
//!     .build()
//!     .await?;
//!
//! let req = Request::new("mock-model", vec![Message::user("hi")]);
//! let resp = client.complete(req).await?;
//! assert_eq!(resp.text(), "hello");
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::UnifiedLlmError;
use crate::middleware::{Middleware, MiddlewareChain};
use crate::providers::ProviderAdapter;
use crate::streaming::EventStream;
use crate::types::{Request, Response};

// ---------------------------------------------------------------------------
// ClientInner (private)
// ---------------------------------------------------------------------------

struct ClientInner {
    providers: HashMap<String, Arc<dyn ProviderAdapter>>,
    /// Ordered list of provider names (insertion order) for default selection.
    provider_order: Vec<String>,
    default_provider: String,
    /// Middleware stored as `Arc` so chain construction is cheap (no cloning).
    middleware: Arc<Vec<Arc<dyn Middleware>>>,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// The main LLM client. Holds provider adapters, middleware, and routing config.
///
/// `Client` is cheaply cloneable — it wraps all internal state in [`Arc`].
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("default_provider", &self.inner.default_provider)
            .field("providers", &self.inner.provider_order)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// Construct from environment variables.
    ///
    /// Reads `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`
    /// (or `GOOGLE_API_KEY`), registering the corresponding adapters for each
    /// key that is present.  At least one key must be set; if none is found,
    /// returns [`UnifiedLlmError::Configuration`].
    ///
    /// The first successfully-registered provider becomes the default.
    pub fn from_env() -> Result<Self, UnifiedLlmError> {
        use crate::providers::anthropic::AnthropicAdapter;
        use crate::providers::gemini::GeminiAdapter;
        use crate::providers::openai::OpenAiAdapter;

        let mut providers: HashMap<String, Arc<dyn ProviderAdapter>> = HashMap::new();
        let mut provider_order: Vec<String> = Vec::new();

        if let Ok(adapter) = OpenAiAdapter::from_env() {
            let name = adapter.name().to_string();
            if !providers.contains_key(&name) {
                provider_order.push(name.clone());
            }
            providers.insert(name, Arc::new(adapter));
        }

        if let Ok(adapter) = AnthropicAdapter::from_env() {
            let name = adapter.name().to_string();
            if !providers.contains_key(&name) {
                provider_order.push(name.clone());
            }
            providers.insert(name, Arc::new(adapter));
        }

        if let Ok(adapter) = GeminiAdapter::from_env() {
            let name = adapter.name().to_string();
            if !providers.contains_key(&name) {
                provider_order.push(name.clone());
            }
            providers.insert(name, Arc::new(adapter));
        }

        if providers.is_empty() {
            return Err(UnifiedLlmError::Configuration {
                message:
                    "no providers configured: set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY"
                        .to_string(),
            });
        }

        let default_provider = provider_order
            .first()
            .cloned()
            .expect("providers non-empty");

        Ok(Client {
            inner: Arc::new(ClientInner {
                providers,
                provider_order,
                default_provider,
                middleware: Arc::new(Vec::new()),
            }),
        })
    }

    /// Returns a builder for fluent client construction.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Resolve the provider adapter for a request.
    fn resolve_adapter(
        &self,
        request: &Request,
    ) -> Result<Arc<dyn ProviderAdapter>, UnifiedLlmError> {
        if self.inner.providers.is_empty() {
            return Err(UnifiedLlmError::Configuration {
                message: "no providers registered".to_string(),
            });
        }
        let name = request
            .provider
            .as_deref()
            .unwrap_or(&self.inner.default_provider);
        self.inner
            .providers
            .get(name)
            .map(Arc::clone)
            .ok_or_else(|| UnifiedLlmError::Configuration {
                message: format!("unknown provider: {name}"),
            })
    }

    /// Send a non-streaming request to the appropriate provider.
    ///
    /// Provider selection:
    /// 1. If `request.provider` is `Some(name)`, use that provider.
    /// 2. Otherwise, use the default provider.
    ///
    /// Returns [`UnifiedLlmError::Configuration`] if the named provider is
    /// not registered.
    pub async fn complete(&self, request: Request) -> Result<Response, UnifiedLlmError> {
        let adapter = self.resolve_adapter(&request)?;
        let chain = MiddlewareChain::from_arcs(Arc::clone(&self.inner.middleware), adapter);
        chain.complete(request).await
    }

    /// Send a streaming request to the appropriate provider.
    ///
    /// Same provider selection logic as [`complete`][Self::complete].
    pub async fn stream(&self, request: Request) -> Result<EventStream, UnifiedLlmError> {
        let adapter = self.resolve_adapter(&request)?;
        let chain = MiddlewareChain::from_arcs(Arc::clone(&self.inner.middleware), adapter);
        chain.stream(request).await
    }

    /// Call `close()` on all registered providers concurrently.
    ///
    /// Errors from individual `close()` calls are logged at WARN level and
    /// discarded — close is best-effort and all providers are closed regardless
    /// of individual failures.
    pub async fn close(&self) {
        let futures: Vec<_> = self
            .inner
            .providers
            .values()
            .map(|adapter| {
                let adapter = Arc::clone(adapter);
                async move {
                    if let Err(e) = adapter.close().await {
                        tracing::warn!(
                            provider = adapter.name(),
                            error = %e,
                            "provider close() returned an error (ignored)"
                        );
                    }
                }
            })
            .collect();
        futures::future::join_all(futures).await;
    }

    /// Returns the name of the default provider.
    pub fn default_provider(&self) -> &str {
        &self.inner.default_provider
    }

    /// Returns `true` if a provider with the given name is registered.
    pub fn has_provider(&self, name: &str) -> bool {
        self.inner.providers.contains_key(name)
    }

    /// Returns the names of all registered providers (unspecified order).
    pub fn provider_names(&self) -> Vec<&str> {
        self.inner
            .provider_order
            .iter()
            .map(String::as_str)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ClientBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`Client`].
pub struct ClientBuilder {
    providers: HashMap<String, Arc<dyn ProviderAdapter>>,
    /// Tracks insertion order so the default can be the first-registered provider.
    provider_order: Vec<String>,
    default_provider: Option<String>,
    middleware: Vec<Box<dyn Middleware>>,
}

impl ClientBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            provider_order: Vec::new(),
            default_provider: None,
            middleware: Vec::new(),
        }
    }

    /// Register a provider adapter under the given name.
    ///
    /// If `name` is already registered, the new adapter replaces the old one.
    pub fn provider(
        mut self,
        name: impl Into<String>,
        adapter: impl ProviderAdapter + 'static,
    ) -> Self {
        let name = name.into();
        self.register_arc(name, Arc::new(adapter));
        self
    }

    /// Register a pre-boxed provider adapter.
    pub fn provider_boxed(
        mut self,
        name: impl Into<String>,
        adapter: Box<dyn ProviderAdapter>,
    ) -> Self {
        let name = name.into();
        // Convert Box<dyn ProviderAdapter> → Arc<dyn ProviderAdapter>
        let arc: Arc<dyn ProviderAdapter> = Arc::from(adapter);
        self.register_arc(name, arc);
        self
    }

    fn register_arc(&mut self, name: String, arc: Arc<dyn ProviderAdapter>) {
        if !self.providers.contains_key(&name) {
            self.provider_order.push(name.clone());
        }
        self.providers.insert(name, arc);
    }

    /// Set the default provider name.
    ///
    /// Must match a registered provider name; validated at [`build`][Self::build].
    /// If not called, the default is the first provider registered.
    pub fn default_provider(mut self, name: impl Into<String>) -> Self {
        self.default_provider = Some(name.into());
        self
    }

    /// Add a middleware layer.
    ///
    /// Middleware is applied in registration order: first-registered = outermost
    /// = called first on request.
    pub fn middleware(mut self, mw: impl Middleware + 'static) -> Self {
        self.middleware.push(Box::new(mw));
        self
    }

    /// Build the [`Client`].
    ///
    /// # Errors
    /// - [`UnifiedLlmError::Configuration`] if no providers are registered.
    /// - [`UnifiedLlmError::Configuration`] if `default_provider` was specified
    ///   but does not match any registered provider.
    /// - Any error returned by a provider's `initialize()` (stops on first
    ///   failure; remaining providers are not initialized).
    pub async fn build(self) -> Result<Client, UnifiedLlmError> {
        if self.providers.is_empty() {
            return Err(UnifiedLlmError::Configuration {
                message: "no providers registered".to_string(),
            });
        }

        // Determine the default provider name.
        let default_provider = match self.default_provider {
            Some(ref name) => {
                if !self.providers.contains_key(name) {
                    return Err(UnifiedLlmError::Configuration {
                        message: format!("default_provider '{name}' is not registered"),
                    });
                }
                name.clone()
            }
            None => self
                .provider_order
                .first()
                .cloned()
                .expect("providers non-empty"),
        };

        // Call initialize() on each provider in insertion order.
        for name in &self.provider_order {
            if let Some(adapter) = self.providers.get(name) {
                adapter.initialize().await?;
            }
        }

        // Convert middleware Vec<Box<dyn Middleware>> → Vec<Arc<dyn Middleware>>.
        let middleware: Vec<Arc<dyn Middleware>> =
            self.middleware.into_iter().map(Arc::from).collect();

        Ok(Client {
            inner: Arc::new(ClientInner {
                providers: self.providers,
                provider_order: self.provider_order,
                default_provider,
                middleware: Arc::new(middleware),
            }),
        })
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockProviderAdapter;
    use crate::types::Message;

    fn user_request(model: &str) -> Request {
        Request::new(model, vec![Message::user("hi")])
    }

    fn user_request_with_provider(model: &str, provider: &str) -> Request {
        let mut req = user_request(model);
        req.provider = Some(provider.to_string());
        req
    }

    // AC-1: build with one mock provider succeeds
    #[tokio::test]
    async fn build_with_mock_succeeds() {
        let client = ClientBuilder::new()
            .provider(
                "mock",
                MockProviderAdapter::default().push_text_response("hi"),
            )
            .build()
            .await
            .unwrap();
        assert!(client.has_provider("mock"));
    }

    // AC-2: complete routes to the provider named in request.provider
    #[tokio::test]
    async fn complete_routes_to_named_provider() {
        let client = ClientBuilder::new()
            .provider(
                "alpha",
                MockProviderAdapter::new("alpha").push_text_response("from-alpha"),
            )
            .provider(
                "beta",
                MockProviderAdapter::new("beta").push_text_response("from-beta"),
            )
            .default_provider("alpha")
            .build()
            .await
            .unwrap();

        let resp = client
            .complete(user_request_with_provider("model", "beta"))
            .await
            .unwrap();
        assert_eq!(resp.text(), "from-beta");
    }

    // AC-3: request.provider == None → use default provider
    #[tokio::test]
    async fn complete_uses_default_provider() {
        let client = ClientBuilder::new()
            .provider(
                "default-one",
                MockProviderAdapter::new("default-one").push_text_response("default-resp"),
            )
            .build()
            .await
            .unwrap();

        let resp = client.complete(user_request("model")).await.unwrap();
        assert_eq!(resp.text(), "default-resp");
    }

    // AC-4: unregistered provider name → Err(Configuration)
    #[tokio::test]
    async fn complete_unknown_provider_returns_config_error() {
        let client = ClientBuilder::new()
            .provider(
                "mock",
                MockProviderAdapter::default().push_text_response("x"),
            )
            .build()
            .await
            .unwrap();

        let err = client
            .complete(user_request_with_provider("model", "nope"))
            .await
            .unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Configuration { .. }));
    }

    // AC-5: has_provider("mock") == true after registering "mock"
    #[tokio::test]
    async fn has_provider_returns_true() {
        let client = ClientBuilder::new()
            .provider("mock", MockProviderAdapter::default())
            .build()
            .await
            .unwrap();
        assert!(client.has_provider("mock"));
        assert!(!client.has_provider("other"));
    }

    // AC-6: provider_names() returns all registered names
    #[tokio::test]
    async fn provider_names_returns_all() {
        let client = ClientBuilder::new()
            .provider("a", MockProviderAdapter::new("a"))
            .provider("b", MockProviderAdapter::new("b"))
            .provider("c", MockProviderAdapter::new("c"))
            .build()
            .await
            .unwrap();
        let mut names = client.provider_names();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    // AC-7: build with no providers → Err(Configuration)
    #[tokio::test]
    async fn build_no_providers_error() {
        let err = ClientBuilder::new().build().await.unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Configuration { .. }));
    }

    // AC-8: mismatched default_provider → Err(Configuration)
    #[tokio::test]
    async fn build_mismatched_default_provider_error() {
        let err = ClientBuilder::new()
            .provider("mock", MockProviderAdapter::default())
            .default_provider("does-not-exist")
            .build()
            .await
            .unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Configuration { .. }));
    }

    // AC-9: build() calls initialize() on the provider
    #[tokio::test]
    async fn build_calls_initialize() {
        let mock = MockProviderAdapter::default();
        // Capture a clone of the Arc counters before building
        let init_check = mock.clone_counters();

        let _client = ClientBuilder::new()
            .provider("mock", mock)
            .build()
            .await
            .unwrap();

        assert_eq!(init_check.initialize_count(), 1);
    }

    // AC-10: Clone produces a valid client
    #[tokio::test]
    async fn client_clone_routes_identically() {
        let client = ClientBuilder::new()
            .provider(
                "mock",
                MockProviderAdapter::default()
                    .push_text_response("a")
                    .push_text_response("b"),
            )
            .build()
            .await
            .unwrap();
        let cloned = client.clone();
        let r1 = client.complete(user_request("model")).await.unwrap();
        let r2 = cloned.complete(user_request("model")).await.unwrap();
        // Both calls go to the same mock queue (shared Arc)
        assert_eq!(r1.text(), "a");
        assert_eq!(r2.text(), "b");
    }

    // AC-11: default_provider() returns correct name
    #[tokio::test]
    async fn default_provider_name() {
        let client = ClientBuilder::new()
            .provider("x", MockProviderAdapter::new("x"))
            .provider("y", MockProviderAdapter::new("y"))
            .default_provider("y")
            .build()
            .await
            .unwrap();
        assert_eq!(client.default_provider(), "y");
    }

    // AC-12: Client is Send + Sync
    #[test]
    fn client_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Client>();
    }

    // Edge: from_env() with no API keys set → Err(Configuration)
    #[test]
    fn from_env_no_keys_returns_config_error() {
        // Save and clear all API-key env vars.
        // SAFETY: This test runs in its own process snapshot; parallel tests
        // that need these vars should use their own env setup.
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

        let result = Client::from_env();

        // Restore the saved values so other tests in this process are unaffected.
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

        let err = result.unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Configuration { .. }));
    }

    // Edge: first registered provider becomes default when no default_provider() call
    #[tokio::test]
    async fn first_registered_is_default() {
        let client = ClientBuilder::new()
            .provider("first", MockProviderAdapter::new("first"))
            .provider("second", MockProviderAdapter::new("second"))
            .build()
            .await
            .unwrap();
        assert_eq!(client.default_provider(), "first");
    }

    // Edge: registering same name twice → second overwrites
    #[tokio::test]
    async fn registering_same_name_overwrites() {
        let client = ClientBuilder::new()
            .provider(
                "mock",
                MockProviderAdapter::new("mock").push_text_response("first-adapter"),
            )
            .provider(
                "mock",
                MockProviderAdapter::new("mock").push_text_response("second-adapter"),
            )
            .build()
            .await
            .unwrap();

        let resp = client.complete(user_request("model")).await.unwrap();
        assert_eq!(resp.text(), "second-adapter");
    }
}
