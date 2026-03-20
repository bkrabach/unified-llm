//! Middleware traits and chain for `unified-llm`.
//!
//! [`Middleware`] forms a chain: each layer receives the request, may modify
//! it, calls `next.next()` to proceed, and may modify the response before
//! returning.
//!
//! [`MiddlewareChain`] executes a stack of middleware followed by the terminal
//! provider adapter. Middleware is applied in registration order for requests
//! (first-registered = outermost = called first); responses pass through in
//! reverse order.
//!
//! # Iterative vs Recursive
//!
//! The chain uses a `ChainNext` helper struct that implements `MiddlewareNext`
//! and holds a shared `Arc` pointer to the middleware vec plus an index. Each
//! call to `ChainNext::next()` allocates a new `ChainNext` with `index + 1`
//! and calls the next middleware. Although this looks like recursion, every
//! `async fn` call through `async_trait` allocates a heap-boxed future — no
//! stack frames accumulate between awaits, so deep chains do not overflow.

use std::sync::Arc;

use crate::error::UnifiedLlmError;
use crate::providers::ProviderAdapter;
use crate::streaming::EventStream;
use crate::types::{Request, Response};

// ---------------------------------------------------------------------------
// MiddlewareNext
// ---------------------------------------------------------------------------

/// Continuation interface for non-streaming middleware.
#[async_trait::async_trait]
pub trait MiddlewareNext: Send + Sync {
    /// Call the next handler in the chain with the given request.
    async fn next(&self, request: Request) -> Result<Response, UnifiedLlmError>;
}

// ---------------------------------------------------------------------------
// MiddlewareStreamNext
// ---------------------------------------------------------------------------

/// Continuation interface for streaming middleware.
#[async_trait::async_trait]
pub trait MiddlewareStreamNext: Send + Sync {
    /// Call the next handler in the chain, returning an event stream.
    async fn next(&self, request: Request) -> Result<EventStream, UnifiedLlmError>;
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/// Trait for request/response middleware.
///
/// Middleware forms a chain: each layer receives the request, may modify it,
/// calls `next.next()` to proceed, and may modify the response before
/// returning.
///
/// Only `process` is required. `process_stream` has a default implementation
/// that passes the request through unchanged — middleware that only intercepts
/// non-streaming requests can leave `process_stream` at its default.
#[async_trait::async_trait]
pub trait Middleware: Send + Sync {
    /// Process a non-streaming request through the middleware.
    async fn process(
        &self,
        request: Request,
        next: &dyn MiddlewareNext,
    ) -> Result<Response, UnifiedLlmError>;

    /// Process a streaming request through the middleware.
    ///
    /// Default: pass through to `next.next(request).await` without
    /// modification. Middleware that only needs to intercept non-streaming
    /// requests can rely on this default.
    async fn process_stream(
        &self,
        request: Request,
        next: &dyn MiddlewareStreamNext,
    ) -> Result<EventStream, UnifiedLlmError> {
        next.next(request).await
    }
}

// ---------------------------------------------------------------------------
// MiddlewareChain internals
// ---------------------------------------------------------------------------

/// Non-streaming chain-next: calls middleware[index] then recurses.
struct ChainNext {
    middlewares: Arc<Vec<Arc<dyn Middleware>>>,
    adapter: Arc<dyn ProviderAdapter>,
    index: usize,
}

#[async_trait::async_trait]
impl MiddlewareNext for ChainNext {
    async fn next(&self, request: Request) -> Result<Response, UnifiedLlmError> {
        if self.index >= self.middlewares.len() {
            // Terminal: call the provider adapter.
            return self.adapter.complete(&request).await;
        }
        let mw = Arc::clone(&self.middlewares[self.index]);
        let next_in_chain = ChainNext {
            middlewares: Arc::clone(&self.middlewares),
            adapter: Arc::clone(&self.adapter),
            index: self.index + 1,
        };
        mw.process(request, &next_in_chain).await
    }
}

/// Streaming chain-next: calls middleware[index].process_stream then recurses.
struct ChainStreamNext {
    middlewares: Arc<Vec<Arc<dyn Middleware>>>,
    adapter: Arc<dyn ProviderAdapter>,
    index: usize,
}

#[async_trait::async_trait]
impl MiddlewareStreamNext for ChainStreamNext {
    async fn next(&self, request: Request) -> Result<EventStream, UnifiedLlmError> {
        if self.index >= self.middlewares.len() {
            // Terminal: call the provider adapter stream.
            return self.adapter.stream(&request).await;
        }
        let mw = Arc::clone(&self.middlewares[self.index]);
        let next_in_chain = ChainStreamNext {
            middlewares: Arc::clone(&self.middlewares),
            adapter: Arc::clone(&self.adapter),
            index: self.index + 1,
        };
        mw.process_stream(request, &next_in_chain).await
    }
}

// ---------------------------------------------------------------------------
// MiddlewareChain
// ---------------------------------------------------------------------------

/// Executes a stack of middleware followed by the provider adapter.
///
/// Middleware is applied in registration order for requests
/// (first-registered = outermost = called first). Responses pass through in
/// reverse order (last-registered = innermost = first to see response).
pub struct MiddlewareChain {
    middlewares: Arc<Vec<Arc<dyn Middleware>>>,
    adapter: Arc<dyn ProviderAdapter>,
}

impl MiddlewareChain {
    /// Create a chain from a vec of boxed middleware and a shared adapter.
    ///
    /// Internally converts `Box<dyn Middleware>` → `Arc<dyn Middleware>` so
    /// the chain can be cheaply shared across concurrent calls.
    pub fn new(middlewares: Vec<Box<dyn Middleware>>, adapter: Arc<dyn ProviderAdapter>) -> Self {
        let arcs: Vec<Arc<dyn Middleware>> = middlewares.into_iter().map(Arc::from).collect();
        Self {
            middlewares: Arc::new(arcs),
            adapter,
        }
    }

    /// Create a chain from pre-made `Arc<dyn Middleware>` slices.
    ///
    /// Used internally by `Client` to avoid re-boxing on every call.
    pub(crate) fn from_arcs(
        middlewares: Arc<Vec<Arc<dyn Middleware>>>,
        adapter: Arc<dyn ProviderAdapter>,
    ) -> Self {
        Self {
            middlewares,
            adapter,
        }
    }

    /// Execute the chain for a non-streaming request.
    pub async fn complete(&self, request: Request) -> Result<Response, UnifiedLlmError> {
        let next = ChainNext {
            middlewares: Arc::clone(&self.middlewares),
            adapter: Arc::clone(&self.adapter),
            index: 0,
        };
        next.next(request).await
    }

    /// Execute the chain for a streaming request.
    pub async fn stream(&self, request: Request) -> Result<EventStream, UnifiedLlmError> {
        let next = ChainStreamNext {
            middlewares: Arc::clone(&self.middlewares),
            adapter: Arc::clone(&self.adapter),
            index: 0,
        };
        next.next(request).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::StreamEvent;
    use crate::types::{FinishReason, Message, Usage};
    use futures::stream;
    use std::sync::Mutex;

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn ok_response(text: &str) -> Response {
        Response {
            id: "r1".to_string(),
            model: "mock".to_string(),
            provider: "mock".to_string(),
            message: Message::assistant(text),
            finish_reason: FinishReason::stop(),
            usage: Usage::default(),
            raw: None,
            warnings: vec![],
            rate_limit: None,
        }
    }

    fn make_request() -> Request {
        crate::types::Request::new("mock-model", vec![Message::user("hi")])
    }

    /// Adapter that always returns the given text.
    struct EchoAdapter {
        text: String,
    }

    #[async_trait::async_trait]
    impl ProviderAdapter for EchoAdapter {
        fn name(&self) -> &str {
            "echo"
        }
        async fn complete(&self, _req: &Request) -> Result<Response, UnifiedLlmError> {
            Ok(ok_response(&self.text))
        }
        async fn stream(&self, _req: &Request) -> Result<EventStream, UnifiedLlmError> {
            let s = stream::iter::<Vec<Result<StreamEvent, UnifiedLlmError>>>(vec![]);
            Ok(Box::pin(s))
        }
    }

    /// Adapter that counts calls.
    struct CountingAdapter {
        count: Arc<Mutex<u32>>,
        response_text: String,
    }

    #[async_trait::async_trait]
    impl ProviderAdapter for CountingAdapter {
        fn name(&self) -> &str {
            "counting"
        }
        async fn complete(&self, _req: &Request) -> Result<Response, UnifiedLlmError> {
            *self.count.lock().unwrap() += 1;
            Ok(ok_response(&self.response_text))
        }
        async fn stream(&self, _req: &Request) -> Result<EventStream, UnifiedLlmError> {
            let s = stream::iter::<Vec<Result<StreamEvent, UnifiedLlmError>>>(vec![]);
            Ok(Box::pin(s))
        }
    }

    /// Middleware that appends a tag to the response text and tracks execution order.
    struct TagMiddleware {
        tag: String,
        order: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl Middleware for TagMiddleware {
        async fn process(
            &self,
            request: Request,
            next: &dyn MiddlewareNext,
        ) -> Result<Response, UnifiedLlmError> {
            // Record pre-call
            self.order.lock().unwrap().push(format!("{}-pre", self.tag));
            let mut resp = next.next(request).await?;
            // Record post-call
            self.order
                .lock()
                .unwrap()
                .push(format!("{}-post", self.tag));
            // Append tag to response text
            let new_text = format!("{} [{}]", resp.text(), self.tag);
            resp.message = Message::assistant(&new_text);
            Ok(resp)
        }
    }

    // AC-2: zero middleware → adapter complete called exactly once
    #[tokio::test]
    async fn zero_middleware_calls_adapter_once() {
        let count = Arc::new(Mutex::new(0u32));
        let adapter = Arc::new(CountingAdapter {
            count: Arc::clone(&count),
            response_text: "answer".to_string(),
        });
        let chain = MiddlewareChain::new(vec![], adapter);
        let resp = chain.complete(make_request()).await.unwrap();
        assert_eq!(*count.lock().unwrap(), 1);
        assert_eq!(resp.text(), "answer");
    }

    // AC-3: two middleware execute in registration order for request,
    //       reverse for response
    #[tokio::test]
    async fn two_middleware_execution_order() {
        let order = Arc::new(Mutex::new(Vec::<String>::new()));
        let adapter = Arc::new(EchoAdapter {
            text: "base".to_string(),
        });

        let mws: Vec<Box<dyn Middleware>> = vec![
            Box::new(TagMiddleware {
                tag: "A".to_string(),
                order: Arc::clone(&order),
            }),
            Box::new(TagMiddleware {
                tag: "B".to_string(),
                order: Arc::clone(&order),
            }),
        ];

        let chain = MiddlewareChain::new(mws, adapter);
        let resp = chain.complete(make_request()).await.unwrap();

        let recorded = order.lock().unwrap().clone();
        // Request: A pre, B pre (outermost first)
        // Response: B post, A post (innermost first)
        assert_eq!(recorded, vec!["A-pre", "B-pre", "B-post", "A-post"]);
        // Response text has both tags: "base [B] [A]"
        assert!(resp.text().contains("[A]"));
        assert!(resp.text().contains("[B]"));
    }

    // AC-4: default process_stream passes through to next
    #[tokio::test]
    async fn default_process_stream_passthrough() {
        struct PassthroughMiddleware;

        #[async_trait::async_trait]
        impl Middleware for PassthroughMiddleware {
            async fn process(
                &self,
                request: Request,
                next: &dyn MiddlewareNext,
            ) -> Result<Response, UnifiedLlmError> {
                next.next(request).await
            }
            // process_stream uses default
        }

        let adapter = Arc::new(EchoAdapter {
            text: "streaming".to_string(),
        });
        let chain = MiddlewareChain::new(vec![Box::new(PassthroughMiddleware)], adapter);
        // stream() should not error — assign to _stream to satisfy must_use
        let _stream = chain.stream(make_request()).await.unwrap();
    }

    // AC-5: Box<dyn ProviderAdapter> compiles
    #[test]
    fn boxed_provider_adapter_compiles() {
        fn _accept(_: Box<dyn ProviderAdapter>) {}
    }

    // AC-6: Box<dyn Middleware> compiles
    #[test]
    fn boxed_middleware_compiles() {
        fn _accept(_: Box<dyn Middleware>) {}
    }

    // Middleware that short-circuits (returns Err without calling next)
    #[tokio::test]
    async fn middleware_short_circuit_skips_adapter() {
        let count = Arc::new(Mutex::new(0u32));

        struct ShortCircuit;
        #[async_trait::async_trait]
        impl Middleware for ShortCircuit {
            async fn process(
                &self,
                _request: Request,
                _next: &dyn MiddlewareNext,
            ) -> Result<Response, UnifiedLlmError> {
                Err(UnifiedLlmError::Authentication {
                    provider: "test".to_string(),
                    message: "blocked".to_string(),
                })
            }
        }

        let adapter = Arc::new(CountingAdapter {
            count: Arc::clone(&count),
            response_text: "never".to_string(),
        });
        let chain = MiddlewareChain::new(vec![Box::new(ShortCircuit)], adapter);
        let err = chain.complete(make_request()).await.unwrap_err();
        assert!(matches!(err, UnifiedLlmError::Authentication { .. }));
        assert_eq!(*count.lock().unwrap(), 0, "adapter should not be called");
    }
}
