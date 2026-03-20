//! Retry policy with exponential backoff and full jitter.
//!
//! [`RetryPolicy`] executes an async operation with configurable retry
//! behaviour. Only errors where [`UnifiedLlmError::is_retryable`] returns
//! `true` are retried; non-retryable errors are returned immediately.
//!
//! The jitter algorithm is "full jitter": each attempt sleeps for a uniformly
//! random duration in `[0, computed_delay]`, which spreads thundering-herd
//! load more effectively than "equal jitter". See the AWS Architecture Blog
//! post "Exponential Backoff And Jitter" for background.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::UnifiedLlmError;

// ---------------------------------------------------------------------------
// RetryConfig
// ---------------------------------------------------------------------------

/// Configuration for exponential backoff retry behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of attempts **including** the first try. Default: `3`.
    ///
    /// If set to `0`, treated as `1` — at least one attempt is always made.
    pub max_attempts: u32,
    /// Delay before the **first** retry. Default: `500 ms`.
    pub initial_delay: Duration,
    /// Multiplier applied to the delay after each retry. Default: `2.0`.
    pub backoff_factor: f64,
    /// Hard cap on computed delay regardless of attempt count. Default: `60 s`.
    pub max_delay: Duration,
    /// If `true`, apply full jitter to each computed delay. Default: `true`.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(500),
            backoff_factor: 2.0,
            max_delay: Duration::from_secs(60),
            jitter: true,
        }
    }
}

// ---------------------------------------------------------------------------
// RetryPolicy
// ---------------------------------------------------------------------------

/// A retry policy that executes an async operation with exponential backoff.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub config: RetryConfig,
}

impl RetryPolicy {
    /// Create a policy with the given configuration.
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Create a policy with default settings: 3 attempts, 500 ms base,
    /// 2× factor, 60 s cap, jitter enabled.
    pub fn default_policy() -> Self {
        Self::new(RetryConfig::default())
    }

    /// Create a policy with `max_attempts = 1` (no retries).
    pub fn no_retry() -> Self {
        Self::new(RetryConfig {
            max_attempts: 1,
            ..RetryConfig::default()
        })
    }

    /// Compute the base (pre-jitter) delay before attempt `attempt_number`.
    ///
    /// - `attempt_number = 1` → delay before the **second** try (first retry).
    /// - `base_delay(n) = min(initial_delay × backoff_factor^(n−1), max_delay)`.
    ///
    /// Does **not** apply jitter. Use [`sleep_duration`][Self::sleep_duration]
    /// for actual sleep durations.
    pub fn base_delay(&self, attempt_number: u32) -> Duration {
        if attempt_number == 0 {
            return self.config.initial_delay;
        }
        let exponent = (attempt_number - 1) as i32;
        let initial_secs = self.config.initial_delay.as_secs_f64();
        let max_secs = self.config.max_delay.as_secs_f64();
        let multiplied_secs = initial_secs * self.config.backoff_factor.powi(exponent);
        // Guard against overflow/infinity before converting to Duration.
        // If the computed value is not finite or already exceeds the cap,
        // return max_delay directly.
        if !multiplied_secs.is_finite() || multiplied_secs >= max_secs {
            return self.config.max_delay;
        }
        Duration::from_secs_f64(multiplied_secs)
    }

    /// Compute the actual sleep duration for attempt `attempt_number`,
    /// optionally floored by a provider-supplied `retry_after_secs` hint.
    ///
    /// `retry_after_secs` is used as a **floor** (not a ceiling). If the
    /// provider says "wait at least 30 s", the sleep duration will be
    /// `max(computed, 30 s)`.
    ///
    /// When `config.jitter = true`, full jitter is applied to the base delay:
    /// a uniformly random value in `[0, base_delay]` is chosen before
    /// comparing against `retry_after_secs`.
    pub fn sleep_duration(&self, attempt_number: u32, retry_after_secs: Option<f64>) -> Duration {
        let base = self.base_delay(attempt_number);

        let after_jitter = if self.config.jitter {
            let base_nanos = base.as_nanos() as u64;
            if base_nanos == 0 {
                Duration::ZERO
            } else {
                use rand::Rng as _;
                let jitter_nanos = rand::rng().random_range(0..=base_nanos);
                Duration::from_nanos(jitter_nanos)
            }
        } else {
            base
        };

        // Provider hint acts as a floor, but is still capped by max_delay.
        match retry_after_secs {
            Some(secs) if secs > 0.0 => {
                let hint = Duration::from_secs_f64(secs);
                after_jitter.max(hint).min(self.config.max_delay)
            }
            _ => after_jitter,
        }
    }

    /// Execute `operation` with retry.
    ///
    /// - Calls `operation()` for attempt 1.
    /// - If the result is `Err(e)` and `e.is_retryable()` and attempts remain,
    ///   sleeps for `sleep_duration(n, e.retry_after())` then retries.
    /// - Non-retryable errors are returned **immediately** without further
    ///   attempts.
    /// - After `config.max_attempts` attempts all fail, returns the last error.
    ///
    /// The `Fn` bound (not `FnMut`) means the closure must be callable multiple
    /// times without consuming state.
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T, UnifiedLlmError>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, UnifiedLlmError>> + Send,
        T: Send,
    {
        // Treat max_attempts == 0 as 1.
        let max_attempts = self.config.max_attempts.max(1);

        let mut last_error: Option<UnifiedLlmError> = None;

        for attempt in 0..max_attempts {
            match operation().await {
                Ok(value) => return Ok(value),
                Err(e) => {
                    let retryable = e.is_retryable();
                    let retry_after = e.retry_after();
                    last_error = Some(e);

                    // Non-retryable or last attempt → return immediately.
                    if !retryable || attempt + 1 >= max_attempts {
                        break;
                    }

                    // Sleep before the next attempt (1-indexed for delay calc).
                    let delay = self.sleep_duration(attempt + 1, retry_after);
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.expect("loop must have run at least once"))
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::default_policy()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn default_policy() -> RetryPolicy {
        RetryPolicy::default_policy()
    }

    fn no_jitter_policy() -> RetryPolicy {
        RetryPolicy::new(RetryConfig {
            jitter: false,
            ..RetryConfig::default()
        })
    }

    // AC-1: default config has max_attempts == 3
    #[test]
    fn default_max_attempts_is_3() {
        assert_eq!(default_policy().config.max_attempts, 3);
    }

    // AC-2: base_delay(1) == initial_delay
    #[test]
    fn base_delay_first_retry() {
        let p = no_jitter_policy();
        assert_eq!(p.base_delay(1), p.config.initial_delay);
    }

    // AC-3: base_delay(3) == min(initial_delay * 4.0, max_delay) for factor=2.0
    #[test]
    fn base_delay_third_retry() {
        let p = no_jitter_policy();
        let expected = (p.config.initial_delay.mul_f64(4.0)).min(p.config.max_delay);
        assert_eq!(p.base_delay(3), expected);
    }

    // AC-4: base_delay(n) never exceeds max_delay
    #[test]
    fn base_delay_never_exceeds_max() {
        let p = no_jitter_policy();
        for n in 1..=100 {
            assert!(
                p.base_delay(n) <= p.config.max_delay,
                "attempt {n}: delay exceeds max_delay"
            );
        }
    }

    // AC-5: sleep_duration(1, Some(30.0)) is at least 30 seconds
    #[test]
    fn sleep_duration_respects_retry_after_floor() {
        let p = no_jitter_policy();
        let d = p.sleep_duration(1, Some(30.0));
        assert!(
            d >= Duration::from_secs(30),
            "sleep_duration should be >= 30s, got {d:?}"
        );
    }

    // AC-6: retryable op that fails twice then succeeds returns Ok
    #[tokio::test(flavor = "current_thread")]
    async fn retryable_fails_twice_then_succeeds() {
        tokio::time::pause();

        let count = Arc::new(Mutex::new(0u32));
        let count_c = Arc::clone(&count);

        let policy = RetryPolicy::new(RetryConfig {
            max_attempts: 3,
            jitter: false,
            initial_delay: Duration::from_millis(1),
            backoff_factor: 1.0,
            max_delay: Duration::from_secs(1),
        });

        let result = policy
            .execute(|| {
                let c = Arc::clone(&count_c);
                async move {
                    let mut guard = c.lock().unwrap();
                    *guard += 1;
                    let n = *guard;
                    drop(guard);
                    if n < 3 {
                        Err(UnifiedLlmError::RateLimit {
                            provider: "test".to_string(),
                            message: "slow".to_string(),
                            retry_after: None,
                        })
                    } else {
                        Ok(n)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 3);
        assert_eq!(*count.lock().unwrap(), 3);
    }

    // AC-7: non-retryable error returned after exactly 1 attempt
    #[tokio::test(flavor = "current_thread")]
    async fn non_retryable_exits_immediately() {
        let count = Arc::new(Mutex::new(0u32));
        let count_c = Arc::clone(&count);

        let policy = RetryPolicy::new(RetryConfig {
            max_attempts: 5,
            jitter: false,
            initial_delay: Duration::ZERO,
            backoff_factor: 1.0,
            max_delay: Duration::ZERO,
        });

        let _ = policy
            .execute(|| {
                let c = Arc::clone(&count_c);
                async move {
                    *c.lock().unwrap() += 1;
                    Err::<(), _>(UnifiedLlmError::Authentication {
                        provider: "test".to_string(),
                        message: "bad key".to_string(),
                    })
                }
            })
            .await;

        assert_eq!(
            *count.lock().unwrap(),
            1,
            "should have called op exactly once"
        );
    }

    // AC-8: always-failing retryable error tried exactly max_attempts times
    #[tokio::test(flavor = "current_thread")]
    async fn retryable_exhausts_all_attempts() {
        tokio::time::pause();

        let count = Arc::new(Mutex::new(0u32));
        let count_c = Arc::clone(&count);

        let policy = RetryPolicy::new(RetryConfig {
            max_attempts: 4,
            jitter: false,
            initial_delay: Duration::from_millis(1),
            backoff_factor: 1.0,
            max_delay: Duration::from_secs(1),
        });

        let result = policy
            .execute(|| {
                let c = Arc::clone(&count_c);
                async move {
                    *c.lock().unwrap() += 1;
                    Err::<(), _>(UnifiedLlmError::RequestTimeout {
                        message: "timeout".to_string(),
                    })
                }
            })
            .await;

        assert!(result.is_err());
        assert_eq!(*count.lock().unwrap(), 4);
    }

    // AC-9: no_retry() calls op exactly once
    #[tokio::test(flavor = "current_thread")]
    async fn no_retry_calls_once() {
        let count = Arc::new(Mutex::new(0u32));
        let count_c = Arc::clone(&count);

        let _ = RetryPolicy::no_retry()
            .execute(|| {
                let c = Arc::clone(&count_c);
                async move {
                    *c.lock().unwrap() += 1;
                    Err::<(), _>(UnifiedLlmError::RateLimit {
                        provider: "test".to_string(),
                        message: "x".to_string(),
                        retry_after: None,
                    })
                }
            })
            .await;

        assert_eq!(*count.lock().unwrap(), 1);
    }

    // AC-10: jitter=false → sleep_duration returns exact base delay
    #[test]
    fn no_jitter_returns_exact_base_delay() {
        let p = no_jitter_policy();
        let base = p.base_delay(1);
        // Run multiple times — must be deterministic
        for _ in 0..10 {
            assert_eq!(p.sleep_duration(1, None), base);
        }
    }

    // Edge: max_attempts=0 → treated as 1
    #[tokio::test(flavor = "current_thread")]
    async fn zero_attempts_treated_as_one() {
        let count = Arc::new(Mutex::new(0u32));
        let count_c = Arc::clone(&count);

        let policy = RetryPolicy::new(RetryConfig {
            max_attempts: 0,
            jitter: false,
            initial_delay: Duration::ZERO,
            backoff_factor: 1.0,
            max_delay: Duration::ZERO,
        });

        let _ = policy
            .execute(|| {
                let c = Arc::clone(&count_c);
                async move {
                    *c.lock().unwrap() += 1;
                    Ok::<(), UnifiedLlmError>(())
                }
            })
            .await;

        assert_eq!(*count.lock().unwrap(), 1);
    }

    // Edge: backoff_factor = 1.0 → all retries use initial_delay
    #[test]
    fn factor_one_no_growth() {
        let p = RetryPolicy::new(RetryConfig {
            max_attempts: 5,
            initial_delay: Duration::from_millis(200),
            backoff_factor: 1.0,
            max_delay: Duration::from_secs(60),
            jitter: false,
        });
        for n in 1..=5 {
            assert_eq!(p.base_delay(n), Duration::from_millis(200));
        }
    }

    // Edge: retry_after_secs == 0.0 → base delay used (max(base, 0) == base)
    #[test]
    fn retry_after_zero_uses_base() {
        let p = no_jitter_policy();
        let base = p.base_delay(1);
        assert_eq!(p.sleep_duration(1, Some(0.0)), base);
    }

    // Edge: initial_delay == 0 → all delays are zero
    #[test]
    fn zero_initial_delay_stays_zero() {
        let p = RetryPolicy::new(RetryConfig {
            initial_delay: Duration::ZERO,
            jitter: false,
            ..RetryConfig::default()
        });
        assert_eq!(p.base_delay(1), Duration::ZERO);
        assert_eq!(p.sleep_duration(1, None), Duration::ZERO);
    }

    // V2-ULM-002: retry_after_secs larger than max_delay → capped at max_delay
    #[test]
    fn retry_after_capped_at_max_delay() {
        let max_delay = Duration::from_secs(60);
        let p = RetryPolicy::new(RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(500),
            backoff_factor: 2.0,
            max_delay,
            jitter: false,
        });
        // Provider says "wait 3600 seconds" — must be capped at max_delay (60s)
        let d = p.sleep_duration(1, Some(3600.0));
        assert!(
            d <= max_delay,
            "sleep_duration with hint=3600s should be capped at max_delay={max_delay:?}, got {d:?}"
        );
    }
}
