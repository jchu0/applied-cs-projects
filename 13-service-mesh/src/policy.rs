//! Traffic management policies.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Retry policy.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Conditions to retry on.
    pub retry_on: Vec<RetryCondition>,
    /// Backoff configuration.
    pub backoff: BackoffConfig,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_on: vec![
                RetryCondition::ConnectionFailure,
                RetryCondition::Status5xx,
            ],
            backoff: BackoffConfig::default(),
        }
    }
}

/// Conditions for retrying requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryCondition {
    /// Connection failure.
    ConnectionFailure,
    /// 5xx status codes.
    Status5xx,
    /// Connection reset.
    Reset,
    /// Connect failure.
    ConnectFailure,
    /// Retriable 4xx codes.
    Retriable4xx,
    /// Specific status code.
    StatusCode(u16),
}

/// Backoff configuration.
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Base interval for backoff.
    pub base_interval: Duration,
    /// Maximum interval.
    pub max_interval: Duration,
    /// Jitter factor (0.0 to 1.0).
    pub jitter: f64,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            base_interval: Duration::from_millis(25),
            max_interval: Duration::from_secs(1),
            jitter: 0.2,
        }
    }
}

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failures before opening circuit.
    pub consecutive_failures: u32,
    /// Monitoring interval.
    pub interval: Duration,
    /// Base ejection time.
    pub base_ejection_time: Duration,
    /// Maximum ejection percentage.
    pub max_ejection_percent: u32,
    /// Successes needed to close circuit.
    pub success_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            consecutive_failures: 5,
            interval: Duration::from_secs(10),
            base_ejection_time: Duration::from_secs(30),
            max_ejection_percent: 50,
            success_threshold: 3,
        }
    }
}

/// Circuit breaker state.
#[derive(Debug, Clone)]
pub enum CircuitState {
    /// Normal operation, requests allowed.
    Closed,
    /// Rejecting requests.
    Open { opened_at: Instant, failures: u32 },
    /// Testing recovery.
    HalfOpen { successes: u32, allowed: u32 },
}

/// Circuit breaker.
pub struct CircuitBreaker {
    /// Configuration.
    config: CircuitBreakerConfig,
    /// Current state.
    state: RwLock<CircuitState>,
    /// Failure count.
    failures: AtomicU32,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: RwLock::new(CircuitState::Closed),
            failures: AtomicU32::new(0),
        }
    }

    /// Check if circuit is open (should reject requests).
    pub fn is_open(&self) -> bool {
        let state = self.state.read();
        match *state {
            CircuitState::Open { opened_at, .. } => {
                // Check if we should transition to half-open
                if opened_at.elapsed() > self.config.base_ejection_time {
                    drop(state);
                    let mut state = self.state.write();
                    *state = CircuitState::HalfOpen {
                        successes: 0,
                        allowed: 1,
                    };
                    false
                } else {
                    true
                }
            }
            CircuitState::HalfOpen { allowed, .. } => allowed == 0,
            CircuitState::Closed => false,
        }
    }

    /// Record a successful request.
    pub fn record_success(&self) {
        let mut state = self.state.write();
        match *state {
            CircuitState::HalfOpen { successes, .. } => {
                let new_successes = successes + 1;
                if new_successes >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    self.failures.store(0, Ordering::SeqCst);
                } else {
                    *state = CircuitState::HalfOpen {
                        successes: new_successes,
                        allowed: 1,
                    };
                }
            }
            CircuitState::Closed => {
                self.failures.store(0, Ordering::SeqCst);
            }
            _ => {}
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::SeqCst) + 1;

        if failures >= self.config.consecutive_failures {
            let mut state = self.state.write();
            *state = CircuitState::Open {
                opened_at: Instant::now(),
                failures,
            };
        }
    }

    /// Get current state.
    pub fn state(&self) -> CircuitState {
        self.state.read().clone()
    }
}

/// Timeout policy.
#[derive(Debug, Clone)]
pub struct TimeoutPolicy {
    /// Request timeout.
    pub request_timeout: Duration,
    /// Idle timeout.
    pub idle_timeout: Duration,
}

impl Default for TimeoutPolicy {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
        }
    }
}

/// Service policy combining traffic management policies.
#[derive(Debug, Clone)]
pub struct ServicePolicy {
    /// Retry policy.
    pub retry: Option<RetryPolicy>,
    /// Timeout.
    pub timeout: Option<Duration>,
    /// Circuit breaker.
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    /// mTLS mode.
    pub mtls_mode: MtlsMode,
    /// Authorization policy.
    pub authorization: AuthorizationPolicy,
}

impl Default for ServicePolicy {
    fn default() -> Self {
        Self {
            retry: Some(RetryPolicy::default()),
            timeout: Some(Duration::from_secs(30)),
            circuit_breaker: Some(CircuitBreakerConfig::default()),
            mtls_mode: MtlsMode::Strict,
            authorization: AuthorizationPolicy::default(),
        }
    }
}

/// mTLS mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtlsMode {
    /// Disable mTLS.
    Disable,
    /// Accept both plaintext and mTLS.
    Permissive,
    /// mTLS only.
    Strict,
}

/// Authorization policy.
#[derive(Debug, Clone)]
pub struct AuthorizationPolicy {
    /// Action (allow or deny).
    pub action: AuthAction,
    /// Authorization rules.
    pub rules: Vec<AuthRule>,
}

impl Default for AuthorizationPolicy {
    fn default() -> Self {
        Self {
            action: AuthAction::Allow,
            rules: Vec::new(),
        }
    }
}

/// Authorization action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthAction {
    /// Allow matching requests.
    Allow,
    /// Deny matching requests.
    Deny,
}

/// Authorization rule.
#[derive(Debug, Clone)]
pub struct AuthRule {
    /// Source principals (SPIFFE IDs).
    pub principals: Vec<String>,
    /// Source namespaces.
    pub namespaces: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_states() {
        let config = CircuitBreakerConfig {
            consecutive_failures: 3,
            success_threshold: 2,
            base_ejection_time: Duration::from_millis(100),
            ..Default::default()
        };

        let cb = CircuitBreaker::new(config);

        // Initially closed
        assert!(!cb.is_open());

        // Record failures
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());

        cb.record_failure();
        assert!(cb.is_open());

        // Wait for ejection time
        std::thread::sleep(Duration::from_millis(150));

        // Should be half-open now
        assert!(!cb.is_open());

        // Record successes
        cb.record_success();
        cb.record_success();

        // Should be closed
        assert!(!cb.is_open());
    }
}
