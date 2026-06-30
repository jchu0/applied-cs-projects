//! Health Checking and Self-Healing
//!
//! Implements health checks (liveness, readiness, startup) and
//! self-healing capabilities for containers.

use super::{Pod, ResourceId, ObjectMeta};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpStream, SocketAddr};
use std::time::{Duration, Instant};
use std::process::Command;

/// Health checker for pods
#[derive(Debug)]
pub struct HealthChecker {
    config: HealthConfig,
    /// History of health check results
    history: HashMap<ResourceId, Vec<ProbeResult>>,
    /// Current health status per pod
    status: HashMap<ResourceId, HealthStatus>,
    /// Consecutive failure counts
    failure_counts: HashMap<ResourceId, u32>,
    /// Consecutive success counts
    success_counts: HashMap<ResourceId, u32>,
}

impl HealthChecker {
    pub fn new(config: HealthConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
            status: HashMap::new(),
            failure_counts: HashMap::new(),
            success_counts: HashMap::new(),
        }
    }

    /// Perform a health check on a pod
    pub fn check(&self, pod: &Pod, health_check: &HealthCheck) -> Result<ProbeResult> {
        let start = Instant::now();

        let result = match &health_check.probe_type {
            ProbeType::Http(http) => self.http_probe(http),
            ProbeType::Tcp(tcp) => self.tcp_probe(tcp),
            ProbeType::Exec(exec) => self.exec_probe(exec),
            ProbeType::Grpc(grpc) => self.grpc_probe(grpc),
        };

        let duration = start.elapsed();

        let success = result.is_ok();
        let message = result.unwrap_or_else(|e| e.to_string());

        Ok(ProbeResult {
            pod_id: pod.metadata.uid.clone(),
            probe_type: health_check.probe_type.clone(),
            success,
            message,
            duration,
            timestamp: Instant::now(),
        })
    }

    /// HTTP health probe
    fn http_probe(&self, config: &HttpProbe) -> std::result::Result<String, String> {
        let addr: SocketAddr = format!("{}:{}", config.host, config.port)
            .parse()
            .map_err(|e| format!("Invalid address: {}", e))?;

        let timeout = Duration::from_secs(config.timeout_seconds as u64);

        let mut stream = TcpStream::connect_timeout(&addr, timeout)
            .map_err(|e| format!("Connection failed: {}", e))?;

        stream.set_read_timeout(Some(timeout)).ok();
        stream.set_write_timeout(Some(timeout)).ok();

        // Build HTTP request
        let request = format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n",
            config.method,
            config.path,
            config.host
        );

        // Add headers
        let mut request_with_headers = request;
        for (key, value) in &config.headers {
            request_with_headers.push_str(&format!("{}: {}\r\n", key, value));
        }
        request_with_headers.push_str("\r\n");

        stream.write_all(request_with_headers.as_bytes())
            .map_err(|e| format!("Write failed: {}", e))?;

        // Read response
        let mut buffer = [0u8; 4096];
        let n = stream.read(&mut buffer)
            .map_err(|e| format!("Read failed: {}", e))?;

        let response = String::from_utf8_lossy(&buffer[..n]);

        // Parse status code
        let status_line = response.lines().next().ok_or("Empty response")?;
        let status_code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or("Invalid status line")?;

        if status_code >= 200 && status_code < 400 {
            Ok(format!("HTTP {} OK", status_code))
        } else {
            Err(format!("HTTP {} error", status_code))
        }
    }

    /// TCP health probe
    fn tcp_probe(&self, config: &TcpProbe) -> std::result::Result<String, String> {
        let addr: SocketAddr = format!("{}:{}", config.host, config.port)
            .parse()
            .map_err(|e| format!("Invalid address: {}", e))?;

        let timeout = Duration::from_secs(config.timeout_seconds as u64);

        TcpStream::connect_timeout(&addr, timeout)
            .map_err(|e| format!("TCP connection failed: {}", e))?;

        Ok("TCP connection successful".to_string())
    }

    /// Exec health probe (runs command in container)
    fn exec_probe(&self, config: &ExecProbe) -> std::result::Result<String, String> {
        if config.command.is_empty() {
            return Err("Empty command".to_string());
        }

        // In a real implementation, this would exec into the container
        // For now, we simulate by running the command locally
        let output = Command::new(&config.command[0])
            .args(&config.command[1..])
            .output()
            .map_err(|e| format!("Exec failed: {}", e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(format!(
                "Command exited with code {}: {}",
                output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    /// gRPC health probe
    fn grpc_probe(&self, config: &GrpcProbe) -> std::result::Result<String, String> {
        // Simplified gRPC health check - just check TCP connectivity
        // In a real implementation, this would use the gRPC health checking protocol
        let addr: SocketAddr = format!("{}:{}", config.host, config.port)
            .parse()
            .map_err(|e| format!("Invalid address: {}", e))?;

        let timeout = Duration::from_secs(config.timeout_seconds as u64);

        TcpStream::connect_timeout(&addr, timeout)
            .map_err(|e| format!("gRPC connection failed: {}", e))?;

        Ok(format!("gRPC service {} is healthy", config.service.as_deref().unwrap_or("default")))
    }

    /// Record a probe result and update status
    pub fn record_result(&mut self, result: ProbeResult) {
        let pod_id = result.pod_id.clone();

        // Update history
        self.history
            .entry(pod_id.clone())
            .or_default()
            .push(result.clone());

        // Trim history to max size
        if let Some(history) = self.history.get_mut(&pod_id) {
            if history.len() > self.config.max_history_size {
                history.remove(0);
            }
        }

        // Update counts
        if result.success {
            *self.success_counts.entry(pod_id.clone()).or_insert(0) += 1;
            self.failure_counts.insert(pod_id.clone(), 0);
        } else {
            *self.failure_counts.entry(pod_id.clone()).or_insert(0) += 1;
            self.success_counts.insert(pod_id.clone(), 0);
        }

        // Update status based on thresholds
        let failures = *self.failure_counts.get(&pod_id).unwrap_or(&0);
        let successes = *self.success_counts.get(&pod_id).unwrap_or(&0);

        let new_status = if failures >= self.config.failure_threshold {
            HealthStatus::Unhealthy
        } else if successes >= self.config.success_threshold {
            HealthStatus::Healthy
        } else {
            self.status.get(&pod_id).cloned().unwrap_or(HealthStatus::Unknown)
        };

        self.status.insert(pod_id, new_status);
    }

    /// Get health status for a pod
    pub fn get_status(&self, pod_id: &ResourceId) -> HealthStatus {
        self.status.get(pod_id).cloned().unwrap_or(HealthStatus::Unknown)
    }

    /// Get probe history for a pod
    pub fn get_history(&self, pod_id: &ResourceId) -> Vec<ProbeResult> {
        self.history.get(pod_id).cloned().unwrap_or_default()
    }

    /// Check if pod should be restarted (liveness failure)
    pub fn should_restart(&self, pod_id: &ResourceId) -> bool {
        let failures = *self.failure_counts.get(pod_id).unwrap_or(&0);
        failures >= self.config.failure_threshold
    }

    /// Check if pod should be removed from service (readiness failure)
    pub fn is_ready(&self, pod_id: &ResourceId) -> bool {
        matches!(self.get_status(pod_id), HealthStatus::Healthy)
    }

    /// Clear health state for a pod
    pub fn clear(&mut self, pod_id: &ResourceId) {
        self.history.remove(pod_id);
        self.status.remove(pod_id);
        self.failure_counts.remove(pod_id);
        self.success_counts.remove(pod_id);
    }
}

/// Health check configuration
#[derive(Clone, Debug)]
pub struct HealthConfig {
    /// Failure threshold before marking unhealthy
    pub failure_threshold: u32,
    /// Success threshold before marking healthy
    pub success_threshold: u32,
    /// Max history entries to keep per pod
    pub max_history_size: usize,
    /// Default probe timeout in seconds
    pub default_timeout_seconds: u32,
    /// Default probe period in seconds
    pub default_period_seconds: u32,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 1,
            max_history_size: 100,
            default_timeout_seconds: 1,
            default_period_seconds: 10,
        }
    }
}

/// Health check definition
#[derive(Clone, Debug)]
pub struct HealthCheck {
    /// Type of probe
    pub probe_type: ProbeType,
    /// Initial delay before first check (seconds)
    pub initial_delay_seconds: u32,
    /// Period between checks (seconds)
    pub period_seconds: u32,
    /// Timeout for each check (seconds)
    pub timeout_seconds: u32,
    /// Number of successes before healthy
    pub success_threshold: u32,
    /// Number of failures before unhealthy
    pub failure_threshold: u32,
}

impl Default for HealthCheck {
    fn default() -> Self {
        Self {
            probe_type: ProbeType::Tcp(TcpProbe::default()),
            initial_delay_seconds: 0,
            period_seconds: 10,
            timeout_seconds: 1,
            success_threshold: 1,
            failure_threshold: 3,
        }
    }
}

impl HealthCheck {
    pub fn http(path: &str, port: u16) -> Self {
        Self {
            probe_type: ProbeType::Http(HttpProbe {
                path: path.to_string(),
                port,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    pub fn tcp(port: u16) -> Self {
        Self {
            probe_type: ProbeType::Tcp(TcpProbe {
                port,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    pub fn exec(command: Vec<String>) -> Self {
        Self {
            probe_type: ProbeType::Exec(ExecProbe { command }),
            ..Default::default()
        }
    }

    pub fn with_initial_delay(mut self, seconds: u32) -> Self {
        self.initial_delay_seconds = seconds;
        self
    }

    pub fn with_period(mut self, seconds: u32) -> Self {
        self.period_seconds = seconds;
        self
    }

    pub fn with_timeout(mut self, seconds: u32) -> Self {
        self.timeout_seconds = seconds;
        self
    }
}

/// Probe type
#[derive(Clone, Debug)]
pub enum ProbeType {
    Http(HttpProbe),
    Tcp(TcpProbe),
    Exec(ExecProbe),
    Grpc(GrpcProbe),
}

/// HTTP probe configuration
#[derive(Clone, Debug)]
pub struct HttpProbe {
    pub path: String,
    pub port: u16,
    pub host: String,
    pub scheme: String,
    pub method: String,
    pub headers: Vec<(String, String)>,
    pub timeout_seconds: u32,
}

impl Default for HttpProbe {
    fn default() -> Self {
        Self {
            path: "/".to_string(),
            port: 80,
            host: "127.0.0.1".to_string(),
            scheme: "HTTP".to_string(),
            method: "GET".to_string(),
            headers: vec![],
            timeout_seconds: 1,
        }
    }
}

/// TCP probe configuration
#[derive(Clone, Debug)]
pub struct TcpProbe {
    pub host: String,
    pub port: u16,
    pub timeout_seconds: u32,
}

impl Default for TcpProbe {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            timeout_seconds: 1,
        }
    }
}

/// Exec probe configuration
#[derive(Clone, Debug)]
pub struct ExecProbe {
    pub command: Vec<String>,
}

/// gRPC probe configuration
#[derive(Clone, Debug)]
pub struct GrpcProbe {
    pub host: String,
    pub port: u16,
    pub service: Option<String>,
    pub timeout_seconds: u32,
}

impl Default for GrpcProbe {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 50051,
            service: None,
            timeout_seconds: 1,
        }
    }
}

/// Health status
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    #[default]
    Unknown,
}

/// Result of a probe
#[derive(Clone, Debug)]
pub struct ProbeResult {
    pub pod_id: ResourceId,
    pub probe_type: ProbeType,
    pub success: bool,
    pub message: String,
    pub duration: Duration,
    pub timestamp: Instant,
}

/// Self-healing controller
#[derive(Debug)]
pub struct SelfHealingController {
    config: SelfHealingConfig,
    restart_counts: HashMap<ResourceId, u32>,
    last_restart: HashMap<ResourceId, Instant>,
}

impl SelfHealingController {
    pub fn new(config: SelfHealingConfig) -> Self {
        Self {
            config,
            restart_counts: HashMap::new(),
            last_restart: HashMap::new(),
        }
    }

    /// Determine action to take based on health status
    pub fn determine_action(&self, pod_id: &ResourceId, status: HealthStatus) -> HealingAction {
        match status {
            HealthStatus::Healthy => HealingAction::None,
            HealthStatus::Unknown => {
                if self.should_start_probe(pod_id) {
                    HealingAction::StartProbe
                } else {
                    HealingAction::None
                }
            }
            HealthStatus::Unhealthy => {
                let restarts = *self.restart_counts.get(pod_id).unwrap_or(&0);

                if restarts >= self.config.max_restarts {
                    if self.config.evict_after_max_restarts {
                        HealingAction::Evict
                    } else {
                        HealingAction::None
                    }
                } else if self.can_restart(pod_id) {
                    HealingAction::Restart
                } else {
                    HealingAction::Wait
                }
            }
        }
    }

    /// Record a restart
    pub fn record_restart(&mut self, pod_id: &ResourceId) {
        *self.restart_counts.entry(pod_id.clone()).or_insert(0) += 1;
        self.last_restart.insert(pod_id.clone(), Instant::now());
    }

    /// Check if pod can be restarted
    fn can_restart(&self, pod_id: &ResourceId) -> bool {
        if let Some(last) = self.last_restart.get(pod_id) {
            last.elapsed() >= self.config.restart_backoff
        } else {
            true
        }
    }

    /// Check if probing should start
    fn should_start_probe(&self, _pod_id: &ResourceId) -> bool {
        true
    }

    /// Reset restart count for a pod
    pub fn reset_restart_count(&mut self, pod_id: &ResourceId) {
        self.restart_counts.remove(pod_id);
        self.last_restart.remove(pod_id);
    }

    /// Get restart count
    pub fn get_restart_count(&self, pod_id: &ResourceId) -> u32 {
        *self.restart_counts.get(pod_id).unwrap_or(&0)
    }

    /// Calculate backoff duration for next restart
    pub fn get_backoff_duration(&self, pod_id: &ResourceId) -> Duration {
        let restarts = *self.restart_counts.get(pod_id).unwrap_or(&0);
        let multiplier = 2u32.pow(restarts.min(6)); // Cap at 64x

        Duration::from_secs(
            (self.config.restart_backoff.as_secs() * multiplier as u64)
                .min(self.config.max_backoff.as_secs())
        )
    }
}

/// Self-healing configuration
#[derive(Clone, Debug)]
pub struct SelfHealingConfig {
    /// Maximum restarts before giving up
    pub max_restarts: u32,
    /// Backoff between restarts
    pub restart_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Whether to evict pod after max restarts
    pub evict_after_max_restarts: bool,
}

impl Default for SelfHealingConfig {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            restart_backoff: Duration::from_secs(10),
            max_backoff: Duration::from_secs(300),
            evict_after_max_restarts: true,
        }
    }
}

/// Action to take for self-healing
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealingAction {
    None,
    StartProbe,
    Restart,
    Wait,
    Evict,
}

/// Pod restart tracker
#[derive(Debug, Default)]
pub struct RestartTracker {
    restarts: HashMap<ResourceId, Vec<RestartRecord>>,
}

impl RestartTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, pod_id: ResourceId, reason: RestartReason) {
        self.restarts
            .entry(pod_id)
            .or_default()
            .push(RestartRecord {
                timestamp: Instant::now(),
                reason,
            });
    }

    pub fn count(&self, pod_id: &ResourceId) -> usize {
        self.restarts.get(pod_id).map(|v| v.len()).unwrap_or(0)
    }

    pub fn recent_restarts(&self, pod_id: &ResourceId, within: Duration) -> usize {
        let now = Instant::now();
        self.restarts.get(pod_id)
            .map(|records| {
                records.iter()
                    .filter(|r| now.duration_since(r.timestamp) <= within)
                    .count()
            })
            .unwrap_or(0)
    }

    pub fn clear(&mut self, pod_id: &ResourceId) {
        self.restarts.remove(pod_id);
    }
}

/// Restart record
#[derive(Clone, Debug)]
pub struct RestartRecord {
    pub timestamp: Instant,
    pub reason: RestartReason,
}

/// Reason for restart
#[derive(Clone, Debug)]
pub enum RestartReason {
    LivenessFailure,
    OOMKilled,
    ContainerError { exit_code: i32 },
    Manual,
    NodeFailure,
}

/// Container crash loop detector
#[derive(Debug)]
pub struct CrashLoopDetector {
    window: Duration,
    threshold: usize,
    crashes: HashMap<ResourceId, Vec<Instant>>,
}

impl CrashLoopDetector {
    pub fn new(window: Duration, threshold: usize) -> Self {
        Self {
            window,
            threshold,
            crashes: HashMap::new(),
        }
    }

    pub fn record_crash(&mut self, pod_id: &ResourceId) {
        self.crashes
            .entry(pod_id.clone())
            .or_default()
            .push(Instant::now());

        // Clean up old crashes
        self.cleanup(pod_id);
    }

    pub fn is_crash_looping(&self, pod_id: &ResourceId) -> bool {
        self.crashes.get(pod_id)
            .map(|crashes| crashes.len() >= self.threshold)
            .unwrap_or(false)
    }

    pub fn get_backoff(&self, pod_id: &ResourceId) -> Duration {
        let crash_count = self.crashes.get(pod_id)
            .map(|c| c.len())
            .unwrap_or(0);

        // Exponential backoff: 10s, 20s, 40s, 80s, 160s (max 5min)
        let secs = (10 * 2u64.pow(crash_count as u32)).min(300);
        Duration::from_secs(secs)
    }

    fn cleanup(&mut self, pod_id: &ResourceId) {
        let now = Instant::now();
        if let Some(crashes) = self.crashes.get_mut(pod_id) {
            crashes.retain(|t| now.duration_since(*t) <= self.window);
        }
    }

    pub fn reset(&mut self, pod_id: &ResourceId) {
        self.crashes.remove(pod_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn create_test_pod() -> Pod {
        Pod {
            metadata: ObjectMeta::new("test-pod", "default"),
            spec: super::super::pod::PodSpec::default(),
            status: super::super::pod::PodStatus::default(),
        }
    }

    #[test]
    fn test_health_checker_new() {
        let config = HealthConfig::default();
        let checker = HealthChecker::new(config);
        assert!(checker.history.is_empty());
    }

    #[test]
    fn test_health_status_default() {
        let config = HealthConfig::default();
        let checker = HealthChecker::new(config);
        let pod = create_test_pod();
        assert_eq!(checker.get_status(&pod.metadata.uid), HealthStatus::Unknown);
    }

    #[test]
    fn test_record_success() {
        let mut checker = HealthChecker::new(HealthConfig::default());
        let pod = create_test_pod();

        let result = ProbeResult {
            pod_id: pod.metadata.uid.clone(),
            probe_type: ProbeType::Tcp(TcpProbe::default()),
            success: true,
            message: "OK".into(),
            duration: Duration::from_millis(10),
            timestamp: Instant::now(),
        };

        checker.record_result(result);
        assert_eq!(checker.get_status(&pod.metadata.uid), HealthStatus::Healthy);
    }

    #[test]
    fn test_record_failures() {
        let config = HealthConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let mut checker = HealthChecker::new(config);
        let pod = create_test_pod();

        // First failure
        checker.record_result(ProbeResult {
            pod_id: pod.metadata.uid.clone(),
            probe_type: ProbeType::Tcp(TcpProbe::default()),
            success: false,
            message: "Failed".into(),
            duration: Duration::from_millis(10),
            timestamp: Instant::now(),
        });
        assert_eq!(checker.get_status(&pod.metadata.uid), HealthStatus::Unknown);

        // Second failure -> unhealthy
        checker.record_result(ProbeResult {
            pod_id: pod.metadata.uid.clone(),
            probe_type: ProbeType::Tcp(TcpProbe::default()),
            success: false,
            message: "Failed".into(),
            duration: Duration::from_millis(10),
            timestamp: Instant::now(),
        });
        assert_eq!(checker.get_status(&pod.metadata.uid), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_should_restart() {
        let config = HealthConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let mut checker = HealthChecker::new(config);
        let pod = create_test_pod();

        for _ in 0..3 {
            checker.record_result(ProbeResult {
                pod_id: pod.metadata.uid.clone(),
                probe_type: ProbeType::Tcp(TcpProbe::default()),
                success: false,
                message: "Failed".into(),
                duration: Duration::from_millis(10),
                timestamp: Instant::now(),
            });
        }

        assert!(checker.should_restart(&pod.metadata.uid));
    }

    #[test]
    fn test_health_check_builders() {
        let http = HealthCheck::http("/health", 8080);
        assert!(matches!(http.probe_type, ProbeType::Http(_)));

        let tcp = HealthCheck::tcp(8080);
        assert!(matches!(tcp.probe_type, ProbeType::Tcp(_)));

        let exec = HealthCheck::exec(vec!["cat".into(), "/tmp/healthy".into()]);
        assert!(matches!(exec.probe_type, ProbeType::Exec(_)));
    }

    #[test]
    fn test_health_check_modifiers() {
        let check = HealthCheck::http("/", 80)
            .with_initial_delay(30)
            .with_period(15)
            .with_timeout(5);

        assert_eq!(check.initial_delay_seconds, 30);
        assert_eq!(check.period_seconds, 15);
        assert_eq!(check.timeout_seconds, 5);
    }

    #[test]
    fn test_self_healing_controller() {
        let controller = SelfHealingController::new(SelfHealingConfig::default());
        let pod_id = ResourceId::generate();

        // Unknown status should start probe
        let action = controller.determine_action(&pod_id, HealthStatus::Unknown);
        assert_eq!(action, HealingAction::StartProbe);

        // Healthy status should do nothing
        let action = controller.determine_action(&pod_id, HealthStatus::Healthy);
        assert_eq!(action, HealingAction::None);
    }

    #[test]
    fn test_self_healing_restart() {
        let mut controller = SelfHealingController::new(SelfHealingConfig {
            max_restarts: 3,
            restart_backoff: Duration::from_millis(1),
            ..Default::default()
        });
        let pod_id = ResourceId::generate();

        // First unhealthy should restart
        let action = controller.determine_action(&pod_id, HealthStatus::Unhealthy);
        assert_eq!(action, HealingAction::Restart);

        // Record restart
        controller.record_restart(&pod_id);
        assert_eq!(controller.get_restart_count(&pod_id), 1);
    }

    #[test]
    fn test_self_healing_max_restarts() {
        let mut controller = SelfHealingController::new(SelfHealingConfig {
            max_restarts: 2,
            restart_backoff: Duration::from_millis(1),
            evict_after_max_restarts: true,
            ..Default::default()
        });
        let pod_id = ResourceId::generate();

        // Exhaust restarts
        controller.record_restart(&pod_id);
        controller.record_restart(&pod_id);

        // Wait for backoff
        thread::sleep(Duration::from_millis(5));

        let action = controller.determine_action(&pod_id, HealthStatus::Unhealthy);
        assert_eq!(action, HealingAction::Evict);
    }

    #[test]
    fn test_restart_tracker() {
        let mut tracker = RestartTracker::new();
        let pod_id = ResourceId::generate();

        tracker.record(pod_id.clone(), RestartReason::LivenessFailure);
        tracker.record(pod_id.clone(), RestartReason::OOMKilled);

        assert_eq!(tracker.count(&pod_id), 2);
        assert_eq!(tracker.recent_restarts(&pod_id, Duration::from_secs(60)), 2);
    }

    #[test]
    fn test_crash_loop_detector() {
        let mut detector = CrashLoopDetector::new(Duration::from_secs(300), 3);
        let pod_id = ResourceId::generate();

        detector.record_crash(&pod_id);
        detector.record_crash(&pod_id);
        assert!(!detector.is_crash_looping(&pod_id));

        detector.record_crash(&pod_id);
        assert!(detector.is_crash_looping(&pod_id));
    }

    #[test]
    fn test_crash_loop_backoff() {
        let detector = CrashLoopDetector::new(Duration::from_secs(300), 5);
        let pod_id = ResourceId::generate();

        // No crashes - 10 seconds
        assert_eq!(detector.get_backoff(&pod_id), Duration::from_secs(10));
    }

    #[test]
    fn test_tcp_probe_config() {
        let probe = TcpProbe {
            host: "192.168.1.1".to_string(),
            port: 3306,
            timeout_seconds: 5,
        };

        assert_eq!(probe.port, 3306);
        assert_eq!(probe.timeout_seconds, 5);
    }

    #[test]
    fn test_http_probe_config() {
        let probe = HttpProbe {
            path: "/healthz".to_string(),
            port: 8080,
            method: "GET".to_string(),
            headers: vec![("Authorization".to_string(), "Bearer token".to_string())],
            ..Default::default()
        };

        assert_eq!(probe.path, "/healthz");
        assert_eq!(probe.headers.len(), 1);
    }

    #[test]
    fn test_clear_health_state() {
        let mut checker = HealthChecker::new(HealthConfig::default());
        let pod = create_test_pod();

        checker.record_result(ProbeResult {
            pod_id: pod.metadata.uid.clone(),
            probe_type: ProbeType::Tcp(TcpProbe::default()),
            success: true,
            message: "OK".into(),
            duration: Duration::from_millis(10),
            timestamp: Instant::now(),
        });

        assert_eq!(checker.get_status(&pod.metadata.uid), HealthStatus::Healthy);

        checker.clear(&pod.metadata.uid);
        assert_eq!(checker.get_status(&pod.metadata.uid), HealthStatus::Unknown);
    }

    #[test]
    fn test_probe_history() {
        let mut checker = HealthChecker::new(HealthConfig::default());
        let pod = create_test_pod();

        for i in 0..5 {
            checker.record_result(ProbeResult {
                pod_id: pod.metadata.uid.clone(),
                probe_type: ProbeType::Tcp(TcpProbe::default()),
                success: i % 2 == 0,
                message: format!("Probe {}", i),
                duration: Duration::from_millis(10),
                timestamp: Instant::now(),
            });
        }

        let history = checker.get_history(&pod.metadata.uid);
        assert_eq!(history.len(), 5);
    }
}
