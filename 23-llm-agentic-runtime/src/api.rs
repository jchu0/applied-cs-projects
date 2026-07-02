//! HTTP/REST API for the agent runtime (Axum).
//!
//! Mirrors the blueprint's API design: start an agent run, poll its status, and
//! cancel it. Runs are executed on background tasks; results are kept in an
//! in-memory registry keyed by task id.
//!
//! Endpoints:
//! - `GET  /health`                 — liveness check
//! - `GET  /tools`                  — list available tool schemas
//! - `POST /agent/run`              — start a run, returns `{task_id, status}`
//! - `GET  /agent/{task_id}`        — poll status (and result once finished)
//! - `POST /agent/{task_id}/cancel` — cancel a running task

use crate::{
    AgentContext, AgentMemory, AgentResult, AgentRuntime, AgentRuntimeConfig, AnthropicClient,
    CalculatorTool, SearchTool, SimplePlanner, ToolRegistry,
};
use axum::{
    extract::{ConnectInfo, Path, State},
    http::{header, Request, StatusCode},
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tower_http::timeout::TimeoutLayer;
use uuid::Uuid;

/// Build the default tool set exposed by the server.
fn default_tools() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(SearchTool::new("search")));
    tools.register(Box::new(CalculatorTool));
    tools
}

/// Request body for `POST /agent/run`.
#[derive(Debug, Deserialize)]
pub struct AgentRequest {
    /// The task for the agent to accomplish.
    pub task: String,
    /// Maximum reasoning steps (default 20).
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// Wall-clock timeout in seconds (default 300).
    #[serde(default = "default_timeout")]
    pub timeout_seconds: f64,
    /// Optional list of tool names to enable.
    pub tools: Option<Vec<String>>,
    /// Optional model id override (used only when an LLM client is configured).
    pub model: Option<String>,
}

fn default_max_steps() -> usize {
    20
}
fn default_timeout() -> f64 {
    300.0
}

/// Response body for the agent endpoints.
#[derive(Debug, Serialize)]
pub struct AgentResponse {
    pub task_id: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<AgentResult>,
}

/// Per-task state held in the registry.
struct TaskRecord {
    status: String,
    result: Option<AgentResult>,
    handle: Option<JoinHandle<()>>,
}

/// Shared application state: the task registry.
#[derive(Clone)]
pub struct AppState {
    tasks: Arc<Mutex<HashMap<String, TaskRecord>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self { tasks: Arc::new(Mutex::new(HashMap::new())) }
    }
}

// ---------------------------------------------------------------------------
// Hardening baseline: API-key auth, in-process rate limiting, request timeout.
// All three are opt-in via env so the existing tests keep working with auth off.
// ---------------------------------------------------------------------------

/// Shared security configuration, resolved once from the environment.
#[derive(Clone)]
struct SecurityConfig {
    /// Valid API keys (empty => auth disabled).
    api_keys: Arc<Vec<String>>,
    /// Requests allowed per rolling minute per caller (0 => disabled).
    rate_limit_per_minute: u32,
    /// Sliding-window request timestamps keyed by caller identity.
    hits: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
}

impl SecurityConfig {
    /// Resolve config from `API_KEYS` and `RATE_LIMIT_PER_MINUTE`.
    fn from_env() -> Self {
        let api_keys: Vec<String> = std::env::var("API_KEYS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if api_keys.is_empty() {
            eprintln!("WARN: API auth disabled (set API_KEYS to enable)");
        }

        let rate_limit_per_minute = std::env::var("RATE_LIMIT_PER_MINUTE")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(120);

        Self {
            api_keys: Arc::new(api_keys),
            rate_limit_per_minute,
            hits: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn auth_enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }
}

/// Constant-time byte comparison (avoids a crypto dependency just for this).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Extract a presented key from `Authorization: Bearer <key>` or `x-api-key`.
fn presented_key<B>(req: &Request<B>) -> Option<String> {
    if let Some(v) = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        if let Some(rest) = v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")) {
            return Some(rest.trim().to_string());
        }
    }
    req.headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
}

/// Middleware: require a valid API key (when auth is enabled). Health is exempt
/// because it is not wired through this layer.
async fn auth_middleware(
    State(cfg): State<SecurityConfig>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if !cfg.auth_enabled() {
        return next.run(req).await;
    }

    let presented = presented_key(&req);
    let ok = presented.as_deref().is_some_and(|k| {
        cfg.api_keys.iter().any(|valid| constant_time_eq(valid.as_bytes(), k.as_bytes()))
    });

    if ok {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "invalid or missing API key",
        )
            .into_response()
    }
}

/// Middleware: in-process sliding-window rate limit keyed by API key or peer IP.
async fn rate_limit_middleware(
    State(cfg): State<SecurityConfig>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let limit = cfg.rate_limit_per_minute;
    if limit == 0 {
        return next.run(req).await;
    }

    // Prefer the API key as the identity; fall back to the peer IP when
    // `ConnectInfo` is present (it is via `into_make_service_with_connect_info`,
    // but not under `oneshot` in tests, where we use a stable fallback).
    let peer_ip = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let identity = presented_key(&req)
        .filter(|_| cfg.auth_enabled())
        .unwrap_or(peer_ip);

    let now = Instant::now();
    let window = Duration::from_secs(60);
    {
        let mut hits = cfg.hits.lock().await;
        let entry = hits.entry(identity).or_default();
        entry.retain(|t| now.duration_since(*t) < window);
        if entry.len() as u32 >= limit {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [(header::RETRY_AFTER, "60")],
                "rate limit exceeded",
            )
                .into_response();
        }
        entry.push(now);
    }

    next.run(req).await
}

/// Build a fresh agent runtime and execute the request to completion.
async fn execute_agent(req: AgentRequest) -> AgentResult {
    let mut runtime = AgentRuntime::new(
        default_tools(),
        Box::new(SimplePlanner::new()),
        AgentMemory::new(50),
        AgentRuntimeConfig::default(),
    );

    // Attach a real LLM client when ANTHROPIC_API_KEY is set; otherwise the
    // runtime falls back to its deterministic heuristic so the service still runs.
    if let Ok(client) = AnthropicClient::from_env() {
        let client = match &req.model {
            Some(m) => client.with_model(m.clone()),
            None => client,
        };
        runtime = runtime.with_llm(Box::new(client));
    }

    let context = AgentContext::new(req.task)
        .with_max_steps(req.max_steps)
        .with_timeout(req.timeout_seconds)
        .with_tools(req.tools.unwrap_or_default());

    runtime.run(context).await
}

async fn health() -> &'static str {
    "ok"
}

async fn list_tools() -> Json<serde_json::Value> {
    let tools = default_tools();
    Json(serde_json::json!({ "tools": tools.get_tool_schemas() }))
}

async fn run_agent(
    State(state): State<AppState>,
    Json(req): Json<AgentRequest>,
) -> Json<AgentResponse> {
    let task_id = Uuid::new_v4().to_string();

    {
        let mut tasks = state.tasks.lock().await;
        tasks.insert(
            task_id.clone(),
            TaskRecord { status: "running".to_string(), result: None, handle: None },
        );
    }

    let tid = task_id.clone();
    let st = state.clone();
    let handle = tokio::spawn(async move {
        let result = execute_agent(req).await;
        let mut tasks = st.tasks.lock().await;
        if let Some(rec) = tasks.get_mut(&tid) {
            // Don't clobber a record the client already cancelled.
            if rec.status != "cancelled" {
                rec.status = if result.success { "completed".to_string() } else { "failed".to_string() };
                rec.result = Some(result);
            }
        }
    });

    {
        let mut tasks = state.tasks.lock().await;
        if let Some(rec) = tasks.get_mut(&task_id) {
            rec.handle = Some(handle);
        }
    }

    Json(AgentResponse { task_id, status: "running".to_string(), result: None })
}

async fn get_agent_status(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<Json<AgentResponse>, StatusCode> {
    let tasks = state.tasks.lock().await;
    match tasks.get(&task_id) {
        None => Err(StatusCode::NOT_FOUND),
        Some(rec) => Ok(Json(AgentResponse {
            task_id,
            status: rec.status.clone(),
            result: rec.result.clone(),
        })),
    }
}

async fn cancel_agent(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<Json<AgentResponse>, StatusCode> {
    let mut tasks = state.tasks.lock().await;
    match tasks.get_mut(&task_id) {
        None => Err(StatusCode::NOT_FOUND),
        Some(rec) => {
            if let Some(h) = rec.handle.take() {
                h.abort();
            }
            if rec.status == "running" {
                rec.status = "cancelled".to_string();
            }
            Ok(Json(AgentResponse {
                task_id,
                status: rec.status.clone(),
                result: rec.result.clone(),
            }))
        }
    }
}

/// Build the router with a fresh in-memory task registry and the hardening
/// baseline (auth + rate limit + timeout) resolved from the environment.
///
/// `/health` is intentionally left open — it is registered outside the
/// protected sub-router so liveness/readiness probes never need a key and are
/// never rate-limited or timed out.
pub fn router() -> Router {
    let cfg = SecurityConfig::from_env();

    let timeout_seconds = std::env::var("REQUEST_TIMEOUT_SECONDS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(30);

    // Everything except /health runs the reason/act flow as an async, polled
    // job (POST /agent/run returns immediately; GET /agent/{id} polls). No
    // handler holds the connection open, so the timeout applies to every
    // protected route with no exemptions needed. If a streaming/SSE/websocket
    // route were added, it would be registered on a separate un-timed sub-router.
    let mut protected = Router::new()
        .route("/tools", get(list_tools))
        .route("/agent/run", post(run_agent))
        .route("/agent/:task_id", get(get_agent_status))
        .route("/agent/:task_id/cancel", post(cancel_agent))
        .layer(from_fn_with_state(cfg.clone(), rate_limit_middleware))
        .layer(from_fn_with_state(cfg.clone(), auth_middleware));

    if timeout_seconds > 0 {
        protected = protected.layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            Duration::from_secs(timeout_seconds),
        ));
    }

    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .with_state(AppState::default())
}

/// Bind to `addr` and serve until the process is stopped.
///
/// Uses `into_make_service_with_connect_info` so the rate limiter can key on the
/// peer IP when no API key is presented.
pub async fn serve(addr: &str) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router().into_make_service_with_connect_info::<SocketAddr>()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    // `API_KEYS` is process-global and read at `router()` build time, so every
    // test must build its router while holding this lock to avoid cross-test
    // races. Callers hold the guard only across the (synchronous) build.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Build a router with auth disabled (env cleared) under the shared lock.
    fn router_auth_off() -> Router {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("API_KEYS");
        router()
    }

    /// Build a router with the given comma-separated keys under the shared lock.
    fn router_with_keys(keys: &str) -> Router {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("API_KEYS", keys);
        let app = router();
        std::env::remove_var("API_KEYS");
        app
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_endpoint() {
        let resp = router_auth_off()
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn tools_endpoint_lists_tools() {
        let resp = router_auth_off()
            .oneshot(Request::builder().uri("/tools").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert!(v["tools"].as_array().unwrap().len() >= 2);
    }

    #[tokio::test]
    async fn run_then_poll_reaches_terminal_state() {
        let app = router_auth_off();

        // Start a run (no API key in tests → deterministic heuristic path).
        let req = serde_json::json!({ "task": "say hello", "max_steps": 10 }).to_string();
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/agent/run")
                    .header("content-type", "application/json")
                    .body(Body::from(req))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["status"], "running");
        let task_id = v["task_id"].as_str().unwrap().to_string();

        // Poll until the background task finishes.
        let mut final_status = String::from("running");
        for _ in 0..100 {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("/agent/{task_id}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let v = body_json(resp).await;
            final_status = v["status"].as_str().unwrap().to_string();
            if final_status != "running" {
                // The result is attached once finished.
                assert!(v.get("result").is_some());
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_ne!(final_status, "running", "task never left running state");
    }

    #[tokio::test]
    async fn auth_enabled_rejects_missing_key() {
        let app = router_with_keys("secret-key-1,secret-key-2");

        // No key => 401 with WWW-Authenticate.
        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/tools").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert!(resp.headers().contains_key("www-authenticate"));

        // Bad key => 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/tools")
                    .header("x-api-key", "nope")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_enabled_accepts_valid_key() {
        let app = router_with_keys("secret-key-1,secret-key-2");

        // Valid Bearer key => 200.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/tools")
                    .header("authorization", "Bearer secret-key-2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_open_when_auth_enabled() {
        let app = router_with_keys("secret-key-1");

        // Health needs no key even when auth is enabled.
        let resp = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_task_is_404() {
        let resp = router_auth_off()
            .oneshot(
                Request::builder()
                    .uri("/agent/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
