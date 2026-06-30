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
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
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

/// Build the router with a fresh in-memory task registry.
pub fn router() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/tools", get(list_tools))
        .route("/agent/run", post(run_agent))
        .route("/agent/:task_id", get(get_agent_status))
        .route("/agent/:task_id/cancel", post(cancel_agent))
        .with_state(AppState::default())
}

/// Bind to `addr` and serve until the process is stopped.
pub async fn serve(addr: &str) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_endpoint() {
        let resp = router()
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn tools_endpoint_lists_tools() {
        let resp = router()
            .oneshot(Request::builder().uri("/tools").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert!(v["tools"].as_array().unwrap().len() >= 2);
    }

    #[tokio::test]
    async fn run_then_poll_reaches_terminal_state() {
        let app = router();

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
    async fn unknown_task_is_404() {
        let resp = router()
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
