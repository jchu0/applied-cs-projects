//! Binary entry point: serve the agent runtime over HTTP.
//!
//! Set `ANTHROPIC_API_KEY` to drive the agent with a real model; without it the
//! runtime falls back to its deterministic heuristic. Override the bind address
//! with `BIND_ADDR` (default `0.0.0.0:8080`).

use llm_agentic_runtime::api;

#[tokio::main]
async fn main() {
    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    println!("LLM Agentic Runtime listening on http://{addr}");
    if let Err(e) = api::serve(&addr).await {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}
