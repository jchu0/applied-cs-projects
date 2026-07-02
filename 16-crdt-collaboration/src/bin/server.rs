//! Runnable collaboration server.
//!
//! Boots the Axum HTTP API together with the real-time WebSocket collaboration
//! endpoint (`GET /ws/:doc_id`). Document state is persisted via SQLite when a
//! database path is configured, and kept in process memory otherwise.
//!
//! Environment variables:
//! - `BIND_ADDR`: socket address to listen on (default `127.0.0.1:8080`).
//! - `DATABASE_PATH` (alias `SQLITE_PATH`): SQLite file path. When set, the
//!   server uses durable storage so documents survive restarts. When unset,
//!   storage is in-memory.
//! - `API_KEYS`: comma-separated API keys; when set, HTTP routes and the
//!   WebSocket handshake require a valid key (health stays open).
//! - `RATE_LIMIT_PER_MINUTE`, `REQUEST_TIMEOUT_SECONDS`: HTTP hardening knobs.

use std::net::SocketAddr;
use std::sync::Arc;

use crdt_collaboration::api::{create_router, ApiState};
use crdt_collaboration::persistent::SqliteBackend;
use crdt_collaboration::server::{CollaborationServer, ServerConfig};
use crdt_collaboration::storage::StorageManager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Select storage backend: durable SQLite if a path is configured, else
    // in-memory (useful for local/dev and ephemeral runs).
    let db_path = std::env::var("DATABASE_PATH")
        .or_else(|_| std::env::var("SQLITE_PATH"))
        .ok();

    let storage = match db_path {
        Some(path) => {
            let backend = SqliteBackend::new(&path)?;
            tracing::info!(path = %path, "using SQLite persistence");
            Arc::new(StorageManager::with_backend(Arc::new(backend)))
        }
        None => {
            tracing::warn!("using in-memory storage (set DATABASE_PATH to persist)");
            Arc::new(StorageManager::new())
        }
    };

    let server = Arc::new(CollaborationServer::new(
        ServerConfig::default(),
        storage.clone(),
    ));
    let state = Arc::new(ApiState::new(server, storage));

    let app = create_router(state);

    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let addr: SocketAddr = bind_addr.parse()?;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "collaboration server listening (HTTP + WebSocket at /ws/:doc_id)");

    // `into_make_service_with_connect_info` exposes the peer address to the
    // rate-limiter middleware for IP-based limiting when no API key is present.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}
