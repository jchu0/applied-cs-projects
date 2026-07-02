//! Message Queue Server binary.
//!
//! Boots a [`Broker`] and serves the wire protocol over TCP. Configuration is
//! read from `MQ_*` environment variables:
//!
//! - `MQ_BROKER_ID`   broker id (default `1`)
//! - `MQ_DATA_DIR`    data directory (default `/tmp/mq/data`)
//! - `MQ_LOG_DIR`     log directory (default `/tmp/mq/logs`)
//! - `MQ_HOST`        bind host (default `127.0.0.1`)
//! - `MQ_PORT`        bind port (default `9092`)
//! - `MQ_AUTH_TOKEN`  if set, clients must authenticate with this token

use message_queue::server::{serve, ServerOptions};
use message_queue::{Broker, BrokerConfig, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging.
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse configuration from environment or use defaults.
    let config = BrokerConfig::default()
        .with_broker_id(
            std::env::var("MQ_BROKER_ID")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
        )
        .with_data_dir(
            std::env::var("MQ_DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/tmp/mq/data")),
        )
        .with_log_dir(
            std::env::var("MQ_LOG_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/tmp/mq/logs")),
        )
        .with_host(std::env::var("MQ_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()))
        .with_port(
            std::env::var("MQ_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9092),
        );

    tracing::info!("Message Queue Server starting...");
    tracing::info!("Broker ID: {}", config.broker_id);
    tracing::info!("Data directory: {:?}", config.data_dir);
    tracing::info!("Listening on: {}", config.socket_addr());

    let addr = config.socket_addr();
    let opts = ServerOptions::from_env(&config.data_dir);
    if opts.auth_token.is_some() {
        tracing::info!("Authentication is ENABLED (MQ_AUTH_TOKEN set)");
    }

    // Create and start broker.
    let broker = Arc::new(Broker::new(config)?);
    broker.start()?;

    tracing::info!("Message Queue Server started successfully");

    // Serve the wire protocol. Ctrl-C triggers a clean shutdown.
    tokio::select! {
        res = serve(broker.clone(), addr, opts) => {
            if let Err(e) = res {
                tracing::error!("server error: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("shutdown signal received");
        }
    }

    broker.stop()?;
    tracing::info!("Message Queue Server stopped");
    Ok(())
}
