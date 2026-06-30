//! Message Queue Server binary.

use message_queue::{Broker, BrokerConfig, Result};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse configuration from environment or use defaults
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
        .with_host(
            std::env::var("MQ_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
        )
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

    // Create and start broker
    let broker = Broker::new(config)?;
    broker.start()?;

    tracing::info!("Message Queue Server started successfully");

    // Wait for shutdown signal
    // In a real implementation, we would:
    // 1. Start a TCP server to accept connections
    // 2. Handle the Kafka protocol
    // 3. Use tokio for async I/O

    // For now, just keep running
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));

        // Check metrics periodically
        let metrics = broker.metrics();
        tracing::debug!(
            "Messages produced: {}, consumed: {}",
            metrics.messages_produced(),
            metrics.messages_consumed()
        );
    }
}
