use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use redis_lite::server::Server;
use redis_lite::config::Config;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Host to bind to
    #[arg(short = 'h', long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 6379)]
    port: u16,

    /// Maximum memory in bytes (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    maxmemory: usize,

    /// Eviction policy
    #[arg(long, default_value = "noeviction")]
    maxmemory_policy: String,

    /// Number of databases
    #[arg(long, default_value_t = 16)]
    databases: usize,

    /// Log level
    #[arg(long, default_value = "info")]
    loglevel: String,

    /// RDB filename
    #[arg(long, default_value = "dump.rdb")]
    dbfilename: String,

    /// Working directory
    #[arg(long, default_value = ".")]
    dir: String,

    /// Enable AOF
    #[arg(long)]
    appendonly: bool,

    /// AOF filename
    #[arg(long, default_value = "appendonly.aof")]
    appendfilename: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set up logging
    let log_level = match args.loglevel.as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(false)
        .with_line_number(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    // Create configuration
    let config = Config {
        bind: args.host.clone(),
        port: args.port,
        maxmemory: args.maxmemory,
        maxmemory_policy: args.maxmemory_policy,
        databases: args.databases,
        dbfilename: args.dbfilename,
        dir: args.dir,
        appendonly: args.appendonly,
        appendfilename: args.appendfilename,
    };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    info!("Starting redis-lite server on {}", addr);

    // Create and run server
    let mut server = Server::new(config)?;
    server.run()?;

    Ok(())
}
