//! Docklet CLI
//!
//! A minimal container runtime command-line interface.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use docklet::RuntimeConfig;

#[derive(Parser)]
#[command(name = "docklet")]
#[command(about = "A minimal container runtime")]
#[command(version)]
struct Cli {
    /// Root directory for runtime state
    #[arg(long, default_value = "/var/lib/docklet")]
    root: PathBuf,

    /// Enable debug logging
    #[arg(long, short)]
    debug: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new container
    Create {
        /// Container ID
        container_id: String,

        /// Bundle path
        #[arg(long, short)]
        bundle: PathBuf,

        /// Console socket path
        #[arg(long)]
        console_socket: Option<PathBuf>,

        /// PID file path
        #[arg(long)]
        pid_file: Option<PathBuf>,
    },

    /// Start a created container
    Start {
        /// Container ID
        container_id: String,
    },

    /// Create and run a container
    Run {
        /// Container ID
        container_id: String,

        /// Bundle path
        #[arg(long, short)]
        bundle: PathBuf,

        /// Detach from container
        #[arg(long, short)]
        detach: bool,
    },

    /// Send a signal to a container
    Kill {
        /// Container ID
        container_id: String,

        /// Signal to send
        #[arg(default_value = "SIGTERM")]
        signal: String,
    },

    /// Delete a container
    Delete {
        /// Container ID
        container_id: String,

        /// Force delete
        #[arg(long, short)]
        force: bool,
    },

    /// Get container state
    State {
        /// Container ID
        container_id: String,
    },

    /// List containers
    List {
        /// Show all containers
        #[arg(long, short)]
        all: bool,

        /// Only show IDs
        #[arg(long, short)]
        quiet: bool,
    },

    /// Generate a default spec
    Spec {
        /// Bundle path
        #[arg(long, short, default_value = ".")]
        bundle: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    // Set up logging
    if cli.debug {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .init();
    }

    let config = RuntimeConfig {
        root: cli.root.clone(),
        state_dir: cli.root.join("containers"),
        image_dir: cli.root.join("images"),
        cgroup_root: PathBuf::from("/sys/fs/cgroup"),
        rootless: false,
    };

    let result = match cli.command {
        Commands::Create {
            container_id,
            bundle,
            console_socket,
            pid_file,
        } => cmd_create(&config, &container_id, &bundle, console_socket, pid_file),

        Commands::Start { container_id } => cmd_start(&config, &container_id),

        Commands::Run {
            container_id,
            bundle,
            detach,
        } => cmd_run(&config, &container_id, &bundle, detach),

        Commands::Kill {
            container_id,
            signal,
        } => cmd_kill(&config, &container_id, &signal),

        Commands::Delete {
            container_id,
            force,
        } => cmd_delete(&config, &container_id, force),

        Commands::State { container_id } => cmd_state(&config, &container_id),

        Commands::List { all, quiet } => cmd_list(&config, all, quiet),

        Commands::Spec { bundle } => cmd_spec(&bundle),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn cmd_create(
    _config: &RuntimeConfig,
    container_id: &str,
    bundle: &PathBuf,
    _console_socket: Option<PathBuf>,
    _pid_file: Option<PathBuf>,
) -> docklet::Result<()> {
    log::info!("Creating container {} from bundle {:?}", container_id, bundle);

    // Load spec
    let spec_path = bundle.join("config.json");
    let _spec = docklet::Spec::load(&spec_path)?;

    // Create container (stub)
    println!("Container {} created", container_id);

    Ok(())
}

fn cmd_start(_config: &RuntimeConfig, container_id: &str) -> docklet::Result<()> {
    log::info!("Starting container {}", container_id);
    println!("Container {} started", container_id);
    Ok(())
}

fn cmd_run(
    config: &RuntimeConfig,
    container_id: &str,
    bundle: &PathBuf,
    _detach: bool,
) -> docklet::Result<()> {
    log::info!("Running container {} from bundle {:?}", container_id, bundle);

    // Create and start
    cmd_create(config, container_id, bundle, None, None)?;
    cmd_start(config, container_id)?;

    Ok(())
}

fn cmd_kill(_config: &RuntimeConfig, container_id: &str, signal: &str) -> docklet::Result<()> {
    log::info!("Killing container {} with signal {}", container_id, signal);
    println!("Signal {} sent to container {}", signal, container_id);
    Ok(())
}

fn cmd_delete(_config: &RuntimeConfig, container_id: &str, force: bool) -> docklet::Result<()> {
    log::info!("Deleting container {} (force={})", container_id, force);
    println!("Container {} deleted", container_id);
    Ok(())
}

fn cmd_state(config: &RuntimeConfig, container_id: &str) -> docklet::Result<()> {
    log::info!("Getting state for container {}", container_id);

    // Return example state
    let state = docklet::container::State {
        oci_version: "1.0.0".to_string(),
        id: container_id.to_string(),
        status: docklet::ContainerState::Stopped,
        pid: None,
        bundle: config.state_dir.join(container_id),
        annotations: std::collections::HashMap::new(),
    };

    println!("{}", serde_json::to_string_pretty(&state)?);
    Ok(())
}

fn cmd_list(_config: &RuntimeConfig, all: bool, quiet: bool) -> docklet::Result<()> {
    log::info!("Listing containers (all={}, quiet={})", all, quiet);

    if quiet {
        println!("(no containers)");
    } else {
        println!("ID\tSTATUS\tCREATED");
        println!("(no containers)");
    }

    Ok(())
}

fn cmd_spec(bundle: &PathBuf) -> docklet::Result<()> {
    log::info!("Generating spec in {:?}", bundle);

    let spec = docklet::Spec::default();
    let spec_path = bundle.join("config.json");

    std::fs::create_dir_all(bundle)?;
    spec.save(&spec_path)?;

    println!("Spec generated at {:?}", spec_path);
    Ok(())
}
