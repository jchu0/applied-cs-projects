use std::collections::HashMap;
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::time::{Duration, Instant};

use mio::{Events, Interest, Poll, Token};
use tracing::{debug, error, info, warn};

use crate::commands::CommandExecutor;
use crate::config::{Config, EvictionPolicy as ConfigEvictionPolicy};
use crate::eviction::{EvictionConfig, EvictionManager, EvictionPolicy};
use crate::persistence::{FsyncPolicy, AOF, RDB};
use crate::resp::RespValue;
use crate::storage::Database;

use super::Connection;

const LISTENER: Token = Token(0);

/// How many keys to sample per active-expiration cycle, per database.
const ACTIVE_EXPIRE_SAMPLE: usize = 20;

/// Determine whether a command mutates keyspace state. Mutating commands are
/// the ones that must be appended to the AOF and can trigger eviction.
fn is_write_command(cmd: &str) -> bool {
    matches!(
        cmd,
        "SET" | "SETNX" | "SETEX" | "PSETEX" | "MSET" | "APPEND" | "GETSET"
            | "INCR" | "INCRBY" | "DECR" | "DECRBY"
            | "LPUSH" | "RPUSH" | "LPOP" | "RPOP" | "LSET"
            | "SADD" | "SREM"
            | "HSET" | "HMSET" | "HDEL"
            | "ZADD" | "ZREM" | "ZINCRBY"
            | "DEL" | "EXPIRE" | "EXPIREAT" | "PEXPIRE" | "PERSIST"
            | "FLUSHDB" | "RENAME" | "RENAMENX"
    )
}

/// Map the string eviction policy from `Config` into the eviction module's
/// `EvictionPolicy` enum.
fn map_policy(cfg: ConfigEvictionPolicy) -> EvictionPolicy {
    match cfg {
        ConfigEvictionPolicy::NoEviction => EvictionPolicy::NoEviction,
        ConfigEvictionPolicy::AllKeysLRU => EvictionPolicy::AllKeysLRU,
        ConfigEvictionPolicy::VolatileLRU => EvictionPolicy::VolatileLRU,
        ConfigEvictionPolicy::AllKeysLFU => EvictionPolicy::AllKeysLFU,
        ConfigEvictionPolicy::VolatileLFU => EvictionPolicy::VolatileLFU,
        ConfigEvictionPolicy::AllKeysRandom => EvictionPolicy::AllKeysRandom,
        ConfigEvictionPolicy::VolatileRandom => EvictionPolicy::VolatileRandom,
        ConfigEvictionPolicy::VolatileTTL => EvictionPolicy::VolatileTTL,
    }
}

/// Redis-lite server
pub struct Server {
    listener: mio::net::TcpListener,
    poll: Poll,
    connections: HashMap<Token, Connection>,
    next_token: usize,
    databases: Vec<Database>,
    /// Per-database eviction managers (one per configured database).
    eviction: Vec<EvictionManager>,
    /// Configured maximum memory (0 = unlimited).
    maxmemory: usize,
    /// Configured eviction policy.
    policy: EvictionPolicy,
    /// Append-only-file handler, present only when `appendonly` is enabled.
    aof: Option<AOF>,
    /// AOF fsync policy.
    aof_fsync: FsyncPolicy,
    /// Timestamp of the last `everysec` AOF fsync.
    last_aof_sync: Instant,
}

impl Server {
    /// Create a new server
    pub fn new(config: Config) -> io::Result<Self> {
        let addr: SocketAddr = format!("{}:{}", config.bind, config.port).parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        let std_listener = TcpListener::bind(addr)?;
        std_listener.set_nonblocking(true)?;

        let poll = Poll::new()?;

        // Register the listener with the poll. We keep the mio listener as the
        // owned source so its registration stays alive for the server's
        // lifetime (dropping the registered source would silently stop
        // delivering accept events).
        let mut listener = mio::net::TcpListener::from_std(std_listener);
        poll.registry().register(&mut listener, LISTENER, Interest::READABLE)?;

        let num_dbs = config.databases.max(1);
        let policy = map_policy(ConfigEvictionPolicy::from(config.maxmemory_policy.as_str()));
        let maxmemory = config.maxmemory;

        // Create databases
        let mut databases = Vec::with_capacity(num_dbs);
        for _ in 0..num_dbs {
            databases.push(Database::new());
        }

        // Resolve persistence file paths relative to the working directory.
        let aof_path = Path::new(&config.dir)
            .join(&config.appendfilename)
            .to_string_lossy()
            .into_owned();
        let rdb_path = Path::new(&config.dir)
            .join(&config.dbfilename)
            .to_string_lossy()
            .into_owned();

        // Startup loading: prefer AOF when append-only is enabled (it is the
        // more recent/complete log), otherwise fall back to the RDB snapshot.
        // Data is always loaded into database 0, matching the on-disk format
        // which only serializes DB 0.
        let aof_fsync = FsyncPolicy::EverySecond;
        let mut aof = None;
        if config.appendonly {
            // Replay any existing AOF through the real command path first.
            let replay = AOF::new(aof_path.clone(), aof_fsync)?;
            match replay.load() {
                Ok(loaded) => {
                    if !loaded.is_empty() {
                        info!("Loaded {} keys from AOF at {}", loaded.len(), aof_path);
                    }
                    databases[0] = loaded;
                }
                Err(e) => warn!("Failed to load AOF ({}): starting empty", e),
            }
            // The handler used for live appends re-opens the same file in append
            // mode, so subsequent writes are added after the replayed contents.
            aof = Some(replay);
        } else if Path::new(&rdb_path).exists() {
            let rdb = RDB::new(rdb_path.clone());
            match rdb.load() {
                Ok(loaded) => {
                    if !loaded.is_empty() {
                        info!("Loaded {} keys from RDB at {}", loaded.len(), rdb_path);
                    }
                    databases[0] = loaded;
                }
                Err(e) => warn!("Failed to load RDB ({}): starting empty", e),
            }
        }

        // Build one eviction manager per database.
        let mut eviction = Vec::with_capacity(num_dbs);
        for _ in 0..num_dbs {
            eviction.push(EvictionManager::new(EvictionConfig {
                max_memory: maxmemory,
                policy,
                sample_size: 5,
            }));
        }

        info!(
            "Server created, listening on {} ({} databases, maxmemory={}, policy={}, appendonly={})",
            addr, num_dbs, maxmemory, policy, config.appendonly
        );

        Ok(Self {
            listener,
            poll,
            connections: HashMap::new(),
            next_token: 1,
            databases,
            eviction,
            maxmemory,
            policy,
            aof,
            aof_fsync,
            last_aof_sync: Instant::now(),
        })
    }

    /// Run the server event loop
    pub fn run(&mut self) -> io::Result<()> {
        let mut events = Events::with_capacity(1024);

        loop {
            // Poll for events with timeout for expiration check
            self.poll.poll(&mut events, Some(Duration::from_millis(100)))?;

            for event in &events {
                match event.token() {
                    LISTENER => {
                        self.accept_connections()?;
                    }
                    token => {
                        if event.is_readable() {
                            self.handle_readable(token)?;
                        }
                        if event.is_writable() {
                            self.handle_writable(token)?;
                        }
                    }
                }
            }

            // Periodic maintenance: active key expiration and timed AOF fsync.
            self.run_periodic_tasks();
        }
    }

    /// Background maintenance run once per event-loop iteration.
    fn run_periodic_tasks(&mut self) {
        // Active expiration: reclaim expired keys across all databases even if
        // no client touches them (complements lazy expiry on access).
        for db in &mut self.databases {
            db.active_expire_cycle(ACTIVE_EXPIRE_SAMPLE);
        }

        // AOF fsync on the "everysec" cadence.
        if let Some(aof) = &self.aof {
            if self.aof_fsync == FsyncPolicy::EverySecond
                && self.last_aof_sync.elapsed() >= Duration::from_secs(1)
            {
                if let Err(e) = aof.sync() {
                    error!("AOF fsync failed: {}", e);
                }
                self.last_aof_sync = Instant::now();
            }
        }
    }

    /// Accept new connections
    fn accept_connections(&mut self) -> io::Result<()> {
        loop {
            match self.listener.accept() {
                Ok((stream, addr)) => {
                    debug!("New connection from {}", addr);

                    let token = Token(self.next_token);
                    self.next_token += 1;

                    // `stream` is already a non-blocking mio::net::TcpStream.
                    let mut connection = Connection::new(stream);

                    // Register with poll
                    self.poll.registry().register(
                        connection.stream_mut(),
                        token,
                        Interest::READABLE | Interest::WRITABLE,
                    )?;

                    self.connections.insert(token, connection);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    break;
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Handle readable event
    fn handle_readable(&mut self, token: Token) -> io::Result<()> {
        // Read data
        {
            let connection = match self.connections.get_mut(&token) {
                Some(conn) => conn,
                None => return Ok(()),
            };

            match connection.read() {
                Ok(0) if connection.is_closed() => {
                    debug!("Connection closed by peer");
                    drop(connection);
                    self.close_connection(token)?;
                    return Ok(());
                }
                Ok(_) => {}
                Err(e) => {
                    error!("Error reading from connection: {}", e);
                    drop(connection);
                    self.close_connection(token)?;
                    return Ok(());
                }
            }
        }

        // Process commands
        loop {
            // Parse command with limited borrow scope
            let command = {
                let connection = match self.connections.get_mut(&token) {
                    Some(conn) => conn,
                    None => return Ok(()),
                };
                match connection.parse_command() {
                    Ok(Some(cmd)) => cmd,
                    Ok(None) => break,
                    Err(e) => {
                        warn!("Error parsing command: {}", e);
                        let response = RespValue::error(format!("ERR {}", e));
                        let _ = connection.write_response(response);
                        break;
                    }
                }
            };

            // Read the connection's currently-selected database index.
            let selected_db = match self.connections.get(&token) {
                Some(conn) => conn.selected_db(),
                None => return Ok(()),
            };

            // Execute command (now self is not borrowed by a connection).
            let (response, new_db) = self.execute_command(command, selected_db);

            // Write response and apply any SELECT database switch.
            let connection = match self.connections.get_mut(&token) {
                Some(conn) => conn,
                None => return Ok(()),
            };
            if let Some(db_index) = new_db {
                connection.set_selected_db(db_index);
            }
            if let Err(e) = connection.write_response(response) {
                error!("Error writing response: {}", e);
                drop(connection);
                self.close_connection(token)?;
                return Ok(());
            }
        }

        // Try to flush
        if let Some(connection) = self.connections.get_mut(&token) {
            if connection.has_pending_writes() {
                let _ = connection.flush();
            }
        }

        Ok(())
    }

    /// Handle writable event
    fn handle_writable(&mut self, token: Token) -> io::Result<()> {
        let connection = match self.connections.get_mut(&token) {
            Some(conn) => conn,
            None => return Ok(()),
        };

        if connection.has_pending_writes() {
            if let Err(e) = connection.flush() {
                error!("Error flushing connection: {}", e);
                self.close_connection(token)?;
            }
        }

        Ok(())
    }

    /// Close a connection
    fn close_connection(&mut self, token: Token) -> io::Result<()> {
        if let Some(mut connection) = self.connections.remove(&token) {
            self.poll.registry().deregister(connection.stream_mut())?;
            debug!("Connection closed");
        }
        Ok(())
    }

    /// Execute a command against the connection's selected database.
    ///
    /// Returns the response plus, for a successful `SELECT`, the new database
    /// index the connection should switch to.
    fn execute_command(
        &mut self,
        command: RespValue,
        selected_db: usize,
    ) -> (RespValue, Option<usize>) {
        // Parse command array
        let args = match command.into_array() {
            Some(args) if !args.is_empty() => args,
            _ => return (RespValue::error("ERR wrong number of arguments"), None),
        };

        // Get command name
        let cmd_name = match args[0].as_str() {
            Some(s) => s.to_uppercase(),
            None => return (RespValue::error("ERR invalid command"), None),
        };

        // Connection/keyspace-management commands handled at the server level.
        match cmd_name.as_str() {
            "SELECT" => return self.cmd_select(&args[1..]),
            "SWAPDB" => return (self.cmd_swapdb(&args[1..]), None),
            _ => {}
        }

        let db_index = selected_db.min(self.databases.len().saturating_sub(1));
        let is_write = is_write_command(&cmd_name);

        // Enforce maxmemory before applying a write. With `noeviction`, reject
        // writes once over the limit; with an eviction policy, free memory
        // first. Reads are always allowed.
        if is_write && self.maxmemory > 0 {
            if let Err(msg) = self.enforce_memory(db_index) {
                return (RespValue::error(format!("OOM {}", msg)), None);
            }
        }

        // Execute against the selected database.
        let db = &mut self.databases[db_index];
        let response = CommandExecutor::execute(&cmd_name, &args[1..], db);

        // Persist successful writes to the AOF and re-check memory afterwards so
        // that the data structure can never grow unbounded past the limit.
        if is_write && !matches!(response, RespValue::Error(_)) {
            if let Some(aof) = &self.aof {
                // Re-serialize the full command (name + args) for replay.
                if let Err(e) = aof.append(&args) {
                    error!("AOF append failed: {}", e);
                }
            }
            if self.maxmemory > 0 {
                let _ = self.enforce_memory(db_index);
            }
        }

        (response, None)
    }

    /// Recompute memory usage for a database and run eviction if it is over the
    /// configured `maxmemory`. Returns an error (used to produce an `OOM`
    /// reply) when the policy is `noeviction` and the limit is exceeded.
    fn enforce_memory(&mut self, db_index: usize) -> Result<(), &'static str> {
        if self.maxmemory == 0 {
            return Ok(());
        }
        let used = self.databases[db_index].memory_usage();
        let manager = &mut self.eviction[db_index];
        manager.set_memory(used);
        if !manager.needs_eviction() {
            return Ok(());
        }
        if self.policy == EvictionPolicy::NoEviction {
            return Err("command not allowed when used memory > 'maxmemory'");
        }
        let db = &mut self.databases[db_index];
        match manager.evict_if_needed(db) {
            Ok(evicted) => {
                if evicted > 0 {
                    debug!("Evicted {} keys from db {}", evicted, db_index);
                }
                Ok(())
            }
            Err(msg) => Err(msg),
        }
    }

    /// Handle the `SELECT <index>` command.
    fn cmd_select(&mut self, args: &[RespValue]) -> (RespValue, Option<usize>) {
        if args.len() != 1 {
            return (
                RespValue::error("ERR wrong number of arguments for 'select' command"),
                None,
            );
        }
        let index = match args[0].as_str().and_then(|s| s.parse::<usize>().ok()) {
            Some(i) => i,
            None => return (RespValue::error("ERR value is not an integer or out of range"), None),
        };
        if index >= self.databases.len() {
            return (RespValue::error("ERR DB index is out of range"), None);
        }
        (RespValue::ok(), Some(index))
    }

    /// Handle the `SWAPDB <index1> <index2>` command.
    fn cmd_swapdb(&mut self, args: &[RespValue]) -> RespValue {
        if args.len() != 2 {
            return RespValue::error("ERR wrong number of arguments for 'swapdb' command");
        }
        let parse = |v: &RespValue| v.as_str().and_then(|s| s.parse::<usize>().ok());
        let (a, b) = match (parse(&args[0]), parse(&args[1])) {
            (Some(a), Some(b)) => (a, b),
            _ => return RespValue::error("ERR invalid first DB index"),
        };
        if a >= self.databases.len() || b >= self.databases.len() {
            return RespValue::error("ERR DB index is out of range");
        }
        if a != b {
            self.databases.swap(a, b);
        }
        RespValue::ok()
    }
}
