use std::collections::HashMap;
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::time::Duration;

use mio::{Events, Interest, Poll, Token};
use tracing::{debug, error, info, warn};

use crate::commands::CommandExecutor;
use crate::config::Config;
use crate::resp::RespValue;
use crate::storage::Database;

use super::Connection;

const LISTENER: Token = Token(0);

/// Redis-lite server
pub struct Server {
    listener: TcpListener,
    poll: Poll,
    connections: HashMap<Token, Connection>,
    next_token: usize,
    databases: Vec<Database>,
    config: Config,
}

impl Server {
    /// Create a new server
    pub fn new(config: Config) -> io::Result<Self> {
        let addr: SocketAddr = format!("{}:{}", config.bind, config.port).parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;

        let poll = Poll::new()?;

        // Register the listener
        let mut mio_listener = mio::net::TcpListener::from_std(listener.try_clone()?);
        poll.registry().register(&mut mio_listener, LISTENER, Interest::READABLE)?;

        // Create databases
        let mut databases = Vec::with_capacity(config.databases);
        for _ in 0..config.databases {
            databases.push(Database::new());
        }

        info!("Server created, listening on {}", addr);

        Ok(Self {
            listener,
            poll,
            connections: HashMap::new(),
            next_token: 1,
            databases,
            config,
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

            // TODO: Periodic tasks (expiration, etc.)
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

                    // Convert std::net::TcpStream to mio::net::TcpStream
                    stream.set_nonblocking(true)?;
                    let mio_stream = mio::net::TcpStream::from_std(stream);
                    let mut connection = Connection::new(mio_stream);

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

            // Execute command (now self is not borrowed)
            let response = self.execute_command(command);

            // Write response
            let connection = match self.connections.get_mut(&token) {
                Some(conn) => conn,
                None => return Ok(()),
            };
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

    /// Execute a command
    fn execute_command(&mut self, command: RespValue) -> RespValue {
        // Parse command array
        let args = match command.into_array() {
            Some(args) if !args.is_empty() => args,
            _ => return RespValue::error("ERR wrong number of arguments"),
        };

        // Get command name
        let cmd_name = match args[0].as_str() {
            Some(s) => s.to_uppercase(),
            None => return RespValue::error("ERR invalid command"),
        };

        // Execute command on database 0 (TODO: track selected DB per connection)
        let db = &mut self.databases[0];
        CommandExecutor::execute(&cmd_name, &args[1..], db)
    }
}
