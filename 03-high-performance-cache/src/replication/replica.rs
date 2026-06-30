//! Replica-specific replication state and operations

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// Replica replication state
pub struct ReplicaState {
    /// Master host
    master_host: String,
    /// Master port
    master_port: u16,
    /// Connection to master
    connection: Option<TcpStream>,
    /// Master's replication ID
    master_repl_id: String,
    /// Current offset
    offset: u64,
    /// Last ping time
    last_ping: Instant,
    /// Replica state
    state: ReplicaConnectionState,
}

/// Replica connection state
#[derive(Debug, Clone, PartialEq)]
pub enum ReplicaConnectionState {
    /// Not connected
    Disconnected,
    /// Connecting to master
    Connecting,
    /// Sending PING
    SendPing,
    /// Authenticating
    Auth,
    /// Sending port info
    SendPort,
    /// Sending capabilities
    SendCapa,
    /// Requesting sync
    SendPsync,
    /// Receiving RDB
    ReceiveRdb,
    /// Connected and streaming
    Connected,
}

impl ReplicaState {
    /// Create a new replica state
    pub fn new(master_host: String, master_port: u16) -> Self {
        Self {
            master_host,
            master_port,
            connection: None,
            master_repl_id: String::from("?"),
            offset: 0,
            last_ping: Instant::now(),
            state: ReplicaConnectionState::Disconnected,
        }
    }

    /// Connect to master
    pub fn connect(&mut self) -> io::Result<()> {
        let addr = format!("{}:{}", self.master_host, self.master_port);
        let stream = TcpStream::connect(&addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        self.connection = Some(stream);
        self.state = ReplicaConnectionState::Connecting;
        Ok(())
    }

    /// Disconnect from master
    pub fn disconnect(&mut self) {
        self.connection = None;
        self.state = ReplicaConnectionState::Disconnected;
    }

    /// Send PING to master
    pub fn send_ping(&mut self) -> io::Result<()> {
        if let Some(ref mut conn) = self.connection {
            conn.write_all(b"*1\r\n$4\r\nPING\r\n")?;
            self.state = ReplicaConnectionState::SendPing;
        }
        Ok(())
    }

    /// Send REPLCONF listening-port
    pub fn send_replconf_port(&mut self, port: u16) -> io::Result<()> {
        if let Some(ref mut conn) = self.connection {
            let cmd = format!(
                "*3\r\n$8\r\nREPLCONF\r\n$14\r\nlistening-port\r\n${}\r\n{}\r\n",
                port.to_string().len(),
                port
            );
            conn.write_all(cmd.as_bytes())?;
            self.state = ReplicaConnectionState::SendPort;
        }
        Ok(())
    }

    /// Send REPLCONF capa
    pub fn send_replconf_capa(&mut self) -> io::Result<()> {
        if let Some(ref mut conn) = self.connection {
            conn.write_all(b"*5\r\n$8\r\nREPLCONF\r\n$4\r\ncapa\r\n$3\r\neof\r\n$4\r\ncapa\r\n$6\r\npsync2\r\n")?;
            self.state = ReplicaConnectionState::SendCapa;
        }
        Ok(())
    }

    /// Send PSYNC command
    pub fn send_psync(&mut self) -> io::Result<()> {
        if let Some(ref mut conn) = self.connection {
            let offset_str = if self.offset == 0 {
                "-1".to_string()
            } else {
                self.offset.to_string()
            };

            let cmd = format!(
                "*3\r\n$5\r\nPSYNC\r\n${}\r\n{}\r\n${}\r\n{}\r\n",
                self.master_repl_id.len(),
                self.master_repl_id,
                offset_str.len(),
                offset_str
            );
            conn.write_all(cmd.as_bytes())?;
            self.state = ReplicaConnectionState::SendPsync;
        }
        Ok(())
    }

    /// Send ACK with current offset
    pub fn send_ack(&mut self) -> io::Result<()> {
        if let Some(ref mut conn) = self.connection {
            let offset_str = self.offset.to_string();
            let cmd = format!(
                "*3\r\n$8\r\nREPLCONF\r\n$3\r\nACK\r\n${}\r\n{}\r\n",
                offset_str.len(),
                offset_str
            );
            conn.write_all(cmd.as_bytes())?;
        }
        Ok(())
    }

    /// Get current state
    pub fn state(&self) -> &ReplicaConnectionState {
        &self.state
    }

    /// Set state
    pub fn set_state(&mut self, state: ReplicaConnectionState) {
        self.state = state;
    }

    /// Get master replication ID
    pub fn master_repl_id(&self) -> &str {
        &self.master_repl_id
    }

    /// Set master replication ID
    pub fn set_master_repl_id(&mut self, repl_id: String) {
        self.master_repl_id = repl_id;
    }

    /// Get current offset
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Update offset
    pub fn add_offset(&mut self, bytes: u64) {
        self.offset += bytes;
    }

    /// Set offset
    pub fn set_offset(&mut self, offset: u64) {
        self.offset = offset;
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.state == ReplicaConnectionState::Connected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replica_state() {
        let replica = ReplicaState::new("localhost".to_string(), 6379);
        assert_eq!(*replica.state(), ReplicaConnectionState::Disconnected);
        assert!(!replica.is_connected());
    }

    #[test]
    fn test_offset_tracking() {
        let mut replica = ReplicaState::new("localhost".to_string(), 6379);
        assert_eq!(replica.offset(), 0);

        replica.add_offset(100);
        assert_eq!(replica.offset(), 100);

        replica.set_offset(200);
        assert_eq!(replica.offset(), 200);
    }
}
