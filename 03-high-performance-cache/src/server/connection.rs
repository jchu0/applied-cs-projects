use std::io::{self, Read, Write};
use mio::net::TcpStream;

use bytes::BytesMut;

use crate::resp::{RespParser, RespValue};

/// Client connection
pub struct Connection {
    stream: TcpStream,
    parser: RespParser,
    write_buffer: BytesMut,
    closed: bool,
    /// Index of the database this connection is currently operating on
    /// (changed via the `SELECT` command; defaults to database 0).
    selected_db: usize,
}

impl Connection {
    /// Create a new connection
    pub fn new(stream: TcpStream) -> Self {
        // mio::net::TcpStream is always non-blocking
        Self {
            stream,
            parser: RespParser::new(),
            write_buffer: BytesMut::with_capacity(4096),
            closed: false,
            selected_db: 0,
        }
    }

    /// Currently selected database index for this connection.
    pub fn selected_db(&self) -> usize {
        self.selected_db
    }

    /// Change the selected database index for this connection.
    pub fn set_selected_db(&mut self, index: usize) {
        self.selected_db = index;
    }

    /// Read data from the connection
    pub fn read(&mut self) -> io::Result<usize> {
        let mut buf = [0u8; 4096];
        let mut total_read = 0;

        loop {
            match self.stream.read(&mut buf) {
                Ok(0) => {
                    self.closed = true;
                    return Ok(total_read);
                }
                Ok(n) => {
                    self.parser.feed(&buf[..n]);
                    total_read += n;
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    return Ok(total_read);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Try to parse a command from the buffer
    pub fn parse_command(&mut self) -> io::Result<Option<RespValue>> {
        match self.parser.parse() {
            Ok(value) => Ok(value),
            Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        }
    }

    /// Write a response
    pub fn write_response(&mut self, response: RespValue) -> io::Result<()> {
        response.serialize_into(&mut self.write_buffer);
        Ok(())
    }

    /// Flush the write buffer
    pub fn flush(&mut self) -> io::Result<()> {
        while !self.write_buffer.is_empty() {
            match self.stream.write(&self.write_buffer) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write to connection",
                    ));
                }
                Ok(n) => {
                    let _ = self.write_buffer.split_to(n);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Check if connection is closed
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Check if there's data to write
    pub fn has_pending_writes(&self) -> bool {
        !self.write_buffer.is_empty()
    }

    /// Get peer address
    pub fn peer_addr(&self) -> io::Result<std::net::SocketAddr> {
        self.stream.peer_addr()
    }

    /// Get mutable reference to stream for mio registration
    pub fn stream_mut(&mut self) -> &mut TcpStream {
        &mut self.stream
    }
}
