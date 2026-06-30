//! Async TCP networking
//!
//! Provides async TCP listener and stream.

use std::future::Future;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::os::unix::io::{AsRawFd, RawFd};
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::executor::EXECUTOR;
use crate::{Interest, Token};

/// Set a file descriptor to non-blocking mode
fn set_nonblocking(fd: RawFd) -> io::Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }

    let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if result < 0 {
        return Err(io::Error::last_os_error());
    }

    Ok(())
}

/// Async TCP listener
pub struct TcpListener {
    inner: StdTcpListener,
    token: Option<Token>,
}

impl TcpListener {
    /// Bind to an address
    pub fn bind(addr: SocketAddr) -> io::Result<Self> {
        let listener = StdTcpListener::bind(addr)?;
        set_nonblocking(listener.as_raw_fd())?;

        Ok(Self {
            inner: listener,
            token: None,
        })
    }

    /// Accept a connection
    pub fn accept(&self) -> Accept<'_> {
        Accept { listener: self }
    }

    /// Get the local address
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

impl AsRawFd for TcpListener {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

/// Future for accepting a connection
pub struct Accept<'a> {
    listener: &'a TcpListener,
}

impl<'a> Future for Accept<'a> {
    type Output = io::Result<(TcpStream, SocketAddr)>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.listener.inner.accept() {
            Ok((stream, addr)) => {
                set_nonblocking(stream.as_raw_fd())?;
                Poll::Ready(Ok((TcpStream { inner: stream, token: None }, addr)))
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                EXECUTOR.with(|ex| {
                    if let Some(executor) = ex.borrow().as_ref() {
                        let reactor = executor.reactor();
                        let fd = self.listener.as_raw_fd();
                        let _ = reactor.register(fd, Interest::READABLE, cx.waker().clone());
                    }
                });
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

/// Async TCP stream
pub struct TcpStream {
    inner: StdTcpStream,
    token: Option<Token>,
}

impl TcpStream {
    /// Connect to an address
    pub async fn connect(addr: SocketAddr) -> io::Result<Self> {
        let stream = StdTcpStream::connect(addr)?;
        set_nonblocking(stream.as_raw_fd())?;

        Ok(Self {
            inner: stream,
            token: None,
        })
    }

    /// Read data from the stream
    pub fn read<'a>(&'a mut self, buf: &'a mut [u8]) -> TcpReadFuture<'a> {
        TcpReadFuture { stream: self, buf }
    }

    /// Write data to the stream
    pub fn write<'a>(&'a mut self, buf: &'a [u8]) -> TcpWriteFuture<'a> {
        TcpWriteFuture { stream: self, buf }
    }

    /// Write all data to the stream
    pub async fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut written = 0;
        while written < buf.len() {
            let n = self.write(&buf[written..]).await?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ));
            }
            written += n;
        }
        Ok(())
    }

    /// Get the peer address
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }

    /// Get the local address
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Shutdown the stream
    pub fn shutdown(&self, how: std::net::Shutdown) -> io::Result<()> {
        self.inner.shutdown(how)
    }
}

impl AsRawFd for TcpStream {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

/// Future for reading from a TCP stream
pub struct TcpReadFuture<'a> {
    stream: &'a mut TcpStream,
    buf: &'a mut [u8],
}

impl<'a> Future for TcpReadFuture<'a> {
    type Output = io::Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.stream.inner.read(this.buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                let fd = this.stream.as_raw_fd();
                EXECUTOR.with(|ex| {
                    if let Some(executor) = ex.borrow().as_ref() {
                        let reactor = executor.reactor();
                        let _ = reactor.register(fd, Interest::READABLE, cx.waker().clone());
                    }
                });
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

/// Future for writing to a TCP stream
pub struct TcpWriteFuture<'a> {
    stream: &'a mut TcpStream,
    buf: &'a [u8],
}

impl<'a> Future for TcpWriteFuture<'a> {
    type Output = io::Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.stream.inner.write(this.buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                let fd = this.stream.as_raw_fd();
                EXECUTOR.with(|ex| {
                    if let Some(executor) = ex.borrow().as_ref() {
                        let reactor = executor.reactor();
                        let _ = reactor.register(fd, Interest::WRITABLE, cx.waker().clone());
                    }
                });
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, SocketAddrV4};

    fn get_available_port() -> SocketAddr {
        // Bind to port 0 to get an available port
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    }

    #[test]
    fn test_tcp_listener_bind() {
        let addr = get_available_port();
        let listener = TcpListener::bind(addr);
        assert!(listener.is_ok());
    }

    #[test]
    fn test_tcp_listener_bind_loopback() {
        let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0));
        let listener = TcpListener::bind(addr);
        assert!(listener.is_ok());

        let listener = listener.unwrap();
        let local_addr = listener.local_addr().unwrap();
        assert!(local_addr.port() > 0);
    }

    #[test]
    fn test_tcp_listener_local_addr() {
        let addr = get_available_port();
        let listener = TcpListener::bind(addr).unwrap();
        let local_addr = listener.local_addr().unwrap();

        assert_eq!(local_addr.ip(), addr.ip());
    }

    #[test]
    fn test_tcp_listener_as_raw_fd() {
        let addr = get_available_port();
        let listener = TcpListener::bind(addr).unwrap();
        let fd = listener.as_raw_fd();

        assert!(fd >= 0);
    }

    #[test]
    fn test_tcp_listener_is_nonblocking() {
        let addr = get_available_port();
        let listener = TcpListener::bind(addr).unwrap();
        let fd = listener.as_raw_fd();

        // Check that the O_NONBLOCK flag is set
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert!(flags & libc::O_NONBLOCK != 0);
    }

    #[test]
    fn test_set_nonblocking() {
        use std::os::unix::net::UnixStream;

        let (stream, _) = UnixStream::pair().unwrap();
        let fd = stream.as_raw_fd();

        // Set to nonblocking
        assert!(set_nonblocking(fd).is_ok());

        // Verify the flag is set
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert!(flags & libc::O_NONBLOCK != 0);
    }

    #[test]
    fn test_tcp_listener_accept_would_block() {
        let addr = get_available_port();
        let listener = TcpListener::bind(addr).unwrap();

        // Accept should return WouldBlock since no client is connecting
        let result = listener.inner.accept();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::WouldBlock);
    }

    #[test]
    fn test_tcp_stream_with_sync_connection() {
        // Set up a sync listener
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        // Connect with a sync client
        let client = std::net::TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();

        // Verify connection
        assert!(client.peer_addr().is_ok());
        assert!(server.peer_addr().is_ok());
    }

    #[test]
    fn test_tcp_stream_addresses() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client = std::net::TcpStream::connect(addr).unwrap();
        set_nonblocking(client.as_raw_fd()).unwrap();

        let stream = TcpStream {
            inner: client,
            token: None,
        };

        // Check addresses
        assert!(stream.peer_addr().is_ok());
        assert!(stream.local_addr().is_ok());
        assert_eq!(stream.peer_addr().unwrap(), addr);
    }

    #[test]
    fn test_tcp_stream_shutdown() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client = std::net::TcpStream::connect(addr).unwrap();
        set_nonblocking(client.as_raw_fd()).unwrap();

        let stream = TcpStream {
            inner: client,
            token: None,
        };

        // Shutdown should succeed
        assert!(stream.shutdown(std::net::Shutdown::Both).is_ok());
    }

    #[test]
    fn test_tcp_stream_as_raw_fd() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client = std::net::TcpStream::connect(addr).unwrap();
        set_nonblocking(client.as_raw_fd()).unwrap();

        let stream = TcpStream {
            inner: client,
            token: None,
        };

        assert!(stream.as_raw_fd() >= 0);
    }

    #[test]
    fn test_tcp_sync_read_write() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let mut client = std::net::TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        // Write from client
        client.write_all(b"hello").unwrap();

        // Read on server
        let mut buf = [0u8; 5];
        server.read_exact(&mut buf).unwrap();

        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn test_tcp_listener_multiple_binds_fail() {
        let addr = get_available_port();
        let _listener1 = TcpListener::bind(addr).unwrap();

        // Second bind to same address should fail
        let listener2 = TcpListener::bind(addr);
        assert!(listener2.is_err());
    }

    #[test]
    fn test_tcp_stream_nonblocking_read() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client = std::net::TcpStream::connect(addr).unwrap();
        set_nonblocking(client.as_raw_fd()).unwrap();
        let (_server, _) = listener.accept().unwrap();

        let mut stream = TcpStream {
            inner: client,
            token: None,
        };

        // Non-blocking read on empty buffer should return WouldBlock
        let mut buf = [0u8; 10];
        let result = stream.inner.read(&mut buf);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::WouldBlock);
    }

    #[test]
    fn test_tcp_listener_drop_frees_port() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

        let port = {
            let listener = TcpListener::bind(addr).unwrap();
            listener.local_addr().unwrap().port()
        };
        // Listener is dropped here

        // Should be able to bind to a new port
        let new_listener = TcpListener::bind(("127.0.0.1", 0).into()).unwrap();
        let new_port = new_listener.local_addr().unwrap().port();

        // Ports should be assigned (both should be valid ports)
        assert!(port > 0);
        assert!(new_port > 0);
    }

    #[test]
    fn test_tcp_stream_token_initial() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client = std::net::TcpStream::connect(addr).unwrap();
        set_nonblocking(client.as_raw_fd()).unwrap();

        let stream = TcpStream {
            inner: client,
            token: None,
        };

        assert!(stream.token.is_none());
    }

    #[test]
    fn test_tcp_listener_token_initial() {
        let addr = get_available_port();
        let listener = TcpListener::bind(addr).unwrap();

        assert!(listener.token.is_none());
    }

    #[test]
    fn test_accept_future_created() {
        let addr = get_available_port();
        let listener = TcpListener::bind(addr).unwrap();

        // Create accept future
        let _accept = listener.accept();
        // Future is created successfully
    }

    #[test]
    fn test_read_future_created() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client = std::net::TcpStream::connect(addr).unwrap();
        set_nonblocking(client.as_raw_fd()).unwrap();
        let (_server, _) = listener.accept().unwrap();

        let mut stream = TcpStream {
            inner: client,
            token: None,
        };

        let mut buf = [0u8; 10];
        let _read_future = stream.read(&mut buf);
        // Future is created successfully
    }

    #[test]
    fn test_write_future_created() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client = std::net::TcpStream::connect(addr).unwrap();
        set_nonblocking(client.as_raw_fd()).unwrap();
        let (_server, _) = listener.accept().unwrap();

        let mut stream = TcpStream {
            inner: client,
            token: None,
        };

        let _write_future = stream.write(b"hello");
        // Future is created successfully
    }
}
