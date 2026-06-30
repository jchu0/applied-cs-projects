//! Async I/O primitives
//!
//! Provides async versions of common I/O operations.

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
        // Try to accept
        match self.listener.inner.accept() {
            Ok((stream, addr)) => {
                set_nonblocking(stream.as_raw_fd())?;
                Poll::Ready(Ok((TcpStream { inner: stream, token: None }, addr)))
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // Register for read events
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
    pub fn read<'a>(&'a mut self, buf: &'a mut [u8]) -> ReadFuture<'a> {
        ReadFuture { stream: self, buf }
    }

    /// Write data to the stream
    pub fn write<'a>(&'a mut self, buf: &'a [u8]) -> WriteFuture<'a> {
        WriteFuture { stream: self, buf }
    }

    /// Get the peer address
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }
}

impl AsRawFd for TcpStream {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

/// Future for reading from a stream
pub struct ReadFuture<'a> {
    stream: &'a mut TcpStream,
    buf: &'a mut [u8],
}

impl<'a> Future for ReadFuture<'a> {
    type Output = io::Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.stream.inner.read(this.buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // Register for read events
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

/// Future for writing to a stream
pub struct WriteFuture<'a> {
    stream: &'a mut TcpStream,
    buf: &'a [u8],
}

impl<'a> Future for WriteFuture<'a> {
    type Output = io::Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.stream.inner.write(this.buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // Register for write events
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

/// Sleep for a duration
pub async fn sleep(duration: std::time::Duration) {
    // TODO: Implement using timer wheel
    std::thread::sleep(duration);
}

/// Timeout future combinator
pub async fn timeout<F, T>(
    _duration: std::time::Duration,
    future: F,
) -> Result<T, TimeoutError>
where
    F: Future<Output = T>,
{
    // TODO: Implement proper timeout using timer wheel
    // For now, just run the future
    Ok(future.await)
}

/// Error returned when a timeout expires
#[derive(Debug, Clone)]
pub struct TimeoutError;

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "operation timed out")
    }
}

impl std::error::Error for TimeoutError {}
