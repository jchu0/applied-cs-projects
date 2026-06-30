//! Async Unix domain sockets
//!
//! Provides async Unix domain socket support.

use std::future::Future;
use std::io::{self, Read, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::{
    SocketAddr, UnixListener as StdUnixListener, UnixStream as StdUnixStream,
};
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::executor::EXECUTOR;
use crate::Interest;

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

/// Async Unix domain listener
pub struct UnixListener {
    inner: StdUnixListener,
}

impl UnixListener {
    /// Bind to a path
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let listener = StdUnixListener::bind(path)?;
        set_nonblocking(listener.as_raw_fd())?;

        Ok(Self { inner: listener })
    }

    /// Accept a connection
    pub fn accept(&self) -> UnixAcceptFuture<'_> {
        UnixAcceptFuture { listener: self }
    }

    /// Get the local address
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

impl AsRawFd for UnixListener {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

/// Future for accepting a Unix connection
pub struct UnixAcceptFuture<'a> {
    listener: &'a UnixListener,
}

impl<'a> Future for UnixAcceptFuture<'a> {
    type Output = io::Result<(UnixStream, SocketAddr)>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.listener.inner.accept() {
            Ok((stream, addr)) => {
                set_nonblocking(stream.as_raw_fd())?;
                Poll::Ready(Ok((UnixStream { inner: stream }, addr)))
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                let fd = self.listener.as_raw_fd();
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

/// Async Unix domain stream
pub struct UnixStream {
    inner: StdUnixStream,
}

impl UnixStream {
    /// Connect to a path
    pub fn connect<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let stream = StdUnixStream::connect(path)?;
        set_nonblocking(stream.as_raw_fd())?;

        Ok(Self { inner: stream })
    }

    /// Read data from the stream
    pub fn read<'a>(&'a mut self, buf: &'a mut [u8]) -> UnixReadFuture<'a> {
        UnixReadFuture { stream: self, buf }
    }

    /// Write data to the stream
    pub fn write<'a>(&'a mut self, buf: &'a [u8]) -> UnixWriteFuture<'a> {
        UnixWriteFuture { stream: self, buf }
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

    /// Create a pair of connected Unix streams
    pub fn pair() -> io::Result<(Self, Self)> {
        let (a, b) = StdUnixStream::pair()?;
        set_nonblocking(a.as_raw_fd())?;
        set_nonblocking(b.as_raw_fd())?;
        Ok((Self { inner: a }, Self { inner: b }))
    }
}

impl AsRawFd for UnixStream {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

/// Future for reading from a Unix stream
pub struct UnixReadFuture<'a> {
    stream: &'a mut UnixStream,
    buf: &'a mut [u8],
}

impl<'a> Future for UnixReadFuture<'a> {
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

/// Future for writing to a Unix stream
pub struct UnixWriteFuture<'a> {
    stream: &'a mut UnixStream,
    buf: &'a [u8],
}

impl<'a> Future for UnixWriteFuture<'a> {
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

    #[test]
    fn test_unix_stream_pair() {
        let (a, b) = UnixStream::pair().unwrap();
        assert!(a.as_raw_fd() >= 0);
        assert!(b.as_raw_fd() >= 0);
    }
}
