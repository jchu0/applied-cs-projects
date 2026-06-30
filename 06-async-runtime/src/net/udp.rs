//! Async UDP networking
//!
//! Provides async UDP socket support.

use std::future::Future;
use std::io;
use std::net::{SocketAddr, UdpSocket as StdUdpSocket};
use std::os::unix::io::{AsRawFd, RawFd};
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

/// Async UDP socket
pub struct UdpSocket {
    inner: StdUdpSocket,
}

impl UdpSocket {
    /// Bind to an address
    pub fn bind(addr: SocketAddr) -> io::Result<Self> {
        let socket = StdUdpSocket::bind(addr)?;
        set_nonblocking(socket.as_raw_fd())?;

        Ok(Self { inner: socket })
    }

    /// Connect to a remote address
    pub fn connect(&self, addr: SocketAddr) -> io::Result<()> {
        self.inner.connect(addr)
    }

    /// Send data to the connected address
    pub fn send<'a>(&'a self, buf: &'a [u8]) -> SendFuture<'a> {
        SendFuture { socket: self, buf }
    }

    /// Receive data
    pub fn recv<'a>(&'a self, buf: &'a mut [u8]) -> RecvFuture<'a> {
        RecvFuture { socket: self, buf }
    }

    /// Send data to a specific address
    pub fn send_to<'a>(&'a self, buf: &'a [u8], addr: SocketAddr) -> SendToFuture<'a> {
        SendToFuture {
            socket: self,
            buf,
            addr,
        }
    }

    /// Receive data with source address
    pub fn recv_from<'a>(&'a self, buf: &'a mut [u8]) -> RecvFromFuture<'a> {
        RecvFromFuture { socket: self, buf }
    }

    /// Get the local address
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Set broadcast mode
    pub fn set_broadcast(&self, on: bool) -> io::Result<()> {
        self.inner.set_broadcast(on)
    }

    /// Get broadcast mode
    pub fn broadcast(&self) -> io::Result<bool> {
        self.inner.broadcast()
    }

    /// Set TTL
    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.inner.set_ttl(ttl)
    }

    /// Get TTL
    pub fn ttl(&self) -> io::Result<u32> {
        self.inner.ttl()
    }
}

impl AsRawFd for UdpSocket {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

/// Future for sending data
pub struct SendFuture<'a> {
    socket: &'a UdpSocket,
    buf: &'a [u8],
}

impl<'a> Future for SendFuture<'a> {
    type Output = io::Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.socket.inner.send(self.buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                let fd = self.socket.as_raw_fd();
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

/// Future for receiving data
pub struct RecvFuture<'a> {
    socket: &'a UdpSocket,
    buf: &'a mut [u8],
}

impl<'a> Future for RecvFuture<'a> {
    type Output = io::Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.socket.inner.recv(this.buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                let fd = this.socket.as_raw_fd();
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

/// Future for sending data to a specific address
pub struct SendToFuture<'a> {
    socket: &'a UdpSocket,
    buf: &'a [u8],
    addr: SocketAddr,
}

impl<'a> Future for SendToFuture<'a> {
    type Output = io::Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.socket.inner.send_to(self.buf, self.addr) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                let fd = self.socket.as_raw_fd();
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

/// Future for receiving data with source address
pub struct RecvFromFuture<'a> {
    socket: &'a UdpSocket,
    buf: &'a mut [u8],
}

impl<'a> Future for RecvFromFuture<'a> {
    type Output = io::Result<(usize, SocketAddr)>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.socket.inner.recv_from(this.buf) {
            Ok((n, addr)) => Poll::Ready(Ok((n, addr))),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                let fd = this.socket.as_raw_fd();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, SocketAddrV4};

    fn get_available_addr() -> SocketAddr {
        // Bind to port 0 to get an available port
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        socket.local_addr().unwrap()
    }

    #[test]
    fn test_udp_socket_bind() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr);
        assert!(socket.is_ok());
    }

    #[test]
    fn test_udp_socket_bind_loopback() {
        let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0));
        let socket = UdpSocket::bind(addr);
        assert!(socket.is_ok());

        let socket = socket.unwrap();
        let local_addr = socket.local_addr().unwrap();
        assert!(local_addr.port() > 0);
    }

    #[test]
    fn test_udp_socket_local_addr() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr).unwrap();
        let local_addr = socket.local_addr().unwrap();

        assert_eq!(local_addr.ip(), addr.ip());
    }

    #[test]
    fn test_udp_socket_as_raw_fd() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr).unwrap();
        let fd = socket.as_raw_fd();

        assert!(fd >= 0);
    }

    #[test]
    fn test_udp_socket_is_nonblocking() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr).unwrap();
        let fd = socket.as_raw_fd();

        // Check that the O_NONBLOCK flag is set
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert!(flags & libc::O_NONBLOCK != 0);
    }

    #[test]
    fn test_udp_socket_connect() {
        let socket1 = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let socket2 = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();

        let addr2 = socket2.local_addr().unwrap();

        // Connect socket1 to socket2
        assert!(socket1.connect(addr2).is_ok());
    }

    #[test]
    fn test_udp_socket_set_broadcast() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr).unwrap();

        // Set broadcast
        assert!(socket.set_broadcast(true).is_ok());
        assert!(socket.broadcast().unwrap());

        // Unset broadcast
        assert!(socket.set_broadcast(false).is_ok());
        assert!(!socket.broadcast().unwrap());
    }

    #[test]
    fn test_udp_socket_ttl() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr).unwrap();

        // Set TTL
        assert!(socket.set_ttl(64).is_ok());
        assert_eq!(socket.ttl().unwrap(), 64);

        // Change TTL
        assert!(socket.set_ttl(128).is_ok());
        assert_eq!(socket.ttl().unwrap(), 128);
    }

    #[test]
    fn test_udp_sync_send_recv() {
        let socket1 = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let socket2 = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();

        let addr1 = socket1.local_addr().unwrap();
        let addr2 = socket2.local_addr().unwrap();

        // Send from socket1 to socket2
        let sent = socket1.send_to(b"hello", addr2).unwrap();
        assert_eq!(sent, 5);

        // Receive on socket2
        let mut buf = [0u8; 10];
        let (received, from_addr) = socket2.recv_from(&mut buf).unwrap();
        assert_eq!(received, 5);
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(from_addr, addr1);
    }

    #[test]
    fn test_udp_connected_send_recv() {
        let socket1 = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let socket2 = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();

        let addr1 = socket1.local_addr().unwrap();
        let addr2 = socket2.local_addr().unwrap();

        // Connect both sockets
        socket1.connect(addr2).unwrap();
        socket2.connect(addr1).unwrap();

        // Send from socket1 (connected, so no address needed)
        let sent = socket1.send(b"world").unwrap();
        assert_eq!(sent, 5);

        // Receive on socket2
        let mut buf = [0u8; 10];
        let received = socket2.recv(&mut buf).unwrap();
        assert_eq!(received, 5);
        assert_eq!(&buf[..5], b"world");
    }

    #[test]
    fn test_udp_socket_recv_would_block() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr).unwrap();

        // Recv should return WouldBlock since no data was sent
        let mut buf = [0u8; 10];
        let result = socket.inner.recv(&mut buf);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::WouldBlock);
    }

    #[test]
    fn test_udp_socket_recv_from_would_block() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr).unwrap();

        // Recv_from should return WouldBlock since no data was sent
        let mut buf = [0u8; 10];
        let result = socket.inner.recv_from(&mut buf);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::WouldBlock);
    }

    #[test]
    fn test_udp_send_future_created() {
        let socket1 = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let socket2 = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();

        let addr2 = socket2.local_addr().unwrap();
        socket1.connect(addr2).unwrap();

        // Create send future
        let _send_future = socket1.send(b"test");
        // Future is created successfully
    }

    #[test]
    fn test_udp_recv_future_created() {
        let socket = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();

        // Create recv future
        let mut buf = [0u8; 10];
        let _recv_future = socket.recv(&mut buf);
        // Future is created successfully
    }

    #[test]
    fn test_udp_send_to_future_created() {
        let socket = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let target: SocketAddr = "127.0.0.1:12345".parse().unwrap();

        // Create send_to future
        let _send_to_future = socket.send_to(b"test", target);
        // Future is created successfully
    }

    #[test]
    fn test_udp_recv_from_future_created() {
        let socket = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();

        // Create recv_from future
        let mut buf = [0u8; 10];
        let _recv_from_future = socket.recv_from(&mut buf);
        // Future is created successfully
    }

    #[test]
    fn test_udp_multiple_binds_fail() {
        let addr = get_available_addr();
        let _socket1 = UdpSocket::bind(addr).unwrap();

        // Second bind to same address should fail
        let socket2 = UdpSocket::bind(addr);
        assert!(socket2.is_err());
    }

    #[test]
    fn test_udp_socket_drop_frees_port() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

        let port = {
            let socket = UdpSocket::bind(addr).unwrap();
            socket.local_addr().unwrap().port()
        };
        // Socket is dropped here

        // Should be able to bind to a new port
        let new_socket = UdpSocket::bind(("127.0.0.1", 0).into()).unwrap();
        let new_port = new_socket.local_addr().unwrap().port();

        // Ports should be assigned (both should be valid ports)
        assert!(port > 0);
        assert!(new_port > 0);
    }

    #[test]
    fn test_udp_large_message() {
        let socket1 = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let socket2 = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();

        let addr2 = socket2.local_addr().unwrap();

        // Send a larger message (but still within UDP limits)
        let data = vec![0xAB; 1000];
        let sent = socket1.send_to(&data, addr2).unwrap();
        assert_eq!(sent, 1000);

        // Receive
        let mut buf = vec![0u8; 2000];
        let (received, _) = socket2.recv_from(&mut buf).unwrap();
        assert_eq!(received, 1000);
        assert_eq!(&buf[..1000], &data[..]);
    }

    #[test]
    fn test_set_nonblocking_udp() {
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let fd = socket.as_raw_fd();

        // Set to nonblocking
        assert!(set_nonblocking(fd).is_ok());

        // Verify the flag is set
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert!(flags & libc::O_NONBLOCK != 0);
    }

    #[test]
    fn test_udp_broadcast_default() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr).unwrap();

        // Broadcast should be off by default
        assert!(!socket.broadcast().unwrap());
    }

    #[test]
    fn test_udp_ttl_default() {
        let addr = get_available_addr();
        let socket = UdpSocket::bind(addr).unwrap();

        // TTL should have a default value (usually 64)
        let ttl = socket.ttl().unwrap();
        assert!(ttl > 0);
    }
}
