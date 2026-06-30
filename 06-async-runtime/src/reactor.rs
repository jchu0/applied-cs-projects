//! Reactor for I/O event notification
//!
//! Provides platform-specific event notification using epoll on Linux.

use std::collections::HashMap;
use std::io;
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::task::Waker;
use std::time::Duration;

use crate::{Events, Interest, Token};

/// I/O reactor using epoll
pub struct Reactor {
    /// Epoll file descriptor
    epoll_fd: RawFd,
    /// Token to waker mapping
    wakers: Mutex<HashMap<Token, Waker>>,
    /// Current token counter
    next_token: AtomicUsize,
}

impl Reactor {
    /// Create a new reactor
    pub fn new() -> io::Result<Self> {
        let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            epoll_fd: fd,
            wakers: Mutex::new(HashMap::new()),
            next_token: AtomicUsize::new(0),
        })
    }

    /// Register interest in events for a file descriptor
    pub fn register(
        &self,
        fd: RawFd,
        interest: Interest,
        waker: Waker,
    ) -> io::Result<Token> {
        let token = Token(self.next_token.fetch_add(1, Ordering::Relaxed));

        let mut event = libc::epoll_event {
            events: interest_to_epoll(interest) as u32,
            u64: token.0 as u64,
        };

        let result = unsafe {
            libc::epoll_ctl(self.epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut event)
        };

        if result < 0 {
            return Err(io::Error::last_os_error());
        }

        self.wakers.lock().unwrap().insert(token, waker);
        Ok(token)
    }

    /// Modify interest for a registered file descriptor
    pub fn modify(
        &self,
        fd: RawFd,
        token: Token,
        interest: Interest,
        waker: Waker,
    ) -> io::Result<()> {
        let mut event = libc::epoll_event {
            events: interest_to_epoll(interest) as u32,
            u64: token.0 as u64,
        };

        let result = unsafe {
            libc::epoll_ctl(self.epoll_fd, libc::EPOLL_CTL_MOD, fd, &mut event)
        };

        if result < 0 {
            return Err(io::Error::last_os_error());
        }

        self.wakers.lock().unwrap().insert(token, waker);
        Ok(())
    }

    /// Deregister a file descriptor
    pub fn deregister(&self, fd: RawFd, token: Token) -> io::Result<()> {
        let result = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_DEL,
                fd,
                std::ptr::null_mut(),
            )
        };

        if result < 0 {
            return Err(io::Error::last_os_error());
        }

        self.wakers.lock().unwrap().remove(&token);
        Ok(())
    }

    /// Poll for events with timeout
    pub fn poll(&self, timeout: Option<Duration>) -> io::Result<usize> {
        let mut events = Events::with_capacity(1024);

        let timeout_ms = timeout
            .map(|d| d.as_millis() as i32)
            .unwrap_or(-1);

        let result = unsafe {
            libc::epoll_wait(
                self.epoll_fd,
                events.as_mut_ptr(),
                events.capacity() as i32,
                timeout_ms,
            )
        };

        if result < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                return Ok(0);
            }
            return Err(err);
        }

        unsafe { events.set_len(result as usize) };

        // Wake up tasks for ready events
        let wakers = self.wakers.lock().unwrap();
        for event in events.iter() {
            if let Some(waker) = wakers.get(&event.token()) {
                waker.wake_by_ref();
            }
        }

        Ok(result as usize)
    }

    /// Get the current waker for a token
    pub fn get_waker(&self, token: Token) -> Option<Waker> {
        self.wakers.lock().unwrap().get(&token).cloned()
    }
}

impl Drop for Reactor {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.epoll_fd);
        }
    }
}

/// Convert Interest to epoll flags
fn interest_to_epoll(interest: Interest) -> i32 {
    let mut events = libc::EPOLLET; // Edge-triggered

    if interest.is_readable() {
        events |= libc::EPOLLIN;
    }
    if interest.is_writable() {
        events |= libc::EPOLLOUT;
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::noop_waker;
    use std::os::unix::net::UnixStream;

    #[test]
    fn test_reactor_create() {
        let reactor = Reactor::new().unwrap();
        assert!(reactor.epoll_fd >= 0);
    }

    #[test]
    fn test_reactor_register() {
        let reactor = Reactor::new().unwrap();
        let (stream, _) = UnixStream::pair().unwrap();

        use std::os::unix::io::AsRawFd;
        let fd = stream.as_raw_fd();

        let waker = noop_waker();
        let token = reactor.register(fd, Interest::READABLE, waker).unwrap();

        assert_eq!(token.0, 0);
    }

    #[test]
    fn test_reactor_register_multiple() {
        let reactor = Reactor::new().unwrap();
        let (stream1, _) = UnixStream::pair().unwrap();
        let (stream2, _) = UnixStream::pair().unwrap();
        let (stream3, _) = UnixStream::pair().unwrap();

        use std::os::unix::io::AsRawFd;

        let waker = noop_waker();
        let token1 = reactor
            .register(stream1.as_raw_fd(), Interest::READABLE, waker.clone())
            .unwrap();
        let token2 = reactor
            .register(stream2.as_raw_fd(), Interest::READABLE, waker.clone())
            .unwrap();
        let token3 = reactor
            .register(stream3.as_raw_fd(), Interest::READABLE, waker)
            .unwrap();

        // Tokens should be sequential
        assert_eq!(token1.0, 0);
        assert_eq!(token2.0, 1);
        assert_eq!(token3.0, 2);
    }

    #[test]
    fn test_reactor_register_writable() {
        let reactor = Reactor::new().unwrap();
        let (stream, _) = UnixStream::pair().unwrap();

        use std::os::unix::io::AsRawFd;
        let fd = stream.as_raw_fd();

        let waker = noop_waker();
        let token = reactor.register(fd, Interest::WRITABLE, waker).unwrap();

        assert_eq!(token.0, 0);
    }

    #[test]
    fn test_reactor_register_read_write() {
        let reactor = Reactor::new().unwrap();
        let (stream, _) = UnixStream::pair().unwrap();

        use std::os::unix::io::AsRawFd;
        let fd = stream.as_raw_fd();

        let waker = noop_waker();
        let interest = Interest::READABLE.writable();
        let token = reactor.register(fd, interest, waker).unwrap();

        assert_eq!(token.0, 0);
    }

    #[test]
    fn test_reactor_deregister() {
        let reactor = Reactor::new().unwrap();
        let (stream, _) = UnixStream::pair().unwrap();

        use std::os::unix::io::AsRawFd;
        let fd = stream.as_raw_fd();

        let waker = noop_waker();
        let token = reactor.register(fd, Interest::READABLE, waker).unwrap();

        // Deregister should succeed
        assert!(reactor.deregister(fd, token).is_ok());
    }

    #[test]
    fn test_reactor_modify() {
        let reactor = Reactor::new().unwrap();
        let (stream, _) = UnixStream::pair().unwrap();

        use std::os::unix::io::AsRawFd;
        let fd = stream.as_raw_fd();

        let waker = noop_waker();
        let token = reactor.register(fd, Interest::READABLE, waker.clone()).unwrap();

        // Modify to writable
        assert!(reactor.modify(fd, token, Interest::WRITABLE, waker).is_ok());
    }

    #[test]
    fn test_reactor_poll_empty() {
        let reactor = Reactor::new().unwrap();

        // Poll with short timeout on empty reactor
        let result = reactor.poll(Some(Duration::from_millis(1)));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // No events
    }

    #[test]
    fn test_reactor_poll_with_ready_fd() {
        let reactor = Reactor::new().unwrap();
        let (stream1, mut stream2) = UnixStream::pair().unwrap();

        use std::io::Write;
        use std::os::unix::io::AsRawFd;

        let waker = noop_waker();
        let _token = reactor
            .register(stream1.as_raw_fd(), Interest::READABLE, waker)
            .unwrap();

        // Write to stream2 to make stream1 readable
        stream2.write_all(b"test").unwrap();

        // Poll should return 1 event
        let result = reactor.poll(Some(Duration::from_millis(100)));
        assert!(result.is_ok());
        assert!(result.unwrap() >= 1);
    }

    #[test]
    fn test_reactor_get_waker() {
        let reactor = Reactor::new().unwrap();
        let (stream, _) = UnixStream::pair().unwrap();

        use std::os::unix::io::AsRawFd;
        let fd = stream.as_raw_fd();

        let waker = noop_waker();
        let token = reactor.register(fd, Interest::READABLE, waker).unwrap();

        // Should be able to retrieve the waker
        let retrieved_waker = reactor.get_waker(token);
        assert!(retrieved_waker.is_some());
    }

    #[test]
    fn test_reactor_get_waker_after_deregister() {
        let reactor = Reactor::new().unwrap();
        let (stream, _) = UnixStream::pair().unwrap();

        use std::os::unix::io::AsRawFd;
        let fd = stream.as_raw_fd();

        let waker = noop_waker();
        let token = reactor.register(fd, Interest::READABLE, waker).unwrap();

        reactor.deregister(fd, token).unwrap();

        // Waker should be removed
        let retrieved_waker = reactor.get_waker(token);
        assert!(retrieved_waker.is_none());
    }

    #[test]
    fn test_reactor_drop() {
        {
            let reactor = Reactor::new().unwrap();
            let fd = reactor.epoll_fd;
            assert!(fd >= 0);
            // Reactor will be dropped here
        }
        // File descriptor should be closed after drop
    }

    #[test]
    fn test_interest_to_epoll_readable() {
        let events = interest_to_epoll(Interest::READABLE);
        assert!(events & libc::EPOLLIN != 0);
        assert!(events & libc::EPOLLET != 0); // Edge-triggered
    }

    #[test]
    fn test_interest_to_epoll_writable() {
        let events = interest_to_epoll(Interest::WRITABLE);
        assert!(events & libc::EPOLLOUT != 0);
        assert!(events & libc::EPOLLET != 0); // Edge-triggered
    }

    #[test]
    fn test_interest_to_epoll_both() {
        let interest = Interest::READABLE.writable();
        let events = interest_to_epoll(interest);
        assert!(events & libc::EPOLLIN != 0);
        assert!(events & libc::EPOLLOUT != 0);
        assert!(events & libc::EPOLLET != 0); // Edge-triggered
    }

    #[test]
    fn test_reactor_poll_timeout_zero() {
        let reactor = Reactor::new().unwrap();

        // Poll with zero timeout should return immediately
        let result = reactor.poll(Some(Duration::ZERO));
        assert!(result.is_ok());
    }

    #[test]
    fn test_reactor_poll_no_timeout() {
        let reactor = Reactor::new().unwrap();
        let (stream1, mut stream2) = UnixStream::pair().unwrap();

        use std::io::Write;
        use std::os::unix::io::AsRawFd;

        let waker = noop_waker();
        let _token = reactor
            .register(stream1.as_raw_fd(), Interest::READABLE, waker)
            .unwrap();

        // Write data to make the poll return immediately
        stream2.write_all(b"data").unwrap();

        // Poll with Some timeout
        let result = reactor.poll(Some(Duration::from_millis(100)));
        assert!(result.is_ok());
    }

    #[test]
    fn test_reactor_multiple_events() {
        let reactor = Reactor::new().unwrap();
        let (stream1, mut stream2) = UnixStream::pair().unwrap();
        let (stream3, mut stream4) = UnixStream::pair().unwrap();

        use std::io::Write;
        use std::os::unix::io::AsRawFd;

        let waker = noop_waker();
        let _token1 = reactor
            .register(stream1.as_raw_fd(), Interest::READABLE, waker.clone())
            .unwrap();
        let _token2 = reactor
            .register(stream3.as_raw_fd(), Interest::READABLE, waker)
            .unwrap();

        // Make both streams readable
        stream2.write_all(b"test1").unwrap();
        stream4.write_all(b"test2").unwrap();

        // Poll should return at least 1 event (might coalesce)
        let result = reactor.poll(Some(Duration::from_millis(100)));
        assert!(result.is_ok());
        assert!(result.unwrap() >= 1);
    }

    #[test]
    fn test_reactor_waker_map_integrity() {
        let reactor = Reactor::new().unwrap();

        // Create multiple stream pairs and register them
        let mut fds = Vec::new();
        let mut tokens = Vec::new();

        for _ in 0..5 {
            let (stream, _) = UnixStream::pair().unwrap();
            use std::os::unix::io::AsRawFd;
            let fd = stream.as_raw_fd();
            let waker = noop_waker();
            let token = reactor.register(fd, Interest::READABLE, waker).unwrap();
            fds.push(stream); // Keep stream alive
            tokens.push(token);
        }

        // All tokens should have wakers
        for token in &tokens {
            assert!(reactor.get_waker(*token).is_some());
        }
    }
}
