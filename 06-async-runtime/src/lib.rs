//! Async Runtime - A custom async runtime implementation
//!
//! This crate provides a Tokio-like async runtime with:
//! - Event-driven I/O using Linux `epoll` (**Linux-only** — a macOS `kqueue`
//!   backend is not yet implemented; see `docs/RUNNING_PLATFORM_SPECIFIC.md`)
//! - Work-stealing task scheduler
//! - Timer wheel for efficient timeout management
//! - Cooperative multitasking

// The reactor calls the Linux `epoll` syscall family unconditionally, so the
// crate only compiles on Linux. Fail with a clear message elsewhere rather than
// a confusing "unresolved import libc::epoll_*" error.
#[cfg(not(target_os = "linux"))]
compile_error!(
    "async-runtime's reactor is built on Linux epoll and only compiles on Linux. \
     Build it on a Linux host/VM/container — see docs/RUNNING_PLATFORM_SPECIFIC.md."
);

pub mod reactor;
pub mod task;
pub mod executor;
pub mod timer;
pub mod io;
pub mod scheduler;
pub mod sync;
pub mod runtime;
pub mod time;
pub mod future;
pub mod net;
pub mod io_util;
pub mod scope;

pub use executor::{spawn, block_on};
pub use runtime::{Runtime, Builder, JoinHandle, JoinError};
pub use sync::{oneshot_channel, mpsc_channel};
pub use sync::{CancellationToken, Cancelled, DropGuard};
pub use sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
pub use sync::{Notify, Notified, Semaphore, SemaphorePermit, Barrier, BarrierWaitResult};
pub use time::{sleep, sleep_until, timeout, timeout_at, interval, Elapsed};
pub use future::{select, join, join3, yield_now, ready, pending, Either};
pub use scope::{scope, scope_detached, Scope, TaskSet};

/// Interest flags for I/O events
#[derive(Clone, Copy, Debug)]
pub struct Interest {
    flags: u8,
}

impl Interest {
    pub const READABLE: Interest = Interest { flags: 0b01 };
    pub const WRITABLE: Interest = Interest { flags: 0b10 };

    pub fn readable(self) -> Self {
        Interest {
            flags: self.flags | 0b01,
        }
    }

    pub fn writable(self) -> Self {
        Interest {
            flags: self.flags | 0b10,
        }
    }

    pub fn is_readable(&self) -> bool {
        self.flags & 0b01 != 0
    }

    pub fn is_writable(&self) -> bool {
        self.flags & 0b10 != 0
    }
}

/// Token for identifying registered I/O sources
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Token(pub usize);

/// Event returned from polling
#[derive(Clone, Copy, Debug)]
pub struct Event {
    token: Token,
    readable: bool,
    writable: bool,
}

impl Event {
    pub fn token(&self) -> Token {
        self.token
    }

    pub fn is_readable(&self) -> bool {
        self.readable
    }

    pub fn is_writable(&self) -> bool {
        self.writable
    }
}

/// Events buffer for polling
pub struct Events {
    inner: Vec<libc::epoll_event>,
}

impl Events {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn as_mut_ptr(&mut self) -> *mut libc::epoll_event {
        self.inner.as_mut_ptr()
    }

    pub unsafe fn set_len(&mut self, len: usize) {
        self.inner.set_len(len);
    }

    pub fn iter(&self) -> impl Iterator<Item = Event> + '_ {
        self.inner.iter().map(|e| Event {
            token: Token(e.u64 as usize),
            readable: e.events as i32 & libc::EPOLLIN != 0,
            writable: e.events as i32 & libc::EPOLLOUT != 0,
        })
    }
}
