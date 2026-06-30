//! Async-aware mutex
//!
//! A mutex that yields to the async runtime while waiting for the lock,
//! rather than blocking the thread.

use std::cell::UnsafeCell;
use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex as SyncMutex;

const UNLOCKED: u32 = 0;
const LOCKED: u32 = 1;

/// An async-aware mutex
///
/// Unlike `std::sync::Mutex`, this mutex will yield to the async runtime
/// when waiting for the lock, allowing other tasks to make progress.
pub struct Mutex<T> {
    /// Lock state (0 = unlocked, 1 = locked)
    state: AtomicU32,
    /// Queue of waiters
    waiters: SyncMutex<Vec<Waker>>,
    /// The protected data
    data: UnsafeCell<T>,
}

// Safety: Mutex provides synchronized access to T
unsafe impl<T: Send> Send for Mutex<T> {}
unsafe impl<T: Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    /// Create a new mutex
    pub const fn new(value: T) -> Self {
        Self {
            state: AtomicU32::new(UNLOCKED),
            waiters: SyncMutex::new(Vec::new()),
            data: UnsafeCell::new(value),
        }
    }

    /// Acquire the lock asynchronously
    pub async fn lock(&self) -> MutexGuard<'_, T> {
        Lock { mutex: self }.await
    }

    /// Try to acquire the lock without blocking
    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        if self
            .state
            .compare_exchange(UNLOCKED, LOCKED, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(MutexGuard { mutex: self })
        } else {
            None
        }
    }

    /// Get the underlying data (requires mutable access)
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// Consume the mutex and return the underlying data
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }
}

/// Future for acquiring the lock
struct Lock<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<'a, T> Future for Lock<'a, T> {
    type Output = MutexGuard<'a, T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Try to acquire the lock
        if self
            .mutex
            .state
            .compare_exchange(UNLOCKED, LOCKED, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return Poll::Ready(MutexGuard { mutex: self.mutex });
        }

        // Register our waker
        self.mutex.waiters.lock().push(cx.waker().clone());

        // Try once more after registering (avoid missed wakeup)
        if self
            .mutex
            .state
            .compare_exchange(UNLOCKED, LOCKED, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            // Remove our waker since we got the lock
            self.mutex.waiters.lock().retain(|w| !w.will_wake(cx.waker()));
            return Poll::Ready(MutexGuard { mutex: self.mutex });
        }

        Poll::Pending
    }
}

/// Guard that releases the lock when dropped
pub struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<T> Deref for MutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Safety: We hold the lock
        unsafe { &*self.mutex.data.get() }
    }
}

impl<T> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: We hold the lock
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        // Release the lock
        self.mutex.state.store(UNLOCKED, Ordering::Release);

        // Wake one waiter
        if let Some(waker) = self.mutex.waiters.lock().pop() {
            waker.wake();
        }
    }
}

/// An async-aware read-write lock
pub struct RwLock<T> {
    /// Lock state: 0 = unlocked, positive = readers, -1 = writer
    state: AtomicI32,
    /// Queue of waiting writers
    writer_waiters: SyncMutex<Vec<Waker>>,
    /// Queue of waiting readers
    reader_waiters: SyncMutex<Vec<Waker>>,
    /// The protected data
    data: UnsafeCell<T>,
}

use std::sync::atomic::AtomicI32;

const RW_UNLOCKED: i32 = 0;
const RW_WRITE_LOCKED: i32 = -1;

// Safety: RwLock provides synchronized access to T
unsafe impl<T: Send> Send for RwLock<T> {}
unsafe impl<T: Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    /// Create a new RwLock
    pub const fn new(value: T) -> Self {
        Self {
            state: AtomicI32::new(RW_UNLOCKED),
            writer_waiters: SyncMutex::new(Vec::new()),
            reader_waiters: SyncMutex::new(Vec::new()),
            data: UnsafeCell::new(value),
        }
    }

    /// Acquire a read lock asynchronously
    pub async fn read(&self) -> RwLockReadGuard<'_, T> {
        ReadLock { rwlock: self }.await
    }

    /// Acquire a write lock asynchronously
    pub async fn write(&self) -> RwLockWriteGuard<'_, T> {
        WriteLock { rwlock: self }.await
    }

    /// Try to acquire a read lock without blocking
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
        loop {
            let state = self.state.load(Ordering::Acquire);
            if state < 0 {
                // Writer holds the lock
                return None;
            }

            // Try to increment reader count
            if self
                .state
                .compare_exchange(state, state + 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return Some(RwLockReadGuard { rwlock: self });
            }
        }
    }

    /// Try to acquire a write lock without blocking
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
        if self
            .state
            .compare_exchange(RW_UNLOCKED, RW_WRITE_LOCKED, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(RwLockWriteGuard { rwlock: self })
        } else {
            None
        }
    }

    /// Get the underlying data (requires mutable access)
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// Consume the lock and return the underlying data
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }
}

/// Future for acquiring a read lock
struct ReadLock<'a, T> {
    rwlock: &'a RwLock<T>,
}

impl<'a, T> Future for ReadLock<'a, T> {
    type Output = RwLockReadGuard<'a, T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            let state = self.rwlock.state.load(Ordering::Acquire);

            if state >= 0 {
                // No writer, try to acquire
                if self
                    .rwlock
                    .state
                    .compare_exchange(state, state + 1, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    return Poll::Ready(RwLockReadGuard { rwlock: self.rwlock });
                }
                continue;
            }

            // Writer holds the lock, register waker
            self.rwlock.reader_waiters.lock().push(cx.waker().clone());

            // Double-check
            let state = self.rwlock.state.load(Ordering::Acquire);
            if state >= 0 {
                if self
                    .rwlock
                    .state
                    .compare_exchange(state, state + 1, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    return Poll::Ready(RwLockReadGuard { rwlock: self.rwlock });
                }
            }

            return Poll::Pending;
        }
    }
}

/// Future for acquiring a write lock
struct WriteLock<'a, T> {
    rwlock: &'a RwLock<T>,
}

impl<'a, T> Future for WriteLock<'a, T> {
    type Output = RwLockWriteGuard<'a, T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Try to acquire
        if self
            .rwlock
            .state
            .compare_exchange(RW_UNLOCKED, RW_WRITE_LOCKED, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return Poll::Ready(RwLockWriteGuard { rwlock: self.rwlock });
        }

        // Register waker
        self.rwlock.writer_waiters.lock().push(cx.waker().clone());

        // Double-check
        if self
            .rwlock
            .state
            .compare_exchange(RW_UNLOCKED, RW_WRITE_LOCKED, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return Poll::Ready(RwLockWriteGuard { rwlock: self.rwlock });
        }

        Poll::Pending
    }
}

/// Guard for read access
pub struct RwLockReadGuard<'a, T> {
    rwlock: &'a RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Safety: Read lock held
        unsafe { &*self.rwlock.data.get() }
    }
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    fn drop(&mut self) {
        let prev = self.rwlock.state.fetch_sub(1, Ordering::Release);

        // If this was the last reader, wake a writer
        if prev == 1 {
            if let Some(waker) = self.rwlock.writer_waiters.lock().pop() {
                waker.wake();
            }
        }
    }
}

/// Guard for write access
pub struct RwLockWriteGuard<'a, T> {
    rwlock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Safety: Write lock held
        unsafe { &*self.rwlock.data.get() }
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: Write lock held
        unsafe { &mut *self.rwlock.data.get() }
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    fn drop(&mut self) {
        self.rwlock.state.store(RW_UNLOCKED, Ordering::Release);

        // Wake all waiting readers
        for waker in self.rwlock.reader_waiters.lock().drain(..) {
            waker.wake();
        }

        // Wake one waiting writer
        if let Some(waker) = self.rwlock.writer_waiters.lock().pop() {
            waker.wake();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutex_try_lock() {
        let mutex = Mutex::new(42);

        let guard = mutex.try_lock().unwrap();
        assert_eq!(*guard, 42);

        // Can't acquire while held
        assert!(mutex.try_lock().is_none());

        drop(guard);

        // Can acquire again
        assert!(mutex.try_lock().is_some());
    }

    #[test]
    fn test_mutex_modify() {
        let mutex = Mutex::new(0);

        {
            let mut guard = mutex.try_lock().unwrap();
            *guard = 42;
        }

        let guard = mutex.try_lock().unwrap();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_rwlock_read() {
        let rwlock = RwLock::new(42);

        let guard1 = rwlock.try_read().unwrap();
        let guard2 = rwlock.try_read().unwrap();

        assert_eq!(*guard1, 42);
        assert_eq!(*guard2, 42);

        // Can't write while reading
        assert!(rwlock.try_write().is_none());
    }

    #[test]
    fn test_rwlock_write() {
        let rwlock = RwLock::new(0);

        {
            let mut guard = rwlock.try_write().unwrap();
            *guard = 42;
        }

        let guard = rwlock.try_read().unwrap();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_rwlock_exclusive() {
        let rwlock = RwLock::new(0);

        let _guard = rwlock.try_write().unwrap();

        // Can't read while writing
        assert!(rwlock.try_read().is_none());
        assert!(rwlock.try_write().is_none());
    }
}
