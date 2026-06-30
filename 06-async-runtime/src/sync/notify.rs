//! Async notification primitive
//!
//! Provides a mechanism for tasks to wait for a notification.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

/// A notification primitive for waking tasks
///
/// `Notify` provides a mechanism for tasks to wait until they are notified.
/// Multiple tasks can wait on the same `Notify`, and they will be woken
/// when `notify_one` or `notify_all` is called.
pub struct Notify {
    inner: Arc<Inner>,
}

struct Inner {
    /// Number of pending notifications (for notify_one)
    pending: AtomicUsize,
    /// Waiters
    waiters: Mutex<Vec<Waker>>,
}

impl Notify {
    /// Create a new `Notify`
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                pending: AtomicUsize::new(0),
                waiters: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Notify one waiting task
    ///
    /// If there are waiting tasks, one will be woken. If there are no
    /// waiting tasks, the notification is stored and the next call to
    /// `notified()` will return immediately.
    pub fn notify_one(&self) {
        // Try to wake a waiter
        if let Some(waker) = self.inner.waiters.lock().pop() {
            waker.wake();
        } else {
            // No waiters, store notification
            self.inner.pending.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Notify all waiting tasks
    ///
    /// All currently waiting tasks will be woken. This does not store
    /// notifications for future waiters.
    pub fn notify_all(&self) {
        let waiters: Vec<_> = self.inner.waiters.lock().drain(..).collect();
        for waker in waiters {
            waker.wake();
        }
    }

    /// Wait for a notification
    ///
    /// Returns a future that completes when this `Notify` is notified.
    pub fn notified(&self) -> Notified {
        Notified {
            inner: self.inner.clone(),
            registered: false,
        }
    }
}

impl Default for Notify {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Notify {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Future returned by `Notify::notified()`
pub struct Notified {
    inner: Arc<Inner>,
    registered: bool,
}

impl Future for Notified {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Check for pending notification
        loop {
            let pending = self.inner.pending.load(Ordering::SeqCst);
            if pending > 0 {
                // Try to consume the notification
                if self
                    .inner
                    .pending
                    .compare_exchange(pending, pending - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    return Poll::Ready(());
                }
                // Retry
                continue;
            }
            break;
        }

        // No pending notification, register waker
        if !self.registered {
            self.inner.waiters.lock().push(cx.waker().clone());
            self.registered = true;

            // Double-check after registering
            let pending = self.inner.pending.load(Ordering::SeqCst);
            if pending > 0 {
                if self
                    .inner
                    .pending
                    .compare_exchange(pending, pending - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    return Poll::Ready(());
                }
            }
        }

        Poll::Pending
    }
}

/// A semaphore for limiting concurrent access
pub struct Semaphore {
    /// Current number of permits available
    permits: AtomicUsize,
    /// Maximum permits
    max_permits: usize,
    /// Waiters
    waiters: Mutex<Vec<(usize, Waker)>>,
}

impl Semaphore {
    /// Create a new semaphore with the given number of permits
    pub fn new(permits: usize) -> Self {
        Self {
            permits: AtomicUsize::new(permits),
            max_permits: permits,
            waiters: Mutex::new(Vec::new()),
        }
    }

    /// Acquire a single permit
    pub async fn acquire(&self) -> SemaphorePermit<'_> {
        self.acquire_many(1).await
    }

    /// Acquire multiple permits
    pub async fn acquire_many(&self, n: usize) -> SemaphorePermit<'_> {
        Acquire {
            semaphore: self,
            needed: n,
        }
        .await;

        SemaphorePermit {
            semaphore: self,
            permits: n,
        }
    }

    /// Try to acquire a single permit without blocking
    pub fn try_acquire(&self) -> Option<SemaphorePermit<'_>> {
        self.try_acquire_many(1)
    }

    /// Try to acquire multiple permits without blocking
    pub fn try_acquire_many(&self, n: usize) -> Option<SemaphorePermit<'_>> {
        loop {
            let current = self.permits.load(Ordering::Acquire);
            if current < n {
                return None;
            }

            if self
                .permits
                .compare_exchange(current, current - n, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Some(SemaphorePermit {
                    semaphore: self,
                    permits: n,
                });
            }
        }
    }

    /// Get the number of available permits
    pub fn available_permits(&self) -> usize {
        self.permits.load(Ordering::Relaxed)
    }

    /// Add permits back (used internally by drop)
    fn add_permits(&self, n: usize) {
        self.permits.fetch_add(n, Ordering::Release);

        // Wake waiters that might now be able to proceed
        let mut waiters = self.waiters.lock();
        let available = self.permits.load(Ordering::Acquire);

        // Wake waiters in order until we can't satisfy any more
        let mut i = 0;
        while i < waiters.len() {
            if waiters[i].0 <= available {
                let (_, waker) = waiters.remove(i);
                waker.wake();
            } else {
                i += 1;
            }
        }
    }
}

/// Future for acquiring semaphore permits
struct Acquire<'a> {
    semaphore: &'a Semaphore,
    needed: usize,
}

impl<'a> Future for Acquire<'a> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            let current = self.semaphore.permits.load(Ordering::Acquire);
            if current >= self.needed {
                if self
                    .semaphore
                    .permits
                    .compare_exchange(
                        current,
                        current - self.needed,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    return Poll::Ready(());
                }
                continue;
            }

            // Not enough permits, register waker
            self.semaphore
                .waiters
                .lock()
                .push((self.needed, cx.waker().clone()));

            // Double-check
            let current = self.semaphore.permits.load(Ordering::Acquire);
            if current >= self.needed {
                continue;
            }

            return Poll::Pending;
        }
    }
}

/// RAII guard for semaphore permits
pub struct SemaphorePermit<'a> {
    semaphore: &'a Semaphore,
    permits: usize,
}

impl Drop for SemaphorePermit<'_> {
    fn drop(&mut self) {
        self.semaphore.add_permits(self.permits);
    }
}

/// A barrier for synchronizing multiple tasks
pub struct Barrier {
    /// Number of tasks to wait for
    num_tasks: usize,
    /// Current count of waiting tasks
    count: AtomicUsize,
    /// Generation (to handle reuse)
    generation: AtomicUsize,
    /// Waiting tasks
    waiters: Mutex<Vec<(usize, Waker)>>,
}

impl Barrier {
    /// Create a new barrier for the given number of tasks
    pub fn new(n: usize) -> Self {
        Self {
            num_tasks: n,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            waiters: Mutex::new(Vec::new()),
        }
    }

    /// Wait at the barrier
    ///
    /// Returns when all tasks have reached the barrier.
    pub async fn wait(&self) -> BarrierWaitResult {
        BarrierWait { barrier: self }.await
    }
}

/// Future for waiting at a barrier
struct BarrierWait<'a> {
    barrier: &'a Barrier,
}

impl<'a> Future for BarrierWait<'a> {
    type Output = BarrierWaitResult;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let gen = self.barrier.generation.load(Ordering::SeqCst);
        let count = self.barrier.count.fetch_add(1, Ordering::SeqCst) + 1;

        if count == self.barrier.num_tasks {
            // We're the last one, wake everyone
            self.barrier.count.store(0, Ordering::SeqCst);
            self.barrier.generation.fetch_add(1, Ordering::SeqCst);

            let waiters: Vec<_> = self.barrier.waiters.lock().drain(..).collect();
            for (_, waker) in waiters {
                waker.wake();
            }

            Poll::Ready(BarrierWaitResult { is_leader: true })
        } else {
            // Not the last, wait
            self.barrier.waiters.lock().push((gen, cx.waker().clone()));

            // Check if we were woken while registering
            if self.barrier.generation.load(Ordering::SeqCst) != gen {
                Poll::Ready(BarrierWaitResult { is_leader: false })
            } else {
                Poll::Pending
            }
        }
    }
}

/// Result from waiting at a barrier
pub struct BarrierWaitResult {
    is_leader: bool,
}

impl BarrierWaitResult {
    /// Returns true if this task was the last to reach the barrier
    pub fn is_leader(&self) -> bool {
        self.is_leader
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notify_pending() {
        let notify = Notify::new();

        // Notify before anyone is waiting
        notify.notify_one();

        // Should be stored
        assert_eq!(notify.inner.pending.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_semaphore_try_acquire() {
        let sem = Semaphore::new(2);

        let _p1 = sem.try_acquire().unwrap();
        let _p2 = sem.try_acquire().unwrap();

        assert!(sem.try_acquire().is_none());
    }

    #[test]
    fn test_semaphore_release() {
        let sem = Semaphore::new(1);

        {
            let _p = sem.try_acquire().unwrap();
            assert!(sem.try_acquire().is_none());
        }

        // After drop, permit should be available
        assert!(sem.try_acquire().is_some());
    }

    #[test]
    fn test_semaphore_many() {
        let sem = Semaphore::new(5);

        let _p = sem.try_acquire_many(3).unwrap();
        assert_eq!(sem.available_permits(), 2);

        assert!(sem.try_acquire_many(3).is_none());
        assert!(sem.try_acquire_many(2).is_some());
    }
}
