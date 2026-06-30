//! Cancellation tokens for cooperative task cancellation
//!
//! Provides hierarchical cancellation tokens that support parent-child relationships.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

/// Inner state of a cancellation token
struct Inner {
    /// Whether cancellation has been requested
    cancelled: AtomicBool,
    /// Wakers waiting for cancellation
    wakers: Mutex<Vec<Waker>>,
    /// Child tokens (for hierarchical cancellation)
    children: Mutex<Vec<CancellationToken>>,
}

/// A token for cooperative cancellation
///
/// Cancellation tokens allow tasks to check if they should stop early.
/// They support hierarchical cancellation - when a parent token is cancelled,
/// all child tokens are also cancelled.
#[derive(Clone)]
pub struct CancellationToken {
    inner: Arc<Inner>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                cancelled: AtomicBool::new(false),
                wakers: Mutex::new(Vec::new()),
                children: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Create a child token that will be cancelled when this token is cancelled
    pub fn child_token(&self) -> CancellationToken {
        let child = CancellationToken::new();

        // If already cancelled, cancel child immediately
        if self.is_cancelled() {
            child.cancel();
        } else {
            self.inner.children.lock().push(child.clone());
        }

        child
    }

    /// Request cancellation of this token and all child tokens
    pub fn cancel(&self) {
        // Use swap to check if already cancelled
        if self.inner.cancelled.swap(true, Ordering::SeqCst) {
            return; // Already cancelled
        }

        // Wake all waiting tasks
        let wakers: Vec<_> = self.inner.wakers.lock().drain(..).collect();
        for waker in wakers {
            waker.wake();
        }

        // Cancel all children
        let children = self.inner.children.lock();
        for child in children.iter() {
            child.cancel();
        }
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Returns a future that completes when the token is cancelled
    pub fn cancelled(&self) -> Cancelled {
        Cancelled {
            token: self.clone(),
        }
    }

    /// Run a future with cancellation support
    ///
    /// Returns `None` if the token was cancelled before the future completed.
    pub async fn run_until_cancelled<F, T>(&self, future: F) -> Option<T>
    where
        F: Future<Output = T>,
    {
        let future = std::pin::pin!(future);
        let cancelled = self.cancelled();
        let cancelled = std::pin::pin!(cancelled);

        // Simple select implementation
        CancellableFuture {
            future,
            cancelled,
        }.await
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Future that completes when a cancellation token is cancelled
pub struct Cancelled {
    token: CancellationToken,
}

impl Future for Cancelled {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.token.is_cancelled() {
            Poll::Ready(())
        } else {
            // Register waker
            self.token.inner.wakers.lock().push(cx.waker().clone());

            // Double-check after registering
            if self.token.is_cancelled() {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }
    }
}

/// Future wrapper that cancels when token is cancelled
struct CancellableFuture<F, C> {
    future: Pin<F>,
    cancelled: Pin<C>,
}

impl<F, C, T> Future for CancellableFuture<F, C>
where
    F: std::ops::DerefMut,
    F::Target: Future<Output = T>,
    C: std::ops::DerefMut,
    C::Target: Future<Output = ()>,
{
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safety: We're not moving the inner futures
        let this = unsafe { self.get_unchecked_mut() };

        // Check cancellation first
        let cancelled = unsafe { Pin::new_unchecked(&mut *this.cancelled) };
        if cancelled.poll(cx).is_ready() {
            return Poll::Ready(None);
        }

        // Poll the actual future
        let future = unsafe { Pin::new_unchecked(&mut *this.future) };
        match future.poll(cx) {
            Poll::Ready(value) => Poll::Ready(Some(value)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Guard that cancels a token when dropped
pub struct DropGuard {
    token: Option<CancellationToken>,
}

impl DropGuard {
    /// Create a new drop guard that will cancel the token when dropped
    pub fn new(token: CancellationToken) -> Self {
        Self { token: Some(token) }
    }

    /// Disarm the guard, preventing cancellation on drop
    pub fn disarm(&mut self) -> Option<CancellationToken> {
        self.token.take()
    }
}

impl Drop for DropGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            token.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_child_cancellation() {
        let parent = CancellationToken::new();
        let child = parent.child_token();

        assert!(!child.is_cancelled());

        parent.cancel();
        assert!(child.is_cancelled());
    }

    #[test]
    fn test_already_cancelled_parent() {
        let parent = CancellationToken::new();
        parent.cancel();

        let child = parent.child_token();
        assert!(child.is_cancelled());
    }

    #[test]
    fn test_drop_guard() {
        let token = CancellationToken::new();
        {
            let _guard = DropGuard::new(token.clone());
        }
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_drop_guard_disarm() {
        let token = CancellationToken::new();
        {
            let mut guard = DropGuard::new(token.clone());
            guard.disarm();
        }
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_multiple_cancel() {
        let token = CancellationToken::new();
        token.cancel();
        token.cancel(); // Should not panic
        assert!(token.is_cancelled());
    }
}
