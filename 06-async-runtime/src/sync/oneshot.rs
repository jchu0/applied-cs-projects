//! One-shot channel
//!
//! A single-use channel for sending one value between tasks.

use std::cell::UnsafeCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

/// Channel state
const EMPTY: u8 = 0;
const SENDING: u8 = 1;
const SENT: u8 = 2;
const CLOSED: u8 = 3;

/// Error when receiving from a closed channel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecvError;

impl std::fmt::Display for RecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "channel closed")
    }
}

impl std::error::Error for RecvError {}

/// Error when sending to a closed channel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendError<T>(pub T);

impl<T> std::fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "channel closed")
    }
}

impl<T: std::fmt::Debug> std::error::Error for SendError<T> {}

/// Shared channel state
struct Inner<T> {
    state: AtomicU8,
    value: UnsafeCell<Option<T>>,
    waker: Mutex<Option<Waker>>,
}

unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

/// Create a new oneshot channel
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let inner = Arc::new(Inner {
        state: AtomicU8::new(EMPTY),
        value: UnsafeCell::new(None),
        waker: Mutex::new(None),
    });

    (
        Sender {
            inner: inner.clone(),
        },
        Receiver { inner },
    )
}

/// Sending half of the channel
pub struct Sender<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Sender<T> {
    /// Send a value
    pub fn send(self, value: T) -> Result<(), SendError<T>> {
        // Check if receiver is still alive
        if Arc::strong_count(&self.inner) == 1 {
            return Err(SendError(value));
        }

        // Try to transition to sending state
        match self.inner.state.compare_exchange(
            EMPTY,
            SENDING,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {
                // Store value
                unsafe {
                    *self.inner.value.get() = Some(value);
                }

                // Mark as sent
                self.inner.state.store(SENT, Ordering::SeqCst);

                // Wake receiver
                if let Some(waker) = self.inner.waker.lock().take() {
                    waker.wake();
                }

                Ok(())
            }
            Err(_) => Err(SendError(value)),
        }
    }

    /// Check if the receiver is still alive
    pub fn is_closed(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        // Mark as closed if not already sent
        let _ = self.inner.state.compare_exchange(
            EMPTY,
            CLOSED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );

        // Wake receiver
        if let Some(waker) = self.inner.waker.lock().take() {
            waker.wake();
        }
    }
}

/// Receiving half of the channel
pub struct Receiver<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Receiver<T> {
    /// Try to receive the value without blocking
    pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
        match self.inner.state.load(Ordering::SeqCst) {
            SENT => {
                let value = unsafe { (*self.inner.value.get()).take() };
                Ok(value.expect("value should be present"))
            }
            CLOSED => Err(TryRecvError::Closed),
            _ => Err(TryRecvError::Empty),
        }
    }

    /// Close the channel
    pub fn close(&mut self) {
        let _ = self.inner.state.compare_exchange(
            EMPTY,
            CLOSED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }
}

impl<T> Future for Receiver<T> {
    type Output = Result<T, RecvError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.inner.state.load(Ordering::SeqCst) {
            SENT => {
                let value = unsafe { (*self.inner.value.get()).take() };
                Poll::Ready(Ok(value.expect("value should be present")))
            }
            CLOSED => Poll::Ready(Err(RecvError)),
            _ => {
                // Register waker
                *self.inner.waker.lock() = Some(cx.waker().clone());

                // Check again to avoid race
                match self.inner.state.load(Ordering::SeqCst) {
                    SENT => {
                        let value = unsafe { (*self.inner.value.get()).take() };
                        Poll::Ready(Ok(value.expect("value should be present")))
                    }
                    CLOSED => Poll::Ready(Err(RecvError)),
                    _ => Poll::Pending,
                }
            }
        }
    }
}

/// Error from try_recv
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvError {
    /// Channel is empty
    Empty,
    /// Channel is closed
    Closed,
}

impl std::fmt::Display for TryRecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TryRecvError::Empty => write!(f, "channel empty"),
            TryRecvError::Closed => write!(f, "channel closed"),
        }
    }
}

impl std::error::Error for TryRecvError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_recv() {
        let (tx, mut rx) = channel::<i32>();
        tx.send(42).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 42);
    }

    #[test]
    fn test_sender_drop() {
        let (tx, mut rx) = channel::<i32>();
        drop(tx);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Closed));
    }

    #[test]
    fn test_receiver_drop() {
        let (tx, rx) = channel::<i32>();
        drop(rx);
        assert!(tx.send(42).is_err());
    }
}
