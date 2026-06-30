//! Multi-producer, single-consumer channel
//!
//! A bounded channel for sending multiple values between tasks.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

/// Error when sending to a closed channel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendError<T>(pub T);

impl<T> std::fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "channel closed")
    }
}

impl<T: std::fmt::Debug> std::error::Error for SendError<T> {}

/// Error when receiving from a closed and empty channel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecvError;

impl std::fmt::Display for RecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "channel closed")
    }
}

impl std::error::Error for RecvError {}

/// Shared channel state
struct Inner<T> {
    buffer: Mutex<VecDeque<T>>,
    capacity: usize,
    closed: AtomicBool,
    sender_count: AtomicUsize,
    send_wakers: Mutex<Vec<Waker>>,
    recv_waker: Mutex<Option<Waker>>,
}

/// Create a new bounded mpsc channel
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let inner = Arc::new(Inner {
        buffer: Mutex::new(VecDeque::with_capacity(capacity)),
        capacity,
        closed: AtomicBool::new(false),
        sender_count: AtomicUsize::new(1),
        send_wakers: Mutex::new(Vec::new()),
        recv_waker: Mutex::new(None),
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
    pub fn send(&self, value: T) -> Send<'_, T> {
        Send {
            sender: self,
            value: Some(value),
        }
    }

    /// Try to send a value without blocking
    pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(TrySendError::Closed(value));
        }

        let mut buffer = self.inner.buffer.lock();
        if buffer.len() >= self.inner.capacity {
            return Err(TrySendError::Full(value));
        }

        buffer.push_back(value);
        drop(buffer);

        // Wake receiver
        if let Some(waker) = self.inner.recv_waker.lock().take() {
            waker.wake();
        }

        Ok(())
    }

    /// Check if the channel is closed
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Relaxed)
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.inner.sender_count.fetch_add(1, Ordering::SeqCst);
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        if self.inner.sender_count.fetch_sub(1, Ordering::SeqCst) == 1 {
            // Last sender, close the channel
            self.inner.closed.store(true, Ordering::SeqCst);

            // Wake receiver
            if let Some(waker) = self.inner.recv_waker.lock().take() {
                waker.wake();
            }
        }
    }
}

/// Future for sending a value
pub struct Send<'a, T> {
    sender: &'a Sender<T>,
    value: Option<T>,
}

// Send is safe to unpin because it doesn't contain self-referential pointers
impl<'a, T> Unpin for Send<'a, T> {}

impl<'a, T> Future for Send<'a, T> {
    type Output = Result<(), SendError<T>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let value = this.value.take().expect("polled after completion");

        if this.sender.inner.closed.load(Ordering::SeqCst) {
            return Poll::Ready(Err(SendError(value)));
        }

        let mut buffer = this.sender.inner.buffer.lock();
        if buffer.len() >= this.sender.inner.capacity {
            // Buffer full, register waker and return value
            this.value = Some(value);
            this.sender.inner.send_wakers.lock().push(cx.waker().clone());
            return Poll::Pending;
        }

        buffer.push_back(value);
        drop(buffer);

        // Wake receiver
        if let Some(waker) = this.sender.inner.recv_waker.lock().take() {
            waker.wake();
        }

        Poll::Ready(Ok(()))
    }
}

/// Receiving half of the channel
pub struct Receiver<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Receiver<T> {
    /// Receive a value
    pub fn recv(&mut self) -> Recv<'_, T> {
        Recv { receiver: self }
    }

    /// Try to receive a value without blocking
    pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
        let mut buffer = self.inner.buffer.lock();

        if let Some(value) = buffer.pop_front() {
            drop(buffer);

            // Wake senders
            for waker in self.inner.send_wakers.lock().drain(..) {
                waker.wake();
            }

            Ok(value)
        } else if self.inner.closed.load(Ordering::SeqCst) {
            Err(TryRecvError::Closed)
        } else {
            Err(TryRecvError::Empty)
        }
    }

    /// Close the channel
    pub fn close(&mut self) {
        self.inner.closed.store(true, Ordering::SeqCst);

        // Wake all senders
        for waker in self.inner.send_wakers.lock().drain(..) {
            waker.wake();
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.close();
    }
}

/// Future for receiving a value
pub struct Recv<'a, T> {
    receiver: &'a mut Receiver<T>,
}

impl<'a, T> Future for Recv<'a, T> {
    type Output = Result<T, RecvError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut buffer = this.receiver.inner.buffer.lock();

        if let Some(value) = buffer.pop_front() {
            drop(buffer);

            // Wake senders
            for waker in this.receiver.inner.send_wakers.lock().drain(..) {
                waker.wake();
            }

            return Poll::Ready(Ok(value));
        }

        if this.receiver.inner.closed.load(Ordering::SeqCst) {
            return Poll::Ready(Err(RecvError));
        }

        // Register waker
        *this.receiver.inner.recv_waker.lock() = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// Error from try_send
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrySendError<T> {
    /// Channel buffer is full
    Full(T),
    /// Channel is closed
    Closed(T),
}

impl<T> std::fmt::Display for TrySendError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrySendError::Full(_) => write!(f, "channel full"),
            TrySendError::Closed(_) => write!(f, "channel closed"),
        }
    }
}

impl<T: std::fmt::Debug> std::error::Error for TrySendError<T> {}

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
        let (tx, mut rx) = channel::<i32>(16);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        tx.try_send(3).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert_eq!(rx.try_recv().unwrap(), 3);
    }

    #[test]
    fn test_sender_clone() {
        let (tx1, mut rx) = channel::<i32>(16);
        let tx2 = tx1.clone();
        tx1.try_send(1).unwrap();
        tx2.try_send(2).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
    }

    #[test]
    fn test_full_channel() {
        let (tx, _rx) = channel::<i32>(2);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        assert!(matches!(tx.try_send(3), Err(TrySendError::Full(3))));
    }

    #[test]
    fn test_all_senders_dropped() {
        let (tx, mut rx) = channel::<i32>(16);
        tx.try_send(1).unwrap();
        drop(tx);
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Closed));
    }
}
