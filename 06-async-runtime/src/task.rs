//! Task system for managing async tasks
//!
//! Tasks are the units of work in the async runtime.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use parking_lot::Mutex;

use crate::executor::EXECUTOR;
use crate::runtime::RUNTIME_SCHEDULER;

/// Task states
const TASK_IDLE: u8 = 0;
const TASK_SCHEDULED: u8 = 1;
const TASK_RUNNING: u8 = 2;
const TASK_COMPLETED: u8 = 3;

/// A spawned async task
pub struct Task {
    /// The future being executed
    future: Mutex<Pin<Box<dyn Future<Output = ()> + Send>>>,
    /// Task state
    state: AtomicU8,
}

impl Task {
    /// Create a new task from a future
    pub fn new<F>(future: F) -> Arc<Self>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        Arc::new(Self {
            future: Mutex::new(Box::pin(future)),
            state: AtomicU8::new(TASK_SCHEDULED),
        })
    }

    /// Poll the task's future
    pub fn poll(self: Arc<Self>) -> bool {
        // Mark as running
        self.state.store(TASK_RUNNING, Ordering::SeqCst);

        // Create waker from task
        let waker = self.clone().into_waker();
        let mut cx = Context::from_waker(&waker);

        // Poll future
        let mut future = self.future.lock();
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                self.state.store(TASK_COMPLETED, Ordering::SeqCst);
                true
            }
            Poll::Pending => {
                self.state.store(TASK_IDLE, Ordering::SeqCst);
                false
            }
        }
    }

    /// Check if task is completed
    pub fn is_completed(&self) -> bool {
        self.state.load(Ordering::SeqCst) == TASK_COMPLETED
    }

    /// Convert task to a waker
    fn into_waker(self: Arc<Self>) -> Waker {
        let ptr = Arc::into_raw(self) as *const ();
        let vtable = &TASK_WAKER_VTABLE;
        unsafe { Waker::from_raw(RawWaker::new(ptr, vtable)) }
    }

    /// Wake up the task (schedule for execution)
    fn wake(self: Arc<Self>) {
        let prev = self.state.swap(TASK_SCHEDULED, Ordering::SeqCst);
        if prev == TASK_IDLE {
            // Try multi-threaded scheduler first (from runtime.rs)
            let scheduled = RUNTIME_SCHEDULER.with(|s| {
                if let Some(scheduler) = s.borrow().as_ref() {
                    scheduler.push(self.clone());
                    true
                } else {
                    false
                }
            });

            // Fall back to single-threaded executor if no scheduler
            if !scheduled {
                EXECUTOR.with(|ex| {
                    if let Some(executor) = ex.borrow().as_ref() {
                        executor.schedule(self);
                    }
                });
            }
        }
    }
}

// Waker vtable for Task
static TASK_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    // clone
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const Task) };
        let cloned = arc.clone();
        std::mem::forget(arc);
        RawWaker::new(Arc::into_raw(cloned) as *const (), &TASK_WAKER_VTABLE)
    },
    // wake
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const Task) };
        arc.wake();
    },
    // wake_by_ref
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const Task) };
        arc.clone().wake();
        std::mem::forget(arc);
    },
    // drop
    |ptr| {
        unsafe { Arc::from_raw(ptr as *const Task) };
    },
);

/// Handle to a spawned task
pub struct JoinHandle<T> {
    /// Result storage
    result: Arc<Mutex<Option<T>>>,
}

impl<T> JoinHandle<T> {
    /// Create a new join handle
    pub fn new(result: Arc<Mutex<Option<T>>>) -> Self {
        Self { result }
    }

    /// Try to get the result (non-blocking)
    pub fn try_join(&self) -> Option<T> {
        self.result.lock().take()
    }
}

impl<T> Future for JoinHandle<T>
where
    T: Unpin,
{
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(result) = self.result.lock().take() {
            Poll::Ready(result)
        } else {
            // TODO: Register waker to be notified when task completes
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Create a waker that does nothing (for testing)
pub fn noop_waker() -> Waker {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(std::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}
