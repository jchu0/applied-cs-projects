//! Multi-threaded async runtime
//!
//! Provides a complete multi-threaded runtime with work-stealing.

use std::cell::RefCell;
use std::future::Future;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use parking_lot::Mutex;

use crate::reactor::Reactor;
use crate::scheduler::{Scheduler, WorkerHandle};
use crate::sync::oneshot;
use crate::task::Task;

thread_local! {
    /// Current multi-threaded scheduler handle for task waking
    pub static RUNTIME_SCHEDULER: RefCell<Option<Arc<Scheduler>>> = RefCell::new(None);
}

/// JoinHandle for awaiting task completion
pub struct JoinHandle<T> {
    receiver: oneshot::Receiver<T>,
}

impl<T> Future for JoinHandle<T> {
    type Output = Result<T, JoinError>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.receiver)
            .poll(cx)
            .map(|result| result.map_err(|_| JoinError::Cancelled))
    }
}

/// Error returned from JoinHandle
#[derive(Debug, Clone, Copy)]
pub enum JoinError {
    /// Task was cancelled
    Cancelled,
    /// Task panicked
    Panicked,
}

impl std::fmt::Display for JoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JoinError::Cancelled => write!(f, "task cancelled"),
            JoinError::Panicked => write!(f, "task panicked"),
        }
    }
}

impl std::error::Error for JoinError {}

/// Multi-threaded runtime
pub struct Runtime {
    scheduler: Arc<Scheduler>,
    reactor: Arc<Reactor>,
    worker_threads: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}

impl Runtime {
    /// Create a new runtime with default settings
    pub fn new() -> io::Result<Self> {
        Builder::new().build()
    }

    /// Spawn a task on the runtime
    pub fn spawn<F, T>(&self, future: F) -> JoinHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        let task = Task::new(async move {
            let result = future.await;
            let _ = tx.send(result);
        });

        self.scheduler.push(task);

        JoinHandle { receiver: rx }
    }

    /// Run a future to completion on the runtime
    pub fn block_on<F, T>(&self, future: F) -> T
    where
        F: Future<Output = T>,
    {
        // Set the scheduler as current for task waking
        RUNTIME_SCHEDULER.with(|s| {
            *s.borrow_mut() = Some(self.scheduler.clone());
        });

        // Pin the future
        let mut future = std::pin::pin!(future);

        // Create a waker that does nothing
        let waker = crate::task::noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);

        let result = loop {
            // Poll the main future
            if let std::task::Poll::Ready(result) = future.as_mut().poll(&mut cx) {
                break result;
            }

            // Poll the reactor
            let _ = self.reactor.poll(Some(Duration::from_millis(1)));

            // Check for shutdown
            if self.shutdown.load(Ordering::Relaxed) {
                panic!("Runtime shutdown while running");
            }
        };

        // Clear the scheduler
        RUNTIME_SCHEDULER.with(|s| {
            *s.borrow_mut() = None;
        });

        result
    }

    /// Shutdown the runtime
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.scheduler.shutdown();

        // Wait for worker threads
        for handle in self.worker_threads {
            let _ = handle.join();
        }
    }

    /// Get reference to the reactor
    pub fn reactor(&self) -> &Arc<Reactor> {
        &self.reactor
    }
}

/// Builder for configuring the runtime
pub struct Builder {
    worker_threads: usize,
    thread_name: String,
    on_thread_start: Option<Arc<dyn Fn() + Send + Sync>>,
    on_thread_stop: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Builder {
    /// Create a new runtime builder
    pub fn new() -> Self {
        Self {
            worker_threads: num_cpus(),
            thread_name: "runtime-worker".to_string(),
            on_thread_start: None,
            on_thread_stop: None,
        }
    }

    /// Set the number of worker threads
    pub fn worker_threads(mut self, n: usize) -> Self {
        self.worker_threads = n;
        self
    }

    /// Set the thread name prefix
    pub fn thread_name(mut self, name: impl Into<String>) -> Self {
        self.thread_name = name.into();
        self
    }

    /// Set callback for thread start
    pub fn on_thread_start<F>(mut self, f: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.on_thread_start = Some(Arc::new(f));
        self
    }

    /// Set callback for thread stop
    pub fn on_thread_stop<F>(mut self, f: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.on_thread_stop = Some(Arc::new(f));
        self
    }

    /// Build the runtime
    pub fn build(self) -> io::Result<Runtime> {
        let reactor = Arc::new(Reactor::new()?);
        let shutdown = Arc::new(AtomicBool::new(false));

        let (scheduler, worker_handles) = Scheduler::new(self.worker_threads);

        // Wrap handles in Arc<Mutex> for thread safety
        let handles: Vec<Arc<Mutex<WorkerHandle>>> = worker_handles
            .into_iter()
            .map(|h| Arc::new(Mutex::new(h)))
            .collect();

        // Spawn worker threads
        let mut worker_threads = Vec::with_capacity(self.worker_threads);

        for (i, handle) in handles.into_iter().enumerate() {
            let thread_name = format!("{}-{}", self.thread_name, i);
            let on_start = self.on_thread_start.clone();
            let on_stop = self.on_thread_stop.clone();
            let reactor = reactor.clone();
            let shutdown = shutdown.clone();

            let thread = thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    if let Some(f) = on_start {
                        f();
                    }

                    worker_loop(handle, reactor, shutdown);

                    if let Some(f) = on_stop {
                        f();
                    }
                })
                .expect("Failed to spawn worker thread");

            worker_threads.push(thread);
        }

        Ok(Runtime {
            scheduler,
            reactor,
            worker_threads,
            shutdown,
        })
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

/// Worker thread main loop
fn worker_loop(
    handle: Arc<Mutex<WorkerHandle>>,
    reactor: Arc<Reactor>,
    shutdown: Arc<AtomicBool>,
) {
    // Set the scheduler as current for task waking on this thread
    {
        let handle_guard = handle.lock();
        let scheduler = handle_guard.scheduler().clone();
        RUNTIME_SCHEDULER.with(|s| {
            *s.borrow_mut() = Some(scheduler);
        });
    }

    let mut idle_count = 0;

    loop {
        // Check for shutdown
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        // Try to get a task
        let task = {
            let handle = handle.lock();
            handle.pop()
        };

        if let Some(task) = task {
            task.poll();

            // Mark task as completed if it finished
            let handle = handle.lock();
            handle.task_completed();

            idle_count = 0;
        } else {
            // No tasks, poll reactor or sleep
            idle_count += 1;

            if idle_count > 61 {
                // Sleep briefly to reduce CPU usage
                let _ = reactor.poll(Some(Duration::from_millis(1)));
                idle_count = 0;
            } else {
                // Yield to other threads
                std::hint::spin_loop();
            }
        }
    }

    // Clear the scheduler on shutdown
    RUNTIME_SCHEDULER.with(|s| {
        *s.borrow_mut() = None;
    });
}

/// Get number of CPUs
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_runtime_spawn() {
        let rt = Runtime::new().unwrap();

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let handle = rt.spawn(async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            42
        });

        let result = rt.block_on(handle);
        assert_eq!(result.unwrap(), 42);

        rt.shutdown();
    }

    #[test]
    fn test_builder_worker_threads() {
        let rt = Builder::new().worker_threads(2).build().unwrap();

        rt.shutdown();
    }
}
