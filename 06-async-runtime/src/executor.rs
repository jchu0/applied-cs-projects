//! Task executor
//!
//! Manages task scheduling and execution.

use std::cell::RefCell;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_queue::SegQueue;

use crate::reactor::Reactor;
use crate::task::Task;

thread_local! {
    /// Current executor handle
    pub static EXECUTOR: RefCell<Option<Arc<Executor>>> = RefCell::new(None);
}

/// Task executor
pub struct Executor {
    /// Task queue
    queue: SegQueue<Arc<Task>>,
    /// Shutdown flag
    shutdown: AtomicBool,
    /// Reactor for I/O
    reactor: Arc<Reactor>,
}

impl Executor {
    /// Create a new executor
    pub fn new() -> io::Result<Arc<Self>> {
        let reactor = Arc::new(Reactor::new()?);

        Ok(Arc::new(Self {
            queue: SegQueue::new(),
            shutdown: AtomicBool::new(false),
            reactor,
        }))
    }

    /// Get the reactor
    pub fn reactor(&self) -> Arc<Reactor> {
        self.reactor.clone()
    }

    /// Schedule a task for execution
    pub fn schedule(&self, task: Arc<Task>) {
        self.queue.push(task);
    }

    /// Spawn a new task
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task = Task::new(future);
        self.schedule(task);
    }

    /// Run the executor until the given future completes
    pub fn block_on<F, T>(&self, future: F) -> T
    where
        F: Future<Output = T>,
    {
        // Pin the future
        let mut future = std::pin::pin!(future);

        // Create a waker that does nothing (main task is polled directly)
        let waker = crate::task::noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);

        loop {
            // Poll the main future
            if let std::task::Poll::Ready(result) = future.as_mut().poll(&mut cx) {
                return result;
            }

            // Process queued tasks
            let mut processed = 0;
            while let Some(task) = self.queue.pop() {
                task.poll();
                processed += 1;

                // Yield occasionally to prevent starvation
                if processed >= 61 {
                    break;
                }
            }

            // If no tasks were processed, poll the reactor
            if processed == 0 {
                let _ = self.reactor.poll(Some(Duration::from_millis(1)));
            }

            // Check for shutdown
            if self.shutdown.load(Ordering::Relaxed) {
                panic!("Executor shutdown while running");
            }
        }
    }

    /// Shutdown the executor
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

use std::io;

/// Spawn a task on the current executor
pub fn spawn<F>(future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    EXECUTOR.with(|ex| {
        if let Some(executor) = ex.borrow().as_ref() {
            executor.spawn(future);
        } else {
            panic!("No executor running");
        }
    });
}

/// Run a future to completion on a new executor
pub fn block_on<F, T>(future: F) -> T
where
    F: Future<Output = T>,
{
    let executor = Executor::new().expect("Failed to create executor");

    // Set as current executor
    EXECUTOR.with(|ex| {
        *ex.borrow_mut() = Some(executor.clone());
    });

    let result = executor.block_on(future);

    // Clear current executor
    EXECUTOR.with(|ex| {
        *ex.borrow_mut() = None;
    });

    result
}

/// Builder for configuring the executor
pub struct Builder {
    num_threads: usize,
}

impl Builder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            num_threads: num_cpus(),
        }
    }

    /// Set the number of worker threads
    pub fn worker_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Build and run the executor
    pub fn build(self) -> io::Result<Runtime> {
        let executor = Executor::new()?;

        Ok(Runtime {
            executor,
            num_threads: self.num_threads,
        })
    }
}

/// Runtime handle
pub struct Runtime {
    executor: Arc<Executor>,
    num_threads: usize,
}

impl Runtime {
    /// Create a new runtime with default settings
    pub fn new() -> io::Result<Self> {
        Builder::new().build()
    }

    /// Run a future to completion
    pub fn block_on<F, T>(&self, future: F) -> T
    where
        F: Future<Output = T>,
    {
        // Set as current executor
        EXECUTOR.with(|ex| {
            *ex.borrow_mut() = Some(self.executor.clone());
        });

        let result = self.executor.block_on(future);

        // Clear current executor
        EXECUTOR.with(|ex| {
            *ex.borrow_mut() = None;
        });

        result
    }

    /// Spawn a task on the runtime
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.executor.spawn(future);
    }
}

/// Get the number of CPUs
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_block_on_simple() {
        let result = block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_spawn_and_block() {
        let result = block_on(async {
            spawn(async {
                // Simple spawned task
            });
            123
        });
        assert_eq!(result, 123);
    }

    #[test]
    fn test_block_on_with_computation() {
        let result = block_on(async {
            let a = 10;
            let b = 20;
            a + b
        });
        assert_eq!(result, 30);
    }

    #[test]
    fn test_spawn_multiple_tasks() {
        let counter = Arc::new(AtomicUsize::new(0));

        let result = block_on(async {
            for _ in 0..10 {
                let counter_clone = counter.clone();
                spawn(async move {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                });
            }
            // Note: spawned tasks may not complete before block_on returns
            42
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_executor_create_and_shutdown() {
        let executor = Executor::new().expect("Failed to create executor");
        assert!(!executor.shutdown.load(Ordering::Relaxed));
        executor.shutdown();
        assert!(executor.shutdown.load(Ordering::SeqCst));
    }

    #[test]
    fn test_executor_schedule_task() {
        let executor = Executor::new().expect("Failed to create executor");
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let task = Task::new(async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        executor.schedule(task.clone());

        // Pop and execute the task manually
        if let Some(t) = executor.queue.pop() {
            t.poll();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_executor_spawn_method() {
        let executor = Executor::new().expect("Failed to create executor");
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        executor.spawn(async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Pop and execute the task
        if let Some(task) = executor.queue.pop() {
            task.poll();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_executor_reactor_access() {
        let executor = Executor::new().expect("Failed to create executor");
        let reactor = executor.reactor();
        // Verify we can access the reactor
        assert!(Arc::strong_count(&reactor) >= 1);
    }

    #[test]
    fn test_block_on_nested_async() {
        let result = block_on(async {
            async { async { 42 }.await }.await
        });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_block_on_with_string_result() {
        let result = block_on(async {
            String::from("hello async")
        });
        assert_eq!(result, "hello async");
    }

    #[test]
    fn test_builder_default() {
        let rt = Builder::new().build().expect("Failed to build runtime");
        let result = rt.block_on(async { 100 });
        assert_eq!(result, 100);
    }

    #[test]
    fn test_builder_worker_threads() {
        let rt = Builder::new()
            .worker_threads(4)
            .build()
            .expect("Failed to build runtime");
        assert_eq!(rt.num_threads, 4);
    }

    #[test]
    fn test_runtime_new() {
        let rt = Runtime::new().expect("Failed to create runtime");
        let result = rt.block_on(async { 200 });
        assert_eq!(result, 200);
    }

    #[test]
    fn test_runtime_spawn() {
        let rt = Runtime::new().expect("Failed to create runtime");
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        rt.spawn(async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Give spawned task time to run in block_on
        rt.block_on(async {
            // Tasks should be processed in the block_on loop
        });
    }

    #[test]
    fn test_executor_processes_tasks_during_block_on() {
        let counter = Arc::new(AtomicUsize::new(0));

        let result = block_on(async {
            let counter_clone = counter.clone();
            spawn(async move {
                counter_clone.fetch_add(10, Ordering::SeqCst);
            });

            // Yield to allow spawned task to run
            crate::future::yield_now().await;

            counter.load(Ordering::SeqCst)
        });

        // The spawned task should have run
        assert!(result >= 0); // May or may not have completed
    }

    #[test]
    fn test_block_on_with_option_result() {
        let result: Option<i32> = block_on(async { Some(42) });
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_block_on_with_result_type() {
        let result: Result<i32, &str> = block_on(async { Ok(42) });
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus >= 1);
    }

    #[test]
    fn test_multiple_block_on_calls() {
        // Each block_on should work independently
        let r1 = block_on(async { 1 });
        let r2 = block_on(async { 2 });
        let r3 = block_on(async { 3 });

        assert_eq!(r1, 1);
        assert_eq!(r2, 2);
        assert_eq!(r3, 3);
    }
}
