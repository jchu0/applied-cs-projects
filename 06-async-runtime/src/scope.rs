//! Structured concurrency primitives
//!
//! Provides scoped task spawning where all spawned tasks must complete
//! before the scope exits.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

use crate::sync::CancellationToken;

/// A scope for structured concurrency
///
/// All tasks spawned within the scope must complete before the scope
/// returns. This guarantees that references to outer data remain valid
/// for the duration of the spawned tasks.
pub struct Scope<'env, T: Send + 'static = ()> {
    /// Number of active tasks
    active_tasks: Arc<AtomicUsize>,
    /// Collected results from tasks
    results: Arc<Mutex<Vec<T>>>,
    /// Waker for when all tasks complete
    completion_waker: Arc<Mutex<Option<Waker>>>,
    /// Cancellation token for the scope
    cancel_token: CancellationToken,
    /// Marker for lifetime
    _marker: PhantomData<&'env ()>,
}

impl<'env, T: Send + 'static> Scope<'env, T> {
    /// Create a new scope
    fn new(cancel_token: CancellationToken) -> Self {
        Self {
            active_tasks: Arc::new(AtomicUsize::new(0)),
            results: Arc::new(Mutex::new(Vec::new())),
            completion_waker: Arc::new(Mutex::new(None)),
            cancel_token,
            _marker: PhantomData,
        }
    }

    /// Spawn a task within this scope
    ///
    /// The task will be cancelled if the scope is cancelled, and its
    /// result will be collected.
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = T> + Send + 'env,
    {
        self.active_tasks.fetch_add(1, Ordering::SeqCst);

        let active_tasks = self.active_tasks.clone();
        let results = self.results.clone();
        let completion_waker = self.completion_waker.clone();
        let cancel_token = self.cancel_token.child_token();

        // Transmute lifetime - safe because we join all before returning
        let future: Pin<Box<dyn Future<Output = T> + Send>> =
            unsafe { std::mem::transmute(Box::pin(future) as Pin<Box<dyn Future<Output = T> + Send + 'env>>) };

        // Spawn as a detached task
        let task = crate::task::Task::new(async move {
            // Run with cancellation support
            if let Some(result) = cancel_token.run_until_cancelled(future).await {
                results.lock().push(result);
            }

            // Decrement active count
            let prev = active_tasks.fetch_sub(1, Ordering::SeqCst);
            if prev == 1 {
                // Last task, wake the scope
                if let Some(waker) = completion_waker.lock().take() {
                    waker.wake();
                }
            }
        });

        // Push to the current scheduler
        crate::runtime::RUNTIME_SCHEDULER.with(|s| {
            if let Some(scheduler) = s.borrow().as_ref() {
                scheduler.push(task);
            } else {
                // Fallback to single-threaded executor
                crate::executor::EXECUTOR.with(|e| {
                    e.borrow().as_ref().unwrap().push(task);
                });
            }
        });
    }

    /// Spawn a task that doesn't return a value
    pub fn spawn_detached<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'env,
    {
        self.active_tasks.fetch_add(1, Ordering::SeqCst);

        let active_tasks = self.active_tasks.clone();
        let completion_waker = self.completion_waker.clone();
        let cancel_token = self.cancel_token.child_token();

        // Transmute lifetime
        let future: Pin<Box<dyn Future<Output = ()> + Send>> =
            unsafe { std::mem::transmute(Box::pin(future) as Pin<Box<dyn Future<Output = ()> + Send + 'env>>) };

        let task = crate::task::Task::new(async move {
            let _ = cancel_token.run_until_cancelled(future).await;

            let prev = active_tasks.fetch_sub(1, Ordering::SeqCst);
            if prev == 1 {
                if let Some(waker) = completion_waker.lock().take() {
                    waker.wake();
                }
            }
        });

        crate::runtime::RUNTIME_SCHEDULER.with(|s| {
            if let Some(scheduler) = s.borrow().as_ref() {
                scheduler.push(task);
            } else {
                crate::executor::EXECUTOR.with(|e| {
                    e.borrow().as_ref().unwrap().push(task);
                });
            }
        });
    }

    /// Cancel all tasks in the scope
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// Check if the scope has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Get number of active tasks
    pub fn active_count(&self) -> usize {
        self.active_tasks.load(Ordering::Relaxed)
    }
}

/// Run a scoped computation
///
/// All tasks spawned within the scope must complete before this function
/// returns. This provides structured concurrency guarantees.
pub async fn scope<'env, T, F, R>(f: F) -> Vec<R>
where
    T: Send + 'static,
    R: Send + 'static,
    F: FnOnce(&Scope<'env, R>) -> T,
{
    let cancel_token = CancellationToken::new();
    let scope = Scope::new(cancel_token);

    // Run the user's function
    let _result = f(&scope);

    // Wait for all tasks to complete
    ScopeCompletion { scope: &scope }.await;

    // Return collected results
    std::mem::take(&mut *scope.results.lock())
}

/// Run a scoped computation without collecting results
pub async fn scope_detached<'env, T, F>(f: F)
where
    F: FnOnce(&Scope<'env, ()>) -> T,
{
    let cancel_token = CancellationToken::new();
    let scope: Scope<'env, ()> = Scope::new(cancel_token);

    // Run the user's function
    let _result = f(&scope);

    // Wait for all tasks to complete
    ScopeCompletion { scope: &scope }.await;
}

/// Future for waiting for scope completion
struct ScopeCompletion<'a, 'env, T: Send + 'static> {
    scope: &'a Scope<'env, T>,
}

impl<T: Send + 'static> Future for ScopeCompletion<'_, '_, T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.scope.active_tasks.load(Ordering::SeqCst) == 0 {
            Poll::Ready(())
        } else {
            *self.scope.completion_waker.lock() = Some(cx.waker().clone());

            // Double-check after registering
            if self.scope.active_tasks.load(Ordering::SeqCst) == 0 {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }
    }
}

/// A task set for collecting multiple spawned tasks
pub struct TaskSet<T: Send + 'static> {
    /// Active task count
    active: Arc<AtomicUsize>,
    /// Results
    results: Arc<Mutex<Vec<T>>>,
    /// Completion waker
    waker: Arc<Mutex<Option<Waker>>>,
}

impl<T: Send + 'static> TaskSet<T> {
    /// Create a new task set
    pub fn new() -> Self {
        Self {
            active: Arc::new(AtomicUsize::new(0)),
            results: Arc::new(Mutex::new(Vec::new())),
            waker: Arc::new(Mutex::new(None)),
        }
    }

    /// Spawn a task in the set
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = T> + Send + 'static,
    {
        self.active.fetch_add(1, Ordering::SeqCst);

        let active = self.active.clone();
        let results = self.results.clone();
        let waker = self.waker.clone();

        let task = crate::task::Task::new(async move {
            let result = future.await;
            results.lock().push(result);

            let prev = active.fetch_sub(1, Ordering::SeqCst);
            if prev == 1 {
                if let Some(w) = waker.lock().take() {
                    w.wake();
                }
            }
        });

        crate::runtime::RUNTIME_SCHEDULER.with(|s| {
            if let Some(scheduler) = s.borrow().as_ref() {
                scheduler.push(task);
            } else {
                crate::executor::EXECUTOR.with(|e| {
                    e.borrow().as_ref().unwrap().push(task);
                });
            }
        });
    }

    /// Wait for all tasks to complete and return results
    pub async fn join_all(self) -> Vec<T> {
        TaskSetJoin {
            active: self.active,
            waker: self.waker,
            results: self.results,
        }
        .await
    }

    /// Get number of active tasks
    pub fn len(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    /// Check if the task set is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Send + 'static> Default for TaskSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Future for joining a task set
struct TaskSetJoin<T> {
    active: Arc<AtomicUsize>,
    waker: Arc<Mutex<Option<Waker>>>,
    results: Arc<Mutex<Vec<T>>>,
}

impl<T> Future for TaskSetJoin<T> {
    type Output = Vec<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.active.load(Ordering::SeqCst) == 0 {
            Poll::Ready(std::mem::take(&mut *self.results.lock()))
        } else {
            *self.waker.lock() = Some(cx.waker().clone());

            if self.active.load(Ordering::SeqCst) == 0 {
                Poll::Ready(std::mem::take(&mut *self.results.lock()))
            } else {
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_set_new() {
        let set: TaskSet<i32> = TaskSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_scope_creation() {
        let token = CancellationToken::new();
        let scope: Scope<'_, i32> = Scope::new(token);
        assert_eq!(scope.active_count(), 0);
        assert!(!scope.is_cancelled());
    }

    #[test]
    fn test_scope_cancel() {
        let token = CancellationToken::new();
        let scope: Scope<'_, ()> = Scope::new(token);
        scope.cancel();
        assert!(scope.is_cancelled());
    }
}
