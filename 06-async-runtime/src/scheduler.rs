//! Work-stealing task scheduler
//!
//! Implements a work-stealing scheduler for efficient multi-threaded task execution.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crossbeam_deque::{Injector, Steal, Stealer, Worker};

use crate::task::Task;

/// Work-stealing scheduler
pub struct Scheduler {
    /// Global task queue
    injector: Injector<Arc<Task>>,
    /// Per-worker stealers
    stealers: Vec<Stealer<Arc<Task>>>,
    /// Number of workers
    num_workers: usize,
    /// Shutdown flag
    shutdown: AtomicBool,
    /// Number of active tasks
    active_tasks: AtomicUsize,
}

impl Scheduler {
    /// Create a new scheduler with the given number of workers
    pub fn new(num_workers: usize) -> (Arc<Self>, Vec<WorkerHandle>) {
        let injector = Injector::new();
        let mut workers = Vec::with_capacity(num_workers);
        let mut stealers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            workers.push(worker);
        }

        let scheduler = Arc::new(Self {
            injector,
            stealers,
            num_workers,
            shutdown: AtomicBool::new(false),
            active_tasks: AtomicUsize::new(0),
        });

        let handles: Vec<_> = workers
            .into_iter()
            .enumerate()
            .map(|(id, worker)| WorkerHandle {
                id,
                worker,
                scheduler: scheduler.clone(),
            })
            .collect();

        (scheduler, handles)
    }

    /// Push a task to the global queue
    pub fn push(&self, task: Arc<Task>) {
        self.active_tasks.fetch_add(1, Ordering::SeqCst);
        self.injector.push(task);
    }

    /// Signal shutdown
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Check if shutdown is requested
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Get number of active tasks
    pub fn active_tasks(&self) -> usize {
        self.active_tasks.load(Ordering::Relaxed)
    }

    /// Decrement active task count
    pub fn task_completed(&self) {
        self.active_tasks.fetch_sub(1, Ordering::SeqCst);
    }

    /// Number of workers
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

/// Handle for a worker thread
pub struct WorkerHandle {
    /// Worker ID
    id: usize,
    /// Local work queue
    worker: Worker<Arc<Task>>,
    /// Reference to scheduler
    scheduler: Arc<Scheduler>,
}

impl WorkerHandle {
    /// Get worker ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Push a task to the local queue
    pub fn push(&self, task: Arc<Task>) {
        self.scheduler.active_tasks.fetch_add(1, Ordering::SeqCst);
        self.worker.push(task);
    }

    /// Pop a task from the local queue, or steal from others
    pub fn pop(&self) -> Option<Arc<Task>> {
        // Try local queue first
        if let Some(task) = self.worker.pop() {
            return Some(task);
        }

        // Try global queue
        loop {
            match self.scheduler.injector.steal_batch_and_pop(&self.worker) {
                Steal::Success(task) => return Some(task),
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }

        // Try stealing from other workers
        let start = self.id;
        for i in 0..self.scheduler.num_workers {
            let idx = (start + i + 1) % self.scheduler.num_workers;
            if idx == self.id {
                continue;
            }

            loop {
                match self.scheduler.stealers[idx].steal() {
                    Steal::Success(task) => return Some(task),
                    Steal::Empty => break,
                    Steal::Retry => continue,
                }
            }
        }

        None
    }

    /// Check if shutdown is requested
    pub fn is_shutdown(&self) -> bool {
        self.scheduler.is_shutdown()
    }

    /// Mark a task as completed
    pub fn task_completed(&self) {
        self.scheduler.task_completed();
    }

    /// Get reference to scheduler
    pub fn scheduler(&self) -> &Arc<Scheduler> {
        &self.scheduler
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::Task;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_scheduler_create() {
        let (scheduler, handles) = Scheduler::new(4);
        assert_eq!(handles.len(), 4);
        assert_eq!(scheduler.num_workers(), 4);
    }

    #[test]
    fn test_push_and_pop() {
        let (scheduler, handles) = Scheduler::new(2);

        // Push task to global queue
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let task = Task::new(async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        scheduler.push(task);

        // Pop from worker
        let task = handles[0].pop();
        assert!(task.is_some());
        assert_eq!(scheduler.active_tasks(), 1);
    }

    #[test]
    fn test_work_stealing() {
        let (scheduler, handles) = Scheduler::new(2);

        // Push task to worker 0's local queue
        let task = Task::new(async {});
        handles[0].push(task);

        // Worker 1 should be able to steal it
        let stolen = handles[1].pop();
        assert!(stolen.is_some());
    }

    #[test]
    fn test_shutdown() {
        let (scheduler, handles) = Scheduler::new(2);

        assert!(!scheduler.is_shutdown());
        scheduler.shutdown();
        assert!(scheduler.is_shutdown());
        assert!(handles[0].is_shutdown());
    }
}
