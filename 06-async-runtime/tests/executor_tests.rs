//! Test suite for the executor module

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_runtime::executor::{Executor, ExecutorConfig};
use async_runtime::future::Future;
use async_runtime::task::{JoinHandle, TaskId};

#[test]
fn test_basic_task_execution() {
    let executor = Executor::new(ExecutorConfig::default());
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let handle = executor.spawn(async move {
        counter_clone.fetch_add(1, Ordering::SeqCst);
        42
    });

    executor.run_until_complete(handle);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[test]
fn test_multiple_tasks() {
    let executor = Executor::new(ExecutorConfig::default());
    let counter = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    for i in 0..10 {
        let counter_clone = counter.clone();
        let handle = executor.spawn(async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            i
        });
        handles.push(handle);
    }

    for handle in handles {
        executor.run_until_complete(handle);
    }

    assert_eq!(counter.load(Ordering::SeqCst), 10);
}

#[test]
fn test_task_spawning_from_task() {
    let executor = Executor::new(ExecutorConfig::default());
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let handle = executor.spawn(async move {
        let counter_inner = counter_clone.clone();

        let inner_handle = async_runtime::spawn(async move {
            counter_inner.fetch_add(1, Ordering::SeqCst);
            10
        });

        let result = inner_handle.await;
        counter_clone.fetch_add(result, Ordering::SeqCst);
        result
    });

    let result = executor.run_until_complete(handle);
    assert_eq!(result, 10);
    assert_eq!(counter.load(Ordering::SeqCst), 11);
}

#[test]
fn test_concurrent_tasks() {
    let executor = Executor::new(ExecutorConfig::default());
    let start_time = Instant::now();

    let handle1 = executor.spawn(async {
        async_runtime::time::sleep(Duration::from_millis(100)).await;
        1
    });

    let handle2 = executor.spawn(async {
        async_runtime::time::sleep(Duration::from_millis(100)).await;
        2
    });

    let handle3 = executor.spawn(async {
        async_runtime::time::sleep(Duration::from_millis(100)).await;
        3
    });

    let results = executor.run_until_complete(async {
        let r1 = handle1.await;
        let r2 = handle2.await;
        let r3 = handle3.await;
        (r1, r2, r3)
    });

    let elapsed = start_time.elapsed();

    assert_eq!(results, (1, 2, 3));
    // Tasks should run concurrently, not taking 300ms
    assert!(elapsed < Duration::from_millis(200));
}

#[test]
fn test_task_cancellation() {
    let executor = Executor::new(ExecutorConfig::default());
    let started = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicUsize::new(0));

    let started_clone = started.clone();
    let completed_clone = completed.clone();

    let handle = executor.spawn(async move {
        started_clone.fetch_add(1, Ordering::SeqCst);
        async_runtime::time::sleep(Duration::from_secs(10)).await;
        completed_clone.fetch_add(1, Ordering::SeqCst);
        42
    });

    // Give the task time to start
    std::thread::sleep(Duration::from_millis(10));

    // Cancel the task
    handle.cancel();

    // Try to wait for it (should return immediately with None)
    let result = executor.try_run_until_complete(handle);

    assert_eq!(started.load(Ordering::SeqCst), 1);
    assert_eq!(completed.load(Ordering::SeqCst), 0);
    assert!(result.is_none());
}

#[test]
fn test_panic_in_task() {
    let executor = Executor::new(ExecutorConfig::default());

    let handle = executor.spawn(async {
        panic!("Task panic!");
    });

    let result = std::panic::catch_unwind(|| {
        executor.run_until_complete(handle)
    });

    assert!(result.is_err());
}

#[test]
fn test_work_stealing() {
    let config = ExecutorConfig {
        num_threads: 4,
        enable_work_stealing: true,
        ..Default::default()
    };

    let executor = Executor::new(config);
    let counter = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Spawn many tasks to trigger work stealing
    for _ in 0..100 {
        let counter_clone = counter.clone();
        let handle = executor.spawn(async move {
            // Simulate some work
            let mut sum = 0u64;
            for i in 0..10000 {
                sum += i;
            }
            counter_clone.fetch_add(1, Ordering::SeqCst);
            sum
        });
        handles.push(handle);
    }

    for handle in handles {
        executor.run_until_complete(handle);
    }

    assert_eq!(counter.load(Ordering::SeqCst), 100);
}

#[test]
fn test_task_priority() {
    let executor = Executor::new(ExecutorConfig::default());
    let execution_order = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Spawn low priority task
    let order_clone = execution_order.clone();
    let low_priority = executor.spawn_with_priority(async move {
        order_clone.lock().unwrap().push("low");
    }, 0);

    // Spawn high priority task
    let order_clone = execution_order.clone();
    let high_priority = executor.spawn_with_priority(async move {
        order_clone.lock().unwrap().push("high");
    }, 10);

    executor.run_until_complete(async {
        high_priority.await;
        low_priority.await;
    });

    let order = execution_order.lock().unwrap();
    assert_eq!(order[0], "high");
    assert_eq!(order[1], "low");
}

#[test]
fn test_local_executor() {
    use std::rc::Rc;
    use std::cell::RefCell;

    let executor = Executor::new_local();

    // Test that we can use non-Send types
    let rc = Rc::new(RefCell::new(0));
    let rc_clone = rc.clone();

    let handle = executor.spawn_local(async move {
        *rc_clone.borrow_mut() += 1;
    });

    executor.run_until_complete(handle);
    assert_eq!(*rc.borrow(), 1);
}

#[test]
fn test_executor_shutdown() {
    let executor = Executor::new(ExecutorConfig::default());
    let counter = Arc::new(AtomicUsize::new(0));

    // Spawn some tasks
    for _ in 0..10 {
        let counter_clone = counter.clone();
        executor.spawn(async move {
            async_runtime::time::sleep(Duration::from_millis(10)).await;
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
    }

    // Shutdown the executor gracefully
    executor.shutdown_graceful(Duration::from_secs(1));

    // All tasks should complete
    assert_eq!(counter.load(Ordering::SeqCst), 10);
}

#[test]
fn test_executor_metrics() {
    let executor = Executor::new(ExecutorConfig::default());

    // Spawn some tasks
    let mut handles = Vec::new();
    for _ in 0..10 {
        let handle = executor.spawn(async {
            async_runtime::time::sleep(Duration::from_millis(10)).await;
        });
        handles.push(handle);
    }

    let metrics = executor.metrics();
    assert_eq!(metrics.total_spawned, 10);
    assert_eq!(metrics.active_tasks, 10);

    for handle in handles {
        executor.run_until_complete(handle);
    }

    let metrics = executor.metrics();
    assert_eq!(metrics.completed_tasks, 10);
    assert_eq!(metrics.active_tasks, 0);
}

#[test]
fn test_task_locals() {
    let executor = Executor::new(ExecutorConfig::default());

    let handle = executor.spawn(async {
        async_runtime::task::set_local("key", "value");

        let value: Option<&str> = async_runtime::task::get_local("key");
        assert_eq!(value, Some("value"));

        // Spawn inner task
        let inner = async_runtime::spawn(async {
            // Task locals should not be inherited
            let value: Option<&str> = async_runtime::task::get_local("key");
            assert_eq!(value, None);
        });

        inner.await;
    });

    executor.run_until_complete(handle);
}

#[test]
fn test_yield_now() {
    let executor = Executor::new(ExecutorConfig::default());
    let counter = Arc::new(AtomicUsize::new(0));

    let counter1 = counter.clone();
    let handle1 = executor.spawn(async move {
        for i in 0..5 {
            counter1.store(i, Ordering::SeqCst);
            async_runtime::task::yield_now().await;
        }
    });

    let counter2 = counter.clone();
    let handle2 = executor.spawn(async move {
        let mut last_seen = 0;
        for _ in 0..5 {
            let current = counter2.load(Ordering::SeqCst);
            // Should see progress from the other task
            assert!(current >= last_seen);
            last_seen = current;
            async_runtime::task::yield_now().await;
        }
    });

    executor.run_until_complete(async {
        handle1.await;
        handle2.await;
    });
}

#[test]
fn test_block_on_nested() {
    let executor = Executor::new(ExecutorConfig::default());

    let result = executor.block_on(async {
        let inner_result = async_runtime::block_on(async {
            async_runtime::time::sleep(Duration::from_millis(10)).await;
            42
        });

        inner_result + 1
    });

    assert_eq!(result, 43);
}

#[test]
#[should_panic(expected = "Cannot block_on from async context")]
fn test_block_on_from_async_panics() {
    let executor = Executor::new(ExecutorConfig::default());

    executor.spawn(async {
        // This should panic
        async_runtime::block_on(async { 42 })
    });

    executor.run();
}