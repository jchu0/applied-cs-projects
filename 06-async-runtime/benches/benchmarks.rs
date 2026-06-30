//! Benchmarks for async runtime

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::{Duration, Instant};

use async_runtime::sync::{mpsc, oneshot};
use async_runtime::timer::TimerWheel;
use async_runtime::task::noop_waker;
use async_runtime::{block_on, spawn};

fn spawn_benchmark(c: &mut Criterion) {
    c.bench_function("spawn_noop", |b| {
        b.iter(|| {
            block_on(async {
                spawn(async {});
            });
        });
    });
}

fn timer_benchmark(c: &mut Criterion) {
    c.bench_function("timer_insert", |b| {
        let mut wheel = TimerWheel::new();
        let waker = noop_waker();
        let start = Instant::now();

        b.iter(|| {
            let deadline = start + Duration::from_millis(black_box(10));
            wheel.insert(deadline, waker.clone());
        });
    });

    c.bench_function("timer_cancel", |b| {
        let mut wheel = TimerWheel::new();
        let waker = noop_waker();
        let start = Instant::now();

        b.iter(|| {
            let deadline = start + Duration::from_millis(black_box(10));
            let handle = wheel.insert(deadline, waker.clone());
            wheel.cancel(handle);
        });
    });
}

fn channel_benchmark(c: &mut Criterion) {
    c.bench_function("oneshot_create", |b| {
        b.iter(|| {
            let (_tx, _rx) = oneshot::channel::<i32>();
        });
    });

    c.bench_function("oneshot_send_recv", |b| {
        b.iter(|| {
            let (tx, mut rx) = oneshot::channel();
            tx.send(black_box(42)).unwrap();
            rx.try_recv().unwrap()
        });
    });

    c.bench_function("mpsc_send_recv", |b| {
        b.iter(|| {
            let (tx, mut rx) = mpsc::channel(16);
            tx.try_send(black_box(42)).unwrap();
            rx.try_recv().unwrap()
        });
    });
}

fn executor_benchmark(c: &mut Criterion) {
    c.bench_function("block_on_ready", |b| {
        b.iter(|| {
            block_on(async { black_box(42) })
        });
    });
}

criterion_group!(
    benches,
    spawn_benchmark,
    timer_benchmark,
    channel_benchmark,
    executor_benchmark
);
criterion_main!(benches);
