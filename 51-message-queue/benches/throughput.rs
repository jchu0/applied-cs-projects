//! Throughput benchmarks for the message queue.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use message_queue::message::Message;
use message_queue::partition::Partition;
use std::time::Duration;
use tempfile::TempDir;

fn benchmark_produce(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("produce");
    group.throughput(Throughput::Elements(1000));
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("single_partition", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                rt.block_on(async {
                    let temp_dir = TempDir::new().unwrap();
                    let partition = Partition::new(
                        "test-topic",
                        0,
                        temp_dir.path(),
                    ).await.unwrap();

                    for i in 0..1000 {
                        let msg = Message::new(
                            format!("key-{}", i).into_bytes(),
                            format!("value-{}", i).into_bytes(),
                        );
                        partition.append(msg).await.unwrap();
                    }
                });
            }
            start.elapsed()
        });
    });

    group.finish();
}

fn benchmark_consume(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("consume");
    group.throughput(Throughput::Elements(1000));

    group.bench_function("sequential_read", |b| {
        b.iter_custom(|iters| {
            // Setup: create partition with messages
            let (partition, _temp_dir) = rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let partition = Partition::new(
                    "test-topic",
                    0,
                    temp_dir.path(),
                ).await.unwrap();

                for i in 0..1000 {
                    let msg = Message::new(
                        format!("key-{}", i).into_bytes(),
                        format!("value-{}", i).into_bytes(),
                    );
                    partition.append(msg).await.unwrap();
                }

                (partition, temp_dir)
            });

            let start = std::time::Instant::now();
            for _ in 0..iters {
                rt.block_on(async {
                    let _ = partition.read(0, 1000).await.unwrap();
                });
            }
            start.elapsed()
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_produce, benchmark_consume);
criterion_main!(benches);
