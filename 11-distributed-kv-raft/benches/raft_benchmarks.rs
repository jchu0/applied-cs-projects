//! Performance benchmarks for the Distributed KV Store with Raft.
//!
//! Run benchmarks with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use distributed_kv_raft::{
    Command, EntryType, KeyValueFSM, LogEntry, RaftConfig, RaftNode, RequestVoteRequest,
    RequestVoteResponse, AppendEntriesRequest, AppendEntriesResponse,
};
use std::time::Duration;

/// Benchmark log entry serialization/deserialization.
fn bench_log_entry_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_entry_serialization");

    // Small entry (typical key-value)
    let small_entry = LogEntry {
        term: 1,
        index: 100,
        command: Command::Put {
            key: b"key".to_vec(),
            value: b"value".to_vec(),
        },
        entry_type: EntryType::Command,
    };

    // Large entry (1KB value)
    let large_entry = LogEntry {
        term: 1,
        index: 100,
        command: Command::Put {
            key: b"key".to_vec(),
            value: vec![0u8; 1024],
        },
        entry_type: EntryType::Command,
    };

    // Very large entry (1MB value)
    let xlarge_entry = LogEntry {
        term: 1,
        index: 100,
        command: Command::Put {
            key: b"key".to_vec(),
            value: vec![0u8; 1024 * 1024],
        },
        entry_type: EntryType::Command,
    };

    group.throughput(Throughput::Elements(1));

    group.bench_function("serialize_small", |b| {
        b.iter(|| bincode::serialize(black_box(&small_entry)).unwrap())
    });

    group.bench_function("serialize_large_1kb", |b| {
        b.iter(|| bincode::serialize(black_box(&large_entry)).unwrap())
    });

    group.bench_function("serialize_xlarge_1mb", |b| {
        b.iter(|| bincode::serialize(black_box(&xlarge_entry)).unwrap())
    });

    let small_bytes = bincode::serialize(&small_entry).unwrap();
    let large_bytes = bincode::serialize(&large_entry).unwrap();

    group.bench_function("deserialize_small", |b| {
        b.iter(|| bincode::deserialize::<LogEntry>(black_box(&small_bytes)).unwrap())
    });

    group.bench_function("deserialize_large_1kb", |b| {
        b.iter(|| bincode::deserialize::<LogEntry>(black_box(&large_bytes)).unwrap())
    });

    group.finish();
}

/// Benchmark FSM operations.
fn bench_fsm_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("fsm_operations");

    group.bench_function("put_small", |b| {
        let mut fsm = KeyValueFSM::new();
        let key = b"key".to_vec();
        let value = b"value".to_vec();
        b.iter(|| {
            fsm.put(black_box(key.clone()), black_box(value.clone()));
        })
    });

    group.bench_function("put_large_1kb", |b| {
        let mut fsm = KeyValueFSM::new();
        let key = b"key".to_vec();
        let value = vec![0u8; 1024];
        b.iter(|| {
            fsm.put(black_box(key.clone()), black_box(value.clone()));
        })
    });

    group.bench_function("get_existing", |b| {
        let mut fsm = KeyValueFSM::new();
        fsm.put(b"key".to_vec(), b"value".to_vec());
        let key = b"key".to_vec();
        b.iter(|| fsm.get(black_box(&key)))
    });

    group.bench_function("get_missing", |b| {
        let fsm = KeyValueFSM::new();
        let key = b"nonexistent".to_vec();
        b.iter(|| fsm.get(black_box(&key)))
    });

    group.bench_function("delete", |b| {
        let mut fsm = KeyValueFSM::new();
        let key = b"key".to_vec();
        // Pre-populate
        for i in 0..1000 {
            fsm.put(format!("key{}", i).into_bytes(), b"value".to_vec());
        }
        b.iter(|| {
            fsm.delete(black_box(&key));
        })
    });

    group.finish();
}

/// Benchmark FSM throughput with many operations.
fn bench_fsm_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("fsm_throughput");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("sequential_puts", size), size, |b, &size| {
            b.iter(|| {
                let mut fsm = KeyValueFSM::new();
                for i in 0..size {
                    fsm.put(
                        format!("key{}", i).into_bytes(),
                        format!("value{}", i).into_bytes(),
                    );
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("sequential_gets", size), size, |b, &size| {
            let mut fsm = KeyValueFSM::new();
            for i in 0..size {
                fsm.put(
                    format!("key{}", i).into_bytes(),
                    format!("value{}", i).into_bytes(),
                );
            }
            b.iter(|| {
                for i in 0..size {
                    black_box(fsm.get(&format!("key{}", i).into_bytes()));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark snapshot creation and restoration.
fn bench_snapshots(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshots");

    for size in [100, 1000, 10000].iter() {
        // Prepare FSM with data
        let mut fsm = KeyValueFSM::new();
        for i in 0..*size {
            fsm.put(
                format!("key{}", i).into_bytes(),
                format!("value{}", i).into_bytes(),
            );
        }

        group.bench_with_input(
            BenchmarkId::new("create_snapshot", size),
            size,
            |b, _| {
                b.iter(|| black_box(fsm.create_snapshot()))
            },
        );

        let snapshot = fsm.create_snapshot();
        group.bench_with_input(
            BenchmarkId::new("restore_snapshot", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut new_fsm = KeyValueFSM::new();
                    new_fsm.restore_snapshot(black_box(&snapshot));
                })
            },
        );
    }

    group.finish();
}

/// Benchmark RPC message creation and serialization.
fn bench_rpc_messages(c: &mut Criterion) {
    let mut group = c.benchmark_group("rpc_messages");

    // RequestVote
    let request_vote = RequestVoteRequest {
        term: 5,
        candidate_id: 1,
        last_log_index: 100,
        last_log_term: 4,
    };

    group.bench_function("serialize_request_vote", |b| {
        b.iter(|| bincode::serialize(black_box(&request_vote)).unwrap())
    });

    // AppendEntries with entries
    let entries: Vec<LogEntry> = (0..10)
        .map(|i| LogEntry {
            term: 5,
            index: 100 + i,
            command: Command::Put {
                key: format!("key{}", i).into_bytes(),
                value: format!("value{}", i).into_bytes(),
            },
            entry_type: EntryType::Command,
        })
        .collect();

    let append_entries = AppendEntriesRequest {
        term: 5,
        leader_id: 1,
        prev_log_index: 99,
        prev_log_term: 4,
        entries: entries.clone(),
        leader_commit: 95,
    };

    group.bench_function("serialize_append_entries_10", |b| {
        b.iter(|| bincode::serialize(black_box(&append_entries)).unwrap())
    });

    // AppendEntries heartbeat (empty)
    let heartbeat = AppendEntriesRequest {
        term: 5,
        leader_id: 1,
        prev_log_index: 99,
        prev_log_term: 4,
        entries: vec![],
        leader_commit: 95,
    };

    group.bench_function("serialize_heartbeat", |b| {
        b.iter(|| bincode::serialize(black_box(&heartbeat)).unwrap())
    });

    // Large batch
    let large_entries: Vec<LogEntry> = (0..100)
        .map(|i| LogEntry {
            term: 5,
            index: 100 + i,
            command: Command::Put {
                key: format!("key{}", i).into_bytes(),
                value: vec![0u8; 100], // 100 byte values
            },
            entry_type: EntryType::Command,
        })
        .collect();

    let large_append = AppendEntriesRequest {
        term: 5,
        leader_id: 1,
        prev_log_index: 99,
        prev_log_term: 4,
        entries: large_entries,
        leader_commit: 95,
    };

    group.bench_function("serialize_append_entries_100", |b| {
        b.iter(|| bincode::serialize(black_box(&large_append)).unwrap())
    });

    group.finish();
}

/// Benchmark Raft node creation and basic operations.
fn bench_raft_node(c: &mut Criterion) {
    let mut group = c.benchmark_group("raft_node");

    group.bench_function("create_node", |b| {
        b.iter(|| {
            let config = RaftConfig::builder().node_id(1).build();
            black_box(RaftNode::new(config))
        })
    });

    group.bench_function("propose_single", |b| {
        let config = RaftConfig::builder().node_id(1).build();
        let mut node = RaftNode::new(config);
        // Make the node a leader so proposals succeed
        node.become_leader();

        b.iter(|| {
            let cmd = Command::Put {
                key: b"key".to_vec(),
                value: b"value".to_vec(),
            };
            black_box(node.propose(cmd))
        })
    });

    group.finish();
}

/// Benchmark log operations.
fn bench_log_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_operations");

    group.bench_function("append_entry", |b| {
        let config = RaftConfig::builder().node_id(1).build();
        let mut node = RaftNode::new(config);
        node.become_leader();

        let mut index = 0u64;
        b.iter(|| {
            index += 1;
            let entry = LogEntry {
                term: 1,
                index,
                command: Command::Put {
                    key: b"key".to_vec(),
                    value: b"value".to_vec(),
                },
                entry_type: EntryType::Command,
            };
            node.log.push(entry);
        })
    });

    // Batch append
    for batch_size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_append", batch_size),
            batch_size,
            |b, &batch_size| {
                let config = RaftConfig::builder().node_id(1).build();
                let mut node = RaftNode::new(config);
                node.become_leader();

                let entries: Vec<LogEntry> = (0..batch_size)
                    .map(|i| LogEntry {
                        term: 1,
                        index: i as u64,
                        command: Command::Put {
                            key: format!("key{}", i).into_bytes(),
                            value: format!("value{}", i).into_bytes(),
                        },
                        entry_type: EntryType::Command,
                    })
                    .collect();

                b.iter(|| {
                    node.log.clear();
                    node.log.extend(black_box(entries.clone()));
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_log_entry_serialization,
    bench_fsm_operations,
    bench_fsm_throughput,
    bench_snapshots,
    bench_rpc_messages,
    bench_raft_node,
    bench_log_operations,
);
criterion_main!(benches);
