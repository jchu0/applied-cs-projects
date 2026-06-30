//! Comprehensive storage tests for the Raft implementation.

use distributed_kv_raft::{
    Command, EntryType, KeyValueFSM, LogEntry, MemoryStorage, Snapshot, Storage,
};
use std::collections::HashMap;
use tempfile::TempDir;

// =============================================================================
// MemoryStorage Tests
// =============================================================================

#[test]
fn test_memory_storage_new() {
    let storage = MemoryStorage::new();
    assert!(storage.last_entry().is_none());
}

#[test]
fn test_memory_storage_default() {
    let storage = MemoryStorage::default();
    assert!(storage.last_entry().is_none());
}

#[test]
fn test_memory_storage_append_single() {
    let mut storage = MemoryStorage::new();

    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    };

    storage.append(&[entry]).unwrap();

    assert!(storage.last_entry().is_some());
    assert_eq!(storage.last_entry().unwrap().index, 1);
}

#[test]
fn test_memory_storage_append_multiple() {
    let mut storage = MemoryStorage::new();

    let entries: Vec<LogEntry> = (1..=10)
        .map(|i| LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        })
        .collect();

    storage.append(&entries).unwrap();

    assert_eq!(storage.last_entry().unwrap().index, 10);
}

#[test]
fn test_memory_storage_get_entries_all() {
    let mut storage = MemoryStorage::new();

    let entries: Vec<LogEntry> = (1..=5)
        .map(|i| LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        })
        .collect();

    storage.append(&entries).unwrap();

    let retrieved = storage.get_entries(1, 5).unwrap();
    assert_eq!(retrieved.len(), 5);
}

#[test]
fn test_memory_storage_get_entries_range() {
    let mut storage = MemoryStorage::new();

    let entries: Vec<LogEntry> = (1..=10)
        .map(|i| LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        })
        .collect();

    storage.append(&entries).unwrap();

    let retrieved = storage.get_entries(3, 7).unwrap();
    assert_eq!(retrieved.len(), 5);
    assert_eq!(retrieved[0].index, 3);
    assert_eq!(retrieved[4].index, 7);
}

#[test]
fn test_memory_storage_get_entries_empty() {
    let storage = MemoryStorage::new();

    let retrieved = storage.get_entries(1, 10).unwrap();
    assert!(retrieved.is_empty());
}

#[test]
fn test_memory_storage_truncate() {
    let mut storage = MemoryStorage::new();

    let entries: Vec<LogEntry> = (1..=10)
        .map(|i| LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        })
        .collect();

    storage.append(&entries).unwrap();
    storage.truncate(6).unwrap();

    let retrieved = storage.get_entries(1, 10).unwrap();
    assert_eq!(retrieved.len(), 5);
    assert!(retrieved.iter().all(|e| e.index < 6));
}

#[test]
fn test_memory_storage_compact() {
    let mut storage = MemoryStorage::new();

    let entries: Vec<LogEntry> = (1..=10)
        .map(|i| LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        })
        .collect();

    storage.append(&entries).unwrap();
    storage.compact(5).unwrap();

    let retrieved = storage.get_entries(1, 10).unwrap();
    assert!(retrieved.iter().all(|e| e.index > 5));
}

// =============================================================================
// KeyValueFSM Tests
// =============================================================================

#[test]
fn test_fsm_new() {
    let fsm = KeyValueFSM::new();
    assert!(fsm.is_empty());
    assert_eq!(fsm.len(), 0);
}

#[test]
fn test_fsm_default() {
    let fsm = KeyValueFSM::default();
    assert!(fsm.is_empty());
}

#[test]
fn test_fsm_put() {
    let mut fsm = KeyValueFSM::new();

    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: b"key1".to_vec(),
            value: b"value1".to_vec(),
        },
        entry_type: EntryType::Command,
    };

    fsm.apply(&entry);

    assert!(!fsm.is_empty());
    assert_eq!(fsm.len(), 1);
    assert_eq!(fsm.get(b"key1"), Some(b"value1".to_vec()));
}

#[test]
fn test_fsm_get_nonexistent() {
    let fsm = KeyValueFSM::new();
    assert_eq!(fsm.get(b"nonexistent"), None);
}

#[test]
fn test_fsm_delete() {
    let mut fsm = KeyValueFSM::new();

    let put_entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: b"key1".to_vec(),
            value: b"value1".to_vec(),
        },
        entry_type: EntryType::Command,
    };
    fsm.apply(&put_entry);

    let delete_entry = LogEntry {
        term: 1,
        index: 2,
        command: Command::Delete {
            key: b"key1".to_vec(),
        },
        entry_type: EntryType::Command,
    };
    fsm.apply(&delete_entry);

    assert!(fsm.is_empty());
    assert_eq!(fsm.get(b"key1"), None);
}

#[test]
fn test_fsm_get_command() {
    let mut fsm = KeyValueFSM::new();

    let put_entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: b"key1".to_vec(),
            value: b"value1".to_vec(),
        },
        entry_type: EntryType::Command,
    };
    fsm.apply(&put_entry);

    let get_entry = LogEntry {
        term: 1,
        index: 2,
        command: Command::Get {
            key: b"key1".to_vec(),
        },
        entry_type: EntryType::Command,
    };
    let result = fsm.apply(&get_entry);

    if let distributed_kv_raft::ApplyResult::Value(Some(value)) = result {
        assert_eq!(value, b"value1".to_vec());
    } else {
        panic!("Expected value");
    }
}

#[test]
fn test_fsm_noop() {
    let mut fsm = KeyValueFSM::new();

    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    };

    let result = fsm.apply(&entry);
    assert!(matches!(result, distributed_kv_raft::ApplyResult::Success));
}

#[test]
fn test_fsm_update_value() {
    let mut fsm = KeyValueFSM::new();

    let entry1 = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: b"key1".to_vec(),
            value: b"value1".to_vec(),
        },
        entry_type: EntryType::Command,
    };
    fsm.apply(&entry1);

    let entry2 = LogEntry {
        term: 1,
        index: 2,
        command: Command::Put {
            key: b"key1".to_vec(),
            value: b"value2".to_vec(),
        },
        entry_type: EntryType::Command,
    };
    fsm.apply(&entry2);

    assert_eq!(fsm.get(b"key1"), Some(b"value2".to_vec()));
}

#[test]
fn test_fsm_keys() {
    let mut fsm = KeyValueFSM::new();

    for i in 0..5 {
        let entry = LogEntry {
            term: 1,
            index: i + 1,
            command: Command::Put {
                key: format!("key{}", i).into_bytes(),
                value: format!("value{}", i).into_bytes(),
            },
            entry_type: EntryType::Command,
        };
        fsm.apply(&entry);
    }

    let keys = fsm.keys();
    assert_eq!(keys.len(), 5);
}

#[test]
fn test_fsm_last_applied() {
    let mut fsm = KeyValueFSM::new();

    assert_eq!(fsm.last_applied_index(), 0);
    assert_eq!(fsm.last_applied_term(), 0);

    let entry = LogEntry {
        term: 5,
        index: 10,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    };
    fsm.apply(&entry);

    assert_eq!(fsm.last_applied_index(), 10);
    assert_eq!(fsm.last_applied_term(), 5);
}

// =============================================================================
// Snapshot Tests
// =============================================================================

#[test]
fn test_fsm_snapshot() {
    let mut fsm = KeyValueFSM::new();

    for i in 0..10 {
        let entry = LogEntry {
            term: 1,
            index: i + 1,
            command: Command::Put {
                key: format!("key{}", i).into_bytes(),
                value: format!("value{}", i).into_bytes(),
            },
            entry_type: EntryType::Command,
        };
        fsm.apply(&entry);
    }

    let snapshot = fsm.snapshot();

    assert_eq!(snapshot.last_included_index, 10);
    assert_eq!(snapshot.last_included_term, 1);
    assert!(!snapshot.data.is_empty());
}

#[test]
fn test_fsm_restore() {
    let mut fsm1 = KeyValueFSM::new();

    for i in 0..5 {
        let entry = LogEntry {
            term: 1,
            index: i + 1,
            command: Command::Put {
                key: format!("key{}", i).into_bytes(),
                value: format!("value{}", i).into_bytes(),
            },
            entry_type: EntryType::Command,
        };
        fsm1.apply(&entry);
    }

    let snapshot = fsm1.snapshot();

    let mut fsm2 = KeyValueFSM::new();
    fsm2.restore(&snapshot).unwrap();

    for i in 0..5 {
        assert_eq!(
            fsm2.get(&format!("key{}", i).into_bytes()),
            Some(format!("value{}", i).into_bytes())
        );
    }
}

#[test]
fn test_snapshot_serialization() {
    let snapshot = Snapshot {
        last_included_index: 100,
        last_included_term: 5,
        data: b"test data".to_vec(),
    };

    let serialized = bincode::serialize(&snapshot).unwrap();
    let deserialized: Snapshot = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.last_included_index, 100);
    assert_eq!(deserialized.last_included_term, 5);
    assert_eq!(deserialized.data, b"test data".to_vec());
}

// =============================================================================
// Large Data Tests
// =============================================================================

#[test]
fn test_fsm_large_values() {
    let mut fsm = KeyValueFSM::new();

    let large_value = vec![b'x'; 1024 * 1024]; // 1MB

    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: b"large_key".to_vec(),
            value: large_value.clone(),
        },
        entry_type: EntryType::Command,
    };
    fsm.apply(&entry);

    assert_eq!(fsm.get(b"large_key"), Some(large_value));
}

#[test]
fn test_storage_many_entries() {
    let mut storage = MemoryStorage::new();

    let entries: Vec<LogEntry> = (1..=10000)
        .map(|i| LogEntry {
            term: (i / 1000 + 1) as u64,
            index: i,
            command: Command::Put {
                key: format!("key{}", i).into_bytes(),
                value: format!("value{}", i).into_bytes(),
            },
            entry_type: EntryType::Command,
        })
        .collect();

    storage.append(&entries).unwrap();

    assert_eq!(storage.last_entry().unwrap().index, 10000);

    let mid_entries = storage.get_entries(5000, 5010).unwrap();
    assert_eq!(mid_entries.len(), 11);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_storage_append_empty() {
    let mut storage = MemoryStorage::new();
    storage.append(&[]).unwrap();
    assert!(storage.last_entry().is_none());
}

#[test]
fn test_storage_truncate_all() {
    let mut storage = MemoryStorage::new();

    let entries: Vec<LogEntry> = (1..=5)
        .map(|i| LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        })
        .collect();

    storage.append(&entries).unwrap();
    storage.truncate(1).unwrap(); // Truncate everything from index 1

    let remaining = storage.get_entries(1, 10).unwrap();
    assert!(remaining.is_empty());
}

#[test]
fn test_storage_compact_all() {
    let mut storage = MemoryStorage::new();

    let entries: Vec<LogEntry> = (1..=5)
        .map(|i| LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        })
        .collect();

    storage.append(&entries).unwrap();
    storage.compact(10).unwrap(); // Compact everything up to index 10

    let remaining = storage.get_entries(1, 10).unwrap();
    assert!(remaining.is_empty());
}

#[test]
fn test_fsm_delete_nonexistent() {
    let mut fsm = KeyValueFSM::new();

    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Delete {
            key: b"nonexistent".to_vec(),
        },
        entry_type: EntryType::Command,
    };

    let result = fsm.apply(&entry);
    assert!(matches!(result, distributed_kv_raft::ApplyResult::Success));
}

#[test]
fn test_fsm_empty_key_value() {
    let mut fsm = KeyValueFSM::new();

    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: vec![],
            value: vec![],
        },
        entry_type: EntryType::Command,
    };
    fsm.apply(&entry);

    assert_eq!(fsm.get(&[]), Some(vec![]));
}

// =============================================================================
// Concurrent-like Access Tests (single-threaded simulation)
// =============================================================================

#[test]
fn test_storage_alternating_operations() {
    let mut storage = MemoryStorage::new();

    for i in 1..=100 {
        let entry = LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        };
        storage.append(&[entry]).unwrap();

        if i % 10 == 0 {
            // Periodic compaction
            storage.compact(i - 5).unwrap();
        }
    }

    let remaining = storage.get_entries(1, 100).unwrap();
    assert!(remaining.iter().all(|e| e.index > 95));
}

#[test]
fn test_fsm_rapid_updates() {
    let mut fsm = KeyValueFSM::new();

    let key = b"counter".to_vec();

    for i in 0..1000 {
        let entry = LogEntry {
            term: 1,
            index: i + 1,
            command: Command::Put {
                key: key.clone(),
                value: format!("{}", i).into_bytes(),
            },
            entry_type: EntryType::Command,
        };
        fsm.apply(&entry);
    }

    assert_eq!(fsm.get(&key), Some(b"999".to_vec()));
}

// =============================================================================
// Binary Data Tests
// =============================================================================

#[test]
fn test_fsm_binary_key() {
    let mut fsm = KeyValueFSM::new();

    let binary_key = vec![0x00, 0x01, 0xFF, 0xFE];
    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: binary_key.clone(),
            value: b"value".to_vec(),
        },
        entry_type: EntryType::Command,
    };
    fsm.apply(&entry);

    assert_eq!(fsm.get(&binary_key), Some(b"value".to_vec()));
}

#[test]
fn test_fsm_binary_value() {
    let mut fsm = KeyValueFSM::new();

    let binary_value = (0..=255).collect::<Vec<u8>>();
    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: b"key".to_vec(),
            value: binary_value.clone(),
        },
        entry_type: EntryType::Command,
    };
    fsm.apply(&entry);

    assert_eq!(fsm.get(b"key"), Some(binary_value));
}

// =============================================================================
// Log Entry Serialization Tests
// =============================================================================

#[test]
fn test_log_entry_serialization() {
    let entry = LogEntry {
        term: 42,
        index: 100,
        command: Command::Put {
            key: b"test_key".to_vec(),
            value: b"test_value".to_vec(),
        },
        entry_type: EntryType::Command,
    };

    let serialized = bincode::serialize(&entry).unwrap();
    let deserialized: LogEntry = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.term, 42);
    assert_eq!(deserialized.index, 100);
    assert_eq!(deserialized.entry_type, EntryType::Command);
}

#[test]
fn test_command_serialization() {
    let commands = vec![
        Command::Put {
            key: b"k".to_vec(),
            value: b"v".to_vec(),
        },
        Command::Get {
            key: b"k".to_vec(),
        },
        Command::Delete {
            key: b"k".to_vec(),
        },
        Command::NoOp,
    ];

    for cmd in commands {
        let serialized = bincode::serialize(&cmd).unwrap();
        let _deserialized: Command = bincode::deserialize(&serialized).unwrap();
    }
}
