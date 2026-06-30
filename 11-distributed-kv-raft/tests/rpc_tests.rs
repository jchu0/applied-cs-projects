//! Comprehensive tests for RPC message types.

use distributed_kv_raft::{
    AppendEntriesRequest, AppendEntriesResponse, ClientRequest, ClientResponse, Command,
    EntryType, InstallSnapshotRequest, InstallSnapshotResponse, LogEntry, RequestVoteRequest,
    RequestVoteResponse,
};

// =============================================================================
// RequestVoteRequest Tests
// =============================================================================

#[test]
fn test_request_vote_request_creation() {
    let request = RequestVoteRequest {
        term: 5,
        candidate_id: 1,
        last_log_index: 100,
        last_log_term: 4,
    };

    assert_eq!(request.term, 5);
    assert_eq!(request.candidate_id, 1);
    assert_eq!(request.last_log_index, 100);
    assert_eq!(request.last_log_term, 4);
}

#[test]
fn test_request_vote_request_clone() {
    let request = RequestVoteRequest {
        term: 3,
        candidate_id: 2,
        last_log_index: 50,
        last_log_term: 2,
    };

    let cloned = request.clone();

    assert_eq!(cloned.term, 3);
    assert_eq!(cloned.candidate_id, 2);
}

#[test]
fn test_request_vote_request_debug() {
    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let debug_str = format!("{:?}", request);
    assert!(debug_str.contains("term"));
    assert!(debug_str.contains("candidate_id"));
}

// =============================================================================
// RequestVoteResponse Tests
// =============================================================================

#[test]
fn test_request_vote_response_granted() {
    let response = RequestVoteResponse {
        term: 5,
        vote_granted: true,
    };

    assert_eq!(response.term, 5);
    assert!(response.vote_granted);
}

#[test]
fn test_request_vote_response_denied() {
    let response = RequestVoteResponse {
        term: 10,
        vote_granted: false,
    };

    assert_eq!(response.term, 10);
    assert!(!response.vote_granted);
}

#[test]
fn test_request_vote_response_clone() {
    let response = RequestVoteResponse {
        term: 3,
        vote_granted: true,
    };

    let cloned = response.clone();

    assert_eq!(cloned.term, 3);
    assert!(cloned.vote_granted);
}

// =============================================================================
// AppendEntriesRequest Tests
// =============================================================================

#[test]
fn test_append_entries_request_heartbeat() {
    let request = AppendEntriesRequest {
        term: 5,
        leader_id: 0,
        prev_log_index: 10,
        prev_log_term: 4,
        entries: vec![],
        leader_commit: 8,
    };

    assert_eq!(request.term, 5);
    assert_eq!(request.leader_id, 0);
    assert!(request.entries.is_empty());
}

#[test]
fn test_append_entries_request_with_entries() {
    let entries = vec![
        LogEntry {
            term: 5,
            index: 11,
            command: Command::Put {
                key: b"key1".to_vec(),
                value: b"value1".to_vec(),
            },
            entry_type: EntryType::Command,
        },
        LogEntry {
            term: 5,
            index: 12,
            command: Command::Put {
                key: b"key2".to_vec(),
                value: b"value2".to_vec(),
            },
            entry_type: EntryType::Command,
        },
    ];

    let request = AppendEntriesRequest {
        term: 5,
        leader_id: 0,
        prev_log_index: 10,
        prev_log_term: 4,
        entries,
        leader_commit: 10,
    };

    assert_eq!(request.entries.len(), 2);
    assert_eq!(request.entries[0].index, 11);
    assert_eq!(request.entries[1].index, 12);
}

#[test]
fn test_append_entries_request_clone() {
    let request = AppendEntriesRequest {
        term: 3,
        leader_id: 1,
        prev_log_index: 5,
        prev_log_term: 2,
        entries: vec![],
        leader_commit: 4,
    };

    let cloned = request.clone();

    assert_eq!(cloned.term, 3);
    assert_eq!(cloned.leader_id, 1);
}

// =============================================================================
// AppendEntriesResponse Tests
// =============================================================================

#[test]
fn test_append_entries_response_success() {
    let response = AppendEntriesResponse {
        term: 5,
        success: true,
        conflict_index: None,
        conflict_term: None,
    };

    assert_eq!(response.term, 5);
    assert!(response.success);
    assert!(response.conflict_index.is_none());
    assert!(response.conflict_term.is_none());
}

#[test]
fn test_append_entries_response_failure_with_conflict() {
    let response = AppendEntriesResponse {
        term: 5,
        success: false,
        conflict_index: Some(10),
        conflict_term: Some(3),
    };

    assert!(!response.success);
    assert_eq!(response.conflict_index, Some(10));
    assert_eq!(response.conflict_term, Some(3));
}

#[test]
fn test_append_entries_response_stale_term() {
    let response = AppendEntriesResponse {
        term: 10,
        success: false,
        conflict_index: None,
        conflict_term: None,
    };

    assert!(!response.success);
    assert_eq!(response.term, 10);
}

// =============================================================================
// InstallSnapshotRequest Tests
// =============================================================================

#[test]
fn test_install_snapshot_request_single_chunk() {
    let request = InstallSnapshotRequest {
        term: 5,
        leader_id: 0,
        last_included_index: 100,
        last_included_term: 4,
        offset: 0,
        data: vec![1, 2, 3, 4, 5],
        done: true,
    };

    assert_eq!(request.term, 5);
    assert_eq!(request.last_included_index, 100);
    assert!(request.done);
    assert_eq!(request.data.len(), 5);
}

#[test]
fn test_install_snapshot_request_chunked() {
    let chunk1 = InstallSnapshotRequest {
        term: 5,
        leader_id: 0,
        last_included_index: 100,
        last_included_term: 4,
        offset: 0,
        data: vec![1, 2, 3],
        done: false,
    };

    let chunk2 = InstallSnapshotRequest {
        term: 5,
        leader_id: 0,
        last_included_index: 100,
        last_included_term: 4,
        offset: 3,
        data: vec![4, 5, 6],
        done: true,
    };

    assert!(!chunk1.done);
    assert!(chunk2.done);
    assert_eq!(chunk2.offset, 3);
}

#[test]
fn test_install_snapshot_request_empty_data() {
    let request = InstallSnapshotRequest {
        term: 1,
        leader_id: 0,
        last_included_index: 0,
        last_included_term: 0,
        offset: 0,
        data: vec![],
        done: true,
    };

    assert!(request.data.is_empty());
}

// =============================================================================
// InstallSnapshotResponse Tests
// =============================================================================

#[test]
fn test_install_snapshot_response() {
    let response = InstallSnapshotResponse { term: 5 };

    assert_eq!(response.term, 5);
}

#[test]
fn test_install_snapshot_response_clone() {
    let response = InstallSnapshotResponse { term: 3 };
    let cloned = response.clone();

    assert_eq!(cloned.term, 3);
}

// =============================================================================
// ClientRequest Tests
// =============================================================================

#[test]
fn test_client_request_put() {
    let request = ClientRequest::Put {
        key: b"test_key".to_vec(),
        value: b"test_value".to_vec(),
    };

    if let ClientRequest::Put { key, value } = request {
        assert_eq!(key, b"test_key".to_vec());
        assert_eq!(value, b"test_value".to_vec());
    } else {
        panic!("Expected Put");
    }
}

#[test]
fn test_client_request_get() {
    let request = ClientRequest::Get {
        key: b"test_key".to_vec(),
    };

    if let ClientRequest::Get { key } = request {
        assert_eq!(key, b"test_key".to_vec());
    } else {
        panic!("Expected Get");
    }
}

#[test]
fn test_client_request_delete() {
    let request = ClientRequest::Delete {
        key: b"test_key".to_vec(),
    };

    if let ClientRequest::Delete { key } = request {
        assert_eq!(key, b"test_key".to_vec());
    } else {
        panic!("Expected Delete");
    }
}

#[test]
fn test_client_request_clone() {
    let request = ClientRequest::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    };

    let cloned = request.clone();

    if let ClientRequest::Put { key, value } = cloned {
        assert_eq!(key, b"key".to_vec());
        assert_eq!(value, b"value".to_vec());
    }
}

// =============================================================================
// ClientResponse Tests
// =============================================================================

#[test]
fn test_client_response_success_with_value() {
    let response = ClientResponse::Success {
        value: Some(b"test_value".to_vec()),
    };

    if let ClientResponse::Success { value } = response {
        assert_eq!(value, Some(b"test_value".to_vec()));
    } else {
        panic!("Expected Success");
    }
}

#[test]
fn test_client_response_success_no_value() {
    let response = ClientResponse::Success { value: None };

    if let ClientResponse::Success { value } = response {
        assert!(value.is_none());
    }
}

#[test]
fn test_client_response_not_leader() {
    let response = ClientResponse::NotLeader {
        leader_hint: Some(2),
    };

    if let ClientResponse::NotLeader { leader_hint } = response {
        assert_eq!(leader_hint, Some(2));
    } else {
        panic!("Expected NotLeader");
    }
}

#[test]
fn test_client_response_not_leader_no_hint() {
    let response = ClientResponse::NotLeader { leader_hint: None };

    if let ClientResponse::NotLeader { leader_hint } = response {
        assert!(leader_hint.is_none());
    }
}

#[test]
fn test_client_response_error() {
    let response = ClientResponse::Error {
        message: "Internal error".to_string(),
    };

    if let ClientResponse::Error { message } = response {
        assert_eq!(message, "Internal error");
    } else {
        panic!("Expected Error");
    }
}

#[test]
fn test_client_response_clone() {
    let response = ClientResponse::Success {
        value: Some(b"data".to_vec()),
    };

    let cloned = response.clone();

    if let ClientResponse::Success { value } = cloned {
        assert_eq!(value, Some(b"data".to_vec()));
    }
}

// =============================================================================
// LogEntry Tests
// =============================================================================

#[test]
fn test_log_entry_command_put() {
    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: b"key".to_vec(),
            value: b"value".to_vec(),
        },
        entry_type: EntryType::Command,
    };

    assert_eq!(entry.term, 1);
    assert_eq!(entry.index, 1);
    assert_eq!(entry.entry_type, EntryType::Command);
}

#[test]
fn test_log_entry_noop() {
    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    };

    assert!(matches!(entry.command, Command::NoOp));
    assert_eq!(entry.entry_type, EntryType::NoOp);
}

#[test]
fn test_log_entry_configuration() {
    let entry = LogEntry {
        term: 2,
        index: 5,
        command: Command::NoOp,
        entry_type: EntryType::Configuration,
    };

    assert_eq!(entry.entry_type, EntryType::Configuration);
}

#[test]
fn test_log_entry_clone() {
    let entry = LogEntry {
        term: 3,
        index: 10,
        command: Command::Put {
            key: b"k".to_vec(),
            value: b"v".to_vec(),
        },
        entry_type: EntryType::Command,
    };

    let cloned = entry.clone();

    assert_eq!(cloned.term, 3);
    assert_eq!(cloned.index, 10);
}

// =============================================================================
// Command Tests
// =============================================================================

#[test]
fn test_command_put() {
    let cmd = Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    };

    assert!(matches!(cmd, Command::Put { .. }));
}

#[test]
fn test_command_get() {
    let cmd = Command::Get {
        key: b"key".to_vec(),
    };

    assert!(matches!(cmd, Command::Get { .. }));
}

#[test]
fn test_command_delete() {
    let cmd = Command::Delete {
        key: b"key".to_vec(),
    };

    assert!(matches!(cmd, Command::Delete { .. }));
}

#[test]
fn test_command_noop() {
    let cmd = Command::NoOp;
    assert!(matches!(cmd, Command::NoOp));
}

#[test]
fn test_command_clone() {
    let cmd = Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    };

    let cloned = cmd.clone();

    if let Command::Put { key, value } = cloned {
        assert_eq!(key, b"key".to_vec());
        assert_eq!(value, b"value".to_vec());
    }
}

// =============================================================================
// EntryType Tests
// =============================================================================

#[test]
fn test_entry_type_command() {
    let entry_type = EntryType::Command;
    assert_eq!(entry_type, EntryType::Command);
}

#[test]
fn test_entry_type_noop() {
    let entry_type = EntryType::NoOp;
    assert_eq!(entry_type, EntryType::NoOp);
}

#[test]
fn test_entry_type_configuration() {
    let entry_type = EntryType::Configuration;
    assert_eq!(entry_type, EntryType::Configuration);
}

#[test]
fn test_entry_type_clone() {
    let entry_type = EntryType::Command;
    let cloned = entry_type.clone();
    assert_eq!(cloned, EntryType::Command);
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_request_vote_request_serialization() {
    let request = RequestVoteRequest {
        term: 5,
        candidate_id: 1,
        last_log_index: 100,
        last_log_term: 4,
    };

    let serialized = bincode::serialize(&request).unwrap();
    let deserialized: RequestVoteRequest = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.term, 5);
    assert_eq!(deserialized.candidate_id, 1);
}

#[test]
fn test_append_entries_request_serialization() {
    let request = AppendEntriesRequest {
        term: 3,
        leader_id: 0,
        prev_log_index: 10,
        prev_log_term: 2,
        entries: vec![LogEntry {
            term: 3,
            index: 11,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        }],
        leader_commit: 10,
    };

    let serialized = bincode::serialize(&request).unwrap();
    let deserialized: AppendEntriesRequest = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.term, 3);
    assert_eq!(deserialized.entries.len(), 1);
}

#[test]
fn test_install_snapshot_request_serialization() {
    let request = InstallSnapshotRequest {
        term: 5,
        leader_id: 0,
        last_included_index: 100,
        last_included_term: 4,
        offset: 0,
        data: vec![1, 2, 3, 4, 5],
        done: true,
    };

    let serialized = bincode::serialize(&request).unwrap();
    let deserialized: InstallSnapshotRequest = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.last_included_index, 100);
    assert_eq!(deserialized.data.len(), 5);
}

#[test]
fn test_client_request_serialization() {
    let request = ClientRequest::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    };

    let serialized = bincode::serialize(&request).unwrap();
    let deserialized: ClientRequest = bincode::deserialize(&serialized).unwrap();

    if let ClientRequest::Put { key, value } = deserialized {
        assert_eq!(key, b"key".to_vec());
        assert_eq!(value, b"value".to_vec());
    }
}
