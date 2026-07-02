//! End-to-end client path tests.
//!
//! These tests exercise the *real* client command path over the in-process
//! `MemoryNetwork` transport: a client submits a command, it is proposed to the
//! leader, replicated to a quorum, committed, applied to the KV state machine,
//! and the applied result is returned to the client. The follower-redirect test
//! confirms that a request sent to a follower is bounced back with a leader hint
//! that the client follows automatically.
//!
//! Scope note: reads here are read-after-commit served from the leader's applied
//! state after a leadership check (see `RaftClusterNode::linearizable_read`).
//! Full linearizability of the client wrapper across leader changes is not
//! asserted; the guarantee tested is that a committed+applied write is visible to
//! a subsequent read routed to the leader.

use std::sync::Arc;
use std::time::Duration;

use distributed_kv_raft::config::{RaftConfig, RaftConfigBuilder};
use distributed_kv_raft::transport::MemoryNetwork;
use distributed_kv_raft::{KVClient, NodeAddress, NodeId, TestCluster};

/// Build a config for `id` given the full set of node ids.
fn create_config(id: NodeId, all_ids: &[NodeId]) -> RaftConfig {
    let mut builder = RaftConfigBuilder::default()
        .id(id)
        .listen_addr(format!("127.0.0.1:{}", 5000 + id))
        .election_timeout(Duration::from_millis(150), Duration::from_millis(300))
        .heartbeat_interval(Duration::from_millis(50));

    for &peer_id in all_ids {
        if peer_id != id {
            builder = builder.peer(peer_id, format!("127.0.0.1:{}", 5000 + peer_id));
        }
    }

    builder.build()
}

/// Node addresses matching the cluster ids, for the client.
fn node_addresses(ids: &[NodeId]) -> Vec<NodeAddress> {
    ids.iter()
        .map(|&id| NodeAddress {
            id,
            addr: format!("127.0.0.1:{}", 5000 + id),
        })
        .collect()
}

/// Start a 3-node cluster over MemoryNetwork and return (cluster, client).
async fn start_cluster_and_client() -> (TestCluster, KVClient, Vec<NodeId>) {
    let ids: Vec<NodeId> = vec![0, 1, 2];
    let configs: Vec<RaftConfig> = ids.iter().map(|&id| create_config(id, &ids)).collect();

    let mut network = MemoryNetwork::new(&ids);
    let client_transport = network.client_transport();

    let mut cluster = TestCluster::new(configs, &mut network);
    cluster.start().await;

    let client = KVClient::new(node_addresses(&ids))
        .with_transport(client_transport as Arc<_>)
        .with_timeout(Duration::from_secs(5))
        .with_retries(6);

    (cluster, client, ids)
}

#[tokio::test]
async fn test_e2e_put_then_get_returns_value() {
    let (mut cluster, mut client, _ids) = start_cluster_and_client().await;

    // Drive an election.
    let leader_id = cluster
        .wait_for_leader(Duration::from_secs(3))
        .await
        .expect("cluster should elect a leader");

    // put(k, v) THROUGH Raft: propose -> replicate -> commit -> apply.
    client
        .put(b"alpha".to_vec(), b"one".to_vec())
        .await
        .expect("put should succeed via the leader");

    // get(k) returns v, served from applied state.
    let got = client
        .get(b"alpha")
        .await
        .expect("get should succeed");
    assert_eq!(got, Some(b"one".to_vec()));

    // Overwrite and re-read to confirm the applied path handles updates.
    client
        .put(b"alpha".to_vec(), b"two".to_vec())
        .await
        .expect("overwrite put should succeed");
    let got = client.get(b"alpha").await.expect("get should succeed");
    assert_eq!(got, Some(b"two".to_vec()));

    // A missing key returns None (applied state, not an error).
    let missing = client.get(b"missing").await.expect("get should succeed");
    assert_eq!(missing, None);

    // Sanity: the elected leader is one of the cluster nodes.
    assert!(cluster.nodes().iter().any(|n| n.id() == leader_id));

    cluster.stop().await;
}

#[tokio::test]
async fn test_e2e_put_to_follower_is_redirected_and_applied() {
    let ids: Vec<NodeId> = vec![0, 1, 2];
    let configs: Vec<RaftConfig> = ids.iter().map(|&id| create_config(id, &ids)).collect();

    let mut network = MemoryNetwork::new(&ids);
    let client_transport = network.client_transport();

    let mut cluster = TestCluster::new(configs, &mut network);
    cluster.start().await;

    let leader_id = cluster
        .wait_for_leader(Duration::from_secs(3))
        .await
        .expect("cluster should elect a leader");

    // Pick a follower to target first, forcing a redirect.
    let follower_id = *ids
        .iter()
        .find(|&&id| id != leader_id)
        .expect("a 3-node cluster has followers");

    // Pin the client's initial leader guess to the FOLLOWER so the first hop
    // hits a non-leader; the follower replies NotLeader{hint=leader} and the
    // client follows the hint automatically on retry.
    let mut client = KVClient::new(node_addresses(&ids))
        .with_transport(client_transport as Arc<_>)
        .with_timeout(Duration::from_secs(5))
        .with_retries(6)
        .with_known_leader(follower_id);

    // put issued to a follower -> redirected to and applied by the leader.
    client
        .put(b"k".to_vec(), b"v".to_vec())
        .await
        .expect("put should be redirected to the leader and applied");

    // The write is visible via a subsequent get served from applied state.
    let got = client.get(b"k").await.expect("get should succeed");
    assert_eq!(got, Some(b"v".to_vec()));

    cluster.stop().await;
}
