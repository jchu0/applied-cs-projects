//! CLUSTER commands implementation
//!
//! Implements Redis CLUSTER commands for cluster mode operation.

use crate::cluster::{ClusterState, ClusterStateFlag, CLUSTER_SLOTS};
use crate::resp::RespValue;

/// Handle CLUSTER command
pub fn cluster(args: &[RespValue], cluster_state: Option<&ClusterState>) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'cluster' command");
    }

    let subcommand = match &args[0] {
        RespValue::BulkString(Some(data)) => {
            String::from_utf8_lossy(data).to_uppercase()
        }
        _ => return RespValue::error("ERR invalid argument"),
    };

    match subcommand.as_str() {
        "INFO" => cluster_info(cluster_state),
        "SLOTS" => cluster_slots(cluster_state),
        "NODES" => cluster_nodes(cluster_state),
        "KEYSLOT" => cluster_keyslot(&args[1..]),
        "COUNTKEYSINSLOT" => cluster_countkeysinslot(&args[1..]),
        "GETKEYSINSLOT" => cluster_getkeysinslot(&args[1..]),
        "MYID" => cluster_myid(cluster_state),
        "MEET" => cluster_meet(&args[1..]),
        "REPLICATE" => cluster_replicate(&args[1..]),
        "ADDSLOTS" => cluster_addslots(&args[1..]),
        "DELSLOTS" => cluster_delslots(&args[1..]),
        "SETSLOT" => cluster_setslot(&args[1..]),
        "FAILOVER" => cluster_failover(&args[1..]),
        "RESET" => cluster_reset(&args[1..]),
        "SAVECONFIG" => cluster_saveconfig(),
        "HELP" => cluster_help(),
        _ => RespValue::error(format!("ERR unknown subcommand '{}'", subcommand)),
    }
}

/// CLUSTER INFO - Get cluster state info
fn cluster_info(cluster_state: Option<&ClusterState>) -> RespValue {
    let info = match cluster_state {
        Some(state) => state.info(),
        None => {
            // Not in cluster mode, return basic info
            let mut info = String::new();
            info.push_str("cluster_state:fail\n");
            info.push_str("cluster_slots_assigned:0\n");
            info.push_str("cluster_slots_ok:0\n");
            info.push_str("cluster_slots_pfail:0\n");
            info.push_str("cluster_slots_fail:0\n");
            info.push_str("cluster_known_nodes:0\n");
            info.push_str("cluster_size:0\n");
            info.push_str("cluster_current_epoch:0\n");
            info.push_str("cluster_my_epoch:0\n");
            info
        }
    };
    RespValue::BulkString(Some(info.into_bytes()))
}

/// CLUSTER SLOTS - Get cluster slot configuration
fn cluster_slots(cluster_state: Option<&ClusterState>) -> RespValue {
    match cluster_state {
        Some(state) => {
            let mut slots: Vec<RespValue> = Vec::new();

            // Group consecutive slots by node
            let mut current_node: Option<String> = None;
            let mut start_slot: u16 = 0;

            for slot in 0..CLUSTER_SLOTS {
                if let Some(node) = state.get_slot_node(slot) {
                    let node_id = node.id.clone();

                    match &current_node {
                        Some(curr) if curr == &node_id => {
                            // Continue with same node
                        }
                        _ => {
                            // New node or first slot
                            if current_node.is_some() && slot > 0 {
                                // Emit previous range
                                if let Some(prev_node) = state.get_node(current_node.as_ref().unwrap()) {
                                    let entry = build_slot_entry(start_slot, slot - 1, prev_node);
                                    slots.push(entry);
                                }
                            }
                            current_node = Some(node_id);
                            start_slot = slot;
                        }
                    }
                }
            }

            // Emit last range if any
            if let Some(node_id) = current_node {
                if let Some(node) = state.get_node(&node_id) {
                    let entry = build_slot_entry(start_slot, CLUSTER_SLOTS - 1, node);
                    slots.push(entry);
                }
            }

            RespValue::Array(Some(slots))
        }
        None => RespValue::Array(Some(vec![])),
    }
}

fn build_slot_entry(start: u16, end: u16, node: &crate::cluster::ClusterNode) -> RespValue {
    let addr_str = node.addr.to_string();
    let addr_parts: Vec<&str> = addr_str.split(':').collect();
    let host = addr_parts.get(0).unwrap_or(&"127.0.0.1").to_string();
    let port = node.addr.port();

    RespValue::Array(Some(vec![
        RespValue::Integer(start as i64),
        RespValue::Integer(end as i64),
        RespValue::Array(Some(vec![
            RespValue::BulkString(Some(host.into_bytes())),
            RespValue::Integer(port as i64),
            RespValue::BulkString(Some(node.id.clone().into_bytes())),
        ])),
    ]))
}

/// CLUSTER NODES - Get cluster node information
fn cluster_nodes(cluster_state: Option<&ClusterState>) -> RespValue {
    match cluster_state {
        Some(state) => {
            let mut nodes_info = String::new();

            for node in state.nodes() {
                // Format: <id> <ip:port@cport> <flags> <master> <ping-sent> <pong-recv> <config-epoch> <link-state> <slot> ...
                let flags = build_flags(&node.flags);
                let master_id = node.master_id.clone().unwrap_or_else(|| "-".to_string());
                let slots: Vec<String> = node.slots.iter().map(|s| s.to_string()).collect();
                let slots_str = if slots.is_empty() {
                    String::new()
                } else {
                    format!(" {}", slots.join(" "))
                };

                nodes_info.push_str(&format!(
                    "{} {}@{} {} {} {} {} {} {}{}\n",
                    node.id,
                    node.addr,
                    node.cport,
                    flags,
                    master_id,
                    node.ping_sent,
                    node.pong_recv,
                    node.config_epoch,
                    node.link_state,
                    slots_str
                ));
            }

            RespValue::BulkString(Some(nodes_info.into_bytes()))
        }
        None => RespValue::BulkString(Some(b"".to_vec())),
    }
}

fn build_flags(flags: &crate::cluster::NodeFlags) -> String {
    let mut parts = Vec::new();
    if flags.myself {
        parts.push("myself");
    }
    if flags.master {
        parts.push("master");
    }
    if flags.slave {
        parts.push("slave");
    }
    if flags.pfail {
        parts.push("fail?");
    }
    if flags.fail {
        parts.push("fail");
    }
    if flags.handshake {
        parts.push("handshake");
    }
    if flags.noaddr {
        parts.push("noaddr");
    }
    if parts.is_empty() {
        "noflags".to_string()
    } else {
        parts.join(",")
    }
}

/// CLUSTER KEYSLOT key - Get hash slot for key
fn cluster_keyslot(args: &[RespValue]) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'cluster|keyslot' command");
    }

    match &args[0] {
        RespValue::BulkString(Some(key)) => {
            let slot = ClusterState::key_hash_slot(key);
            RespValue::Integer(slot as i64)
        }
        _ => RespValue::error("ERR invalid key"),
    }
}

/// CLUSTER COUNTKEYSINSLOT slot - Count keys in slot
fn cluster_countkeysinslot(args: &[RespValue]) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'cluster|countkeysinslot' command");
    }

    // Parse slot number
    let _slot = match &args[0] {
        RespValue::BulkString(Some(data)) => {
            match String::from_utf8_lossy(data).parse::<u16>() {
                Ok(s) if s < CLUSTER_SLOTS => s,
                _ => return RespValue::error("ERR Invalid or out of range slot"),
            }
        }
        RespValue::Integer(s) if *s >= 0 && (*s as u16) < CLUSTER_SLOTS => *s as u16,
        _ => return RespValue::error("ERR Invalid or out of range slot"),
    };

    // In a real implementation, this would count keys in the slot
    // For now, return 0 as a placeholder
    RespValue::Integer(0)
}

/// CLUSTER GETKEYSINSLOT slot count - Get keys in slot
fn cluster_getkeysinslot(args: &[RespValue]) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'cluster|getkeysinslot' command");
    }

    // Parse slot number
    let _slot = match &args[0] {
        RespValue::BulkString(Some(data)) => {
            match String::from_utf8_lossy(data).parse::<u16>() {
                Ok(s) if s < CLUSTER_SLOTS => s,
                _ => return RespValue::error("ERR Invalid or out of range slot"),
            }
        }
        RespValue::Integer(s) if *s >= 0 && (*s as u16) < CLUSTER_SLOTS => *s as u16,
        _ => return RespValue::error("ERR Invalid or out of range slot"),
    };

    // In a real implementation, this would return keys in the slot
    // For now, return empty array as a placeholder
    RespValue::Array(Some(vec![]))
}

/// CLUSTER MYID - Get this node's ID
fn cluster_myid(cluster_state: Option<&ClusterState>) -> RespValue {
    match cluster_state {
        Some(_state) => {
            // In a real implementation, return the node's ID
            RespValue::BulkString(Some(b"0000000000000000000000000000000000000000".to_vec()))
        }
        None => RespValue::error("ERR This instance has cluster support disabled"),
    }
}

/// CLUSTER MEET ip port - Add node to cluster
fn cluster_meet(args: &[RespValue]) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'cluster|meet' command");
    }

    let _ip = match &args[0] {
        RespValue::BulkString(Some(data)) => String::from_utf8_lossy(data).to_string(),
        _ => return RespValue::error("ERR invalid IP address"),
    };

    let _port = match &args[1] {
        RespValue::BulkString(Some(data)) => {
            match String::from_utf8_lossy(data).parse::<u16>() {
                Ok(p) => p,
                Err(_) => return RespValue::error("ERR invalid port"),
            }
        }
        RespValue::Integer(p) if *p > 0 && *p <= 65535 => *p as u16,
        _ => return RespValue::error("ERR invalid port"),
    };

    // In a real implementation, this would initiate cluster handshake
    RespValue::ok()
}

/// CLUSTER REPLICATE node-id - Configure this node as replica
fn cluster_replicate(args: &[RespValue]) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'cluster|replicate' command");
    }

    let _node_id = match &args[0] {
        RespValue::BulkString(Some(data)) => String::from_utf8_lossy(data).to_string(),
        _ => return RespValue::error("ERR invalid node ID"),
    };

    // In a real implementation, this would configure replication
    RespValue::ok()
}

/// CLUSTER ADDSLOTS slot [slot ...] - Assign slots to this node
fn cluster_addslots(args: &[RespValue]) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'cluster|addslots' command");
    }

    for arg in args {
        let _slot = match arg {
            RespValue::BulkString(Some(data)) => {
                match String::from_utf8_lossy(data).parse::<u16>() {
                    Ok(s) if s < CLUSTER_SLOTS => s,
                    _ => return RespValue::error("ERR Invalid or out of range slot"),
                }
            }
            RespValue::Integer(s) if *s >= 0 && (*s as u16) < CLUSTER_SLOTS => *s as u16,
            _ => return RespValue::error("ERR Invalid or out of range slot"),
        };
        // In a real implementation, would assign the slot
    }

    RespValue::ok()
}

/// CLUSTER DELSLOTS slot [slot ...] - Remove slot assignment
fn cluster_delslots(args: &[RespValue]) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'cluster|delslots' command");
    }

    for arg in args {
        let _slot = match arg {
            RespValue::BulkString(Some(data)) => {
                match String::from_utf8_lossy(data).parse::<u16>() {
                    Ok(s) if s < CLUSTER_SLOTS => s,
                    _ => return RespValue::error("ERR Invalid or out of range slot"),
                }
            }
            RespValue::Integer(s) if *s >= 0 && (*s as u16) < CLUSTER_SLOTS => *s as u16,
            _ => return RespValue::error("ERR Invalid or out of range slot"),
        };
        // In a real implementation, would remove slot assignment
    }

    RespValue::ok()
}

/// CLUSTER SETSLOT slot IMPORTING|MIGRATING|STABLE|NODE node-id
fn cluster_setslot(args: &[RespValue]) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'cluster|setslot' command");
    }

    // Parse slot
    let _slot = match &args[0] {
        RespValue::BulkString(Some(data)) => {
            match String::from_utf8_lossy(data).parse::<u16>() {
                Ok(s) if s < CLUSTER_SLOTS => s,
                _ => return RespValue::error("ERR Invalid or out of range slot"),
            }
        }
        RespValue::Integer(s) if *s >= 0 && (*s as u16) < CLUSTER_SLOTS => *s as u16,
        _ => return RespValue::error("ERR Invalid or out of range slot"),
    };

    // Parse subcommand
    let subcommand = match &args[1] {
        RespValue::BulkString(Some(data)) => {
            String::from_utf8_lossy(data).to_uppercase()
        }
        _ => return RespValue::error("ERR invalid argument"),
    };

    match subcommand.as_str() {
        "IMPORTING" | "MIGRATING" | "NODE" => {
            if args.len() < 3 {
                return RespValue::error(format!(
                    "ERR wrong number of arguments for 'cluster|setslot|{}' command",
                    subcommand.to_lowercase()
                ));
            }
            // In a real implementation, would handle slot migration
            RespValue::ok()
        }
        "STABLE" => RespValue::ok(),
        _ => RespValue::error(format!("ERR unknown subcommand '{}'", subcommand)),
    }
}

/// CLUSTER FAILOVER [FORCE|TAKEOVER]
fn cluster_failover(args: &[RespValue]) -> RespValue {
    let option: String = if args.is_empty() {
        "normal".to_string()
    } else {
        match &args[0] {
            RespValue::BulkString(Some(data)) => {
                let opt = String::from_utf8_lossy(data).to_uppercase();
                match opt.as_str() {
                    "FORCE" | "TAKEOVER" => opt.to_string(),
                    _ => return RespValue::error(format!("ERR unknown option '{}'", opt)),
                }
            }
            _ => return RespValue::error("ERR invalid argument"),
        }
    };

    // In a real implementation, would initiate failover
    let _ = option;
    RespValue::ok()
}

/// CLUSTER RESET [HARD|SOFT]
fn cluster_reset(args: &[RespValue]) -> RespValue {
    let _hard = if args.is_empty() {
        false
    } else {
        match &args[0] {
            RespValue::BulkString(Some(data)) => {
                let opt = String::from_utf8_lossy(data).to_uppercase();
                match opt.as_str() {
                    "HARD" => true,
                    "SOFT" => false,
                    _ => return RespValue::error(format!("ERR unknown option '{}'", opt)),
                }
            }
            _ => return RespValue::error("ERR invalid argument"),
        }
    };

    // In a real implementation, would reset cluster state
    RespValue::ok()
}

/// CLUSTER SAVECONFIG - Save cluster config to disk
fn cluster_saveconfig() -> RespValue {
    // In a real implementation, would persist cluster configuration
    RespValue::ok()
}

/// CLUSTER HELP - Show cluster command help
fn cluster_help() -> RespValue {
    let help = vec![
        RespValue::BulkString(Some(b"CLUSTER INFO -- Return info about the cluster".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER SLOTS -- Return slot info".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER NODES -- Return node info".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER KEYSLOT <key> -- Return slot for key".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER COUNTKEYSINSLOT <slot> -- Count keys in slot".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER GETKEYSINSLOT <slot> <count> -- Get keys in slot".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER MYID -- Return this node's ID".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER MEET <ip> <port> -- Add node to cluster".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER REPLICATE <node-id> -- Configure as replica".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER ADDSLOTS <slot> [slot ...] -- Assign slots".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER DELSLOTS <slot> [slot ...] -- Remove slots".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER SETSLOT <slot> <subcommand> -- Configure slot".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER FAILOVER [FORCE|TAKEOVER] -- Initiate failover".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER RESET [HARD|SOFT] -- Reset cluster".to_vec())),
        RespValue::BulkString(Some(b"CLUSTER SAVECONFIG -- Save config".to_vec())),
    ];
    RespValue::Array(Some(help))
}

/// Handle READONLY command - Set client to readonly mode
pub fn readonly(_args: &[RespValue]) -> RespValue {
    RespValue::ok()
}

/// Handle READWRITE command - Set client to read-write mode
pub fn readwrite(_args: &[RespValue]) -> RespValue {
    RespValue::ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_keyslot() {
        // Simple key
        let result = cluster_keyslot(&[RespValue::BulkString(Some(b"foo".to_vec()))]);
        match result {
            RespValue::Integer(slot) => {
                assert!(slot >= 0 && slot < CLUSTER_SLOTS as i64);
            }
            _ => panic!("Expected integer"),
        }
    }

    #[test]
    fn test_cluster_keyslot_hash_tag() {
        // Keys with hash tag should have same slot
        let result1 = cluster_keyslot(&[RespValue::BulkString(Some(b"user:{123}:name".to_vec()))]);
        let result2 = cluster_keyslot(&[RespValue::BulkString(Some(b"user:{123}:email".to_vec()))]);

        match (result1, result2) {
            (RespValue::Integer(slot1), RespValue::Integer(slot2)) => {
                assert_eq!(slot1, slot2, "Keys with same hash tag should have same slot");
            }
            _ => panic!("Expected integers"),
        }
    }

    #[test]
    fn test_cluster_info_no_cluster() {
        let result = cluster_info(None);
        match result {
            RespValue::BulkString(Some(data)) => {
                let info = String::from_utf8_lossy(&data);
                assert!(info.contains("cluster_state:fail"));
            }
            _ => panic!("Expected bulk string"),
        }
    }

    #[test]
    fn test_cluster_help() {
        let result = cluster_help();
        match result {
            RespValue::Array(Some(arr)) => {
                assert!(arr.len() > 10);
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_cluster_addslots() {
        let result = cluster_addslots(&[
            RespValue::BulkString(Some(b"0".to_vec())),
            RespValue::BulkString(Some(b"1".to_vec())),
        ]);
        assert_eq!(result, RespValue::ok());
    }

    #[test]
    fn test_cluster_addslots_invalid() {
        let result = cluster_addslots(&[RespValue::BulkString(Some(b"99999".to_vec()))]);
        match result {
            RespValue::Error(msg) => {
                assert!(msg.contains("Invalid or out of range"));
            }
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn test_cluster_unknown_subcommand() {
        let result = cluster(&[RespValue::BulkString(Some(b"UNKNOWN".to_vec()))], None);
        match result {
            RespValue::Error(msg) => {
                assert!(msg.contains("unknown subcommand"));
            }
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn test_cluster_meet() {
        let result = cluster_meet(&[
            RespValue::BulkString(Some(b"127.0.0.1".to_vec())),
            RespValue::BulkString(Some(b"7001".to_vec())),
        ]);
        assert_eq!(result, RespValue::ok());
    }

    #[test]
    fn test_cluster_setslot_stable() {
        let result = cluster_setslot(&[
            RespValue::BulkString(Some(b"0".to_vec())),
            RespValue::BulkString(Some(b"STABLE".to_vec())),
        ]);
        assert_eq!(result, RespValue::ok());
    }

    #[test]
    fn test_cluster_reset() {
        let result = cluster_reset(&[]);
        assert_eq!(result, RespValue::ok());

        let result = cluster_reset(&[RespValue::BulkString(Some(b"HARD".to_vec()))]);
        assert_eq!(result, RespValue::ok());

        let result = cluster_reset(&[RespValue::BulkString(Some(b"SOFT".to_vec()))]);
        assert_eq!(result, RespValue::ok());
    }

    #[test]
    fn test_readonly_readwrite() {
        assert_eq!(readonly(&[]), RespValue::ok());
        assert_eq!(readwrite(&[]), RespValue::ok());
    }
}
