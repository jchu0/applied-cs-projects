//! Cluster module for distributed Redis-lite
//!
//! Implements Redis cluster protocol basics including:
//! - Hash slot mapping
//! - Node discovery
//! - MOVED/ASK redirections

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;

/// Number of hash slots in Redis cluster
pub const CLUSTER_SLOTS: u16 = 16384;

/// Cluster node state
#[derive(Debug, Clone, PartialEq)]
pub enum NodeState {
    /// Node is online
    Online,
    /// Node is in handshake
    Handshake,
    /// Node has failed
    Fail,
    /// Node is suspected to have failed
    PFail,
}

/// Cluster node flags
#[derive(Debug, Clone)]
pub struct NodeFlags {
    pub myself: bool,
    pub master: bool,
    pub slave: bool,
    pub pfail: bool,
    pub fail: bool,
    pub handshake: bool,
    pub noaddr: bool,
}

impl Default for NodeFlags {
    fn default() -> Self {
        Self {
            myself: false,
            master: false,
            slave: false,
            pfail: false,
            fail: false,
            handshake: false,
            noaddr: false,
        }
    }
}

/// Cluster node information
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Node ID (40 hex chars)
    pub id: String,
    /// Node address
    pub addr: SocketAddr,
    /// Cluster bus port (usually port + 10000)
    pub cport: u16,
    /// Node flags
    pub flags: NodeFlags,
    /// Master ID (if slave)
    pub master_id: Option<String>,
    /// Assigned slots
    pub slots: HashSet<u16>,
    /// Ping sent time
    pub ping_sent: u64,
    /// Pong received time
    pub pong_recv: u64,
    /// Config epoch
    pub config_epoch: u64,
    /// Link state
    pub link_state: String,
}

/// Cluster state
pub struct ClusterState {
    /// This node's ID
    myself_id: String,
    /// All known nodes
    nodes: HashMap<String, ClusterNode>,
    /// Slot to node mapping
    slots: [Option<String>; CLUSTER_SLOTS as usize],
    /// Current config epoch
    current_epoch: u64,
    /// Cluster state
    state: ClusterStateFlag,
    /// Size (number of master nodes)
    size: usize,
}

/// Cluster state flag
#[derive(Debug, Clone, PartialEq)]
pub enum ClusterStateFlag {
    Ok,
    Fail,
}

impl ClusterState {
    /// Create a new cluster state
    pub fn new(node_id: String) -> Self {
        Self {
            myself_id: node_id,
            nodes: HashMap::new(),
            slots: [const { None }; CLUSTER_SLOTS as usize],
            current_epoch: 0,
            state: ClusterStateFlag::Ok,
            size: 0,
        }
    }

    /// Add a node to the cluster
    pub fn add_node(&mut self, node: ClusterNode) {
        if node.flags.master {
            self.size += 1;
        }
        self.nodes.insert(node.id.clone(), node);
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, node_id: &str) -> Option<ClusterNode> {
        if let Some(node) = self.nodes.remove(node_id) {
            if node.flags.master {
                self.size = self.size.saturating_sub(1);
            }
            // Clear slots assigned to this node
            for slot in 0..CLUSTER_SLOTS {
                if self.slots[slot as usize].as_ref() == Some(&node_id.to_string()) {
                    self.slots[slot as usize] = None;
                }
            }
            Some(node)
        } else {
            None
        }
    }

    /// Assign slot to node
    pub fn assign_slot(&mut self, slot: u16, node_id: &str) {
        if slot < CLUSTER_SLOTS {
            self.slots[slot as usize] = Some(node_id.to_string());
            if let Some(node) = self.nodes.get_mut(node_id) {
                node.slots.insert(slot);
            }
        }
    }

    /// Get node responsible for slot
    pub fn get_slot_node(&self, slot: u16) -> Option<&ClusterNode> {
        if slot >= CLUSTER_SLOTS {
            return None;
        }
        self.slots[slot as usize]
            .as_ref()
            .and_then(|id| self.nodes.get(id))
    }

    /// Calculate hash slot for key
    pub fn key_hash_slot(key: &[u8]) -> u16 {
        // Check for hash tag {xxx}
        if let Some(start) = key.iter().position(|&b| b == b'{') {
            if let Some(end) = key[start + 1..].iter().position(|&b| b == b'}') {
                if end > 0 {
                    return crc16(&key[start + 1..start + 1 + end]) % CLUSTER_SLOTS;
                }
            }
        }
        crc16(key) % CLUSTER_SLOTS
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: &str) -> Option<&ClusterNode> {
        self.nodes.get(node_id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &ClusterNode> {
        self.nodes.values()
    }

    /// Get master nodes
    pub fn masters(&self) -> impl Iterator<Item = &ClusterNode> {
        self.nodes.values().filter(|n| n.flags.master)
    }

    /// Get cluster size (number of masters)
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get cluster state
    pub fn state(&self) -> &ClusterStateFlag {
        &self.state
    }

    /// Update cluster state based on slot coverage
    pub fn update_state(&mut self) {
        let covered = self.slots.iter().filter(|s| s.is_some()).count();
        self.state = if covered == CLUSTER_SLOTS as usize {
            ClusterStateFlag::Ok
        } else {
            ClusterStateFlag::Fail
        };
    }

    /// Get cluster info string
    pub fn info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("cluster_state:{}\n", match self.state {
            ClusterStateFlag::Ok => "ok",
            ClusterStateFlag::Fail => "fail",
        }));
        info.push_str(&format!("cluster_slots_assigned:{}\n",
            self.slots.iter().filter(|s| s.is_some()).count()));
        info.push_str(&format!("cluster_slots_ok:{}\n",
            self.slots.iter().filter(|s| s.is_some()).count()));
        info.push_str(&format!("cluster_slots_pfail:0\n"));
        info.push_str(&format!("cluster_slots_fail:0\n"));
        info.push_str(&format!("cluster_known_nodes:{}\n", self.nodes.len()));
        info.push_str(&format!("cluster_size:{}\n", self.size));
        info.push_str(&format!("cluster_current_epoch:{}\n", self.current_epoch));
        info.push_str(&format!("cluster_my_epoch:{}\n", self.current_epoch));
        info
    }
}

/// CRC16 implementation for hash slot calculation
fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for byte in data {
        crc = ((crc << 8) & 0xFF00) ^ CRC16_TABLE[((crc >> 8) as u8 ^ byte) as usize];
    }
    crc
}

/// CRC16 lookup table (XMODEM)
const CRC16_TABLE: [u16; 256] = [
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
    0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
    0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
    0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
    0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
    0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
    0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
    0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
    0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
    0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
    0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
    0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
    0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
    0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
    0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
    0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
    0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
    0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
    0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
    0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
    0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
    0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
    0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_slot() {
        // Simple key
        let slot = ClusterState::key_hash_slot(b"foo");
        assert!(slot < CLUSTER_SLOTS);

        // Key with hash tag
        let slot1 = ClusterState::key_hash_slot(b"user:{123}:name");
        let slot2 = ClusterState::key_hash_slot(b"user:{123}:email");
        assert_eq!(slot1, slot2);
    }

    #[test]
    fn test_cluster_state() {
        let mut cluster = ClusterState::new("node1".to_string());
        assert_eq!(cluster.size(), 0);

        // Add master node
        let node = ClusterNode {
            id: "node1".to_string(),
            addr: "127.0.0.1:6379".parse().unwrap(),
            cport: 16379,
            flags: NodeFlags {
                myself: true,
                master: true,
                ..Default::default()
            },
            master_id: None,
            slots: HashSet::new(),
            ping_sent: 0,
            pong_recv: 0,
            config_epoch: 1,
            link_state: "connected".to_string(),
        };
        cluster.add_node(node);
        assert_eq!(cluster.size(), 1);
    }
}
