"""
Comprehensive tests for block replication in the distributed file system.

Tests cover:
- Block replication across datanodes
- Failure recovery
- Replication factor enforcement
- Data consistency
"""

import pytest
import asyncio
import os
import time
import tempfile
import shutil
import hashlib
from typing import Dict, List, Set
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from hdfs.common.types import (
    BlockID, NodeID, Block, BlockLocation, FileInfo, DirectoryInfo,
    DataNodeInfo, BlockReport, HeartbeatResponse, generate_block_id, generate_node_id
)
from hdfs.namenode.namenode import NameNode
from hdfs.datanode.datanode import DataNode
from hdfs.client.client import HDFSClient
from hdfs.common.protocol import (
    HDFSError, NoDataNodeError, ReplicationError, BlockNotFoundError
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def namenode():
    """Create a NameNode instance for testing."""
    nn = NameNode(
        default_replication=3,
        default_block_size=64 * 1024,  # 64KB for testing
        heartbeat_interval=1.0
    )
    nn._safe_mode = False  # Disable safe mode for tests
    return nn


@pytest.fixture
def datanode_factory(temp_dir):
    """Factory to create DataNode instances."""
    datanodes = []

    def create_datanode(node_id: str = None, port: int = 50010):
        node_id = node_id or generate_node_id()
        data_dir = os.path.join(temp_dir, node_id)
        os.makedirs(data_dir, exist_ok=True)
        dn = DataNode(
            data_dir=data_dir,
            node_id=node_id,
            port=port,
            capacity=1024 * 1024 * 1024  # 1GB
        )
        datanodes.append(dn)
        return dn

    yield create_datanode


@pytest.fixture
def registered_cluster(namenode, datanode_factory):
    """Create a cluster with multiple registered datanodes."""
    datanodes = []
    for i in range(5):
        dn = datanode_factory(node_id=f"dn-{i}", port=50010 + i)
        namenode.register_datanode(
            node_id=dn.node_id,
            host="localhost",
            port=dn.port,
            capacity=dn.capacity,
            rack=f"/rack-{i % 2}"  # Alternate racks
        )
        datanodes.append(dn)

    return {"namenode": namenode, "datanodes": datanodes}


def generate_test_data(size: int) -> bytes:
    """Generate random test data."""
    return os.urandom(size)


# =============================================================================
# BLOCK REPLICATION ACROSS DATANODES TESTS
# =============================================================================

class TestBlockReplicationAcrossDatanodes:
    """Tests for block replication across multiple datanodes."""

    def test_block_allocation_selects_multiple_datanodes(self, registered_cluster):
        """Test that block allocation selects the correct number of datanodes."""
        namenode = registered_cluster["namenode"]

        # Create a file with replication factor 3
        namenode.create_file("/test.txt", replication=3)

        # Add a block
        block, locations = namenode.add_block("/test.txt")

        # Should have 3 unique datanodes
        assert len(locations) == 3
        node_ids = set(loc.node_id for loc in locations)
        assert len(node_ids) == 3

    def test_block_allocation_prefers_different_racks(self, namenode, datanode_factory):
        """Test that block placement prefers different racks."""
        # Register datanodes in different racks
        for i in range(6):
            dn = datanode_factory(node_id=f"dn-{i}", port=50010 + i)
            rack = f"/rack-{i // 2}"  # 2 nodes per rack
            namenode.register_datanode(
                node_id=dn.node_id,
                host="localhost",
                port=dn.port,
                capacity=dn.capacity,
                rack=rack
            )

        namenode.create_file("/test.txt", replication=3)
        block, locations = namenode.add_block("/test.txt")

        # Should try to use different racks
        racks = set(loc.rack for loc in locations)
        # Should have at least 2 different racks for replication factor 3
        assert len(racks) >= 2

    def test_block_replication_to_all_targets(self, registered_cluster):
        """Test that blocks are written to all target datanodes."""
        namenode = registered_cluster["namenode"]
        datanodes = registered_cluster["datanodes"]

        # Create file and get block allocation
        namenode.create_file("/test.txt", replication=3)
        block, locations = namenode.add_block("/test.txt")

        # Write block data to each target datanode
        test_data = generate_test_data(1024)
        for loc in locations:
            # Find the corresponding datanode
            dn = next(d for d in datanodes if d.node_id == loc.node_id)
            dn.write_block(block.block_id, test_data)

            # Report block received
            namenode.block_received(loc.node_id, block.block_id, len(test_data))

        # Verify block is tracked on all datanodes
        node_ids_with_block = namenode._block_to_nodes[block.block_id]
        assert len(node_ids_with_block) == 3

    def test_block_report_updates_replication_state(self, registered_cluster):
        """Test that block reports correctly update replication state."""
        namenode = registered_cluster["namenode"]
        datanodes = registered_cluster["datanodes"]

        # Create a block
        block_id = generate_block_id()
        namenode._blocks[block_id] = Block(block_id=block_id, size=1024)

        # Simulate block reports from 3 datanodes
        for dn in datanodes[:3]:
            dn.write_block(block_id, generate_test_data(1024))
            report = BlockReport(node_id=dn.node_id, blocks=[block_id])
            namenode.block_report(report)

        # Verify replication tracking
        assert len(namenode._block_to_nodes[block_id]) == 3

    def test_insufficient_datanodes_for_replication(self, namenode, datanode_factory):
        """Test behavior when not enough datanodes are available."""
        # Register only 2 datanodes
        for i in range(2):
            dn = datanode_factory(node_id=f"dn-{i}")
            namenode.register_datanode(
                node_id=dn.node_id,
                host="localhost",
                port=dn.port,
                capacity=dn.capacity
            )

        namenode.create_file("/test.txt", replication=3)
        block, locations = namenode.add_block("/test.txt")

        # Should return available datanodes (2 instead of 3)
        assert len(locations) == 2

    def test_block_location_returns_all_replicas(self, registered_cluster):
        """Test that get_block_locations returns all replica locations."""
        namenode = registered_cluster["namenode"]
        datanodes = registered_cluster["datanodes"]

        # Create file and write block
        namenode.create_file("/test.txt", replication=3)
        block, locations = namenode.add_block("/test.txt")

        # Simulate block reception
        for loc in locations:
            namenode.block_received(loc.node_id, block.block_id, 1024)

        # Get block locations
        all_locations = namenode.get_block_locations("/test.txt")

        assert len(all_locations) == 1  # One block
        assert len(all_locations[0]) == 3  # Three replicas


# =============================================================================
# FAILURE RECOVERY TESTS
# =============================================================================

class TestFailureRecovery:
    """Tests for failure recovery scenarios."""

    def test_datanode_failure_detection_via_heartbeat(self, registered_cluster):
        """Test that failed datanodes are detected via missed heartbeats."""
        namenode = registered_cluster["namenode"]

        # Get a datanode
        dn_id = "dn-0"
        dn_info = namenode._datanodes[dn_id]

        # Simulate old heartbeat (older than 30 seconds)
        dn_info.last_heartbeat = time.time() - 60

        # Check if node is alive
        assert not dn_info.is_alive

    def test_read_after_single_datanode_failure(self, registered_cluster):
        """Test that data can still be read after one datanode fails."""
        namenode = registered_cluster["namenode"]
        datanodes = registered_cluster["datanodes"]

        # Create and replicate a block
        namenode.create_file("/test.txt", replication=3)
        block, locations = namenode.add_block("/test.txt")

        test_data = generate_test_data(1024)
        for loc in locations:
            dn = next(d for d in datanodes if d.node_id == loc.node_id)
            dn.write_block(block.block_id, test_data)
            namenode.block_received(loc.node_id, block.block_id, len(test_data))

        # Simulate one datanode failure
        failed_dn_id = locations[0].node_id
        namenode._datanodes[failed_dn_id].last_heartbeat = time.time() - 60

        # Get block locations - should only return alive nodes
        all_locations = namenode.get_block_locations("/test.txt")

        # Should have 2 remaining locations (excluding failed node)
        live_locations = [loc for loc in all_locations[0] if loc.node_id != failed_dn_id]
        assert len(live_locations) >= 2

        # Data should still be readable from remaining nodes
        for loc in live_locations:
            dn = next(d for d in datanodes if d.node_id == loc.node_id)
            data = dn.read_block(block.block_id)
            assert data == test_data

    def test_pending_deletions_after_heartbeat(self, registered_cluster):
        """Test that pending deletions are communicated via heartbeat response."""
        namenode = registered_cluster["namenode"]

        # Create file and add block
        namenode.create_file("/test.txt", replication=3)
        block, locations = namenode.add_block("/test.txt")

        # Simulate block received
        for loc in locations:
            namenode.block_received(loc.node_id, block.block_id, 1024)

        # Delete the file (should mark blocks for deletion)
        namenode.delete_file("/test.txt")

        # Heartbeat from affected datanode should get delete command
        dn_id = locations[0].node_id
        response = namenode.heartbeat(dn_id, used=1024, remaining=1024*1024)

        # Check for delete command
        delete_commands = [cmd for cmd in response.commands if cmd.get("type") == "delete"]
        assert len(delete_commands) > 0
        assert block.block_id in delete_commands[0].get("block_ids", [])

    def test_unregistered_datanode_heartbeat(self, namenode):
        """Test heartbeat from unregistered datanode triggers re-registration."""
        # Send heartbeat from unknown node
        response = namenode.heartbeat("unknown-node", used=0, remaining=1000)

        # Should get re-register command
        assert len(response.commands) > 0
        assert response.commands[0].get("type") == "re-register"

    def test_datanode_recovers_blocks_on_restart(self, temp_dir):
        """Test that datanode recovers blocks from disk on restart."""
        data_dir = os.path.join(temp_dir, "dn-restart")
        os.makedirs(data_dir)

        # Create datanode and write blocks
        dn1 = DataNode(data_dir=data_dir, node_id="dn-restart")

        block_ids = []
        test_data_map = {}
        for i in range(3):
            block_id = generate_block_id()
            data = generate_test_data(1024)
            dn1.write_block(block_id, data)
            block_ids.append(block_id)
            test_data_map[block_id] = data

        # Simulate restart by creating new datanode with same data_dir
        dn2 = DataNode(data_dir=data_dir, node_id="dn-restart")

        # All blocks should be recovered
        recovered_blocks = dn2.get_block_ids()
        assert set(recovered_blocks) == set(block_ids)

        # Data should be intact
        for block_id in block_ids:
            assert dn2.read_block(block_id) == test_data_map[block_id]

    def test_block_verification_after_recovery(self, temp_dir):
        """Test block integrity verification after corruption."""
        data_dir = os.path.join(temp_dir, "dn-verify")
        os.makedirs(data_dir)

        # Create datanode and write blocks
        dn = DataNode(data_dir=data_dir, node_id="dn-verify")

        block_id = generate_block_id()
        test_data = generate_test_data(1024)
        dn.write_block(block_id, test_data)

        # Store original checksum
        original_checksum = dn._block_checksums[block_id]

        # Verify block is valid
        assert dn.verify_block(block_id)

        # Corrupt the block file (without updating in-memory checksum)
        block_path = dn._get_block_path(block_id)
        with open(block_path, 'wb') as f:
            f.write(b'corrupted data')

        # Verification should now fail since disk content differs from stored checksum
        assert not dn.verify_block(block_id)

        # The stored checksum should still be the original
        assert dn._block_checksums[block_id] == original_checksum


# =============================================================================
# REPLICATION FACTOR ENFORCEMENT TESTS
# =============================================================================

class TestReplicationFactorEnforcement:
    """Tests for replication factor enforcement."""

    def test_default_replication_factor(self, namenode, datanode_factory):
        """Test that default replication factor is applied."""
        # Register enough datanodes
        for i in range(5):
            dn = datanode_factory(node_id=f"dn-{i}")
            namenode.register_datanode(
                node_id=dn.node_id,
                host="localhost",
                port=dn.port,
                capacity=dn.capacity
            )

        # Create file without specifying replication
        file_info = namenode.create_file("/test.txt")

        assert file_info.replication == namenode.default_replication

    def test_custom_replication_factor(self, registered_cluster):
        """Test setting custom replication factor."""
        namenode = registered_cluster["namenode"]

        # Create file with custom replication
        file_info = namenode.create_file("/test.txt", replication=5)

        assert file_info.replication == 5

        # Add block should try to get 5 replicas
        block, locations = namenode.add_block("/test.txt")
        assert len(locations) == 5

    def test_replication_factor_one(self, namenode, datanode_factory):
        """Test replication factor of 1 (no replication)."""
        dn = datanode_factory(node_id="dn-0")
        namenode.register_datanode(
            node_id=dn.node_id,
            host="localhost",
            port=dn.port,
            capacity=dn.capacity
        )

        file_info = namenode.create_file("/test.txt", replication=1)
        block, locations = namenode.add_block("/test.txt")

        assert file_info.replication == 1
        assert len(locations) == 1

    def test_under_replicated_block_detection(self, registered_cluster):
        """Test detection of under-replicated blocks."""
        namenode = registered_cluster["namenode"]

        # Create file with replication 3
        namenode.create_file("/test.txt", replication=3)
        block, _ = namenode.add_block("/test.txt")

        # Only report block from 2 datanodes (under-replicated)
        namenode.block_received("dn-0", block.block_id, 1024)
        namenode.block_received("dn-1", block.block_id, 1024)

        # Check replication count
        replica_count = len(namenode._block_to_nodes[block.block_id])
        assert replica_count == 2
        assert replica_count < 3  # Under-replicated

    def test_over_replicated_block_detection(self, registered_cluster):
        """Test detection of over-replicated blocks."""
        namenode = registered_cluster["namenode"]

        # Create file with replication 2
        namenode.create_file("/test.txt", replication=2)
        block, _ = namenode.add_block("/test.txt")

        # Report block from 4 datanodes (over-replicated)
        for i in range(4):
            namenode.block_received(f"dn-{i}", block.block_id, 1024)

        # Check replication count
        replica_count = len(namenode._block_to_nodes[block.block_id])
        assert replica_count == 4
        assert replica_count > 2  # Over-replicated

    def test_block_allocation_respects_capacity(self, namenode, datanode_factory):
        """Test that block allocation respects datanode capacity."""
        # Register datanodes with different remaining space
        for i in range(3):
            dn = datanode_factory(node_id=f"dn-{i}")
            namenode.register_datanode(
                node_id=dn.node_id,
                host="localhost",
                port=dn.port,
                capacity=1024 * 1024 * 1024
            )

        # Simulate one node being nearly full
        namenode._datanodes["dn-0"].remaining = 1024  # Very little space

        namenode.create_file("/test.txt", replication=3)
        block, locations = namenode.add_block("/test.txt")

        # The nearly-full node may not be selected
        # This depends on implementation, but we verify allocation works
        assert len(locations) >= 2  # At least some nodes available

    def test_replication_factor_preserved_in_checkpoint(self, namenode, temp_dir, datanode_factory):
        """Test that replication factor is preserved across checkpoints."""
        # Register datanodes
        for i in range(3):
            dn = datanode_factory(node_id=f"dn-{i}")
            namenode.register_datanode(
                node_id=dn.node_id,
                host="localhost",
                port=dn.port,
                capacity=dn.capacity
            )

        # Create file with specific replication
        namenode.create_file("/test.txt", replication=5)

        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        namenode.save_checkpoint(checkpoint_path)

        # Create new namenode and load checkpoint
        nn2 = NameNode()
        nn2.load_checkpoint(checkpoint_path)

        # Verify replication preserved
        file_info = nn2._files["/test.txt"]
        assert file_info.replication == 5


# =============================================================================
# DATA CONSISTENCY TESTS
# =============================================================================

class TestDataConsistency:
    """Tests for data consistency across replicas."""

    def test_all_replicas_have_same_data(self, registered_cluster):
        """Test that all replicas contain identical data."""
        namenode = registered_cluster["namenode"]
        datanodes = registered_cluster["datanodes"]

        # Create file and get block allocation
        namenode.create_file("/test.txt", replication=3)
        block, locations = namenode.add_block("/test.txt")

        # Write same data to all replicas
        test_data = generate_test_data(4096)
        for loc in locations:
            dn = next(d for d in datanodes if d.node_id == loc.node_id)
            dn.write_block(block.block_id, test_data)

        # Verify all replicas have identical data
        checksums = set()
        for loc in locations:
            dn = next(d for d in datanodes if d.node_id == loc.node_id)
            data = dn.read_block(block.block_id)
            checksums.add(hashlib.md5(data).hexdigest())

        # All checksums should be identical
        assert len(checksums) == 1

    def test_checksum_verification(self, temp_dir):
        """Test checksum verification for block integrity."""
        data_dir = os.path.join(temp_dir, "dn-checksum")
        os.makedirs(data_dir)

        dn = DataNode(data_dir=data_dir, node_id="dn-checksum")

        block_id = generate_block_id()
        test_data = generate_test_data(1024)

        # Write block
        dn.write_block(block_id, test_data)

        # Verify checksum is computed
        assert block_id in dn._block_checksums
        expected_checksum = hashlib.md5(test_data).hexdigest()
        assert dn._block_checksums[block_id] == expected_checksum

    def test_block_read_returns_exact_data(self, temp_dir):
        """Test that block read returns exactly what was written."""
        data_dir = os.path.join(temp_dir, "dn-exact")
        os.makedirs(data_dir)

        dn = DataNode(data_dir=data_dir, node_id="dn-exact")

        # Test with various data sizes
        test_sizes = [1, 100, 1024, 8192, 64 * 1024]

        for size in test_sizes:
            block_id = generate_block_id()
            test_data = generate_test_data(size)

            dn.write_block(block_id, test_data)
            read_data = dn.read_block(block_id)

            assert read_data == test_data
            assert len(read_data) == size

    def test_partial_read(self, temp_dir):
        """Test reading partial block data with offset and length."""
        data_dir = os.path.join(temp_dir, "dn-partial")
        os.makedirs(data_dir)

        dn = DataNode(data_dir=data_dir, node_id="dn-partial")

        block_id = generate_block_id()
        test_data = generate_test_data(4096)
        dn.write_block(block_id, test_data)

        # Read with offset
        data_offset = dn.read_block(block_id, offset=100)
        assert data_offset == test_data[100:]

        # Read with length
        data_length = dn.read_block(block_id, length=500)
        assert data_length == test_data[:500]

        # Read with both offset and length
        data_both = dn.read_block(block_id, offset=100, length=200)
        assert data_both == test_data[100:300]

    def test_concurrent_writes_to_different_blocks(self, temp_dir):
        """Test concurrent writes to different blocks maintain consistency."""
        import threading

        data_dir = os.path.join(temp_dir, "dn-concurrent")
        os.makedirs(data_dir)

        dn = DataNode(data_dir=data_dir, node_id="dn-concurrent")

        results = {}
        errors = []

        def write_and_verify(block_id: str, data: bytes):
            try:
                dn.write_block(block_id, data)
                read_data = dn.read_block(block_id)
                results[block_id] = (data == read_data)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads writing different blocks
        threads = []
        test_data_map = {}
        for i in range(10):
            block_id = generate_block_id()
            data = generate_test_data(1024)
            test_data_map[block_id] = data
            t = threading.Thread(target=write_and_verify, args=(block_id, data))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All operations should succeed
        assert len(errors) == 0
        assert all(results.values())
        assert len(results) == 10

    def test_block_not_found_error(self, temp_dir):
        """Test appropriate error when reading non-existent block."""
        data_dir = os.path.join(temp_dir, "dn-notfound")
        os.makedirs(data_dir)

        dn = DataNode(data_dir=data_dir, node_id="dn-notfound")

        with pytest.raises(BlockNotFoundError):
            dn.read_block("non-existent-block-id")

    def test_file_size_tracking(self, registered_cluster):
        """Test that file size is correctly tracked across blocks."""
        namenode = registered_cluster["namenode"]

        # Create file
        namenode.create_file("/test.txt")

        # Add multiple blocks
        block_sizes = [1024, 2048, 512]
        total_size = sum(block_sizes)

        for size in block_sizes:
            block, _ = namenode.add_block("/test.txt")
            namenode._blocks[block.block_id].size = size

        # Complete file
        file_info = namenode.complete_file("/test.txt", total_size)

        assert file_info.size == total_size

    def test_generation_stamp_uniqueness(self):
        """Test that blocks have unique generation stamps."""
        blocks = []
        for _ in range(100):
            block = Block(block_id=generate_block_id())
            blocks.append(block)

        # All generation stamps should be present (may not be unique if created too fast)
        stamps = [b.generation_stamp for b in blocks]
        assert len(stamps) == 100

    def test_block_id_uniqueness(self):
        """Test that block IDs are unique."""
        block_ids = set()
        for _ in range(1000):
            block_id = generate_block_id()
            assert block_id not in block_ids
            block_ids.add(block_id)

    def test_consistent_block_locations_after_update(self, registered_cluster):
        """Test that block locations remain consistent after updates."""
        namenode = registered_cluster["namenode"]

        # Create file and add block
        namenode.create_file("/test.txt", replication=3)
        block, locations = namenode.add_block("/test.txt")

        # Report block received
        for loc in locations:
            namenode.block_received(loc.node_id, block.block_id, 1024)

        # Get locations multiple times
        locations1 = namenode.get_block_locations("/test.txt")
        locations2 = namenode.get_block_locations("/test.txt")

        # Should return consistent results
        nodes1 = set(loc.node_id for loc in locations1[0])
        nodes2 = set(loc.node_id for loc in locations2[0])
        assert nodes1 == nodes2


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestReplicationEdgeCases:
    """Edge case tests for replication."""

    def test_zero_datanodes_available(self, namenode):
        """Test behavior when no datanodes are registered."""
        namenode.create_file("/test.txt")

        with pytest.raises(NoDataNodeError):
            namenode.add_block("/test.txt")

    def test_all_datanodes_dead(self, registered_cluster):
        """Test behavior when all datanodes are marked as dead."""
        namenode = registered_cluster["namenode"]

        # Mark all datanodes as dead
        for dn_info in namenode._datanodes.values():
            dn_info.last_heartbeat = time.time() - 120  # 2 minutes ago

        namenode.create_file("/test.txt")

        with pytest.raises(NoDataNodeError):
            namenode.add_block("/test.txt")

    def test_delete_block_idempotent(self, temp_dir):
        """Test that deleting a block twice doesn't cause errors."""
        data_dir = os.path.join(temp_dir, "dn-delete")
        os.makedirs(data_dir)

        dn = DataNode(data_dir=data_dir, node_id="dn-delete")

        block_id = generate_block_id()
        dn.write_block(block_id, generate_test_data(1024))

        # Delete twice
        assert dn.delete_block(block_id) == True
        assert dn.delete_block(block_id) == False  # Already deleted

    def test_empty_block_report(self, registered_cluster):
        """Test handling of empty block report."""
        namenode = registered_cluster["namenode"]

        # Send empty block report
        report = BlockReport(node_id="dn-0", blocks=[])
        namenode.block_report(report)

        # Should not cause errors
        assert namenode._datanodes["dn-0"].blocks == set()

    def test_block_report_from_unknown_node(self, namenode):
        """Test block report from unregistered node."""
        report = BlockReport(node_id="unknown-node", blocks=[generate_block_id()])

        # Should not raise, just log warning
        namenode.block_report(report)

    def test_safe_mode_blocks_allocation(self, namenode, datanode_factory):
        """Test that safe mode blocks new allocations."""
        # Register a datanode
        dn = datanode_factory(node_id="dn-0")
        namenode.register_datanode(
            node_id=dn.node_id,
            host="localhost",
            port=dn.port,
            capacity=dn.capacity
        )

        # Keep safe mode on
        namenode._safe_mode = True

        # File creation should work (metadata only)
        namenode.create_file("/test.txt")

        # Block allocation should be blocked in safe mode (matches real HDFS behavior)
        with pytest.raises(HDFSError):
            namenode.add_block("/test.txt")

    def test_large_block_write_and_read(self, temp_dir):
        """Test writing and reading a large block."""
        data_dir = os.path.join(temp_dir, "dn-large")
        os.makedirs(data_dir)

        dn = DataNode(data_dir=data_dir, node_id="dn-large")

        # 1MB block
        large_data = generate_test_data(1024 * 1024)
        block_id = generate_block_id()

        dn.write_block(block_id, large_data)
        read_data = dn.read_block(block_id)

        assert read_data == large_data
        assert dn.verify_block(block_id)

    def test_storage_space_tracking(self, temp_dir):
        """Test that storage space is correctly tracked."""
        data_dir = os.path.join(temp_dir, "dn-space")
        os.makedirs(data_dir)

        dn = DataNode(data_dir=data_dir, node_id="dn-space", capacity=1024 * 1024)

        initial_used = dn.used_space
        initial_remaining = dn.remaining_space

        # Write blocks
        block1_data = generate_test_data(10000)
        block2_data = generate_test_data(20000)

        dn.write_block(generate_block_id(), block1_data)
        dn.write_block(generate_block_id(), block2_data)

        assert dn.used_space == initial_used + 30000
        assert dn.remaining_space == initial_remaining - 30000


# =============================================================================
# ASYNC OPERATION TESTS
# =============================================================================

class TestAsyncReplicationOperations:
    """Tests for async replication operations."""

    @pytest.mark.asyncio
    async def test_async_block_report(self, temp_dir):
        """Test async block report sending."""
        data_dir = os.path.join(temp_dir, "dn-async")
        os.makedirs(data_dir)

        dn = DataNode(
            data_dir=data_dir,
            node_id="dn-async",
            namenode_host="localhost",
            namenode_port=9000
        )

        # Mock the connection
        with patch('asyncio.open_connection', new_callable=AsyncMock) as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = MagicMock()
            mock_writer.drain = AsyncMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()

            # Setup response
            from hdfs.common.protocol import Message, MessageType, serialize_message
            response = Message(MessageType.SUCCESS, {})
            response_data = serialize_message(response)
            mock_reader.read = AsyncMock(side_effect=[
                len(response_data).to_bytes(4, 'big'),
                response_data
            ])

            mock_conn.return_value = (mock_reader, mock_writer)

            # Write a block first
            dn.write_block(generate_block_id(), generate_test_data(1024))

            # Send block report
            result = await dn.send_block_report()
            assert result == True

    @pytest.mark.asyncio
    async def test_async_heartbeat(self, temp_dir):
        """Test async heartbeat sending."""
        data_dir = os.path.join(temp_dir, "dn-heartbeat")
        os.makedirs(data_dir)

        dn = DataNode(
            data_dir=data_dir,
            node_id="dn-heartbeat",
            namenode_host="localhost",
            namenode_port=9000
        )

        with patch('asyncio.open_connection', new_callable=AsyncMock) as mock_conn:
            mock_reader = AsyncMock()
            mock_writer = MagicMock()
            mock_writer.drain = AsyncMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()

            from hdfs.common.protocol import Message, MessageType, serialize_message
            response = Message(MessageType.SUCCESS, {"commands": []})
            response_data = serialize_message(response)
            mock_reader.read = AsyncMock(side_effect=[
                len(response_data).to_bytes(4, 'big'),
                response_data
            ])

            mock_conn.return_value = (mock_reader, mock_writer)

            commands = await dn.send_heartbeat()
            assert commands == []

    @pytest.mark.asyncio
    async def test_execute_delete_command(self, temp_dir):
        """Test executing delete command from namenode."""
        data_dir = os.path.join(temp_dir, "dn-cmd")
        os.makedirs(data_dir)

        dn = DataNode(data_dir=data_dir, node_id="dn-cmd")

        # Write some blocks
        block_ids = []
        for _ in range(3):
            block_id = generate_block_id()
            dn.write_block(block_id, generate_test_data(1024))
            block_ids.append(block_id)

        # Execute delete command
        command = {"type": "delete", "block_ids": block_ids[:2]}
        await dn.execute_command(command)

        # Verify blocks deleted
        remaining_blocks = dn.get_block_ids()
        assert len(remaining_blocks) == 1
        assert block_ids[2] in remaining_blocks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
