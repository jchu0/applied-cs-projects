"""Unit tests for the NameNode."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock

from fixtures import (
    create_mock_block, create_mock_file_info, create_mock_datanode_info,
    generate_test_file_path, DataNodeMock, assert_files_equal
)

from hdfs.namenode.namenode import NameNode
from hdfs.common.types import (
    BlockID, NodeID, Block, FileInfo, DirectoryInfo,
    DataNodeInfo, generate_block_id
)
from hdfs.common.protocol import (
    HDFSError, FileNotFoundError, FileExistsError,
    DirectoryNotEmptyError, NoDataNodeError
)


class TestNameNode:
    """Test cases for NameNode functionality."""

    def test_initialization(self):
        """Test NameNode initialization."""
        nn = NameNode(
            default_replication=3,
            default_block_size=128 * 1024 * 1024,
            heartbeat_interval=3.0
        )

        assert nn.default_replication == 3
        assert nn.default_block_size == 128 * 1024 * 1024
        assert nn.heartbeat_interval == 3.0
        assert nn._safe_mode is True
        assert "/" in nn._directories
        assert len(nn._files) == 0
        assert len(nn._datanodes) == 0

    def test_create_file(self):
        """Test file creation."""
        nn = NameNode()
        nn._safe_mode = False  # Disable safe mode for testing

        # Create a file
        file_info = nn.create_file("/test.txt", replication=2, block_size=64 * 1024 * 1024)

        assert file_info.path == "/test.txt"
        assert file_info.replication == 2
        assert file_info.block_size == 64 * 1024 * 1024
        assert "/test.txt" in nn._files

        # Try to create existing file without overwrite
        with pytest.raises(FileExistsError):
            nn.create_file("/test.txt")

        # Create with overwrite
        file_info2 = nn.create_file("/test.txt", overwrite=True)
        assert file_info2.path == "/test.txt"

    def test_create_file_in_nonexistent_directory(self):
        """Test creating file in non-existent directory."""
        nn = NameNode()
        nn._safe_mode = False  # Disable safe mode for testing

        with pytest.raises(FileNotFoundError):
            nn.create_file("/nonexistent/test.txt")

    def test_delete_file(self):
        """Test file deletion."""
        nn = NameNode()
        nn._safe_mode = False  # Disable safe mode for testing

        # Create and delete a file
        nn.create_file("/test.txt")
        assert "/test.txt" in nn._files

        nn.delete_file("/test.txt")
        assert "/test.txt" not in nn._files

        # Try to delete non-existent file
        with pytest.raises(FileNotFoundError):
            nn.delete_file("/nonexistent.txt")

    def test_create_directory(self):
        """Test directory creation."""
        nn = NameNode()

        # Create directory
        nn.create_directory("/test_dir")
        assert "/test_dir" in nn._directories

        # Create nested directory
        nn.create_directory("/test_dir/nested")
        assert "/test_dir/nested" in nn._directories

        # Try to create existing directory
        with pytest.raises(FileExistsError):
            nn.create_directory("/test_dir")

    def test_delete_directory(self):
        """Test directory deletion."""
        nn = NameNode()

        # Create and delete empty directory
        nn.create_directory("/test_dir")
        nn.delete_directory("/test_dir")
        assert "/test_dir" not in nn._directories

        # Try to delete non-empty directory
        nn.create_directory("/parent")
        nn.create_directory("/parent/child")

        with pytest.raises(DirectoryNotEmptyError):
            nn.delete_directory("/parent")

        # Delete child first, then parent
        nn.delete_directory("/parent/child")
        nn.delete_directory("/parent")
        assert "/parent" not in nn._directories

    def test_list_directory(self):
        """Test directory listing."""
        nn = NameNode()
        nn._safe_mode = False  # Disable safe mode for testing

        # Create some files and directories
        nn.create_directory("/dir1")
        nn.create_directory("/dir2")
        nn.create_file("/file1.txt")
        nn.create_file("/file2.txt")

        # List root directory
        contents = nn.list_directory("/")
        assert "dir1" in contents
        assert "dir2" in contents
        assert "file1.txt" in contents
        assert "file2.txt" in contents

        # List empty directory
        empty_contents = nn.list_directory("/dir1")
        assert len(empty_contents) == 0

    def test_register_datanode(self):
        """Test DataNode registration."""
        nn = NameNode()

        # Register DataNodes
        nn.register_datanode("dn1", "localhost", 50010)
        nn.register_datanode("dn2", "localhost", 50020)

        assert "dn1" in nn._datanodes
        assert "dn2" in nn._datanodes

        dn1_info = nn._datanodes["dn1"]
        assert dn1_info.node_id == "dn1"
        assert dn1_info.host == "localhost"
        assert dn1_info.port == 50010

    def test_handle_heartbeat(self):
        """Test heartbeat handling."""
        nn = NameNode()

        # Register DataNode
        nn.register_datanode("dn1", "localhost", 50010)

        # Send heartbeat
        response = nn.handle_heartbeat(
            "dn1",
            capacity=1000000,
            used=500000,
            remaining=500000
        )

        assert response is not None
        dn_info = nn._datanodes["dn1"]
        assert dn_info.capacity == 1000000
        assert dn_info.used == 500000
        assert dn_info.remaining == 500000

    def test_handle_block_report(self):
        """Test block report handling."""
        nn = NameNode()

        # Register DataNode
        nn.register_datanode("dn1", "localhost", 50010)

        # Create some blocks
        block_ids = [generate_block_id() for _ in range(5)]

        # Send block report
        nn.handle_block_report("dn1", block_ids)

        # Check block mappings
        for block_id in block_ids:
            assert "dn1" in nn._block_to_nodes[block_id]

    def test_allocate_blocks(self):
        """Test block allocation."""
        nn = NameNode()

        # Register DataNodes
        for i in range(3):
            nn.register_datanode(f"dn{i}", "localhost", 50010 + i)

        # Exit safe mode for allocation
        nn._safe_mode = False

        # Allocate blocks
        blocks = nn.allocate_blocks("/test.txt", 3, 1024)

        assert len(blocks) == 3
        for block in blocks:
            assert block.size == 1024
            assert block.block_id is not None

    def test_get_block_locations(self):
        """Test getting block locations."""
        nn = NameNode()

        # Register DataNodes
        nn.register_datanode("dn1", "localhost", 50010)
        nn.register_datanode("dn2", "localhost", 50020)

        # Create a block and add locations
        block_id = generate_block_id()
        nn._block_to_nodes[block_id].add("dn1")
        nn._block_to_nodes[block_id].add("dn2")

        # Get locations
        locations = nn.get_block_locations(block_id)

        assert len(locations) == 2
        node_ids = [loc.node_id for loc in locations]
        assert "dn1" in node_ids
        assert "dn2" in node_ids

    def test_replication_monitoring(self):
        """Test under-replicated block detection."""
        nn = NameNode(default_replication=3)
        nn._safe_mode = False  # Disable safe mode for testing

        # Register DataNodes
        for i in range(5):
            nn.register_datanode(f"dn{i}", "localhost", 50010 + i)

        # Create a file with blocks
        nn.create_file("/test.txt", replication=3)
        block = create_mock_block()
        nn._blocks[block.block_id] = block

        # Add block to the file's block list
        nn._files["/test.txt"].blocks.append(block.block_id)

        # Add only 2 replicas (under-replicated)
        nn._block_to_nodes[block.block_id].add("dn1")
        nn._block_to_nodes[block.block_id].add("dn2")

        # Check for under-replication
        under_replicated = nn._check_under_replicated_blocks()
        assert block.block_id in under_replicated

        # Add third replica
        nn._block_to_nodes[block.block_id].add("dn3")

        # Should no longer be under-replicated
        under_replicated = nn._check_under_replicated_blocks()
        assert block.block_id not in under_replicated

    def test_safe_mode(self):
        """Test safe mode behavior."""
        nn = NameNode()

        # Initially in safe mode
        assert nn._safe_mode is True

        # Register DataNodes
        for i in range(3):
            nn.register_datanode(f"dn{i}", "localhost", 50010 + i)

        # Create some blocks
        block_ids = []
        for i in range(10):
            block_id = generate_block_id()
            block = Block(block_id=block_id, size=1024, generation_stamp=1)
            nn._blocks[block_id] = block
            block_ids.append(block_id)

        # Report 99% of blocks (should stay in safe mode)
        for i in range(9):
            nn._block_to_nodes[block_ids[i]].add("dn1")

        nn._check_safe_mode_exit()
        assert nn._safe_mode is True

        # Report the last block (should exit safe mode)
        nn._block_to_nodes[block_ids[9]].add("dn1")
        nn._check_safe_mode_exit()
        assert nn._safe_mode is False

    def test_get_file_info(self):
        """Test getting file information."""
        nn = NameNode()
        nn._safe_mode = False  # Disable safe mode for testing

        # Create a file
        nn.create_file("/test.txt", replication=2, block_size=64 * 1024 * 1024)

        # Get file info
        file_info = nn.get_file_info("/test.txt")
        assert file_info is not None
        assert file_info.path == "/test.txt"
        assert file_info.replication == 2

        # Get non-existent file
        with pytest.raises(FileNotFoundError):
            nn.get_file_info("/nonexistent.txt")

    def test_rename_file(self):
        """Test file renaming."""
        nn = NameNode()
        nn._safe_mode = False  # Disable safe mode for testing

        # Create a file
        nn.create_file("/old_name.txt")

        # Rename file
        nn.rename("/old_name.txt", "/new_name.txt")

        assert "/old_name.txt" not in nn._files
        assert "/new_name.txt" in nn._files

        # Try to rename to existing file
        nn.create_file("/another.txt")
        with pytest.raises(FileExistsError):
            nn.rename("/new_name.txt", "/another.txt")

    def test_get_statistics(self):
        """Test filesystem statistics."""
        nn = NameNode()
        nn._safe_mode = False  # Disable safe mode for testing

        # Register DataNodes
        for i in range(3):
            nn.register_datanode(
                f"dn{i}", "localhost", 50010 + i,
                capacity=1000000,
                used=300000,
                remaining=700000
            )

        # Create some files
        for i in range(5):
            nn.create_file(f"/file{i}.txt")

        stats = nn.get_statistics()
        assert stats['total_files'] == 5
        assert stats['total_directories'] == 1  # Just root
        assert stats['total_datanodes'] == 3
        assert stats['total_capacity'] == 3000000
        assert stats['total_used'] == 900000

    def test_checkpoint(self):
        """Test namespace checkpointing."""
        nn = NameNode()
        nn._safe_mode = False  # Disable safe mode for testing

        # Create some state
        nn.create_directory("/test_dir")
        nn.create_file("/test.txt")
        nn.register_datanode("dn1", "localhost", 50010)

        # Save checkpoint
        import tempfile
        with tempfile.NamedTemporaryFile() as f:
            nn.save_checkpoint(f.name)

            # Create new NameNode and load checkpoint
            nn2 = NameNode()
            nn2.load_checkpoint(f.name)

            # Verify state was restored
            assert "/test_dir" in nn2._directories
            assert "/test.txt" in nn2._files
            assert "dn1" in nn2._datanodes