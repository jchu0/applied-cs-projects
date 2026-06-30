"""Unit tests for the DataNode."""

import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock

from fixtures import (
    temp_directory, generate_test_data, create_mock_block
)

from hdfs.datanode.datanode import DataNode
from hdfs.common.types import BlockID, Block, generate_block_id
from hdfs.common.protocol import HDFSError


class TestDataNode:
    """Test cases for DataNode functionality."""

    def test_initialization(self):
        """Test DataNode initialization."""
        with temp_directory() as data_dir:
            dn = DataNode(
                node_id="test-dn",
                data_dir=data_dir,
                namenode_host="localhost",
                namenode_port=9000
            )

            assert dn.node_id == "test-dn"
            assert dn.data_dir == data_dir
            assert dn.namenode_host == "localhost"
            assert dn.namenode_port == 9000
            assert os.path.exists(data_dir)

    def test_store_block(self):
        """Test block storage."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Store a block
            block_id = generate_block_id()
            data = generate_test_data(1024)

            dn.store_block(block_id, data)

            # Verify block is stored
            assert block_id in dn._blocks
            assert dn._blocks[block_id] == len(data)

            # Verify file exists (in blocks subdirectory)
            block_path = os.path.join(data_dir, "blocks", block_id)
            assert os.path.exists(block_path)

            # Read and verify data
            with open(block_path, 'rb') as f:
                stored_data = f.read()
            assert stored_data == data

    def test_retrieve_block(self):
        """Test block retrieval."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Store a block
            block_id = generate_block_id()
            data = generate_test_data(1024)
            dn.store_block(block_id, data)

            # Retrieve the block
            retrieved_data = dn.retrieve_block(block_id)
            assert retrieved_data == data

            # Try to retrieve non-existent block
            with pytest.raises(HDFSError):
                dn.retrieve_block(generate_block_id())

    def test_delete_block(self):
        """Test block deletion."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Store a block
            block_id = generate_block_id()
            data = generate_test_data(1024)
            dn.store_block(block_id, data)

            # Delete the block
            dn.delete_block(block_id)

            # Verify block is deleted
            assert block_id not in dn._blocks

            # Verify file is deleted
            block_path = os.path.join(data_dir, f"blk_{block_id}")
            assert not os.path.exists(block_path)

            # Try to delete non-existent block (should not raise)
            dn.delete_block(generate_block_id())

    def test_get_block_report(self):
        """Test block report generation."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Store multiple blocks
            block_ids = []
            for i in range(5):
                block_id = generate_block_id()
                data = generate_test_data(1024)
                dn.store_block(block_id, data)
                block_ids.append(block_id)

            # Get block report
            report = dn.get_block_report()
            assert len(report) == 5
            assert set(report) == set(block_ids)

    def test_get_storage_info(self):
        """Test storage information retrieval."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Store some blocks
            total_size = 0
            for i in range(3):
                block_id = generate_block_id()
                data = generate_test_data(1024 * (i + 1))
                dn.store_block(block_id, data)
                total_size += len(data)

            # Get storage info
            storage_info = dn.get_storage_info()

            assert storage_info['used'] == total_size
            assert storage_info['capacity'] > 0
            assert storage_info['remaining'] == storage_info['capacity'] - storage_info['used']

    @pytest.mark.asyncio
    async def test_heartbeat(self):
        """Test heartbeat sending."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Mock the network connection
            mock_reader = AsyncMock()
            # Use Mock for writer since writer.write() is synchronous in asyncio
            mock_writer = Mock()

            # Mock deserialize_message to return success response
            from hdfs.common.protocol import Message, MessageType, serialize_message

            success_msg = Message(MessageType.SUCCESS, {"commands": []})
            response_data = serialize_message(success_msg)

            # Set up mock reader to return length + data
            mock_reader.read = AsyncMock(side_effect=[
                len(response_data).to_bytes(4, 'big'),
                response_data
            ])
            # write() is synchronous, drain()/wait_closed() are async
            mock_writer.write = Mock()
            mock_writer.drain = AsyncMock()
            mock_writer.close = Mock()
            mock_writer.wait_closed = AsyncMock()

            with patch('asyncio.open_connection', AsyncMock(return_value=(mock_reader, mock_writer))):
                # Send heartbeat
                commands = await dn.send_heartbeat()

                # Verify heartbeat was sent (writer.write was called)
                assert mock_writer.write.called
                assert commands == []

    @pytest.mark.asyncio
    async def test_block_report(self):
        """Test block report sending."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Store some blocks
            block_ids = []
            for i in range(3):
                block_id = generate_block_id()
                data = generate_test_data(1024)
                dn.store_block(block_id, data)
                block_ids.append(block_id)

            # Mock the network connection
            mock_reader = AsyncMock()
            # Use Mock for writer since writer.write() is synchronous in asyncio
            mock_writer = Mock()

            from hdfs.common.protocol import Message, MessageType, serialize_message

            success_msg = Message(MessageType.SUCCESS, {})
            response_data = serialize_message(success_msg)

            mock_reader.read = AsyncMock(side_effect=[
                len(response_data).to_bytes(4, 'big'),
                response_data
            ])
            # write() is synchronous, drain()/wait_closed() are async
            mock_writer.write = Mock()
            mock_writer.drain = AsyncMock()
            mock_writer.close = Mock()
            mock_writer.wait_closed = AsyncMock()

            with patch('asyncio.open_connection', AsyncMock(return_value=(mock_reader, mock_writer))):
                # Send block report
                result = await dn.send_block_report()

                # Verify block report was sent
                assert result is True
                assert mock_writer.write.called

    def test_block_corruption_detection(self):
        """Test detection of corrupted blocks."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Store a block
            block_id = generate_block_id()
            data = generate_test_data(1024)
            dn.store_block(block_id, data)

            # Verify block is valid initially
            assert dn.verify_block(block_id) is True

            # Corrupt the block file (blocks are stored in blocks/ subdirectory)
            block_path = os.path.join(data_dir, "blocks", block_id)
            with open(block_path, 'wb') as f:
                f.write(b'corrupted data')

            # verify_block should detect the corruption via checksum mismatch
            assert dn.verify_block(block_id) is False

            # scan_blocks should report this block as corrupted
            corrupted = dn.scan_blocks()
            assert block_id in corrupted

    def test_disk_space_management(self):
        """Test disk space checks."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Get initial storage info
            initial_info = dn.get_storage_info()

            # Store blocks until we use some space
            for i in range(5):
                block_id = generate_block_id()
                data = generate_test_data(1024 * 100)  # 100KB each
                dn.store_block(block_id, data)

            # Check storage info updated
            final_info = dn.get_storage_info()
            assert final_info['used'] > initial_info['used']
            assert final_info['remaining'] < initial_info['remaining']

    def test_concurrent_block_operations(self):
        """Test concurrent block operations."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Concurrent stores
            block_ids = [generate_block_id() for _ in range(10)]
            data_list = [generate_test_data(1024) for _ in range(10)]

            import threading

            def store_block_thread(block_id, data):
                dn.store_block(block_id, data)

            threads = []
            for block_id, data in zip(block_ids, data_list):
                t = threading.Thread(target=store_block_thread, args=(block_id, data))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Verify all blocks stored
            for block_id in block_ids:
                assert block_id in dn._blocks

    def test_block_scanner(self):
        """Test background block scanning for corruption."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Store blocks
            good_block_id = generate_block_id()
            good_data = generate_test_data(1024)
            dn.store_block(good_block_id, good_data)

            bad_block_id = generate_block_id()
            bad_data = generate_test_data(1024)
            dn.store_block(bad_block_id, bad_data)

            # Corrupt one block (blocks are stored in blocks/ subdirectory)
            block_path = os.path.join(data_dir, "blocks", bad_block_id)
            with open(block_path, 'wb') as f:
                f.write(b'corrupted')

            # Run block scanner
            corrupted = dn.scan_blocks()

            # Should detect the corrupted block
            assert bad_block_id in corrupted
            assert good_block_id not in corrupted

    def test_pipeline_write(self):
        """Test pipeline write for replication."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Mock downstream DataNode
            downstream = Mock()
            downstream.store_block = Mock()

            # Store block with pipeline
            block_id = generate_block_id()
            data = generate_test_data(1024)

            with patch.object(dn, '_forward_to_downstream') as mock_forward:
                dn.store_block_pipeline(block_id, data, [downstream])

                # Verify local storage
                assert block_id in dn._blocks

                # Verify forwarding
                mock_forward.assert_called_once()

    def test_block_recovery(self):
        """Test block recovery after failure."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir)

            # Store blocks
            block_ids = []
            for i in range(3):
                block_id = generate_block_id()
                data = generate_test_data(1024)
                dn.store_block(block_id, data)
                block_ids.append(block_id)

            # Simulate restart by creating new DataNode
            dn2 = DataNode("test-dn", data_dir)
            dn2.recover_blocks()

            # Verify blocks recovered
            for block_id in block_ids:
                assert block_id in dn2._blocks

    def test_bandwidth_throttling(self):
        """Test bandwidth throttling for transfers."""
        with temp_directory() as data_dir:
            dn = DataNode("test-dn", data_dir, max_bandwidth=1024 * 100)  # 100KB/s

            # Store large block and measure time
            block_id = generate_block_id()
            data = generate_test_data(1024 * 50)  # 50KB

            import time
            start_time = time.time()
            dn.store_block_throttled(block_id, data)
            elapsed = time.time() - start_time

            # Should take approximately 0.5 seconds at 100KB/s
            # Allow some tolerance
            assert elapsed >= 0.3  # At least 0.3 seconds