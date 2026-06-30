"""Unit tests for the HDFS Client."""

import pytest
import asyncio
import hashlib
import time
from unittest.mock import Mock, AsyncMock, patch
from io import BytesIO

from fixtures import (
    generate_test_data, create_mock_file_info, create_mock_block,
    hdfs_cluster, temp_directory
)

from hdfs.client.client import HDFSClient
from hdfs.common.types import BlockID, BlockLocation, FileInfo
from hdfs.common.protocol import HDFSError, FileNotFoundError


class TestHDFSClient:
    """Test cases for HDFS Client functionality."""

    def test_initialization(self):
        """Test client initialization."""
        client = HDFSClient(
            namenode_host="localhost",
            namenode_port=9000,
            block_size=64 * 1024 * 1024,
            replication=2
        )

        assert client.namenode_host == "localhost"
        assert client.namenode_port == 9000
        assert client.default_block_size == 64 * 1024 * 1024
        assert client.default_replication == 2

    @pytest.mark.asyncio
    async def test_create_file(self):
        """Test file creation through client."""
        client = HDFSClient()

        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {
                'type': 'create_file_response',
                'file_info': {
                    'path': '/test.txt',
                    'replication': 3,
                    'block_size': 128 * 1024 * 1024
                }
            }

            # Create file
            await client.create('/test.txt', replication=3)

            # Verify request sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            assert call_args['type'] == 'create_file'
            assert call_args['path'] == '/test.txt'

    @pytest.mark.asyncio
    async def test_write_file(self):
        """Test writing data to a file."""
        client = HDFSClient(block_size=1024)  # Small block size for testing

        # Mock NameNode responses for create flow
        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_nn:
            # Return sequence: CREATE_FILE, ADD_BLOCK (x2), COMPLETE_FILE
            mock_nn.side_effect = [
                {'status': 'ok'},  # CREATE_FILE
                {'block_id': 'block1', 'locations': [{'host': 'localhost', 'port': 50010}]},  # ADD_BLOCK
                {'block_id': 'block2', 'locations': [{'host': 'localhost', 'port': 50020}]},  # ADD_BLOCK
                {'status': 'ok'},  # COMPLETE_FILE
            ]

            # Mock DataNode connections
            with patch.object(client, '_send_to_datanode', new_callable=AsyncMock) as mock_dn:
                mock_dn.return_value = {'status': 'ok'}

                # Write data
                data = generate_test_data(1536)  # Will be split into 2 blocks
                await client.write('/test.txt', data)

                # Verify blocks were written to DataNodes
                assert mock_dn.call_count == 2

    @pytest.mark.asyncio
    async def test_read_file(self):
        """Test reading data from a file."""
        client = HDFSClient()

        # Mock NameNode responses - locations is list of lists (replicas per block)
        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_nn:
            mock_nn.return_value = {
                'locations': [
                    [{'host': 'localhost', 'port': 50010, 'block_id': 'block1'}],  # block 1 replicas
                    [{'host': 'localhost', 'port': 50020, 'block_id': 'block2'}],  # block 2 replicas
                ]
            }

            # Mock DataNode reads
            with patch.object(client, '_send_to_datanode', new_callable=AsyncMock) as mock_dn:
                block1_data = generate_test_data(1024)
                block2_data = generate_test_data(1024)
                mock_dn.side_effect = [
                    {'data': block1_data.hex()},
                    {'data': block2_data.hex()},
                ]

                # Read file
                data = await client.read('/test.txt')

                # Verify data
                assert len(data) == 2048
                assert data == block1_data + block2_data
                assert mock_dn.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_file(self):
        """Test file deletion."""
        client = HDFSClient()

        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {'status': 'ok'}

            # Delete file
            await client.delete('/test.txt')

            # Verify request
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            assert call_args['type'] == 'delete_file'
            assert call_args['path'] == '/test.txt'

    @pytest.mark.asyncio
    async def test_mkdir(self):
        """Test directory creation."""
        client = HDFSClient()

        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {'status': 'ok'}

            # Create directory
            await client.mkdir('/test_dir')

            # Verify request
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            assert call_args['type'] == 'mkdir'
            assert call_args['path'] == '/test_dir'

    @pytest.mark.asyncio
    async def test_list_directory(self):
        """Test directory listing."""
        client = HDFSClient()

        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {
                'entries': ['file1.txt', 'file2.txt', 'subdir']
            }

            # List directory
            entries = await client.listdir('/')

            # Verify
            assert len(entries) == 3
            assert 'file1.txt' in entries
            assert 'subdir' in entries

    @pytest.mark.asyncio
    async def test_get_file_status(self):
        """Test getting file status."""
        client = HDFSClient()

        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {
                'file_info': {
                    'path': '/test.txt',
                    'size': 1024,
                    'replication': 3,
                    'block_size': 128 * 1024 * 1024,
                    'modification_time': 1234567890
                }
            }

            # Get status
            status = await client.get_file_status('/test.txt')

            # Verify
            assert status['path'] == '/test.txt'
            assert status['size'] == 1024
            assert status['replication'] == 3

    @pytest.mark.asyncio
    async def test_rename(self):
        """Test file/directory renaming."""
        client = HDFSClient()

        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {'status': 'ok'}

            # Rename
            await client.rename('/old.txt', '/new.txt')

            # Verify
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            assert call_args['type'] == 'rename_file'
            assert call_args['src'] == '/old.txt'
            assert call_args['dst'] == '/new.txt'

    @pytest.mark.asyncio
    async def test_append_to_file(self):
        """Test appending data to existing file."""
        client = HDFSClient()

        # Test append by mocking read, delete, and create
        with patch.object(client, 'read', new_callable=AsyncMock) as mock_read:
            with patch.object(client, 'delete', new_callable=AsyncMock) as mock_delete:
                with patch.object(client, 'create', new_callable=AsyncMock) as mock_create:
                    # Mock existing file content
                    existing_data = b'existing content'
                    mock_read.return_value = existing_data
                    mock_delete.return_value = True
                    mock_create.return_value = True

                    # Append new data
                    new_data = generate_test_data(1024)
                    await client.append('/test.txt', new_data)

                    # Verify the workflow: read -> delete -> create with combined data
                    mock_read.assert_called_once_with('/test.txt')
                    mock_delete.assert_called_once_with('/test.txt')
                    mock_create.assert_called_once()
                    # Verify combined data was written
                    call_args = mock_create.call_args
                    assert call_args[0][0] == '/test.txt'
                    assert call_args[0][1] == existing_data + new_data

    @pytest.mark.asyncio
    async def test_retry_on_datanode_failure(self):
        """Test retry logic when DataNode fails."""
        client = HDFSClient()

        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_nn:
            # Provide multiple DataNode locations
            mock_nn.return_value = {
                'locations': [
                    {'node_id': 'dn1', 'host': 'localhost', 'port': 50010},
                    {'node_id': 'dn2', 'host': 'localhost', 'port': 50020},
                    {'node_id': 'dn3', 'host': 'localhost', 'port': 50030}
                ]
            }

            with patch.object(client, '_read_block_from_datanode', new_callable=AsyncMock) as mock_read:
                # First two attempts fail, third succeeds
                data = generate_test_data(1024)
                mock_read.side_effect = [
                    ConnectionError("DN1 failed"),
                    ConnectionError("DN2 failed"),
                    data
                ]

                # Should succeed on third attempt
                result = await client._read_block_with_retry('block1')
                assert result == data
                assert mock_read.call_count == 3

    @pytest.mark.asyncio
    async def test_streaming_read(self):
        """Test streaming read for large files."""
        client = HDFSClient(block_size=1024)  # Small blocks for testing

        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_nn:
            # Mock large file with multiple blocks
            num_blocks = 10
            blocks = [{'block_id': f'block{i}', 'size': 1024} for i in range(num_blocks)]

            mock_nn.return_value = {
                'file_info': {
                    'path': '/large.txt',
                    'blocks': blocks
                },
                'locations': {
                    f'block{i}': [{'host': 'localhost', 'port': 50010}]
                    for i in range(num_blocks)
                }
            }

            with patch.object(client, '_read_block_from_datanode', new_callable=AsyncMock) as mock_read:
                mock_read.return_value = generate_test_data(1024)

                # Stream read - with 1024 byte blocks and 2048 chunk_size, we get 2 blocks per chunk
                chunks = []
                async for chunk in client.stream_read('/large.txt', chunk_size=2048):
                    chunks.append(chunk)

                # Should read in chunks - with 1024 byte blocks and 2048 chunk_size
                # blocks_per_chunk = 2048 // 1024 = 2
                assert len(chunks) == 5  # 10 blocks / 2 blocks per chunk
                assert mock_read.call_count == num_blocks

    @pytest.mark.asyncio
    async def test_concurrent_writes(self):
        """Test concurrent writes to different files."""
        client = HDFSClient(block_size=1024 * 1024)  # 1MB blocks so 1024 bytes fits in one block

        with patch.object(client, '_send_to_namenode', new_callable=AsyncMock) as mock_nn:
            with patch.object(client, '_send_to_datanode', new_callable=AsyncMock) as mock_dn:
                # Each write does: CREATE_FILE, ADD_BLOCK, COMPLETE_FILE
                mock_nn.return_value = {
                    'block_id': 'block1',
                    'locations': [{'host': 'localhost', 'port': 50010}]
                }
                mock_dn.return_value = {'status': 'ok'}

                # Write multiple files concurrently
                tasks = []
                for i in range(5):
                    data = generate_test_data(1024)
                    task = client.write(f'/file{i}.txt', data)
                    tasks.append(task)

                await asyncio.gather(*tasks)

                # Each file writes to one DataNode (5 files * 1 block each)
                assert mock_dn.call_count == 5

    def test_client_side_caching(self):
        """Test client-side metadata caching."""
        client = HDFSClient(enable_cache=True, cache_ttl=60)

        # Mock cache
        client._cache = {}

        # Add to cache
        file_info = create_mock_file_info('/cached.txt')
        client._cache['/cached.txt'] = {
            'data': file_info,
            'timestamp': time.time()
        }

        # Retrieve from cache
        cached = client._get_from_cache('/cached.txt')
        assert cached is not None
        assert cached.path == '/cached.txt'

        # Expired cache
        client._cache['/expired.txt'] = {
            'data': file_info,
            'timestamp': time.time() - 120  # 2 minutes ago
        }

        expired = client._get_from_cache('/expired.txt')
        assert expired is None

    @pytest.mark.asyncio
    async def test_checksum_verification(self):
        """Test checksum verification for data integrity."""
        client = HDFSClient(verify_checksum=True)

        with patch.object(client, '_read_block_from_datanode', new_callable=AsyncMock) as mock_read:
            data = generate_test_data(1024)
            checksum = hashlib.md5(data).hexdigest()

            mock_read.return_value = (data, checksum)

            # Read with correct checksum
            result = await client._read_block_with_checksum('block1')
            assert result == data

            # Read with incorrect checksum
            mock_read.return_value = (data, 'wrong_checksum')

            with pytest.raises(HDFSError):
                await client._read_block_with_checksum('block1')