"""Integration tests for the HDFS system."""

import pytest
import asyncio
import os
import time
import random
from typing import List

from fixtures import (
    hdfs_cluster, generate_test_data, temp_directory,
    generate_test_file_path
)

from hdfs.common.protocol import HDFSError, FileNotFoundError


class TestHDFSIntegration:
    """Integration tests for the complete HDFS system."""

    @pytest.mark.asyncio
    async def test_end_to_end_file_operations(self):
        """Test complete file lifecycle: create, write, read, delete."""
        async with hdfs_cluster(num_datanodes=3) as cluster:
            client = cluster['client']

            # Create file
            await client.create('/test.txt')

            # Write data
            test_data = generate_test_data(1024 * 100)  # 100KB
            await client.write('/test.txt', test_data)

            # Read data back
            read_data = await client.read('/test.txt')
            assert read_data == test_data

            # Delete file
            await client.delete('/test.txt')

            # Verify deletion
            with pytest.raises(FileNotFoundError):
                await client.read('/test.txt')

    @pytest.mark.asyncio
    async def test_large_file_handling(self):
        """Test handling of large files spanning multiple blocks."""
        async with hdfs_cluster(
            num_datanodes=3,
            default_block_size=1024 * 1024  # 1MB blocks
        ) as cluster:
            client = cluster['client']

            # Write large file (5MB)
            large_data = generate_test_data(5 * 1024 * 1024)
            await client.write('/large_file.txt', large_data)

            # Read back and verify
            read_data = await client.read('/large_file.txt')
            assert read_data == large_data

            # Check that file spans multiple blocks
            file_info = await client.get_file_status('/large_file.txt')
            assert len(file_info['blocks']) >= 5

    @pytest.mark.asyncio
    async def test_replication(self):
        """Test that blocks are properly replicated across DataNodes."""
        async with hdfs_cluster(num_datanodes=3) as cluster:
            client = cluster['client']
            namenode = cluster['namenode']

            # Create file with replication factor 3
            await client.create('/replicated.txt', replication=3)

            # Write data
            data = generate_test_data(1024)
            await client.write('/replicated.txt', data)

            # Wait for replication
            await asyncio.sleep(1)

            # Check replication
            file_info = namenode.get_file_info('/replicated.txt')
            for block_id in file_info.blocks:
                # blocks is a list of block_id strings
                locations = namenode.get_block_locations(block_id)
                assert len(locations) == 3

    @pytest.mark.asyncio
    async def test_datanode_failure_handling(self):
        """Test system behavior when a DataNode fails."""
        async with hdfs_cluster(num_datanodes=4) as cluster:
            client = cluster['client']
            namenode = cluster['namenode']
            datanodes = cluster['datanodes']

            # Write file
            data = generate_test_data(1024)
            await client.write('/test.txt', data)

            # Simulate DataNode failure
            failed_dn = datanodes[0]
            await cluster['dn_servers'][0].stop()

            # Wait for heartbeat timeout
            await asyncio.sleep(namenode.heartbeat_interval * 2)

            # Simulate heartbeats from surviving nodes (they would do this in real system)
            import time
            for i, dn in enumerate(datanodes[1:], 1):  # Skip failed node
                namenode._datanodes[dn.node_id].last_heartbeat = time.time()

            # Manually trigger dead node detection with short timeout for testing
            # (in real system, this happens via background loop with 30s timeout)
            namenode.check_and_remove_dead_nodes(timeout=namenode.heartbeat_interval)

            # Should still be able to read file
            read_data = await client.read('/test.txt')
            assert read_data == data

            # Check that NameNode detected failure
            assert failed_dn.node_id not in namenode._datanodes

    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self):
        """Test concurrent operations on different files."""
        async with hdfs_cluster(num_datanodes=3) as cluster:
            client = cluster['client']

            # Create multiple files concurrently
            num_files = 10
            tasks = []

            for i in range(num_files):
                data = generate_test_data(1024 * (i + 1))
                task = self._write_file_async(client, f'/file{i}.txt', data)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            assert all(results)

            # Read files concurrently
            read_tasks = []
            for i in range(num_files):
                task = client.read(f'/file{i}.txt')
                read_tasks.append(task)

            read_results = await asyncio.gather(*read_tasks)
            assert len(read_results) == num_files

    async def _write_file_async(self, client, path: str, data: bytes) -> bool:
        """Helper for concurrent write operations."""
        try:
            await client.create(path)
            await client.write(path, data)
            return True
        except Exception:
            return False

    @pytest.mark.asyncio
    async def test_directory_operations(self):
        """Test directory creation, listing, and deletion."""
        async with hdfs_cluster(num_datanodes=3) as cluster:
            client = cluster['client']

            # Create directory structure
            await client.mkdir('/dir1')
            await client.mkdir('/dir1/subdir1')
            await client.mkdir('/dir1/subdir2')
            await client.mkdir('/dir2')

            # Create files in directories
            await client.write('/dir1/file1.txt', b'data1')
            await client.write('/dir1/subdir1/file2.txt', b'data2')
            await client.write('/dir2/file3.txt', b'data3')

            # List directories
            dir1_contents = await client.listdir('/dir1')
            assert 'file1.txt' in dir1_contents
            assert 'subdir1' in dir1_contents
            assert 'subdir2' in dir1_contents

            # Delete directory (should fail if not empty)
            with pytest.raises(HDFSError):
                await client.rmdir('/dir1')

            # Delete files first
            await client.delete('/dir1/file1.txt')
            await client.delete('/dir1/subdir1/file2.txt')
            await client.rmdir('/dir1/subdir1')
            await client.rmdir('/dir1/subdir2')
            await client.rmdir('/dir1')

            # Verify deletion
            root_contents = await client.listdir('/')
            assert 'dir1' not in root_contents
            assert 'dir2' in root_contents

    @pytest.mark.asyncio
    async def test_append_operation(self):
        """Test appending data to existing files."""
        async with hdfs_cluster(num_datanodes=3) as cluster:
            client = cluster['client']

            # Create and write initial data
            initial_data = b'Initial data\n'
            await client.write('/append_test.txt', initial_data)

            # Append more data
            append_data = b'Appended data\n'
            await client.append('/append_test.txt', append_data)

            # Read and verify
            final_data = await client.read('/append_test.txt')
            assert final_data == initial_data + append_data

    @pytest.mark.asyncio
    async def test_safe_mode_behavior(self):
        """Test system behavior in safe mode."""
        async with hdfs_cluster(num_datanodes=3) as cluster:
            namenode = cluster['namenode']
            client = cluster['client']

            # Force safe mode
            namenode._safe_mode = True

            # Write operations should fail in safe mode
            with pytest.raises(HDFSError):
                await client.write('/safe_mode_test.txt', b'data')

            # Exit safe mode
            namenode._safe_mode = False

            # Operations should work now
            await client.write('/safe_mode_test.txt', b'data')
            data = await client.read('/safe_mode_test.txt')
            assert data == b'data'

    @pytest.mark.asyncio
    async def test_block_corruption_recovery(self):
        """Test recovery from block corruption."""
        async with hdfs_cluster(num_datanodes=3) as cluster:
            client = cluster['client']
            datanodes = cluster['datanodes']

            # Write file with replication
            data = generate_test_data(1024)
            await client.write('/corruption_test.txt', data)

            # Corrupt block on one DataNode
            dn = datanodes[0]
            if dn._blocks:
                block_id = list(dn._blocks.keys())[0]
                block_path = os.path.join(dn.data_dir, f'blk_{block_id}')
                with open(block_path, 'wb') as f:
                    f.write(b'corrupted')

            # Should still be able to read from other replicas
            read_data = await client.read('/corruption_test.txt')
            assert read_data == data

    @pytest.mark.asyncio
    async def test_quota_management(self):
        """Test space and namespace quota enforcement."""
        async with hdfs_cluster(num_datanodes=3) as cluster:
            namenode = cluster['namenode']
            client = cluster['client']

            # Set space quota on directory
            await client.mkdir('/quota_dir')
            namenode.set_quota('/quota_dir', space_quota=1024 * 1024)  # 1MB

            # Write within quota
            small_data = generate_test_data(512 * 1024)  # 512KB
            await client.write('/quota_dir/small.txt', small_data)

            # Try to exceed quota
            large_data = generate_test_data(1024 * 1024)  # 1MB
            with pytest.raises(HDFSError):
                await client.write('/quota_dir/large.txt', large_data)

    @pytest.mark.asyncio
    async def test_load_balancing(self):
        """Test that block placement is balanced across DataNodes."""
        async with hdfs_cluster(num_datanodes=5) as cluster:
            client = cluster['client']
            namenode = cluster['namenode']

            # Write multiple files
            for i in range(20):
                data = generate_test_data(1024 * 100)  # 100KB each
                await client.write(f'/balanced_{i}.txt', data)

            # Check block distribution
            block_counts = {}
            for dn_id in namenode._datanodes:
                count = sum(1 for blocks in namenode._block_to_nodes.values()
                           if dn_id in blocks)
                block_counts[dn_id] = count

            # Distribution should be reasonably balanced
            counts = list(block_counts.values())
            avg_count = sum(counts) / len(counts)
            for count in counts:
                assert abs(count - avg_count) < avg_count * 0.5  # Within 50% of average

    @pytest.mark.asyncio
    async def test_metadata_persistence(self):
        """Test that metadata persists across NameNode restarts."""
        with temp_directory() as checkpoint_dir:
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.json')
            original_block_count = 0

            # Create initial cluster
            async with hdfs_cluster(num_datanodes=3) as cluster:
                client = cluster['client']
                namenode = cluster['namenode']

                # Create files and directories
                await client.mkdir('/persistent_dir')
                await client.write('/persistent_dir/file.txt', b'persistent data')
                original_block_count = len(namenode._block_to_nodes)

                # Save checkpoint
                namenode.save_checkpoint(checkpoint_file)

            # Create new cluster with restored checkpoint
            async with hdfs_cluster(num_datanodes=3) as cluster2:
                namenode2 = cluster2['namenode']

                # Load checkpoint
                namenode2.load_checkpoint(checkpoint_file)

                # Verify metadata persisted (directory structure and file info)
                # Note: Actual data read requires DataNodes to hold the blocks,
                # which is not the case with fresh DataNodes. This tests metadata only.
                file_info = namenode2.get_file_info('/persistent_dir/file.txt')
                assert file_info is not None
                assert file_info.path == '/persistent_dir/file.txt'
                assert file_info.size == len(b'persistent data')

                # Verify directory structure persisted
                entries = namenode2.list_directory('/persistent_dir')
                assert len(entries) == 1
                # entries are FileInfo objects or path strings depending on implementation
                entry = entries[0]
                entry_path = entry.path if hasattr(entry, 'path') else str(entry)
                assert 'file.txt' in entry_path

                # Verify block mappings persisted
                assert len(namenode2._block_to_nodes) == original_block_count

    @pytest.mark.asyncio
    async def test_stress_test(self):
        """Stress test with many concurrent operations."""
        async with hdfs_cluster(num_datanodes=5) as cluster:
            client = cluster['client']

            # Parameters
            num_operations = 100
            max_concurrent = 20

            async def random_operation(idx: int):
                """Perform a random HDFS operation."""
                operation = random.choice(['write', 'read', 'delete', 'mkdir'])

                try:
                    if operation == 'write':
                        data = generate_test_data(random.randint(1024, 10240))
                        await client.write(f'/stress_{idx}.txt', data)
                    elif operation == 'read':
                        # Try to read existing file
                        await client.read(f'/stress_{random.randint(0, idx)}.txt')
                    elif operation == 'delete':
                        await client.delete(f'/stress_{random.randint(0, idx)}.txt')
                    elif operation == 'mkdir':
                        await client.mkdir(f'/stress_dir_{idx}')
                    return True
                except Exception:
                    return False

            # Run operations with concurrency limit
            semaphore = asyncio.Semaphore(max_concurrent)

            async def limited_operation(idx: int):
                async with semaphore:
                    return await random_operation(idx)

            tasks = [limited_operation(i) for i in range(num_operations)]
            results = await asyncio.gather(*tasks)

            # Most operations should succeed
            # Note: 40% threshold accounts for random read/delete operations
            # that may reference non-existent files (by design, to test chaos resilience)
            success_rate = sum(results) / len(results)
            assert success_rate > 0.4  # At least 40% success rate for chaos test