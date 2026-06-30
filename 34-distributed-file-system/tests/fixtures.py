"""Test fixtures and helpers for HDFS tests."""

import os
import asyncio
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, MagicMock
from contextlib import contextmanager, asynccontextmanager
import random
import string

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from hdfs.common.types import (
    BlockID, NodeID, Block, BlockLocation, FileInfo, DirectoryInfo,
    DataNodeInfo, generate_block_id
)
from hdfs.namenode.namenode import NameNode
from hdfs.datanode.datanode import DataNode
from hdfs.client.client import HDFSClient


class TestNameNodeServer:
    """Test NameNode server for testing."""

    def __init__(self, namenode: NameNode, port: int = 0):
        self.namenode = namenode
        self.port = port
        self.server = None

    async def start(self):
        """Start the test server."""
        self.server = await asyncio.start_server(
            self.handle_client, 'localhost', self.port
        )
        addrs = self.server.sockets[0].getsockname()
        self.port = addrs[1]

    async def handle_client(self, reader, writer):
        """Handle client connections with proper protocol handling."""
        from hdfs.common.protocol import (
            Message, MessageType, deserialize_message, serialize_message,
            HDFSError
        )
        from hdfs.common.types import BlockReport
        import logging
        logger = logging.getLogger(__name__)

        try:
            while True:
                # Read message length
                length_data = await reader.readexactly(4)
                if not length_data:
                    break

                length = int.from_bytes(length_data, 'big')
                data = await reader.readexactly(length)

                # Process message
                message = deserialize_message(data)
                response = await self._process_message(message)

                # Send response
                response_data = serialize_message(response)
                writer.write(len(response_data).to_bytes(4, 'big'))
                writer.write(response_data)
                await writer.drain()

        except asyncio.IncompleteReadError:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"NameNode handler error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_message(self, message) -> 'Message':
        """Process incoming message."""
        from hdfs.common.protocol import Message, MessageType, HDFSError
        from hdfs.common.types import BlockReport

        try:
            if message.msg_type == MessageType.CREATE_FILE:
                result = self.namenode.create_file(**message.payload)
                return Message(MessageType.SUCCESS, {"path": result.path})

            elif message.msg_type == MessageType.ADD_BLOCK:
                block, locations = self.namenode.add_block(message.payload["path"])
                # Also track that these locations will have the block
                # (In real HDFS this happens via block_received callback)
                for loc in locations:
                    self.namenode._block_to_nodes[block.block_id].add(loc.node_id)
                return Message(MessageType.SUCCESS, {
                    "block_id": block.block_id,
                    "locations": [
                        {"node_id": loc.node_id, "host": loc.host, "port": loc.port}
                        for loc in locations
                    ]
                })

            elif message.msg_type == MessageType.COMPLETE_FILE:
                result = self.namenode.complete_file(**message.payload)
                return Message(MessageType.SUCCESS, {"size": result.size})

            elif message.msg_type == MessageType.GET_BLOCK_LOCATIONS:
                locations = self.namenode.get_block_locations_for_file(message.payload["path"])
                return Message(MessageType.SUCCESS, {
                    "locations": [
                        [{"block_id": loc.block_id, "host": loc.host, "port": loc.port}
                         for loc in block_locs]
                        for block_locs in locations
                    ]
                })

            elif message.msg_type == MessageType.LIST_DIR:
                result = self.namenode.list_directory(message.payload["path"])
                return Message(MessageType.SUCCESS, {"entries": result})

            elif message.msg_type == MessageType.GET_FILE_INFO:
                result = self.namenode.get_file_info_dict(message.payload["path"])
                if result is None:
                    return Message(MessageType.ERROR, {"error": "Path not found"})
                # Also include blocks for files
                path = message.payload["path"]
                if path in self.namenode._files:
                    file_info = self.namenode._files[path]
                    result["blocks"] = file_info.blocks
                return Message(MessageType.SUCCESS, {"info": result, "file_info": result})

            elif message.msg_type == MessageType.MKDIR:
                self.namenode.mkdir(**message.payload)
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.DELETE_FILE:
                self.namenode.delete_file(message.payload["path"])
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.DELETE_DIR:
                recursive = message.payload.get("recursive", False)
                self.namenode.delete_directory(message.payload["path"], recursive=recursive)
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.RENAME_FILE:
                self.namenode.rename(message.payload["src"], message.payload["dst"])
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.REGISTER_DATANODE:
                self.namenode.register_datanode(**message.payload)
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.HEARTBEAT:
                response = self.namenode.heartbeat(**message.payload)
                return Message(MessageType.SUCCESS, {"commands": response.commands})

            elif message.msg_type == MessageType.BLOCK_REPORT:
                report = BlockReport(**message.payload)
                self.namenode.block_report(report)
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.BLOCK_RECEIVED:
                self.namenode.block_received(**message.payload)
                return Message(MessageType.SUCCESS, {})

            else:
                return Message(MessageType.ERROR, {"error": f"Unknown message type: {message.msg_type}"})

        except HDFSError as e:
            return Message(MessageType.ERROR, {"error": str(e)})
        except Exception as e:
            return Message(MessageType.ERROR, {"error": str(e)})

    async def stop(self):
        """Stop the test server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()


class TestDataNodeServer:
    """Test DataNode server for testing."""

    def __init__(self, datanode: DataNode, port: int = 0):
        self.datanode = datanode
        self.port = port
        self.server = None

    async def start(self):
        """Start the test server."""
        self.server = await asyncio.start_server(
            self.handle_client, 'localhost', self.port
        )
        addrs = self.server.sockets[0].getsockname()
        self.port = addrs[1]

    async def handle_client(self, reader, writer):
        """Handle client connections with proper protocol handling."""
        from hdfs.common.protocol import (
            Message, MessageType, deserialize_message, serialize_message
        )
        import logging
        logger = logging.getLogger(__name__)

        try:
            while True:
                # Read message length
                length_data = await reader.readexactly(4)
                if not length_data:
                    break

                length = int.from_bytes(length_data, 'big')
                data = await reader.readexactly(length)

                # Process message
                message = deserialize_message(data)
                response = await self._process_message(message)

                # Send response
                response_data = serialize_message(response)
                writer.write(len(response_data).to_bytes(4, 'big'))
                writer.write(response_data)
                await writer.drain()

        except asyncio.IncompleteReadError:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"DataNode handler error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_message(self, message) -> 'Message':
        """Process incoming message."""
        from hdfs.common.protocol import Message, MessageType

        try:
            if message.msg_type == MessageType.WRITE_BLOCK:
                block_id = message.payload["block_id"]
                data = bytes.fromhex(message.payload["data"])
                self.datanode.store_block(block_id, data)
                return Message(MessageType.SUCCESS, {"block_id": block_id})

            elif message.msg_type == MessageType.READ_BLOCK:
                block_id = message.payload["block_id"]
                try:
                    data = self.datanode.retrieve_block(block_id)
                    return Message(MessageType.SUCCESS, {"data": data.hex()})
                except Exception:
                    return Message(MessageType.ERROR, {"error": f"Block not found: {block_id}"})

            elif message.msg_type == MessageType.DELETE_BLOCK:
                block_id = message.payload["block_id"]
                self.datanode.delete_block(block_id)
                return Message(MessageType.SUCCESS, {})

            else:
                return Message(MessageType.ERROR, {"error": f"Unknown message type: {message.msg_type}"})

        except Exception as e:
            return Message(MessageType.ERROR, {"error": str(e)})

    async def stop(self):
        """Stop the test server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()


@contextmanager
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@asynccontextmanager
async def hdfs_cluster(num_datanodes: int = 3, **kwargs):
    """Create a test HDFS cluster."""
    namenode = NameNode(**kwargs)
    datanodes = []
    datanode_servers = []
    temp_dirs = []

    # Create NameNode server
    nn_server = TestNameNodeServer(namenode)
    await nn_server.start()

    # Exit safe mode for testing
    namenode._safe_mode = False

    try:
        # Create DataNodes
        for i in range(num_datanodes):
            # Create temp directory that persists for the cluster lifetime
            data_dir = tempfile.mkdtemp()
            temp_dirs.append(data_dir)

            dn = DataNode(
                node_id=f"datanode-{i}",
                data_dir=data_dir,
                namenode_host='localhost',
                namenode_port=nn_server.port
            )
            datanodes.append(dn)

            # Start DataNode server
            dn_server = TestDataNodeServer(dn)
            await dn_server.start()
            datanode_servers.append(dn_server)

            # Register with NameNode
            namenode.register_datanode(
                f"datanode-{i}",
                'localhost',
                dn_server.port
            )

        # Create client with matching block size
        client = HDFSClient(
            namenode_host='localhost',
            namenode_port=nn_server.port,
            block_size=namenode.default_block_size,
            replication=namenode.default_replication
        )

        yield {
            'namenode': namenode,
            'datanodes': datanodes,
            'client': client,
            'nn_server': nn_server,
            'dn_servers': datanode_servers
        }

    finally:
        # Cleanup
        await nn_server.stop()
        for server in datanode_servers:
            await server.stop()
        # Clean up temp directories
        for data_dir in temp_dirs:
            shutil.rmtree(data_dir, ignore_errors=True)


def generate_test_data(size: int) -> bytes:
    """Generate random test data of specified size."""
    return os.urandom(size)


def generate_test_file_path() -> str:
    """Generate a random file path for testing."""
    filename = ''.join(random.choices(string.ascii_lowercase, k=10))
    return f"/test/{filename}.txt"


def create_mock_block(block_id: Optional[BlockID] = None, size: int = 1024) -> Block:
    """Create a mock block for testing."""
    if block_id is None:
        block_id = generate_block_id()

    return Block(
        block_id=block_id,
        size=size,
        generation_stamp=1
    )


def create_mock_file_info(path: str, num_blocks: int = 3) -> FileInfo:
    """Create a mock FileInfo for testing."""
    blocks = [create_mock_block() for _ in range(num_blocks)]

    file_info = FileInfo(
        path=path,
        replication=3,
        block_size=128 * 1024 * 1024
    )
    file_info.blocks = blocks
    file_info.size = sum(b.size for b in blocks)

    return file_info


def create_mock_datanode_info(node_id: str) -> DataNodeInfo:
    """Create a mock DataNodeInfo for testing."""
    return DataNodeInfo(
        node_id=node_id,
        host='localhost',
        port=50000 + hash(node_id) % 1000,
        capacity=1024 * 1024 * 1024 * 100,  # 100GB
        used=1024 * 1024 * 1024 * 10,  # 10GB
        remaining=1024 * 1024 * 1024 * 90  # 90GB
    )


class MockNetwork:
    """Mock network for testing distributed operations."""

    def __init__(self):
        self.messages: List[Tuple[str, str, dict]] = []
        self.latency = 0.001  # 1ms default latency
        self.failure_rate = 0.0  # No failures by default

    async def send(self, from_node: str, to_node: str, message: dict):
        """Simulate sending a message."""
        if random.random() < self.failure_rate:
            raise ConnectionError("Network failure")

        await asyncio.sleep(self.latency)
        self.messages.append((from_node, to_node, message))

    def clear(self):
        """Clear message history."""
        self.messages.clear()

    def set_failure_rate(self, rate: float):
        """Set network failure rate for chaos testing."""
        self.failure_rate = min(1.0, max(0.0, rate))

    def set_latency(self, latency: float):
        """Set network latency."""
        self.latency = max(0.0, latency)


class DataNodeMock:
    """Mock DataNode for testing."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.blocks: Dict[BlockID, bytes] = {}
        self.capacity = 1024 * 1024 * 1024 * 100  # 100GB
        self.used = 0

    def store_block(self, block_id: BlockID, data: bytes):
        """Store a block."""
        self.blocks[block_id] = data
        self.used += len(data)

    def retrieve_block(self, block_id: BlockID) -> Optional[bytes]:
        """Retrieve a block."""
        return self.blocks.get(block_id)

    def delete_block(self, block_id: BlockID):
        """Delete a block."""
        if block_id in self.blocks:
            self.used -= len(self.blocks[block_id])
            del self.blocks[block_id]

    def get_block_report(self) -> List[BlockID]:
        """Get list of blocks stored."""
        return list(self.blocks.keys())


class NameNodeMock:
    """Mock NameNode for testing."""

    def __init__(self):
        self.files = {}
        self.directories = {'/': DirectoryInfo('/')}
        self.blocks = {}
        self.block_locations = {}

    def create_file(self, path: str, **kwargs) -> FileInfo:
        """Create a file."""
        file_info = FileInfo(path=path, **kwargs)
        self.files[path] = file_info
        return file_info

    def get_file_info(self, path: str) -> Optional[FileInfo]:
        """Get file information."""
        return self.files.get(path)

    def delete_file(self, path: str):
        """Delete a file."""
        if path in self.files:
            del self.files[path]


def assert_blocks_equal(block1: Block, block2: Block):
    """Assert that two blocks are equal."""
    assert block1.block_id == block2.block_id
    assert block1.size == block2.size
    assert block1.generation_stamp == block2.generation_stamp


def assert_files_equal(file1: FileInfo, file2: FileInfo):
    """Assert that two FileInfo objects are equal."""
    assert file1.path == file2.path
    assert file1.replication == file2.replication
    assert file1.block_size == file2.block_size
    assert len(file1.blocks) == len(file2.blocks)

    for b1, b2 in zip(file1.blocks, file2.blocks):
        assert_blocks_equal(b1, b2)