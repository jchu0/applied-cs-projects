"""Pytest configuration and fixtures for HDFS tests."""

import os
import sys

# Add tests directory to path so 'from fixtures import ...' works
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# Add src directory to path for hdfs imports
src_dir = os.path.join(os.path.dirname(tests_dir), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import fixtures to make them available
from fixtures import (
    TestNameNodeServer,
    TestDataNodeServer,
    temp_directory,
    hdfs_cluster,
    generate_test_data,
    generate_test_file_path,
    create_mock_block,
    create_mock_file_info,
    create_mock_datanode_info,
    MockNetwork,
    DataNodeMock,
    NameNodeMock,
    assert_blocks_equal,
    assert_files_equal,
)

import pytest

# Re-export fixtures as pytest fixtures for convenience
@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    with temp_directory() as d:
        yield d

@pytest.fixture
def mock_network():
    """Provide a mock network."""
    return MockNetwork()

@pytest.fixture
def datanode_mock():
    """Provide a mock DataNode."""
    return DataNodeMock("test-datanode-0")

@pytest.fixture
def namenode_mock():
    """Provide a mock NameNode."""
    return NameNodeMock()
