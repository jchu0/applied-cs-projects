"""HDFS-lite - Distributed file system for large-scale data storage."""

from .common import (
    BlockID,
    NodeID,
    Block,
    BlockLocation,
    FileInfo,
    DirectoryInfo,
    DataNodeInfo,
    ReplicationPolicy,
)
from .namenode import NameNode, NameNodeServer
from .datanode import DataNode, DataNodeServer
from .client import HDFSClient

__version__ = "0.1.0"

__all__ = [
    # Common types
    "BlockID",
    "NodeID",
    "Block",
    "BlockLocation",
    "FileInfo",
    "DirectoryInfo",
    "DataNodeInfo",
    "ReplicationPolicy",
    # NameNode
    "NameNode",
    "NameNodeServer",
    # DataNode
    "DataNode",
    "DataNodeServer",
    # Client
    "HDFSClient",
]
