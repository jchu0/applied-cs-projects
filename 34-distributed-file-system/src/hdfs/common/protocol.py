"""Communication protocol for HDFS."""

import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


class MessageType(Enum):
    """Message types for HDFS protocol."""
    # NameNode operations
    CREATE_FILE = "create_file"
    OPEN_FILE = "open_file"
    DELETE_FILE = "delete_file"
    RENAME_FILE = "rename_file"
    MKDIR = "mkdir"
    DELETE_DIR = "delete_dir"
    LIST_DIR = "list_dir"
    GET_FILE_INFO = "get_file_info"

    # Block operations
    ADD_BLOCK = "add_block"
    COMPLETE_FILE = "complete_file"
    GET_BLOCK_LOCATIONS = "get_block_locations"
    REPORT_BAD_BLOCKS = "report_bad_blocks"

    # DataNode operations
    REGISTER_DATANODE = "register_datanode"
    HEARTBEAT = "heartbeat"
    BLOCK_REPORT = "block_report"
    BLOCK_RECEIVED = "block_received"

    # Data transfer
    READ_BLOCK = "read_block"
    WRITE_BLOCK = "write_block"
    COPY_BLOCK = "copy_block"
    DELETE_BLOCK = "delete_block"

    # Responses
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class Message:
    """Protocol message."""
    msg_type: MessageType
    payload: Dict[str, Any]
    request_id: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "type": self.msg_type.value,
            "payload": self.payload,
            "request_id": self.request_id
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for test compatibility."""
        if key == "type":
            return self.msg_type.value
        elif key == "payload":
            return self.payload
        elif key == "request_id":
            return self.request_id
        elif key in self.payload:
            return self.payload[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for test compatibility."""
        try:
            return self[key]
        except KeyError:
            return default

    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create from dictionary."""
        return cls(
            msg_type=MessageType(data["type"]),
            payload=data.get("payload", {}),
            request_id=data.get("request_id", "")
        )


def serialize_message(message: Message) -> bytes:
    """Serialize message to bytes."""
    data = message.to_dict()
    json_str = json.dumps(data)
    return json_str.encode('utf-8')


def deserialize_message(data: bytes) -> Message:
    """Deserialize message from bytes."""
    json_str = data.decode('utf-8')
    dict_data = json.loads(json_str)
    return Message.from_dict(dict_data)


class HDFSError(Exception):
    """Base HDFS error."""
    pass


class FileNotFoundError(HDFSError):
    """File not found."""
    pass


class FileExistsError(HDFSError):
    """File already exists."""
    pass


class DirectoryNotEmptyError(HDFSError):
    """Directory not empty."""
    pass


class NoDataNodeError(HDFSError):
    """No DataNode available."""
    pass


class BlockNotFoundError(HDFSError):
    """Block not found."""
    pass


class ReplicationError(HDFSError):
    """Replication failed."""
    pass
