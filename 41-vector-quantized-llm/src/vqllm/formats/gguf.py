"""GGUF format support for llama.cpp compatible model storage.

GGUF (GPT-Generated Unified Format) is the format used by llama.cpp for
efficient storage and loading of quantized LLM models.

Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

import struct
import numpy as np
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)

# GGUF magic number and version
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3


class GGUFQuantType(IntEnum):
    """GGUF quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 29


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass
class GGUFTensorInfo:
    """Information about a tensor in the GGUF file."""
    name: str
    n_dims: int
    dims: Tuple[int, ...]
    type: GGUFQuantType
    offset: int


@dataclass
class GGUFMetadata:
    """GGUF file metadata."""
    # General
    architecture: str = "llama"
    quantization_version: int = 2
    alignment: int = 32
    name: str = ""
    author: str = ""
    version: str = ""
    description: str = ""
    license: str = ""

    # Model architecture
    context_length: int = 2048
    embedding_length: int = 4096
    block_count: int = 32
    feed_forward_length: int = 11008
    attention_head_count: int = 32
    attention_head_count_kv: int = 32
    attention_layer_norm_rms_epsilon: float = 1e-5
    rope_freq_base: float = 10000.0
    rope_dimension_count: int = 128

    # Tokenizer
    tokenizer_model: str = "llama"
    vocab_size: int = 32000
    bos_token_id: int = 1
    eos_token_id: int = 2
    padding_token_id: int = 0

    # Quantization info
    file_type: int = 0  # GGML file type

    # Extra key-value pairs
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        result = {
            "general.architecture": self.architecture,
            "general.quantization_version": self.quantization_version,
            "general.alignment": self.alignment,
            "general.name": self.name,
            "general.author": self.author,
            "general.version": self.version,
            "general.description": self.description,
            "general.license": self.license,
            f"{self.architecture}.context_length": self.context_length,
            f"{self.architecture}.embedding_length": self.embedding_length,
            f"{self.architecture}.block_count": self.block_count,
            f"{self.architecture}.feed_forward_length": self.feed_forward_length,
            f"{self.architecture}.attention.head_count": self.attention_head_count,
            f"{self.architecture}.attention.head_count_kv": self.attention_head_count_kv,
            f"{self.architecture}.attention.layer_norm_rms_epsilon": self.attention_layer_norm_rms_epsilon,
            f"{self.architecture}.rope.freq_base": self.rope_freq_base,
            f"{self.architecture}.rope.dimension_count": self.rope_dimension_count,
            "tokenizer.ggml.model": self.tokenizer_model,
            "tokenizer.ggml.vocab_size": self.vocab_size,
            "tokenizer.ggml.bos_token_id": self.bos_token_id,
            "tokenizer.ggml.eos_token_id": self.eos_token_id,
            "tokenizer.ggml.padding_token_id": self.padding_token_id,
            "general.file_type": self.file_type,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GGUFMetadata":
        """Create metadata from dictionary."""
        arch = data.get("general.architecture", "llama")

        return cls(
            architecture=arch,
            quantization_version=data.get("general.quantization_version", 2),
            alignment=data.get("general.alignment", 32),
            name=data.get("general.name", ""),
            author=data.get("general.author", ""),
            version=data.get("general.version", ""),
            description=data.get("general.description", ""),
            license=data.get("general.license", ""),
            context_length=data.get(f"{arch}.context_length", 2048),
            embedding_length=data.get(f"{arch}.embedding_length", 4096),
            block_count=data.get(f"{arch}.block_count", 32),
            feed_forward_length=data.get(f"{arch}.feed_forward_length", 11008),
            attention_head_count=data.get(f"{arch}.attention.head_count", 32),
            attention_head_count_kv=data.get(f"{arch}.attention.head_count_kv", 32),
            attention_layer_norm_rms_epsilon=data.get(
                f"{arch}.attention.layer_norm_rms_epsilon", 1e-5
            ),
            rope_freq_base=data.get(f"{arch}.rope.freq_base", 10000.0),
            rope_dimension_count=data.get(f"{arch}.rope.dimension_count", 128),
            tokenizer_model=data.get("tokenizer.ggml.model", "llama"),
            vocab_size=data.get("tokenizer.ggml.vocab_size", 32000),
            bos_token_id=data.get("tokenizer.ggml.bos_token_id", 1),
            eos_token_id=data.get("tokenizer.ggml.eos_token_id", 2),
            padding_token_id=data.get("tokenizer.ggml.padding_token_id", 0),
            file_type=data.get("general.file_type", 0),
        )


def _get_block_size(qtype: GGUFQuantType) -> int:
    """Get the block size for a quantization type."""
    block_sizes = {
        GGUFQuantType.F32: 1,
        GGUFQuantType.F16: 1,
        GGUFQuantType.Q4_0: 32,
        GGUFQuantType.Q4_1: 32,
        GGUFQuantType.Q5_0: 32,
        GGUFQuantType.Q5_1: 32,
        GGUFQuantType.Q8_0: 32,
        GGUFQuantType.Q8_1: 32,
        GGUFQuantType.Q2_K: 256,
        GGUFQuantType.Q3_K: 256,
        GGUFQuantType.Q4_K: 256,
        GGUFQuantType.Q5_K: 256,
        GGUFQuantType.Q6_K: 256,
        GGUFQuantType.Q8_K: 256,
    }
    return block_sizes.get(qtype, 1)


def _get_type_size(qtype: GGUFQuantType) -> float:
    """Get bytes per element for a quantization type."""
    type_sizes = {
        GGUFQuantType.F32: 4.0,
        GGUFQuantType.F16: 2.0,
        GGUFQuantType.Q4_0: 0.5 + 2/32,  # 4 bits + scale per 32 elements
        GGUFQuantType.Q4_1: 0.5 + 4/32,  # 4 bits + scale + min per 32 elements
        GGUFQuantType.Q5_0: 0.625 + 2/32,
        GGUFQuantType.Q5_1: 0.625 + 4/32,
        GGUFQuantType.Q8_0: 1.0 + 2/32,
        GGUFQuantType.Q8_1: 1.0 + 4/32,
        GGUFQuantType.I8: 1.0,
        GGUFQuantType.I16: 2.0,
        GGUFQuantType.I32: 4.0,
        GGUFQuantType.I64: 8.0,
        GGUFQuantType.F64: 8.0,
        GGUFQuantType.BF16: 2.0,
    }
    return type_sizes.get(qtype, 4.0)


class GGUFReader:
    """Read GGUF format files."""

    def __init__(self, path: str):
        """Initialize reader with file path."""
        self.path = path
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, GGUFTensorInfo] = {}
        self._file: Optional[BinaryIO] = None
        self._data_offset = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> None:
        """Open the GGUF file and read header."""
        self._file = open(self.path, "rb")
        self._read_header()

    def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    def _read_header(self) -> None:
        """Read GGUF header."""
        # Read magic
        magic = struct.unpack("<I", self._file.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic:#x}, expected {GGUF_MAGIC:#x}")

        # Read version
        version = struct.unpack("<I", self._file.read(4))[0]
        if version > GGUF_VERSION:
            logger.warning(f"GGUF version {version} is newer than supported {GGUF_VERSION}")

        # Read counts
        n_tensors = struct.unpack("<Q", self._file.read(8))[0]
        n_kv = struct.unpack("<Q", self._file.read(8))[0]

        logger.debug(f"GGUF: version={version}, tensors={n_tensors}, kv_pairs={n_kv}")

        # Read metadata key-value pairs
        for _ in range(n_kv):
            key = self._read_string()
            value = self._read_value()
            self.metadata[key] = value

        # Read tensor info
        for _ in range(n_tensors):
            info = self._read_tensor_info()
            self.tensors[info.name] = info

        # Calculate data offset (aligned)
        alignment = self.metadata.get("general.alignment", 32)
        current = self._file.tell()
        self._data_offset = ((current + alignment - 1) // alignment) * alignment

    def _read_string(self) -> str:
        """Read a length-prefixed string."""
        length = struct.unpack("<Q", self._file.read(8))[0]
        data = self._file.read(length)
        return data.decode("utf-8")

    def _read_value(self) -> Any:
        """Read a typed value."""
        vtype = struct.unpack("<I", self._file.read(4))[0]

        if vtype == GGUFValueType.UINT8:
            return struct.unpack("<B", self._file.read(1))[0]
        elif vtype == GGUFValueType.INT8:
            return struct.unpack("<b", self._file.read(1))[0]
        elif vtype == GGUFValueType.UINT16:
            return struct.unpack("<H", self._file.read(2))[0]
        elif vtype == GGUFValueType.INT16:
            return struct.unpack("<h", self._file.read(2))[0]
        elif vtype == GGUFValueType.UINT32:
            return struct.unpack("<I", self._file.read(4))[0]
        elif vtype == GGUFValueType.INT32:
            return struct.unpack("<i", self._file.read(4))[0]
        elif vtype == GGUFValueType.FLOAT32:
            return struct.unpack("<f", self._file.read(4))[0]
        elif vtype == GGUFValueType.BOOL:
            return struct.unpack("<B", self._file.read(1))[0] != 0
        elif vtype == GGUFValueType.STRING:
            return self._read_string()
        elif vtype == GGUFValueType.ARRAY:
            arr_type = struct.unpack("<I", self._file.read(4))[0]
            arr_len = struct.unpack("<Q", self._file.read(8))[0]
            return [self._read_array_value(arr_type) for _ in range(arr_len)]
        elif vtype == GGUFValueType.UINT64:
            return struct.unpack("<Q", self._file.read(8))[0]
        elif vtype == GGUFValueType.INT64:
            return struct.unpack("<q", self._file.read(8))[0]
        elif vtype == GGUFValueType.FLOAT64:
            return struct.unpack("<d", self._file.read(8))[0]
        else:
            raise ValueError(f"Unknown value type: {vtype}")

    def _read_array_value(self, vtype: int) -> Any:
        """Read an array element value."""
        if vtype == GGUFValueType.UINT8:
            return struct.unpack("<B", self._file.read(1))[0]
        elif vtype == GGUFValueType.INT8:
            return struct.unpack("<b", self._file.read(1))[0]
        elif vtype == GGUFValueType.UINT16:
            return struct.unpack("<H", self._file.read(2))[0]
        elif vtype == GGUFValueType.INT16:
            return struct.unpack("<h", self._file.read(2))[0]
        elif vtype == GGUFValueType.UINT32:
            return struct.unpack("<I", self._file.read(4))[0]
        elif vtype == GGUFValueType.INT32:
            return struct.unpack("<i", self._file.read(4))[0]
        elif vtype == GGUFValueType.FLOAT32:
            return struct.unpack("<f", self._file.read(4))[0]
        elif vtype == GGUFValueType.BOOL:
            return struct.unpack("<B", self._file.read(1))[0] != 0
        elif vtype == GGUFValueType.STRING:
            return self._read_string()
        elif vtype == GGUFValueType.UINT64:
            return struct.unpack("<Q", self._file.read(8))[0]
        elif vtype == GGUFValueType.INT64:
            return struct.unpack("<q", self._file.read(8))[0]
        elif vtype == GGUFValueType.FLOAT64:
            return struct.unpack("<d", self._file.read(8))[0]
        else:
            raise ValueError(f"Unknown array value type: {vtype}")

    def _read_tensor_info(self) -> GGUFTensorInfo:
        """Read tensor information."""
        name = self._read_string()
        n_dims = struct.unpack("<I", self._file.read(4))[0]
        dims = tuple(struct.unpack("<Q", self._file.read(8))[0] for _ in range(n_dims))
        qtype = GGUFQuantType(struct.unpack("<I", self._file.read(4))[0])
        offset = struct.unpack("<Q", self._file.read(8))[0]

        return GGUFTensorInfo(name=name, n_dims=n_dims, dims=dims, type=qtype, offset=offset)

    def get_tensor(self, name: str) -> np.ndarray:
        """Read a tensor from the file."""
        if name not in self.tensors:
            raise KeyError(f"Tensor '{name}' not found")

        info = self.tensors[name]

        # Seek to tensor data
        self._file.seek(self._data_offset + info.offset)

        # Calculate size
        n_elements = 1
        for d in info.dims:
            n_elements *= d

        # Read based on type
        if info.type == GGUFQuantType.F32:
            data = np.frombuffer(self._file.read(n_elements * 4), dtype=np.float32)
        elif info.type == GGUFQuantType.F16:
            data = np.frombuffer(self._file.read(n_elements * 2), dtype=np.float16)
        elif info.type == GGUFQuantType.I8:
            data = np.frombuffer(self._file.read(n_elements), dtype=np.int8)
        elif info.type == GGUFQuantType.I16:
            data = np.frombuffer(self._file.read(n_elements * 2), dtype=np.int16)
        elif info.type == GGUFQuantType.I32:
            data = np.frombuffer(self._file.read(n_elements * 4), dtype=np.int32)
        elif info.type == GGUFQuantType.I64:
            data = np.frombuffer(self._file.read(n_elements * 8), dtype=np.int64)
        elif info.type == GGUFQuantType.BF16:
            # Read as uint16, interpret as bfloat16
            raw = np.frombuffer(self._file.read(n_elements * 2), dtype=np.uint16)
            # Convert bfloat16 to float32
            data = np.zeros(n_elements, dtype=np.float32)
            data.view(np.uint32)[:] = raw.astype(np.uint32) << 16
        else:
            # Quantized types - read raw bytes
            block_size = _get_block_size(info.type)
            type_size = _get_type_size(info.type)
            n_bytes = int(n_elements * type_size)
            data = np.frombuffer(self._file.read(n_bytes), dtype=np.uint8)

        return data.reshape(info.dims) if info.n_dims > 1 and len(data.shape) == 1 else data

    def get_metadata(self) -> GGUFMetadata:
        """Get parsed metadata."""
        return GGUFMetadata.from_dict(self.metadata)

    def list_tensors(self) -> List[str]:
        """List all tensor names."""
        return list(self.tensors.keys())


class GGUFWriter:
    """Write GGUF format files."""

    def __init__(self, path: str, metadata: Optional[GGUFMetadata] = None):
        """Initialize writer with file path."""
        self.path = path
        self.metadata = metadata or GGUFMetadata()
        self._kv_data: Dict[str, Tuple[int, bytes]] = {}
        self._tensors: List[Tuple[str, np.ndarray, GGUFQuantType]] = []

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata key-value pair."""
        vtype, data = self._encode_value(value)
        self._kv_data[key] = (vtype, data)

    def add_tensor(
        self,
        name: str,
        data: np.ndarray,
        qtype: Optional[GGUFQuantType] = None
    ) -> None:
        """Add a tensor to the file."""
        # Infer type from dtype if not specified
        if qtype is None:
            if data.dtype == np.float32:
                qtype = GGUFQuantType.F32
            elif data.dtype == np.float16:
                qtype = GGUFQuantType.F16
            elif data.dtype == np.int8:
                qtype = GGUFQuantType.I8
            elif data.dtype == np.int16:
                qtype = GGUFQuantType.I16
            elif data.dtype == np.int32:
                qtype = GGUFQuantType.I32
            elif data.dtype == np.int64:
                qtype = GGUFQuantType.I64
            elif data.dtype == np.uint8:
                qtype = GGUFQuantType.I8
            else:
                qtype = GGUFQuantType.F32
                data = data.astype(np.float32)

        self._tensors.append((name, data, qtype))

    def _encode_value(self, value: Any) -> Tuple[int, bytes]:
        """Encode a value to bytes."""
        if isinstance(value, bool):
            return GGUFValueType.BOOL, struct.pack("<B", 1 if value else 0)
        elif isinstance(value, int):
            if value < 0:
                if value >= -128:
                    return GGUFValueType.INT8, struct.pack("<b", value)
                elif value >= -32768:
                    return GGUFValueType.INT16, struct.pack("<h", value)
                elif value >= -2147483648:
                    return GGUFValueType.INT32, struct.pack("<i", value)
                else:
                    return GGUFValueType.INT64, struct.pack("<q", value)
            else:
                if value <= 255:
                    return GGUFValueType.UINT8, struct.pack("<B", value)
                elif value <= 65535:
                    return GGUFValueType.UINT16, struct.pack("<H", value)
                elif value <= 4294967295:
                    return GGUFValueType.UINT32, struct.pack("<I", value)
                else:
                    return GGUFValueType.UINT64, struct.pack("<Q", value)
        elif isinstance(value, float):
            return GGUFValueType.FLOAT32, struct.pack("<f", value)
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            return GGUFValueType.STRING, struct.pack("<Q", len(encoded)) + encoded
        elif isinstance(value, (list, tuple)):
            if not value:
                return GGUFValueType.ARRAY, struct.pack("<IQ", GGUFValueType.UINT8, 0)
            # Use first element to determine type
            elem_type, _ = self._encode_value(value[0])
            elements = b"".join(self._encode_value(v)[1] for v in value)
            return GGUFValueType.ARRAY, struct.pack("<IQ", elem_type, len(value)) + elements
        else:
            raise TypeError(f"Cannot encode type {type(value)}")

    def _encode_string(self, s: str) -> bytes:
        """Encode a length-prefixed string."""
        encoded = s.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def write(self) -> None:
        """Write the GGUF file."""
        # Prepare metadata
        meta_dict = self.metadata.to_dict()
        for key, value in meta_dict.items():
            if key not in self._kv_data:
                vtype, data = self._encode_value(value)
                self._kv_data[key] = (vtype, data)

        with open(self.path, "wb") as f:
            # Write header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self._tensors)))
            f.write(struct.pack("<Q", len(self._kv_data)))

            # Write key-value pairs
            for key, (vtype, data) in self._kv_data.items():
                f.write(self._encode_string(key))
                f.write(struct.pack("<I", vtype))
                f.write(data)

            # Calculate tensor offsets
            alignment = self.metadata.alignment
            header_end = f.tell()

            # Write tensor info
            offset = 0
            tensor_data_list = []
            for name, data, qtype in self._tensors:
                f.write(self._encode_string(name))
                f.write(struct.pack("<I", len(data.shape)))
                for dim in data.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", int(qtype)))
                f.write(struct.pack("<Q", offset))

                # Calculate size
                tensor_bytes = data.tobytes()
                tensor_data_list.append(tensor_bytes)
                size = len(tensor_bytes)
                # Align offset
                offset += size
                padding = (alignment - (offset % alignment)) % alignment
                offset += padding

            # Align to data section
            current = f.tell()
            data_start = ((current + alignment - 1) // alignment) * alignment
            f.write(b"\x00" * (data_start - current))

            # Write tensor data
            for i, (name, data, qtype) in enumerate(self._tensors):
                tensor_bytes = tensor_data_list[i]
                f.write(tensor_bytes)

                # Align
                size = len(tensor_bytes)
                padding = (alignment - (size % alignment)) % alignment
                f.write(b"\x00" * padding)

        logger.info(f"Wrote GGUF file: {self.path}")


def convert_to_gguf(
    weights: Dict[str, np.ndarray],
    output_path: str,
    metadata: Optional[GGUFMetadata] = None,
    qtype: GGUFQuantType = GGUFQuantType.F16
) -> None:
    """Convert model weights to GGUF format.

    Args:
        weights: Dictionary of tensor name to numpy array
        output_path: Output file path
        metadata: Optional metadata (uses defaults if not provided)
        qtype: Quantization type for all tensors
    """
    writer = GGUFWriter(output_path, metadata)

    for name, tensor in weights.items():
        # Convert to appropriate dtype
        if qtype == GGUFQuantType.F32:
            tensor = tensor.astype(np.float32)
        elif qtype == GGUFQuantType.F16:
            tensor = tensor.astype(np.float16)
        elif qtype in (GGUFQuantType.I8, GGUFQuantType.Q8_0):
            tensor = tensor.astype(np.int8)

        writer.add_tensor(name, tensor, qtype)

    writer.write()


def load_from_gguf(path: str) -> Tuple[Dict[str, np.ndarray], GGUFMetadata]:
    """Load model weights from GGUF format.

    Args:
        path: Path to GGUF file

    Returns:
        Tuple of (weights dict, metadata)
    """
    weights = {}

    with GGUFReader(path) as reader:
        metadata = reader.get_metadata()

        for name in reader.list_tensors():
            weights[name] = reader.get_tensor(name)

    return weights, metadata
