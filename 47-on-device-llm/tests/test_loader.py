"""Tests for the real GGUFLoader binary parser.

The rest of the suite exercises MockGGUFLoader; these tests write minimal
valid GGUF files to disk and parse them through the real header/metadata/
tensor path (mmap access, dequantization, config extraction) documented in
the README, and cover the error paths.
"""

import struct

import numpy as np
import pytest

from on_device_llm.loader import GGUFLoader, TensorInfo, ModelConfig
from on_device_llm.quantization import GGMLType, quantize_q8_0


GGUF_MAGIC = 0x46554747

# GGUF value type ids (mirrors GGUFLoader constants).
VT_UINT32 = 4
VT_FLOAT32 = 6
VT_STRING = 8


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _meta_uint32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", VT_UINT32) + struct.pack("<I", value)


def _meta_float32(key: str, value: float) -> bytes:
    return _string(key) + struct.pack("<I", VT_FLOAT32) + struct.pack("<f", value)


def _meta_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", VT_STRING) + _string(value)


def _tensor_header(name: str, shape, dtype: GGMLType) -> bytes:
    out = _string(name)
    out += struct.pack("<I", len(shape))
    for dim in shape:
        out += struct.pack("<Q", dim)
    out += struct.pack("<I", int(dtype))
    out += struct.pack("<Q", 0)  # relative offset (ignored by loader)
    return out


def _align32(n: int) -> int:
    return (n + 31) // 32 * 32


def build_and_write(tmp_path, meta_entries, tensors, version=3):
    """Serialize a GGUF file and return (path, expected_payloads)."""
    body = struct.pack("<I", GGUF_MAGIC)
    body += struct.pack("<I", version)
    body += struct.pack("<Q", len(tensors))       # n_tensors
    body += struct.pack("<Q", len(meta_entries))  # n_metadata
    for entry in meta_entries:
        body += entry
    for name, shape, dtype, _payload in tensors:
        body += _tensor_header(name, shape, dtype)

    # Data section is aligned to 32 bytes.
    data_offset = _align32(len(body))
    body += b"\x00" * (data_offset - len(body))

    expected = {}
    for name, shape, dtype, payload in tensors:
        # Each tensor is 32-byte aligned within the data section.
        cur = len(body)
        aligned = _align32(cur)
        body += b"\x00" * (aligned - cur)
        body += payload
        expected[name] = (shape, dtype, payload)

    path = tmp_path / "model.gguf"
    path.write_bytes(body)
    return str(path), expected


class TestGGUFLoaderRoundTrip:
    def test_parse_header_metadata_and_config(self, tmp_path):
        meta = [
            _meta_string("general.architecture", "llama"),
            _meta_uint32("llama.vocab_size", 128),
            _meta_uint32("llama.embedding_length", 16),
            _meta_uint32("llama.block_count", 2),
            _meta_uint32("llama.attention.head_count", 4),
            _meta_uint32("llama.attention.head_count_kv", 2),
            _meta_uint32("llama.feed_forward_length", 32),
            _meta_uint32("llama.context_length", 64),
            _meta_float32("llama.rope.freq_base", 10000.0),
            _meta_float32("llama.attention.layer_norm_rms_epsilon", 1e-5),
        ]
        # One F32 tensor: shape (4,), contiguous float32 payload.
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tensors = [("token_embd.weight", (4,), GGMLType.F32, vec.tobytes())]
        path, _ = build_and_write(tmp_path, meta, tensors)

        with GGUFLoader(path) as loader:
            assert loader.metadata["general.architecture"] == "llama"
            cfg = loader.config
            assert isinstance(cfg, ModelConfig)
            assert cfg.vocab_size == 128
            assert cfg.hidden_size == 16
            assert cfg.num_layers == 2
            assert cfg.num_heads == 4
            assert cfg.num_kv_heads == 2
            assert cfg.head_dim == 4  # 16 / 4

    def test_get_tensor_f32_roundtrip(self, tmp_path):
        vec = np.arange(8, dtype=np.float32)
        tensors = [("w", (8,), GGMLType.F32, vec.tobytes())]
        path, _ = build_and_write(tmp_path, [], tensors)
        with GGUFLoader(path) as loader:
            out = loader.get_tensor("w")
            np.testing.assert_array_equal(out, vec)

    def test_get_tensor_q8_0_dequantizes(self, tmp_path):
        w = (np.arange(32, dtype=np.float32) - 16) * 0.5
        payload, _ = quantize_q8_0(w)
        tensors = [("qw", (32,), GGMLType.Q8_0, payload)]
        path, _ = build_and_write(tmp_path, [], tensors)
        with GGUFLoader(path) as loader:
            out = loader.get_tensor("qw")
            assert out.shape == (32,)
            # Q8_0 is lossy but should track the original closely.
            assert np.max(np.abs(out - w)) < 0.1

    def test_list_and_info(self, tmp_path):
        vec = np.ones(4, dtype=np.float32)
        tensors = [("a", (4,), GGMLType.F32, vec.tobytes())]
        path, _ = build_and_write(tmp_path, [], tensors)
        with GGUFLoader(path) as loader:
            assert loader.list_tensors() == ["a"]
            info = loader.get_tensor_info("a")
            assert isinstance(info, TensorInfo)
            assert info.shape == (4,)
            assert info.dtype == GGMLType.F32
            assert loader.get_tensor_info("missing") is None


class TestGGUFLoaderErrors:
    def test_bad_magic_raises(self, tmp_path):
        path = tmp_path / "bad.gguf"
        path.write_bytes(struct.pack("<I", 0xDEADBEEF) + b"\x00" * 32)
        with pytest.raises(ValueError, match="magic"):
            GGUFLoader(str(path))

    def test_old_version_raises(self, tmp_path):
        body = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 1) + b"\x00" * 16
        path = tmp_path / "old.gguf"
        path.write_bytes(body)
        with pytest.raises(ValueError, match="version"):
            GGUFLoader(str(path))

    def test_unknown_tensor_raises_keyerror(self, tmp_path):
        vec = np.ones(4, dtype=np.float32)
        tensors = [("a", (4,), GGMLType.F32, vec.tobytes())]
        path, _ = build_and_write(tmp_path, [], tensors)
        with GGUFLoader(path) as loader:
            with pytest.raises(KeyError):
                loader.get_tensor("nope")

    def test_kquant_get_tensor_raises_not_implemented(self, tmp_path):
        # Q4_K is a disclosed unsupported path; getting such a tensor must
        # raise NotImplementedError rather than fabricate data.
        tensors = [("kq", (256,), GGMLType.Q4_K, b"\x00" * 256)]
        path, _ = build_and_write(tmp_path, [], tensors)
        with GGUFLoader(path) as loader:
            with pytest.raises(NotImplementedError):
                loader.get_tensor("kq")
