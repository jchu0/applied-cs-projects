"""Tests for GGUF format support."""

import unittest
import tempfile
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vqllm.formats.gguf import (
    GGUFReader,
    GGUFWriter,
    GGUFMetadata,
    GGUFQuantType,
    GGUFValueType,
    GGUF_MAGIC,
    GGUF_VERSION,
    convert_to_gguf,
    load_from_gguf,
)


class TestGGUFMetadata(unittest.TestCase):
    """Tests for GGUF metadata."""

    def test_default_metadata(self):
        """Test default metadata values."""
        metadata = GGUFMetadata()
        self.assertEqual(metadata.architecture, "llama")
        self.assertEqual(metadata.context_length, 2048)
        self.assertEqual(metadata.embedding_length, 4096)
        self.assertEqual(metadata.block_count, 32)

    def test_custom_metadata(self):
        """Test custom metadata values."""
        metadata = GGUFMetadata(
            architecture="mistral",
            context_length=4096,
            embedding_length=8192,
            block_count=64,
            name="test-model",
        )
        self.assertEqual(metadata.architecture, "mistral")
        self.assertEqual(metadata.context_length, 4096)
        self.assertEqual(metadata.name, "test-model")

    def test_to_dict(self):
        """Test metadata to dictionary conversion."""
        metadata = GGUFMetadata(
            architecture="llama",
            name="test-model",
            context_length=2048,
        )
        d = metadata.to_dict()

        self.assertEqual(d["general.architecture"], "llama")
        self.assertEqual(d["general.name"], "test-model")
        self.assertEqual(d["llama.context_length"], 2048)

    def test_from_dict(self):
        """Test metadata from dictionary."""
        d = {
            "general.architecture": "mistral",
            "general.name": "test-model",
            "mistral.context_length": 4096,
            "mistral.embedding_length": 8192,
        }
        metadata = GGUFMetadata.from_dict(d)

        self.assertEqual(metadata.architecture, "mistral")
        self.assertEqual(metadata.name, "test-model")
        self.assertEqual(metadata.context_length, 4096)
        self.assertEqual(metadata.embedding_length, 8192)

    def test_extra_metadata(self):
        """Test extra metadata key-value pairs."""
        metadata = GGUFMetadata(extra={"custom.key": "value", "custom.number": 42})
        d = metadata.to_dict()

        self.assertEqual(d["custom.key"], "value")
        self.assertEqual(d["custom.number"], 42)


class TestGGUFWriter(unittest.TestCase):
    """Tests for GGUF writer."""

    def setUp(self):
        """Set up temporary file for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test.gguf")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)

    def test_write_empty_file(self):
        """Test writing an empty GGUF file."""
        metadata = GGUFMetadata(name="empty-model")
        writer = GGUFWriter(self.temp_file, metadata)
        writer.write()

        self.assertTrue(os.path.exists(self.temp_file))
        self.assertGreater(os.path.getsize(self.temp_file), 0)

    def test_write_f32_tensor(self):
        """Test writing F32 tensor."""
        writer = GGUFWriter(self.temp_file)
        tensor = np.random.randn(32, 64).astype(np.float32)
        writer.add_tensor("test.weight", tensor, GGUFQuantType.F32)
        writer.write()

        self.assertTrue(os.path.exists(self.temp_file))

    def test_write_f16_tensor(self):
        """Test writing F16 tensor."""
        writer = GGUFWriter(self.temp_file)
        tensor = np.random.randn(32, 64).astype(np.float16)
        writer.add_tensor("test.weight", tensor, GGUFQuantType.F16)
        writer.write()

        self.assertTrue(os.path.exists(self.temp_file))

    def test_write_int8_tensor(self):
        """Test writing INT8 tensor."""
        writer = GGUFWriter(self.temp_file)
        tensor = np.random.randint(-128, 127, size=(32, 64), dtype=np.int8)
        writer.add_tensor("test.weight", tensor, GGUFQuantType.I8)
        writer.write()

        self.assertTrue(os.path.exists(self.temp_file))

    def test_write_multiple_tensors(self):
        """Test writing multiple tensors."""
        writer = GGUFWriter(self.temp_file)
        writer.add_tensor("layer.0.weight", np.random.randn(32, 64).astype(np.float32))
        writer.add_tensor("layer.0.bias", np.random.randn(32).astype(np.float32))
        writer.add_tensor("layer.1.weight", np.random.randn(64, 32).astype(np.float32))
        writer.write()

        self.assertTrue(os.path.exists(self.temp_file))

    def test_add_metadata(self):
        """Test adding custom metadata."""
        writer = GGUFWriter(self.temp_file)
        writer.add_metadata("custom.string", "test")
        writer.add_metadata("custom.int", 42)
        writer.add_metadata("custom.float", 3.14)
        writer.add_metadata("custom.bool", True)
        writer.write()

        self.assertTrue(os.path.exists(self.temp_file))

    def test_auto_infer_dtype(self):
        """Test automatic dtype inference."""
        writer = GGUFWriter(self.temp_file)
        # Don't specify qtype, let it infer
        writer.add_tensor("f32", np.random.randn(10).astype(np.float32))
        writer.add_tensor("f16", np.random.randn(10).astype(np.float16))
        writer.add_tensor("i8", np.random.randint(-128, 127, size=10, dtype=np.int8))
        writer.write()

        self.assertTrue(os.path.exists(self.temp_file))


class TestGGUFReader(unittest.TestCase):
    """Tests for GGUF reader."""

    def setUp(self):
        """Set up temporary file for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test.gguf")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)

    def test_read_header(self):
        """Test reading GGUF header."""
        # Write a file first
        metadata = GGUFMetadata(name="test-model", architecture="llama")
        writer = GGUFWriter(self.temp_file, metadata)
        writer.add_tensor("test.weight", np.random.randn(32, 64).astype(np.float32))
        writer.write()

        # Read it back
        with GGUFReader(self.temp_file) as reader:
            self.assertIn("general.name", reader.metadata)
            self.assertEqual(reader.metadata["general.name"], "test-model")

    def test_read_tensor_f32(self):
        """Test reading F32 tensor."""
        original = np.random.randn(32, 64).astype(np.float32)

        writer = GGUFWriter(self.temp_file)
        writer.add_tensor("test.weight", original, GGUFQuantType.F32)
        writer.write()

        with GGUFReader(self.temp_file) as reader:
            loaded = reader.get_tensor("test.weight")
            np.testing.assert_array_almost_equal(original, loaded)

    def test_read_tensor_f16(self):
        """Test reading F16 tensor."""
        original = np.random.randn(32, 64).astype(np.float16)

        writer = GGUFWriter(self.temp_file)
        writer.add_tensor("test.weight", original, GGUFQuantType.F16)
        writer.write()

        with GGUFReader(self.temp_file) as reader:
            loaded = reader.get_tensor("test.weight")
            np.testing.assert_array_almost_equal(original, loaded, decimal=3)

    def test_read_tensor_int8(self):
        """Test reading INT8 tensor."""
        original = np.random.randint(-128, 127, size=(32, 64), dtype=np.int8)

        writer = GGUFWriter(self.temp_file)
        writer.add_tensor("test.weight", original, GGUFQuantType.I8)
        writer.write()

        with GGUFReader(self.temp_file) as reader:
            loaded = reader.get_tensor("test.weight")
            np.testing.assert_array_equal(original, loaded)

    def test_list_tensors(self):
        """Test listing tensor names."""
        writer = GGUFWriter(self.temp_file)
        writer.add_tensor("layer.0.weight", np.random.randn(32, 64).astype(np.float32))
        writer.add_tensor("layer.0.bias", np.random.randn(32).astype(np.float32))
        writer.add_tensor("layer.1.weight", np.random.randn(64, 32).astype(np.float32))
        writer.write()

        with GGUFReader(self.temp_file) as reader:
            names = reader.list_tensors()
            self.assertEqual(len(names), 3)
            self.assertIn("layer.0.weight", names)
            self.assertIn("layer.0.bias", names)
            self.assertIn("layer.1.weight", names)

    def test_read_metadata(self):
        """Test reading metadata."""
        metadata = GGUFMetadata(
            name="test-model",
            architecture="llama",
            context_length=4096,
            vocab_size=50000,
        )
        writer = GGUFWriter(self.temp_file, metadata)
        writer.add_tensor("test.weight", np.random.randn(32).astype(np.float32))
        writer.write()

        with GGUFReader(self.temp_file) as reader:
            loaded_meta = reader.get_metadata()
            self.assertEqual(loaded_meta.name, "test-model")
            self.assertEqual(loaded_meta.architecture, "llama")
            self.assertEqual(loaded_meta.context_length, 4096)
            self.assertEqual(loaded_meta.vocab_size, 50000)

    def test_tensor_not_found(self):
        """Test error when tensor not found."""
        writer = GGUFWriter(self.temp_file)
        writer.add_tensor("test.weight", np.random.randn(32).astype(np.float32))
        writer.write()

        with GGUFReader(self.temp_file) as reader:
            with self.assertRaises(KeyError):
                reader.get_tensor("nonexistent")


class TestGGUFRoundTrip(unittest.TestCase):
    """Tests for GGUF round-trip (write then read)."""

    def setUp(self):
        """Set up temporary file for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test.gguf")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)

    def test_roundtrip_multiple_tensors(self):
        """Test round-trip with multiple tensors."""
        tensors = {
            "embed.weight": np.random.randn(1000, 256).astype(np.float32),
            "layer.0.attn.q": np.random.randn(256, 256).astype(np.float32),
            "layer.0.attn.k": np.random.randn(256, 256).astype(np.float32),
            "layer.0.attn.v": np.random.randn(256, 256).astype(np.float32),
            "layer.0.ffn.w1": np.random.randn(256, 1024).astype(np.float32),
            "layer.0.ffn.w2": np.random.randn(1024, 256).astype(np.float32),
        }

        writer = GGUFWriter(self.temp_file)
        for name, tensor in tensors.items():
            writer.add_tensor(name, tensor, GGUFQuantType.F32)
        writer.write()

        with GGUFReader(self.temp_file) as reader:
            for name, original in tensors.items():
                loaded = reader.get_tensor(name)
                np.testing.assert_array_almost_equal(original, loaded)

    def test_roundtrip_large_tensor(self):
        """Test round-trip with large tensor."""
        # 4096 x 4096 = 16M elements = 64MB in F32
        original = np.random.randn(4096, 4096).astype(np.float32)

        writer = GGUFWriter(self.temp_file)
        writer.add_tensor("large.weight", original, GGUFQuantType.F32)
        writer.write()

        with GGUFReader(self.temp_file) as reader:
            loaded = reader.get_tensor("large.weight")
            np.testing.assert_array_almost_equal(original, loaded)

    def test_roundtrip_mixed_dtypes(self):
        """Test round-trip with mixed dtypes."""
        tensors = [
            ("f32_tensor", np.random.randn(32, 64).astype(np.float32), GGUFQuantType.F32),
            ("f16_tensor", np.random.randn(32, 64).astype(np.float16), GGUFQuantType.F16),
            ("i8_tensor", np.random.randint(-128, 127, size=(32, 64), dtype=np.int8), GGUFQuantType.I8),
        ]

        writer = GGUFWriter(self.temp_file)
        for name, tensor, qtype in tensors:
            writer.add_tensor(name, tensor, qtype)
        writer.write()

        with GGUFReader(self.temp_file) as reader:
            for name, original, qtype in tensors:
                loaded = reader.get_tensor(name)
                if qtype == GGUFQuantType.F16:
                    np.testing.assert_array_almost_equal(original, loaded, decimal=3)
                elif qtype == GGUFQuantType.I8:
                    np.testing.assert_array_equal(original, loaded)
                else:
                    np.testing.assert_array_almost_equal(original, loaded)


class TestConvertLoadFunctions(unittest.TestCase):
    """Tests for convert_to_gguf and load_from_gguf functions."""

    def setUp(self):
        """Set up temporary file for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test.gguf")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)

    def test_convert_to_gguf_f16(self):
        """Test convert_to_gguf with F16."""
        weights = {
            "layer.0.weight": np.random.randn(32, 64).astype(np.float32),
            "layer.0.bias": np.random.randn(32).astype(np.float32),
        }
        metadata = GGUFMetadata(name="test-model")

        convert_to_gguf(weights, self.temp_file, metadata, GGUFQuantType.F16)

        self.assertTrue(os.path.exists(self.temp_file))

    def test_load_from_gguf(self):
        """Test load_from_gguf."""
        weights = {
            "layer.0.weight": np.random.randn(32, 64).astype(np.float32),
            "layer.0.bias": np.random.randn(32).astype(np.float32),
        }
        metadata = GGUFMetadata(name="test-model", context_length=4096)

        convert_to_gguf(weights, self.temp_file, metadata, GGUFQuantType.F32)

        loaded_weights, loaded_meta = load_from_gguf(self.temp_file)

        self.assertEqual(loaded_meta.name, "test-model")
        self.assertEqual(loaded_meta.context_length, 4096)
        self.assertEqual(len(loaded_weights), 2)
        np.testing.assert_array_almost_equal(
            weights["layer.0.weight"], loaded_weights["layer.0.weight"]
        )

    def test_convert_without_metadata(self):
        """Test convert_to_gguf without explicit metadata."""
        weights = {"test.weight": np.random.randn(32, 64).astype(np.float32)}

        convert_to_gguf(weights, self.temp_file)

        self.assertTrue(os.path.exists(self.temp_file))


class TestGGUFQuantTypes(unittest.TestCase):
    """Tests for GGUF quantization types."""

    def test_quant_type_values(self):
        """Test quantization type enum values."""
        self.assertEqual(GGUFQuantType.F32, 0)
        self.assertEqual(GGUFQuantType.F16, 1)
        self.assertEqual(GGUFQuantType.Q4_0, 2)
        self.assertEqual(GGUFQuantType.Q8_0, 8)
        self.assertEqual(GGUFQuantType.I8, 24)

    def test_value_type_values(self):
        """Test value type enum values."""
        self.assertEqual(GGUFValueType.UINT8, 0)
        self.assertEqual(GGUFValueType.STRING, 8)
        self.assertEqual(GGUFValueType.ARRAY, 9)


class TestGGUFConstants(unittest.TestCase):
    """Tests for GGUF constants."""

    def test_magic_number(self):
        """Test GGUF magic number."""
        # "GGUF" in little-endian
        self.assertEqual(GGUF_MAGIC, 0x46554747)

    def test_version(self):
        """Test GGUF version."""
        self.assertEqual(GGUF_VERSION, 3)


if __name__ == "__main__":
    unittest.main()
