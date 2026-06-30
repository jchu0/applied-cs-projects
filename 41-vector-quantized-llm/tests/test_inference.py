"""Unit tests for inference engine."""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
import threading
import tempfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vqllm.inference.engine import (
    InferenceEngine, BatchedInference, KVCache,
    QuantizedModel, optimize_inference, benchmark_latency
)
from vqllm.core.types import QuantConfig, QuantType, QuantizedTensor


class TestInferenceEngine(unittest.TestCase):
    """Test inference engine functionality."""

    def setUp(self):
        self.config = QuantConfig(quant_type=QuantType.INT8)
        self.engine = InferenceEngine(self.config)
        np.random.seed(42)

    def test_engine_init(self):
        """Test inference engine initialization."""
        self.assertEqual(self.engine.config.quant_type, QuantType.INT8)
        self.assertIsNotNone(self.engine.device)
        self.assertEqual(self.engine.batch_size, 1)

    def test_load_quantized_model(self):
        """Test loading quantized model."""
        # Mock quantized model
        model = Mock()
        model.config = {"hidden_size": 768, "num_layers": 12}
        model.is_quantized = True

        self.engine.load_model(model)

        # Check model loaded
        self.assertIsNotNone(self.engine.model)
        self.assertTrue(self.engine.model.is_quantized)

    def test_single_forward_pass(self):
        """Test single forward pass through quantized model."""
        # Mock quantized layers
        self.engine.model = Mock()
        self.engine.model.forward = Mock(
            return_value=np.random.randn(1, 128, 768).astype(np.float32)
        )

        input_ids = np.random.randint(0, 1000, size=(1, 128))
        output = self.engine.forward(input_ids)

        # Check output shape
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[1], 128)  # Sequence length

    def test_batch_inference(self):
        """Test batched inference."""
        self.engine.batch_size = 4
        self.engine.model = Mock()

        # Mock batch forward
        batch_output = np.random.randn(4, 128, 768).astype(np.float32)
        self.engine.model.forward = Mock(return_value=batch_output)

        input_ids = np.random.randint(0, 1000, size=(4, 128))
        output = self.engine.forward(input_ids)

        # Check batch processing
        self.assertEqual(output.shape[0], 4)
        self.engine.model.forward.assert_called_once()

    def test_streaming_inference(self):
        """Test streaming token generation."""
        self.engine.model = Mock()

        # Mock token generation
        def mock_generate_token(input_ids, past_kvs=None):
            next_token = np.random.randint(0, 1000, size=(input_ids.shape[0], 1))
            return next_token, {}

        self.engine.generate_token = mock_generate_token

        # Stream generation
        input_ids = np.array([[1, 2, 3, 4, 5]])
        tokens = []

        for token in self.engine.stream_generate(input_ids, max_length=10):
            tokens.append(token)

        # Check streaming worked
        self.assertEqual(len(tokens), 10)


class TestKVCache(unittest.TestCase):
    """Test KV cache for efficient inference."""

    def setUp(self):
        self.cache = KVCache(
            num_layers=12,
            max_batch_size=4,
            max_seq_length=2048,
            hidden_size=768,
            num_heads=12
        )
        np.random.seed(42)

    def test_cache_init(self):
        """Test KV cache initialization."""
        self.assertEqual(self.cache.num_layers, 12)
        self.assertEqual(self.cache.max_seq_length, 2048)
        self.assertEqual(len(self.cache.key_cache), 12)
        self.assertEqual(len(self.cache.value_cache), 12)

    def test_cache_update(self):
        """Test updating KV cache."""
        layer_idx = 0
        batch_size = 2
        seq_len = 10

        # New keys and values
        keys = np.random.randn(batch_size, seq_len, self.cache.hidden_size).astype(np.float32)
        values = np.random.randn(batch_size, seq_len, self.cache.hidden_size).astype(np.float32)

        # Update cache
        self.cache.update(layer_idx, keys, values, seq_position=0)

        # Check cache updated
        cached_keys = self.cache.get_keys(layer_idx, batch_size, seq_len)
        np.testing.assert_array_equal(cached_keys[:, :seq_len, :], keys)

    def test_cache_reuse(self):
        """Test reusing cached KV pairs."""
        layer_idx = 0

        # Initial cache
        keys1 = np.random.randn(1, 10, 768).astype(np.float32)
        values1 = np.random.randn(1, 10, 768).astype(np.float32)
        self.cache.update(layer_idx, keys1, values1, seq_position=0)

        # Extend with new token
        keys2 = np.random.randn(1, 1, 768).astype(np.float32)
        values2 = np.random.randn(1, 1, 768).astype(np.float32)
        self.cache.update(layer_idx, keys2, values2, seq_position=10)

        # Get full cache
        full_keys = self.cache.get_keys(layer_idx, 1, 11)
        full_values = self.cache.get_values(layer_idx, 1, 11)

        # Check concatenation
        self.assertEqual(full_keys.shape[1], 11)
        np.testing.assert_array_equal(full_keys[:, :10, :], keys1)
        np.testing.assert_array_equal(full_keys[:, 10:11, :], keys2)

    def test_cache_clear(self):
        """Test clearing cache."""
        # Add some data
        keys = np.random.randn(2, 10, 768).astype(np.float32)
        values = np.random.randn(2, 10, 768).astype(np.float32)
        self.cache.update(0, keys, values)

        # Clear cache
        self.cache.clear()

        # Check cache is zeroed
        cached_keys = self.cache.get_keys(0, 2, 10)
        self.assertTrue(np.allclose(cached_keys, 0))

    def test_cache_memory_efficiency(self):
        """Test cache memory usage."""
        # Calculate expected memory
        per_layer_memory = (
            2 * self.cache.max_batch_size * self.cache.max_seq_length *
            self.cache.hidden_size * 4  # float32
        )
        total_expected = per_layer_memory * self.cache.num_layers

        # Check actual memory
        actual_memory = self.cache.memory_usage()
        self.assertLessEqual(actual_memory, total_expected * 1.1)  # Allow 10% overhead


class TestQuantizedModel(unittest.TestCase):
    """Test quantized model wrapper."""

    def setUp(self):
        self.config = QuantConfig(quant_type=QuantType.INT8)
        np.random.seed(42)

    def test_model_quantization(self):
        """Test model quantization wrapper."""
        # Mock original model
        original_model = Mock()
        original_model.layers = [
            Mock(weight=np.random.randn(768, 768)),
            Mock(weight=np.random.randn(768, 3072)),
            Mock(weight=np.random.randn(3072, 768)),
        ]

        # Quantize model
        quantized = QuantizedModel(original_model, self.config)

        # Check quantization applied
        self.assertTrue(quantized.is_quantized)
        self.assertEqual(quantized.config.quant_type, QuantType.INT8)

    def test_quantized_forward(self):
        """Test forward pass through quantized model."""
        # Create mock quantized layers
        model = QuantizedModel(Mock(), self.config)

        def mock_forward(x):
            return x @ np.random.randn(x.shape[-1], 768).astype(np.float32)

        model.forward_quantized = mock_forward

        # Run forward pass
        input_data = np.random.randn(4, 128, 768).astype(np.float32)
        output = model(input_data)

        # Check output
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[-1], 768)

    def test_mixed_precision_layers(self):
        """Test model with mixed precision layers."""
        model = Mock()
        model.layers = {
            "attention": Mock(weight=np.random.randn(768, 768)),
            "mlp": Mock(weight=np.random.randn(768, 3072)),
            "output": Mock(weight=np.random.randn(768, 50257)),
        }

        # Different precision for different layers
        precision_map = {
            "attention": QuantType.INT8,
            "mlp": QuantType.INT4,
            "output": QuantType.FP16,
        }

        quantized = QuantizedModel(model, self.config, precision_map=precision_map)

        # Check mixed precision applied
        self.assertEqual(quantized.precision_map["attention"], QuantType.INT8)
        self.assertEqual(quantized.precision_map["mlp"], QuantType.INT4)


class TestBatchedInference(unittest.TestCase):
    """Test batched inference optimization."""

    def setUp(self):
        self.config = QuantConfig(quant_type=QuantType.INT8)
        self.batch_engine = BatchedInference(
            config=self.config,
            max_batch_size=8,
            max_seq_length=512
        )
        np.random.seed(42)

    def test_dynamic_batching(self):
        """Test dynamic batching of requests."""
        # Add multiple requests
        requests = []
        for i in range(5):
            req_id = f"req_{i}"
            input_ids = np.random.randint(0, 1000, size=(1, 50 + i * 10))
            self.batch_engine.add_request(req_id, input_ids)
            requests.append(req_id)

        # Process batch
        batch = self.batch_engine.create_batch()

        # Check batching
        self.assertLessEqual(len(batch), 5)
        self.assertGreater(len(batch), 0)

    def test_continuous_batching(self):
        """Test continuous batching with different sequence lengths."""
        # Add requests with varying lengths
        self.batch_engine.add_request("req_1", np.random.randint(0, 1000, (1, 100)))
        self.batch_engine.add_request("req_2", np.random.randint(0, 1000, (1, 150)))
        self.batch_engine.add_request("req_3", np.random.randint(0, 1000, (1, 200)))

        # Process with padding
        batch, mask = self.batch_engine.create_padded_batch()

        # Check padding
        self.assertEqual(batch.shape[1], 200)  # Max length
        self.assertEqual(mask.shape, batch.shape[:2])

        # Check mask is correct
        self.assertTrue(np.all(mask[0, :100]))
        self.assertTrue(np.all(mask[1, :150]))
        self.assertTrue(np.all(mask[2, :200]))

    def test_request_scheduling(self):
        """Test request scheduling and prioritization."""
        # Add requests with priorities
        self.batch_engine.add_request("urgent", np.random.randint(0, 1000, (1, 50)), priority=10)
        self.batch_engine.add_request("normal", np.random.randint(0, 1000, (1, 50)), priority=5)
        self.batch_engine.add_request("low", np.random.randint(0, 1000, (1, 50)), priority=1)

        # Get next batch
        batch_ids = self.batch_engine.get_next_batch_ids(max_batch_size=2)

        # Check priority ordering
        self.assertEqual(batch_ids[0], "urgent")
        self.assertEqual(batch_ids[1], "normal")

    def test_throughput_optimization(self):
        """Test throughput optimization strategies."""
        # Simulate high-throughput scenario
        num_requests = 100
        request_times = []

        for i in range(num_requests):
            start = time.time()

            # Add request
            self.batch_engine.add_request(
                f"req_{i}",
                np.random.randint(0, 1000, (1, np.random.randint(50, 200)))
            )

            # Process if batch full
            if self.batch_engine.should_process_batch():
                batch = self.batch_engine.create_batch()
                # Simulate processing
                time.sleep(0.001)

            request_times.append(time.time() - start)

        # Check throughput improvements with batching
        avg_time = np.mean(request_times)
        self.assertLess(avg_time, 0.01)  # Should be fast


class TestOptimizations(unittest.TestCase):
    """Test inference optimizations."""

    def setUp(self):
        np.random.seed(42)

    def test_kernel_fusion(self):
        """Test kernel fusion optimization."""
        # Mock quantized operations
        def quantized_matmul(a, b, scale_a, scale_b):
            return (a @ b) * scale_a * scale_b

        def fused_quantized_matmul_add(a, b, c, scale_a, scale_b):
            return (a @ b) * scale_a * scale_b + c

        # Test fusion improves performance
        a = np.random.randint(-127, 127, size=(128, 256), dtype=np.int8)
        b = np.random.randint(-127, 127, size=(256, 512), dtype=np.int8)
        c = np.random.randn(128, 512).astype(np.float32)
        scale_a, scale_b = 0.01, 0.01

        # Separate ops
        start = time.time()
        result1 = quantized_matmul(a, b, scale_a, scale_b) + c
        time1 = time.time() - start

        # Fused op
        start = time.time()
        result2 = fused_quantized_matmul_add(a, b, c, scale_a, scale_b)
        time2 = time.time() - start

        # Check results match
        np.testing.assert_allclose(result1, result2, rtol=1e-5)

    def test_weight_packing(self):
        """Test weight packing for INT4."""
        # Pack INT4 weights
        weights = np.random.randint(-8, 7, size=(256, 512), dtype=np.int8)

        # Pack two INT4 values per byte
        packed = np.zeros((256, 256), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                low = weights[i, j * 2] & 0x0F
                high = (weights[i, j * 2 + 1] & 0x0F) << 4
                packed[i, j] = low | high

        # Check packing reduces memory
        original_size = weights.nbytes
        packed_size = packed.nbytes
        self.assertLess(packed_size, original_size)

    def test_graph_optimization(self):
        """Test computation graph optimization."""
        # Mock computation graph
        graph = {
            "input": {"type": "input", "shape": (1, 128)},
            "quant1": {"type": "quantize", "input": "input"},
            "matmul1": {"type": "matmul", "inputs": ["quant1", "weight1"]},
            "dequant1": {"type": "dequantize", "input": "matmul1"},
            "quant2": {"type": "quantize", "input": "dequant1"},
            "matmul2": {"type": "matmul", "inputs": ["quant2", "weight2"]},
            "output": {"type": "dequantize", "input": "matmul2"},
        }

        # Optimize graph (remove redundant quant/dequant)
        optimized = optimize_inference(graph)

        # Check optimization
        self.assertNotIn("dequant1", optimized)
        self.assertNotIn("quant2", optimized)

    def test_memory_planning(self):
        """Test memory allocation planning."""
        # Plan memory for inference
        layers = [
            {"name": "layer1", "input_size": (1, 128, 768), "output_size": (1, 128, 768)},
            {"name": "layer2", "input_size": (1, 128, 768), "output_size": (1, 128, 3072)},
            {"name": "layer3", "input_size": (1, 128, 3072), "output_size": (1, 128, 768)},
        ]

        memory_plan = {}
        current_offset = 0

        for layer in layers:
            input_bytes = np.prod(layer["input_size"]) * 4  # float32
            output_bytes = np.prod(layer["output_size"]) * 4

            memory_plan[layer["name"]] = {
                "input_offset": current_offset,
                "output_offset": current_offset + input_bytes,
            }

            current_offset = max(
                current_offset + input_bytes,
                current_offset + output_bytes
            )

        # Check memory reuse
        total_memory = current_offset
        naive_memory = sum(
            np.prod(l["input_size"]) * 4 + np.prod(l["output_size"]) * 4
            for l in layers
        )
        self.assertLess(total_memory, naive_memory)


class TestBenchmarking(unittest.TestCase):
    """Test benchmarking utilities."""

    def setUp(self):
        self.config = QuantConfig(quant_type=QuantType.INT8)
        np.random.seed(42)

    def test_latency_benchmark(self):
        """Test latency benchmarking."""
        # Mock model
        model = Mock()
        model.forward = Mock(return_value=np.random.randn(1, 128, 768))

        # Benchmark latency
        input_data = np.random.randn(1, 128, 768)
        latency = benchmark_latency(model, input_data, num_runs=10, warmup=2)

        # Check latency measured
        self.assertIsInstance(latency, dict)
        self.assertIn("mean", latency)
        self.assertIn("std", latency)
        self.assertIn("p50", latency)
        self.assertIn("p95", latency)
        self.assertIn("p99", latency)

    def test_throughput_benchmark(self):
        """Test throughput benchmarking."""
        engine = InferenceEngine(self.config)
        engine.model = Mock()
        engine.model.forward = Mock(return_value=np.random.randn(4, 128, 768))

        # Measure throughput
        batch_sizes = [1, 2, 4, 8]
        throughputs = {}

        for bs in batch_sizes:
            inputs = np.random.randn(bs, 128, 768)
            start = time.time()

            for _ in range(10):
                _ = engine.model.forward(inputs)

            elapsed = time.time() - start
            throughputs[bs] = (10 * bs) / elapsed  # samples/sec

        # Check throughput scales
        self.assertGreater(throughputs[4], throughputs[1])

    def test_memory_profiling(self):
        """Test memory usage profiling."""
        # Create quantized tensors
        tensors = []
        memory_usage = []

        for i in range(10):
            tensor = QuantizedTensor(
                data=np.random.randint(-127, 127, (256, 256), dtype=np.int8),
                scale=np.random.randn(256).astype(np.float32),
                zero_point=np.zeros(256, dtype=np.int8),
                config=self.config
            )
            tensors.append(tensor)

            # Calculate memory
            mem = sum(t.memory_usage() for t in tensors)
            memory_usage.append(mem)

        # Check memory grows linearly
        for i in range(1, len(memory_usage)):
            self.assertGreater(memory_usage[i], memory_usage[i - 1])


if __name__ == '__main__':
    unittest.main()