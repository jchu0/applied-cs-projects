# AI Benchmark Suite

Comprehensive benchmarking framework for AI/ML workloads with automated reporting.

## Features

- **8 Workload Types**: LLM inference, training, embedding, etc.
- **Multi-Framework**: PyTorch, TensorFlow, ONNX support
- **Automated Reports**: HTML and Markdown output
- **GPU Profiling**: Memory, utilization, power metrics
- **Reproducible**: Deterministic seeding and configurations

## Installation

```bash
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Run all benchmarks
aibench run --all

# Run specific benchmark
aibench run --benchmark llm-inference

# Generate report
aibench report --format html --output results.html
```

### Python API

```python
from aibench import BenchmarkSuite, LLMInferenceBenchmark

# Create suite
suite = BenchmarkSuite()

# Add benchmarks
suite.add(LLMInferenceBenchmark(
    model="gpt2",
    batch_sizes=[1, 4, 16],
    sequence_lengths=[128, 512]
))

# Run and get results
results = suite.run()

# Generate report
suite.generate_report("results.html", format="html")
```

## Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `llm-inference` | LLM token generation throughput |
| `llm-training` | LLM fine-tuning performance |
| `embedding` | Embedding model throughput |
| `vision` | Image classification/detection |
| `audio` | Speech recognition |
| `matmul` | Matrix multiplication |
| `memory` | GPU memory bandwidth |
| `communication` | Multi-GPU communication |

## Custom Benchmarks

```python
from aibench import Benchmark

class MyBenchmark(Benchmark):
    name = "my-benchmark"

    def setup(self):
        self.model = load_model()

    def run(self, config):
        start = time.time()
        self.model(config.input)
        return {"latency_ms": (time.time() - start) * 1000}

suite.add(MyBenchmark())
```

## Report Output

```
┌─────────────────┬──────────┬────────────┬─────────┐
│ Benchmark       │ Latency  │ Throughput │ Memory  │
├─────────────────┼──────────┼────────────┼─────────┤
│ llm-inference   │ 45.2ms   │ 22.1 tok/s │ 4.2 GB  │
│ embedding       │ 2.3ms    │ 435 vec/s  │ 1.1 GB  │
│ matmul (4096)   │ 0.8ms    │ 171 TFLOPS │ 0.5 GB  │
└─────────────────┴──────────┴────────────┴─────────┘
```

## Testing

```bash
pytest tests/ -v  # 171 tests
```
