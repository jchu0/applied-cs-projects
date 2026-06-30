"""Test fixtures and helpers for ML compiler tests."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlcompiler.ir.builder import IRBuilder
from mlcompiler.ir.types import TensorType, DataType, Shape
from mlcompiler.ir.module import Module


def create_test_module(name='test_module'):
    """Create a simple test module.

    Args:
        name: Module name

    Returns:
        Module instance with basic operations
    """
    builder = IRBuilder()
    builder.begin_block(name)

    x = builder.create_input(
        TensorType(DataType.FLOAT32, Shape([32, 128])),
        name='x'
    )
    y = builder.create_input(
        TensorType(DataType.FLOAT32, Shape([32, 128])),
        name='y'
    )

    z = builder.add_binary_op('add', x, y)
    result = builder.add_relu(z)
    builder.set_return(result)

    return builder.get_module()


def create_test_graph(num_nodes=10, branching_factor=2):
    """Create a test computation graph.

    Args:
        num_nodes: Number of nodes in the graph
        branching_factor: Average number of outputs per node

    Returns:
        GraphBuilder instance with generated graph
    """
    from mlcompiler.ir.builder import GraphBuilder

    graph = GraphBuilder()
    nodes = []

    # Create nodes
    for i in range(num_nodes):
        node = graph.add_node(f'op_{i}', op_type='test_op')
        nodes.append(node)

    # Create edges with branching
    for i, node in enumerate(nodes[:-1]):
        # Connect to next nodes based on branching factor
        for j in range(min(branching_factor, num_nodes - i - 1)):
            target_idx = min(i + j + 1, num_nodes - 1)
            graph.add_edge(node, nodes[target_idx])

    return graph


def create_cnn_model(num_classes=1000):
    """Create a CNN model for testing.

    Args:
        num_classes: Number of output classes

    Returns:
        Module representing a CNN model
    """
    builder = IRBuilder()
    builder.begin_block('cnn_model')

    # Input image [batch, channels, height, width]
    x = builder.create_input(
        TensorType(DataType.FLOAT32, Shape([32, 3, 224, 224])),
        name='image'
    )

    # Conv Block 1
    conv1_w = builder.create_weight(
        TensorType(DataType.FLOAT32, Shape([64, 3, 7, 7])),
        name='conv1_weight'
    )
    conv1 = builder.add_convolution(x, conv1_w, stride=(2, 2), padding=(3, 3))
    conv1 = builder.add_batch_norm(conv1)
    conv1 = builder.add_relu(conv1)
    conv1 = builder.add_pooling(conv1, pool_type='max', kernel_size=(3, 3), stride=(2, 2))

    # Conv Block 2
    conv2_w = builder.create_weight(
        TensorType(DataType.FLOAT32, Shape([128, 64, 3, 3])),
        name='conv2_weight'
    )
    conv2 = builder.add_convolution(conv1, conv2_w, stride=(2, 2), padding=(1, 1))
    conv2 = builder.add_batch_norm(conv2)
    conv2 = builder.add_relu(conv2)

    # Conv Block 3
    conv3_w = builder.create_weight(
        TensorType(DataType.FLOAT32, Shape([256, 128, 3, 3])),
        name='conv3_weight'
    )
    conv3 = builder.add_convolution(conv2, conv3_w, stride=(2, 2), padding=(1, 1))
    conv3 = builder.add_batch_norm(conv3)
    conv3 = builder.add_relu(conv3)

    # Global Average Pooling
    gap = builder.add_reduce(conv3, reduce_type='mean', axes=[2, 3])

    # Classifier
    fc_w = builder.create_weight(
        TensorType(DataType.FLOAT32, Shape([256, num_classes])),
        name='fc_weight'
    )
    fc_b = builder.create_weight(
        TensorType(DataType.FLOAT32, Shape([num_classes])),
        name='fc_bias'
    )

    logits = builder.add_matmul(gap, fc_w)
    logits = builder.add_bias_add(logits, fc_b)
    output = builder.add_softmax(logits)

    builder.set_return(output)
    return builder.get_module()


def create_transformer_model(seq_len=100, d_model=768, num_heads=12, num_layers=6):
    """Create a transformer model for testing.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers

    Returns:
        Module representing a transformer model
    """
    builder = IRBuilder()
    builder.begin_block('transformer_model')

    # Input [batch, seq_len, d_model]
    x = builder.create_input(
        TensorType(DataType.FLOAT32, Shape([32, seq_len, d_model])),
        name='input'
    )

    head_dim = d_model // num_heads

    for layer_idx in range(num_layers):
        # Multi-head attention
        q_w = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([d_model, d_model])),
            name=f'layer_{layer_idx}_q_weight'
        )
        k_w = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([d_model, d_model])),
            name=f'layer_{layer_idx}_k_weight'
        )
        v_w = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([d_model, d_model])),
            name=f'layer_{layer_idx}_v_weight'
        )
        o_w = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([d_model, d_model])),
            name=f'layer_{layer_idx}_o_weight'
        )

        # Compute Q, K, V
        q = builder.add_matmul(x, q_w)
        k = builder.add_matmul(x, k_w)
        v = builder.add_matmul(x, v_w)

        # Reshape for multi-head
        q = builder.add_reshape(q, Shape([32, seq_len, num_heads, head_dim]))
        k = builder.add_reshape(k, Shape([32, seq_len, num_heads, head_dim]))
        v = builder.add_reshape(v, Shape([32, seq_len, num_heads, head_dim]))

        # Transpose for attention
        q = builder.add_transpose(q, axes=[0, 2, 1, 3])
        k = builder.add_transpose(k, axes=[0, 2, 1, 3])
        v = builder.add_transpose(v, axes=[0, 2, 1, 3])

        # Attention
        scores = builder.add_matmul(q, builder.add_transpose(k, axes=[0, 1, 3, 2]))
        scores = builder.add_scalar_multiply(scores, 1.0 / np.sqrt(head_dim))
        scores = builder.add_softmax(scores, axis=-1)

        attn = builder.add_matmul(scores, v)
        attn = builder.add_transpose(attn, axes=[0, 2, 1, 3])
        attn = builder.add_reshape(attn, Shape([32, seq_len, d_model]))

        # Output projection
        attn = builder.add_matmul(attn, o_w)

        # Residual and layer norm
        x = builder.add_binary_op('add', x, attn)
        x = builder.add_layer_norm(x)

        # Feed-forward network
        ff_w1 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([d_model, d_model * 4])),
            name=f'layer_{layer_idx}_ff_weight1'
        )
        ff_w2 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([d_model * 4, d_model])),
            name=f'layer_{layer_idx}_ff_weight2'
        )

        ff = builder.add_matmul(x, ff_w1)
        ff = builder.add_gelu(ff)
        ff = builder.add_matmul(ff, ff_w2)

        # Residual and layer norm
        x = builder.add_binary_op('add', x, ff)
        x = builder.add_layer_norm(x)

    builder.set_return(x)
    return builder.get_module()


def create_test_weights(shape, seed=42):
    """Create test weight tensors.

    Args:
        shape: Weight shape
        seed: Random seed for reproducibility

    Returns:
        Numpy array with initialized weights
    """
    np.random.seed(seed)

    # Xavier/Glorot initialization
    fan_in = np.prod(shape[:-1]) if len(shape) > 1 else shape[0]
    fan_out = shape[-1]

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    weights = np.random.uniform(-limit, limit, shape).astype(np.float32)

    return weights


def create_test_data(shape, distribution='normal', seed=42):
    """Create test input data.

    Args:
        shape: Data shape
        distribution: 'normal', 'uniform', or 'zeros'
        seed: Random seed

    Returns:
        Numpy array with test data
    """
    np.random.seed(seed)

    if distribution == 'normal':
        data = np.random.randn(*shape).astype(np.float32)
    elif distribution == 'uniform':
        data = np.random.rand(*shape).astype(np.float32)
    elif distribution == 'zeros':
        data = np.zeros(shape, dtype=np.float32)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return data


def compare_outputs(output1, output2, tolerance=1e-5):
    """Compare two outputs for numerical equivalence.

    Args:
        output1: First output tensor
        output2: Second output tensor
        tolerance: Numerical tolerance for comparison

    Returns:
        bool: True if outputs are equivalent within tolerance
    """
    if output1.shape != output2.shape:
        return False

    diff = np.abs(output1 - output2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    return max_diff < tolerance and mean_diff < tolerance / 10


def benchmark_module(module, num_runs=100, warmup_runs=10):
    """Benchmark a compiled module.

    Args:
        module: Compiled module to benchmark
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs

    Returns:
        dict: Benchmark results including mean, std, min, max times
    """
    import time

    times = []

    # Warmup
    for _ in range(warmup_runs):
        module.run()

    # Benchmark
    for _ in range(num_runs):
        start = time.perf_counter()
        module.run()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }


class ModelFactory:
    """Factory for creating test models."""

    @staticmethod
    def create_mlp(input_dim=784, hidden_dims=[256, 128], output_dim=10):
        """Create a multi-layer perceptron model."""
        builder = IRBuilder()
        builder.begin_block('mlp')

        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, input_dim])),
            name='input'
        )

        current = x
        for i, hidden_dim in enumerate(hidden_dims):
            prev_dim = input_dim if i == 0 else hidden_dims[i-1]

            w = builder.create_weight(
                TensorType(DataType.FLOAT32, Shape([prev_dim, hidden_dim])),
                name=f'weight_{i}'
            )
            b = builder.create_weight(
                TensorType(DataType.FLOAT32, Shape([hidden_dim])),
                name=f'bias_{i}'
            )

            current = builder.add_matmul(current, w)
            current = builder.add_bias_add(current, b)
            current = builder.add_relu(current)

        # Output layer
        w_out = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([hidden_dims[-1], output_dim])),
            name='weight_out'
        )
        b_out = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([output_dim])),
            name='bias_out'
        )

        output = builder.add_matmul(current, w_out)
        output = builder.add_bias_add(output, b_out)
        output = builder.add_softmax(output)

        builder.set_return(output)
        return builder.get_module()

    @staticmethod
    def create_rnn(input_dim=128, hidden_dim=256, seq_len=100):
        """Create an RNN model."""
        builder = IRBuilder()
        builder.begin_block('rnn')

        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, seq_len, input_dim])),
            name='input'
        )

        w_ih = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([input_dim, hidden_dim])),
            name='input_hidden_weight'
        )
        w_hh = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([hidden_dim, hidden_dim])),
            name='hidden_hidden_weight'
        )
        b = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([hidden_dim])),
            name='bias'
        )

        h = builder.create_zeros(Shape([32, hidden_dim]))

        outputs = []
        for t in range(seq_len):
            x_t = builder.get_slice(x, axis=1, index=t)

            i_h = builder.add_matmul(x_t, w_ih)
            h_h = builder.add_matmul(h, w_hh)
            h = builder.add_binary_op('add', i_h, h_h)
            h = builder.add_bias_add(h, b)
            h = builder.add_tanh(h)

            outputs.append(h)

        output = builder.stack(outputs, axis=1)
        builder.set_return(output)
        return builder.get_module()

    @staticmethod
    def create_autoencoder(input_dim=784, latent_dim=32):
        """Create an autoencoder model."""
        builder = IRBuilder()
        builder.begin_block('autoencoder')

        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, input_dim])),
            name='input'
        )

        # Encoder
        enc_w1 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([input_dim, 256])),
            name='encoder_weight1'
        )
        enc_w2 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([256, 128])),
            name='encoder_weight2'
        )
        enc_w3 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([128, latent_dim])),
            name='encoder_weight3'
        )

        h = builder.add_matmul(x, enc_w1)
        h = builder.add_relu(h)
        h = builder.add_matmul(h, enc_w2)
        h = builder.add_relu(h)
        latent = builder.add_matmul(h, enc_w3)

        # Decoder
        dec_w1 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([latent_dim, 128])),
            name='decoder_weight1'
        )
        dec_w2 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([128, 256])),
            name='decoder_weight2'
        )
        dec_w3 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([256, input_dim])),
            name='decoder_weight3'
        )

        h = builder.add_matmul(latent, dec_w1)
        h = builder.add_relu(h)
        h = builder.add_matmul(h, dec_w2)
        h = builder.add_relu(h)
        output = builder.add_matmul(h, dec_w3)
        output = builder.add_sigmoid(output)

        builder.set_return(output)
        return builder.get_module()


# Export commonly used fixtures
__all__ = [
    'create_test_module',
    'create_test_graph',
    'create_cnn_model',
    'create_transformer_model',
    'create_test_weights',
    'create_test_data',
    'compare_outputs',
    'benchmark_module',
    'ModelFactory'
]