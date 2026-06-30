//! Attention benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use long_context_attention::{
    AttentionConfig, TensorShape, StandardAttention, FlashAttention,
    StreamingInference, StreamingConfig, AttentionStrategy,
};

fn bench_standard_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_attention");

    for seq_len in [64, 128, 256, 512].iter() {
        let config = AttentionConfig::new(8, 64).with_causal(true);
        let attention = StandardAttention::new(config);

        let batch = 1;
        let num_heads = 8;
        let head_dim = 64;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let key = query.clone();
        let value = query.clone();
        let shape = TensorShape::new(batch, *seq_len, num_heads, head_dim);

        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    attention.forward(
                        black_box(&query),
                        black_box(&key),
                        black_box(&value),
                        black_box(shape),
                        black_box(shape),
                        None,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention");

    for seq_len in [64, 128, 256, 512, 1024].iter() {
        let config = AttentionConfig::new(8, 64).with_causal(true);
        let attention = FlashAttention::new(config);

        let batch = 1;
        let num_heads = 8;
        let head_dim = 64;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let key = query.clone();
        let value = query.clone();
        let shape = TensorShape::new(batch, *seq_len, num_heads, head_dim);

        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    attention.forward(
                        black_box(&query),
                        black_box(&key),
                        black_box(&value),
                        black_box(shape),
                        black_box(shape),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_streaming_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_prefill");

    for seq_len in [64, 128, 256].iter() {
        let config = StreamingConfig::new(1, 1, 1024, 8, 64);

        let batch = 1;
        let num_heads = 8;
        let head_dim = 64;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let key = query.clone();
        let value = query.clone();

        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, &seq_len| {
                b.iter(|| {
                    let mut inference = StreamingInference::new(config.clone());
                    inference.prefill(
                        black_box(&query),
                        black_box(&key),
                        black_box(&value),
                        black_box(seq_len),
                        0,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_streaming_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_decode");

    let config = StreamingConfig::new(1, 1, 1024, 8, 64);

    let batch = 1;
    let num_heads = 8;
    let head_dim = 64;

    // Prefill setup
    let prefill_len = 128;
    let prefill_size = batch * prefill_len * num_heads * head_dim;
    let prefill_q: Vec<f32> = (0..prefill_size).map(|i| (i as f32) * 0.001).collect();

    // Decode token
    let decode_size = batch * 1 * num_heads * head_dim;
    let decode_q: Vec<f32> = (0..decode_size).map(|i| (i as f32) * 0.001).collect();

    group.bench_function("single_token_decode", |b| {
        b.iter_batched(
            || {
                let mut inference = StreamingInference::new(config.clone());
                inference.prefill(&prefill_q, &prefill_q, &prefill_q, prefill_len, 0).unwrap();
                inference
            },
            |mut inference| {
                inference.decode_step(
                    black_box(&decode_q),
                    black_box(&decode_q),
                    black_box(&decode_q),
                    0,
                )
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_standard_attention,
    bench_flash_attention,
    bench_streaming_prefill,
    bench_streaming_decode,
);
criterion_main!(benches);
