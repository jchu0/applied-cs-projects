use criterion::{criterion_group, criterion_main, Criterion};

fn simd_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, simd_benchmark);
criterion_main!(benches);
