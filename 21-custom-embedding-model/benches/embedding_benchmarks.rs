use criterion::{criterion_group, criterion_main, Criterion};

fn embedding_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, embedding_benchmark);
criterion_main!(benches);
