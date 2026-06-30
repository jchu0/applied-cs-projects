use criterion::{criterion_group, criterion_main, Criterion};

fn query_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            // Placeholder benchmark
            let _x = 1 + 1;
        })
    });
}

criterion_group!(benches, query_benchmark);
criterion_main!(benches);
