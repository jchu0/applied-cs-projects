//! GEMM benchmarks for performance measurement.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gpu_gemm_optimization::{Matrix, naive_gemm, tiled_gemm};

fn benchmark_naive_gemm(c: &mut Criterion) {
    let size = 128;
    let a = Matrix::random(size, size);
    let b = Matrix::random(size, size);
    let mut result = Matrix::zeros(size, size);

    c.bench_function("naive_gemm_128x128", |bencher| {
        bencher.iter(|| {
            naive_gemm(black_box(&a), black_box(&b), black_box(&mut result)).unwrap();
        });
    });
}

fn benchmark_tiled_gemm(c: &mut Criterion) {
    let size = 128;
    let a = Matrix::random(size, size);
    let b = Matrix::random(size, size);
    let mut result = Matrix::zeros(size, size);

    c.bench_function("tiled_gemm_128x128", |bencher| {
        bencher.iter(|| {
            tiled_gemm(black_box(&a), black_box(&b), black_box(&mut result), 16).unwrap();
        });
    });
}

criterion_group!(benches, benchmark_naive_gemm, benchmark_tiled_gemm);
criterion_main!(benches);
