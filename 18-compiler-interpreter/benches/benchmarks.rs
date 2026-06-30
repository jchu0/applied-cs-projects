//! Benchmarks for the Python compiler/interpreter.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use py_compiler::run;

fn benchmark_fibonacci(c: &mut Criterion) {
    let code = r#"
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
result = fib(20)
"#;

    c.bench_function("fibonacci_20", |b| {
        b.iter(|| run(black_box(code)))
    });
}

fn benchmark_loop(c: &mut Criterion) {
    let code = r#"
total = 0
for i in range(1000):
    total = total + i
"#;

    c.bench_function("loop_1000", |b| {
        b.iter(|| run(black_box(code)))
    });
}

fn benchmark_class_creation(c: &mut Criterion) {
    let code = r#"
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

points = []
for i in range(100):
    points.append(Point(i, i * 2))
"#;

    c.bench_function("class_creation_100", |b| {
        b.iter(|| run(black_box(code)))
    });
}

criterion_group!(benches, benchmark_fibonacci, benchmark_loop, benchmark_class_creation);
criterion_main!(benches);
