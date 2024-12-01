use criterion::{criterion_group, criterion_main, Criterion};
use neighbourhood::KdIndexTree;

pub mod util;
use util::random_points;

pub fn buildup(c: &mut Criterion) {
    const NUM_POINTS: usize = 200_000;
    const SEED: u64 = 0;

    c.bench_function("BuildUp", |b| {
        let points: Vec<[f64; 3]> = random_points(NUM_POINTS, -10., 10., SEED);
        b.iter(|| {
            let kd_tree = KdIndexTree::new(&points);
            std::hint::black_box(kd_tree);
        });
    });
}

fn neighbourhood_query(c: &mut Criterion) {
    const NUM_POINTS: usize = 200_000;
    const SEED: u64 = 0;
    const EPSILON: f64 = 0.5;
    c.bench_function("NeighbourhoodQuery", |b| {
        let points: Vec<[f64; 3]> = random_points(NUM_POINTS, -10., 10., SEED);
        let kd_tree = KdIndexTree::new(&points);
        b.iter(|| {
            let neighbours = kd_tree.neighbourhood_by_index(&[0., 0., 0.], EPSILON);
            std::hint::black_box(neighbours);
        });
    });
}

criterion_group!(benches, buildup, neighbourhood_query);
criterion_main!(benches);
