use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use neighbourhood::KdTree;

pub mod util;
use util::random_points;

pub fn buildup(c: &mut Criterion) {
    const NUM_POINTS: usize = 200_000;
    const SEED: u64 = 0;

    c.bench_function("BuildUp", |b| {
        let points: Vec<[f64; 3]> = random_points(NUM_POINTS, -10., 10., SEED);
        b.iter(|| {
            let kd_tree = KdTree::new(points.clone());
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
        let kd_tree = KdTree::new(points);
        b.iter(|| {
            let neighbours = kd_tree.neighbourhood(&[0., 0., 0.], EPSILON);
            std::hint::black_box(neighbours);
        });
    });
}

fn count_neighbourhood_query(c: &mut Criterion) {
    const NUM_POINTS: usize = 200_000;
    const SEED: u64 = 0;
    const EPSILON: f64 = 0.5;
    c.bench_function("CountNeighbourhoodQuery", |b| {
        let points: Vec<[f64; 3]> = random_points(NUM_POINTS, -10., 10., SEED);
        let kd_tree = KdTree::new(points);
        b.iter(|| {
            let neighbours = kd_tree.count_neighbourhood(&[0., 0., 0.], EPSILON);
            std::hint::black_box(neighbours);
        });
    });
}

fn optimal_brute_force_size(c: &mut Criterion) {
    const NUM_POINTS: usize = 200_000;
    const EPSILON: f64 = 0.5;

    let mut group = c.benchmark_group("BruteForceSize");
    for brute_force_size in 1..=50 {
        group.bench_with_input(
            BenchmarkId::from_parameter(brute_force_size),
            &brute_force_size,
            |b, brute_force_size| {
                let points: Vec<[f64; 3]> = random_points(NUM_POINTS, -10., 10., 0);
                let mut kd_tree = KdTree::new(points.clone());
                kd_tree.brute_force_size = *brute_force_size;
                b.iter(|| {
                    for p in &points[99_000..101_000] {
                        let neighbours = kd_tree.neighbourhood(p, EPSILON);
                        std::hint::black_box(neighbours);
                    }
                });
            },
        );
    }
}

fn knn_query(c: &mut Criterion) {
    const NUM_POINTS: usize = 200_000;
    const SEED: u64 = 0;
    const K: usize = 10;
    c.bench_function("KnnQuery", |b| {
        let points: Vec<[f64; 3]> = random_points(NUM_POINTS, -10., 10., SEED);
        let kd_tree = KdTree::new(points);
        b.iter(|| {
            let neighbours = kd_tree.knn(&[0., 0., 0.], K);
            std::hint::black_box(neighbours);
        });
    });
}

criterion_group!(
    benches,
    buildup,
    neighbourhood_query,
    count_neighbourhood_query,
    optimal_brute_force_size,
    knn_query
);
criterion_main!(benches);
