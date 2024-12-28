// Some benchmark functions are commented out.
#![allow(dead_code)]

use kiddo::immutable::float::kdtree::ImmutableKdTree;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

#[derive(Default)]
struct Timings {
    nh_kdtree: Vec<Duration>,
    nh_kdindextree: Vec<Duration>,
    kiddo_kdtree: Vec<Duration>,
    kdtree_kdtree: Vec<Duration>,
    kd_tree_kdtree: Vec<Duration>,
}

#[derive(Default)]
struct BenchmarkResult {
    construction: HashMap<usize, Timings>,
    neighbourhood_query: HashMap<(usize, ordered_float::NotNan<f64>), Timings>,
    knn_query: HashMap<(usize, usize), Timings>,
}

impl BenchmarkResult {
    fn add_nh_kdtree_cnstr_time(&mut self, size: usize, timing: Duration) {
        self.construction
            .entry(size)
            .or_default()
            .nh_kdtree
            .push(timing);
    }

    fn add_nh_kdindextree_cnstr_time(&mut self, size: usize, timing: Duration) {
        self.construction
            .entry(size)
            .or_default()
            .nh_kdindextree
            .push(timing);
    }

    fn add_kiddo_cnstr_time(&mut self, size: usize, timing: Duration) {
        self.construction
            .entry(size)
            .or_default()
            .kiddo_kdtree
            .push(timing);
    }

    fn add_kdtree_cnstr_time(&mut self, size: usize, timing: Duration) {
        self.construction
            .entry(size)
            .or_default()
            .kdtree_kdtree
            .push(timing);
    }

    fn add_kd_tree_cnstr_time(&mut self, size: usize, timing: Duration) {
        self.construction
            .entry(size)
            .or_default()
            .kd_tree_kdtree
            .push(timing);
    }

    fn add_nh_kdtree_nh_query_time(&mut self, size: usize, eps: f64, timing: Duration) {
        let eps = ordered_float::NotNan::new(eps).unwrap();
        self.neighbourhood_query
            .entry((size, eps))
            .or_default()
            .nh_kdtree
            .push(timing);
    }

    fn add_nh_kdindextree_nh_query_time(&mut self, size: usize, eps: f64, timing: Duration) {
        let eps = ordered_float::NotNan::new(eps).unwrap();
        self.neighbourhood_query
            .entry((size, eps))
            .or_default()
            .nh_kdindextree
            .push(timing);
    }

    fn add_kiddo_nh_query_time(&mut self, size: usize, eps: f64, timing: Duration) {
        let eps = ordered_float::NotNan::new(eps).unwrap();
        self.neighbourhood_query
            .entry((size, eps))
            .or_default()
            .kiddo_kdtree
            .push(timing);
    }

    fn add_kdtree_nh_query_time(&mut self, size: usize, eps: f64, timing: Duration) {
        let eps = ordered_float::NotNan::new(eps).unwrap();
        self.neighbourhood_query
            .entry((size, eps))
            .or_default()
            .kdtree_kdtree
            .push(timing);
    }

    fn add_kd_tree_nh_query_time(&mut self, size: usize, eps: f64, timing: Duration) {
        let eps = ordered_float::NotNan::new(eps).unwrap();
        self.neighbourhood_query
            .entry((size, eps))
            .or_default()
            .kd_tree_kdtree
            .push(timing);
    }

    fn add_nh_kdtree_knn_query_time(&mut self, size: usize, k: usize, timing: Duration) {
        self.knn_query
            .entry((size, k))
            .or_default()
            .nh_kdtree
            .push(timing);
    }

    fn add_nh_kdindextree_knn_query_time(&mut self, size: usize, k: usize, timing: Duration) {
        self.knn_query
            .entry((size, k))
            .or_default()
            .nh_kdindextree
            .push(timing);
    }

    fn add_kiddo_knn_query_time(&mut self, size: usize, k: usize, timing: Duration) {
        self.knn_query
            .entry((size, k))
            .or_default()
            .kiddo_kdtree
            .push(timing);
    }

    fn add_kdtree_knn_query_time(&mut self, size: usize, k: usize, timing: Duration) {
        self.knn_query
            .entry((size, k))
            .or_default()
            .kdtree_kdtree
            .push(timing);
    }

    fn add_kd_tree_knn_query_time(&mut self, size: usize, k: usize, timing: Duration) {
        self.knn_query
            .entry((size, k))
            .or_default()
            .kd_tree_kdtree
            .push(timing);
    }


    fn print_cnstr_timings(&self) {
        let mut keys: Vec<_> = self.construction.keys().collect();
        keys.sort_by_key(|k| *k);

        let default_timing = Duration::default();
        for k in keys {
            let timings = self.construction.get(k).unwrap();
            let nh_kdtree_min = timings
                .nh_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let nh_kdindextree_min = timings
                .nh_kdindextree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let kiddo_kdtree_min = timings
                .kiddo_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let kdtree_kdtree_min = timings
                .kdtree_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let kd_tree_kdtree = timings
                .kd_tree_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);

            println!(
                "{:<10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|",
                k,
                format!("{:.5}", nh_kdtree_min.as_secs_f32()),
                format!("{:.5}", nh_kdindextree_min.as_secs_f32()),
                format!("{:.5}", kiddo_kdtree_min.as_secs_f32()),
                format!("{:.5}", kdtree_kdtree_min.as_secs_f32()),
                format!("{:.5}", kd_tree_kdtree.as_secs_f32())
            );
        }
    }

    fn print_neighbourhood_query_timings(&self, query_size: usize) {
        let mut keys: Vec<_> = self.neighbourhood_query.keys().collect();
        keys.sort_by_key(|k| k.0);

        let default_timing = Duration::default();
        for k in keys {
            let timings = self.neighbourhood_query.get(k).unwrap();

            let nh_kdtree_min = timings
                .nh_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let nh_kdindextree_min = timings
                .nh_kdindextree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let kiddo_kdtree_min = timings
                .kiddo_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let kdtree_kdtree_min = timings
                .kdtree_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let kd_tree_kdtree = timings
                .kd_tree_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);

            println!(
                "{:<10} - {:<10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|",
                k.0,
                k.1,
                format!(
                    "{:.5}",
                    nh_kdtree_min.as_nanos().div_ceil(query_size as u128)
                ),
                format!(
                    "{:.5}",
                    nh_kdindextree_min.as_nanos().div_ceil(query_size as u128)
                ),
                format!(
                    "{:.5}",
                    kiddo_kdtree_min.as_nanos().div_ceil(query_size as u128)
                ),
                format!(
                    "{:.5}",
                    kdtree_kdtree_min.as_nanos().div_ceil(query_size as u128)
                ),
                format!(
                    "{:.5}",
                    kd_tree_kdtree.as_nanos().div_ceil(query_size as u128)
                )
            );
        }
    }

    fn print_knn_query_timings(&self, query_size: usize) {
        let mut keys: Vec<_> = self.knn_query.keys().collect();
        keys.sort_by_key(|k| k.0);

        let default_timing = Duration::default();
        for k in keys {
            let timings = self.knn_query.get(k).unwrap();

            let nh_kdtree_min = timings
                .nh_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let nh_kdindextree_min = timings
                .nh_kdindextree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let kiddo_kdtree_min = timings
                .kiddo_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let kdtree_kdtree_min = timings
                .kdtree_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);
            let kd_tree_kdtree = timings
                .kd_tree_kdtree
                .iter()
                .min_by_key(|v| v.as_nanos())
                .unwrap_or(&default_timing);

            println!(
                "{:<10} - {:<10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|",
                k.0,
                k.1,
                format!(
                    "{:.5}",
                    nh_kdtree_min.as_nanos().div_ceil(query_size as u128)
                ),
                format!(
                    "{:.5}",
                    nh_kdindextree_min.as_nanos().div_ceil(query_size as u128)
                ),
                format!(
                    "{:.5}",
                    kiddo_kdtree_min.as_nanos().div_ceil(query_size as u128)
                ),
                format!(
                    "{:.5}",
                    kdtree_kdtree_min.as_nanos().div_ceil(query_size as u128)
                ),
                format!(
                    "{:.5}",
                    kd_tree_kdtree.as_nanos().div_ceil(query_size as u128)
                )
            );
        }
    }
}

fn main() {
    let mut timings = BenchmarkResult::default();
    let query_size = 20_000;
    let eps = &[0.02, 0.05, 0.1, 0.2];
    let knns = &[5, 10, 20, 100];
    for size in [100_000, 1_000_000, 10_000_000, 100_000_000] {
        for seed in 0..1 {
            let points = compare::make_points(size, -10., 10., seed);
            let query_points = compare::make_points(query_size, -10., 10., u64::MAX - seed);

            benchmark_nh_kdtree(&mut timings, points.clone(), &query_points, eps, knns);
            benchmark_nh_kdindextree(&mut timings, points.clone(), &query_points, eps, knns);
            benchmark_kiddo_kdtree(&mut timings, points.clone(), &query_points, eps, knns);
            //benchmark_kdtree_kdtree(&mut timings, points.clone(), &query_points, eps, knns);
            benchmark_kd_tree_kdtree(&mut timings, points.clone(), &query_points, eps, knns);
        }
    }
    timings.print_cnstr_timings();
    println!();
    timings.print_neighbourhood_query_timings(query_size);
    println!();
    timings.print_knn_query_timings(query_size);
}

fn benchmark_nh_kdtree(
    timings: &mut BenchmarkResult,
    points: Vec<[f64; 3]>,
    query_points: &[[f64; 3]],
    epsilons: &[f64],
    knns: &[usize],
) {
    // Construction
    let now = Instant::now();
    let nh_kd_tree = neighbourhood::KdTree::new(points);
    let timing = now.elapsed();
    timings.add_nh_kdtree_cnstr_time(nh_kd_tree.len(), timing);
    let nh_kd_tree = std::hint::black_box(nh_kd_tree);

    // Neighbourhood query
    for eps in epsilons {
        let now = Instant::now();
        for p in query_points {
            let r = nh_kd_tree.neighbourhood(p, *eps);
            std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_nh_kdtree_nh_query_time(nh_kd_tree.len(), *eps, timing);
    }

    // Knn query
    for knn in knns {
        let now = Instant::now();
        for p in query_points {
            let r = nh_kd_tree.knn(p, *knn);
            std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_nh_kdtree_knn_query_time(nh_kd_tree.len(), *knn, timing);
    }
}

fn benchmark_nh_kdindextree(
    timings: &mut BenchmarkResult,
    points: Vec<[f64; 3]>,
    query_points: &[[f64; 3]],
    epsilons: &[f64],
    knns: &[usize],
) {
    // Construction
    let now = Instant::now();
    let nh_kd_index_tree = neighbourhood::KdIndexTree::new(&points);
    let timing = now.elapsed();
    timings.add_nh_kdindextree_cnstr_time(nh_kd_index_tree.len(), timing);
    let nh_kd_index_tree = std::hint::black_box(nh_kd_index_tree);

    // Neighbourhood query
    for eps in epsilons {
        let now = Instant::now();
        for p in query_points {
            let r = nh_kd_index_tree.neighbourhood_by_index(p, *eps);
            std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_nh_kdindextree_nh_query_time(nh_kd_index_tree.len(), *eps, timing);
    }

    // Knn query
    for knn in knns {
        let now = Instant::now();
        for p in query_points {
            let r = nh_kd_index_tree.knn_by_index(p, *knn);
            std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_nh_kdindextree_knn_query_time(nh_kd_index_tree.len(), *knn, timing);
    }
}

fn benchmark_kiddo_kdtree(
    timings: &mut BenchmarkResult,
    points: Vec<[f64; 3]>,
    query_points: &[[f64; 3]],
    epsilons: &[f64],
    knns: &[usize],
) {
    // Construction
    let now = Instant::now();
    let kiddo_kdtree: ImmutableKdTree<f64, u64, 3, 32> =
        kiddo::ImmutableKdTree::new_from_slice(&points);
    let timing = now.elapsed();
    timings.add_kiddo_cnstr_time(points.len(), timing);
    let kiddo_kdtree = std::hint::black_box(kiddo_kdtree);

    //Neighbourhood query
    for eps in epsilons {
        let now = Instant::now();
        for p in query_points {
            let r = kiddo_kdtree.within_unsorted::<kiddo::SquaredEuclidean>(p, *eps * *eps);
            std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_kiddo_nh_query_time(points.len(), *eps, timing);
    }

    // Knn query
    for knn in knns {
        let now = Instant::now();
        for p in query_points {
            let r = kiddo_kdtree.nearest_n::<kiddo::SquaredEuclidean>(p, core::num::NonZero::new(*knn).unwrap());
            std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_kiddo_knn_query_time(points.len(), *knn, timing);
    }
}

fn benchmark_kdtree_kdtree(
    timings: &mut BenchmarkResult,
    points: Vec<[f64; 3]>,
    query_points: &[[f64; 3]],
    epsilons: &[f64],
    knns: &[usize],
) {
    // Construction benchmarks
    let now = Instant::now();
    let mut kdtree_kdtree = kdtree::KdTree::with_capacity(3, points.len());
    for (index, p) in points.iter().enumerate() {
        kdtree_kdtree.add(*p, index).unwrap();
    }
    let timing = now.elapsed();
    timings.add_kdtree_cnstr_time(points.len(), timing);
    let kdtree_kdtree = std::hint::black_box(kdtree_kdtree);

    //Neighbourhood query
    for eps in epsilons {
        let now = Instant::now();
        for p in query_points {
            let r = kdtree_kdtree.within(p, *eps * *eps, &kdtree::distance::squared_euclidean);
            let _ = std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_kdtree_nh_query_time(points.len(), *eps, timing);
    }

    // Knn query
    for knn in knns {
        let now = Instant::now();
        for p in query_points {
            let r = kdtree_kdtree.nearest(p, *knn, &kdtree::distance::squared_euclidean);
            let _ = std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_kdtree_knn_query_time(points.len(), *knn, timing);
    }
}

fn benchmark_kd_tree_kdtree(
    timings: &mut BenchmarkResult,
    points: Vec<[f64; 3]>,
    query_points: &[[f64; 3]],
    epsilons: &[f64],
    knns: &[usize],
) {
    let size = points.len();
    //Construction benchmarks
    let now = Instant::now();
    let kdt_kd_tree = kd_tree::KdTree::build_by_ordered_float(points);
    let timing = now.elapsed();
    timings.add_kd_tree_cnstr_time(size, timing);
    let kdt_kd_tree = std::hint::black_box(&kdt_kd_tree);

    //Neighbourhood query
    for eps in epsilons {
        let now = Instant::now();
        for p in query_points {
            let r = kdt_kd_tree.within_radius(p, *eps);
            std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_kd_tree_nh_query_time(size, *eps, timing);
    }

    // Knn query
    for knn in knns {
        let now = Instant::now();
        for p in query_points {
            let r = kdt_kd_tree.nearests(p, *knn);
            std::hint::black_box(r);
        }
        let timing = now.elapsed();
        timings.add_kd_tree_knn_query_time(size, *knn, timing);
    }
}
