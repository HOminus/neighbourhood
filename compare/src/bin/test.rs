use clap::Parser;
use std::fmt::Debug;

use compare::{make_points, print_mismatch_founds};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 1000000)]
    num_points: usize,

    #[arg(long, default_value_t = 3)]
    dim: usize,

    #[arg(long, default_value_t = 5)]
    iterations: u64,

    #[arg(long)]
    kdtree: Vec<String>,
}

fn run_test<const N: usize>(num_points: usize, iterations: u64, kdtrees: &[String]) {
    for seed in 0..iterations {
        println!("Seed {seed}");
        let points: Vec<[f64; N]> = make_points(num_points, -10., 10., seed);

        let mut kdtree_list = vec![
            Box::new(compare::nh::KdTree::new(&points))
                as Box<dyn compare::UnifiedKdTreeTestApi<f64, N>>,
            Box::new(compare::nh::KdIndexTree::new(&points))
                as Box<dyn compare::UnifiedKdTreeTestApi<f64, N>>,
        ];

        for name in kdtrees {
            match name.as_str() {
                "kiddo" => {
                    let kdtree = Box::new(compare::kiddo::KdTree::new(&points));
                    kdtree_list.push(kdtree as Box<dyn compare::UnifiedKdTreeTestApi<f64, N>>);
                }
                "kdtree" => {
                    let kdtree = Box::new(compare::kdtree::KdTree::new(&points));
                    kdtree_list.push(kdtree as Box<dyn compare::UnifiedKdTreeTestApi<f64, N>>);
                }
                "kd_tree" => {
                    let kdtree = Box::new(compare::kd_tree::KdTree::new(&points));
                    kdtree_list.push(kdtree as Box<dyn compare::UnifiedKdTreeTestApi<f64, N>>);
                }
                _ => panic!("No kdtree implementation with name {name}"),
            }
        }

        let eps = 0.1;
        for p in points.iter() {
            let result: Vec<_> = kdtree_list
                .iter()
                .map(|t| t.query_within(p, eps, &points))
                .collect();

            for i in 1..result.len() {
                if result[0] != result[i] {
                    println!("Point and Eps: {p:?} {eps:?}");
                    print_mismatch_founds(p, &result[0], &result[i]);
                    panic!()
                }
            }
        }
    }
}

fn main() {
    let args = Args::parse();

    match args.dim {
        1 => run_test::<1>(args.num_points, args.iterations, &args.kdtree),
        2 => run_test::<2>(args.num_points, args.iterations, &args.kdtree),
        3 => run_test::<3>(args.num_points, args.iterations, &args.kdtree),
        4 => run_test::<4>(args.num_points, args.iterations, &args.kdtree),
        5 => run_test::<5>(args.num_points, args.iterations, &args.kdtree),
        6 => run_test::<6>(args.num_points, args.iterations, &args.kdtree),
        7 => run_test::<7>(args.num_points, args.iterations, &args.kdtree),
        8 => run_test::<8>(args.num_points, args.iterations, &args.kdtree),
        9 => run_test::<9>(args.num_points, args.iterations, &args.kdtree),
        10 => run_test::<10>(args.num_points, args.iterations, &args.kdtree),
        11 => run_test::<11>(args.num_points, args.iterations, &args.kdtree),
        12 => run_test::<12>(args.num_points, args.iterations, &args.kdtree),
        13 => run_test::<13>(args.num_points, args.iterations, &args.kdtree),
        14 => run_test::<14>(args.num_points, args.iterations, &args.kdtree),
        15 => run_test::<15>(args.num_points, args.iterations, &args.kdtree),
        16 => run_test::<16>(args.num_points, args.iterations, &args.kdtree),
        _ => panic!(),
    }
}
