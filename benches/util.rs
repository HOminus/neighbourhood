extern crate alloc;

use rand::{
    distributions::{
        uniform::{SampleBorrow, SampleUniform},
        Distribution, Uniform,
    },
    SeedableRng,
};

pub fn random_points<
    S: SampleUniform + Default + Copy,
    const N: usize,
    B: SampleBorrow<S> + Sized,
>(
    num_points: usize,
    min: B,
    max: B,
    seed: u64,
) -> alloc::vec::Vec<[S; N]> {
    let mut points = alloc::vec::Vec::with_capacity(num_points);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    let distr = Uniform::new_inclusive(min, max);
    for _ in 0..num_points {
        let mut p: [S; N] = [S::default(); N];
        for pi in p.iter_mut() {
            *pi = distr.sample(&mut rng);
        }
        points.push(p);
    }
    points
}
