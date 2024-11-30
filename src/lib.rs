#![no_std]

extern crate alloc;

pub mod kd_index_tree;
pub mod kd_tree;

pub use kd_index_tree::KdIndexTree;
pub use kd_tree::KdTree;

use num_traits::Float;

#[derive(Debug, Copy, Clone)]
struct NeighbourhoodParams<'a, T, const N: usize> {
    point: &'a [T; N],
    epsilon: T,
    row: usize,
}

impl<T, const N: usize> NeighbourhoodParams<'_, T, N> {
    fn next_row(&mut self) {
        self.row = (self.row + 1) % N;
    }
}

fn distance<T: Float, const N: usize>(v1: &[T; N], v2: &[T; N]) -> T {
    let dst_squared = v1
        .iter()
        .zip(v2.iter())
        .fold(T::zero(), |acc, (v1, v2)| acc + (*v1 - *v2).powi(2));
    dst_squared.sqrt()
}

fn norm<T: Float, const N: usize>(v: &[T; N]) -> T {
    let norm_squared = v.iter().fold(T::zero(), |acc, x| acc + (*x).powi(2));
    norm_squared.sqrt()
}
