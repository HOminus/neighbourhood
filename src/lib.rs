#![doc = include_str!("../README.md")]
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

#[inline]
fn distance<T: Float, const N: usize>(v1: &[T; N], v2: &[T; N]) -> T {
    let mut dst = T::zero();
    for i in 0..N {
        dst = dst + (v1[i] - v2[i]).powi(2);
    }
    dst.sqrt()
}

#[inline]
fn norm<T: Float, const N: usize>(v: &[T; N]) -> T {
    let mut norm = T::zero();
    for i in 0..N {
        norm = norm + v[i].powi(2);
    }
    norm.sqrt()
}
