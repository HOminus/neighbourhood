#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

pub mod kd_index_tree;
pub mod kd_tree;

use core::marker::PhantomData;

pub use kd_index_tree::KdIndexTree;
pub use kd_tree::KdTree;

use num_traits::Float;

#[derive(Debug, Copy, Clone)]
struct NeighbourhoodParams<'a, T, const N: usize> {
    point: &'a [T; N],
    epsilon: T,
    brute_force_size: usize,
}

#[derive(Debug, Copy, Clone)]
struct KnnParams<'a, T, const N: usize> {
    point: &'a [T; N],
    k: core::num::NonZero<usize>,
    brute_force_size: usize,
}

#[derive(Copy, Clone)]
struct FilteredKnnParams<'a, T, const N: usize, P, F: Fn(P) -> bool> {
    point: &'a [T; N],
    k: core::num::NonZero<usize>,
    brute_force_size: usize,
    filter: &'a F,
    _p: PhantomData<P>,
}

#[inline]
fn distance<T: Float, const N: usize>(v1: &[T; N], v2: &[T; N]) -> T {
    let mut dst = T::zero();
    for i in 0..N {
        dst = dst + (v1[i] - v2[i]).powi(2);
    }
    dst.sqrt()
}

#[allow(clippy::needless_range_loop)]
#[inline]
fn norm<T: Float, const N: usize>(v: &[T; N]) -> T {
    let mut norm = T::zero();
    for i in 0..N {
        norm = norm + v[i].powi(2);
    }
    norm.sqrt()
}
