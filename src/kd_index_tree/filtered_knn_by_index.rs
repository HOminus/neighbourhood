use super::KdIndexTree;
use crate::{distance, norm, FilteredKnnParams};
use alloc::{vec, vec::Vec};
use core::marker::PhantomData;
use num_traits::Float;

impl<'a, T: Float + Clone, const N: usize> KdIndexTree<'a, T, N> {
    pub fn filtered_knn_by_index<F: Fn(usize) -> bool>(
        &self,
        point: &[T; N],
        k: usize,
        filter: &F,
    ) -> Vec<(T, usize)> {
        if k == 0 {
            return vec![];
        }
        let mut subtree_distance = [T::zero(); N];
        let mut result = Vec::with_capacity(k);

        let params = FilteredKnnParams {
            point,
            k: core::num::NonZero::new(k).unwrap(),
            brute_force_size: self.brute_force_size,
            filter,
            _p: PhantomData,
        };

        Self::find_filtered_knn_by_index_recursive(
            self.data,
            &self.indices,
            &params,
            &mut subtree_distance,
            &mut result,
            0,
        );
        result
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn dispatch_find_filtered_knn_by_index_recursive_on_subtrees(
        full_data: &[[T; N]],
        subtree1: &[usize],
        subtree2: &[usize],
        split_point: &[T; N],
        params: &FilteredKnnParams<T, N, usize, impl Fn(usize) -> bool>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<(T, usize)>,
        row: usize,
    ) {
        let next_row = (row + 1) % N;
        Self::find_filtered_knn_by_index_recursive(
            full_data,
            subtree1,
            params,
            subtree_distance,
            result,
            next_row,
        );

        let row_value = subtree_distance[row];
        subtree_distance[row] = params.point[row] - split_point[row];
        let dst = norm(subtree_distance);
        if result.len() < params.k.get() || dst < result.last().unwrap().0 {
            Self::find_filtered_knn_by_index_recursive(
                full_data,
                subtree2,
                params,
                subtree_distance,
                result,
                next_row,
            );
        }
        subtree_distance[row] = row_value;
    }

    fn find_filtered_knn_by_index_recursive(
        full_data: &[[T; N]],
        subtree: &[usize],
        params: &FilteredKnnParams<T, N, usize, impl Fn(usize) -> bool>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<(T, usize)>,
        row: usize,
    ) {
        if subtree.len() <= params.brute_force_size.max(1) {
            for index in subtree {
                let node_point = &full_data[*index];
                Self::knn_try_filtered_insert(params, result, node_point, *index);
            }
        } else {
            let split_index = subtree.len() / 2;
            let split_node_index = subtree[split_index];
            let split_node = &full_data[split_node_index];

            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_node[row] {
                Self::dispatch_find_filtered_knn_by_index_recursive_on_subtrees(
                    full_data,
                    subtree1,
                    subtree2,
                    split_node,
                    params,
                    subtree_distance,
                    result,
                    row,
                );
            } else if params.point[row] > split_node[row] {
                Self::dispatch_find_filtered_knn_by_index_recursive_on_subtrees(
                    full_data,
                    subtree2,
                    subtree1,
                    split_node,
                    params,
                    subtree_distance,
                    result,
                    row,
                );
            }
            Self::knn_try_filtered_insert(params, result, split_node, split_node_index);
        }
    }

    #[inline]
    fn knn_try_filtered_insert(
        params: &FilteredKnnParams<T, N, usize, impl Fn(usize) -> bool>,
        result: &mut Vec<(T, usize)>,
        point: &[T; N],
        index: usize,
    ) {
        let dst = distance(point, params.point);
        if result.is_empty() {
            if (params.filter)(index) {
                result.push((dst, index));
            }
        } else if result.len() < params.k.get() {
            if (params.filter)(index) {
                let pos = result
                    .iter()
                    .position(|p| dst < p.0)
                    .unwrap_or(result.len());
                result.insert(pos, (dst, index));
            }
        } else if params.k.get() <= 20 {
            if (params.filter)(index) {
                let pos = result
                    .iter()
                    .position(|p| dst < p.0)
                    .unwrap_or(result.len());
                result.insert(pos, (dst, index));
                result.pop().unwrap();
            }
        } else if (params.filter)(index) {
            let pos = result
                .binary_search_by(|(lhs, _)| lhs.partial_cmp(&dst).unwrap())
                .unwrap_or_else(|i| i);
            result.insert(pos, (dst, index));
            let _ = result.pop();
        }
    }
}
