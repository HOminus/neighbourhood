use super::KdIndexTree;
use crate::{distance, norm, NeighbourhoodParams};
use alloc::{vec, vec::Vec};
use num_traits::Float;

impl<'a, T: Float + Clone, const N: usize> KdIndexTree<'a, T, N> {
    /// Returns the index of all points with a distance less than or equals to
    /// epsilon from p. The list of indices can bes used toghether with [Self::data]
    /// to retrieve the points.
    pub fn neighbourhood_by_index(&self, point: &[T; N], epsilon: T) -> Vec<usize> {
        let mut result = vec![];
        let mut subtree_distance = [T::zero(); N];

        let params = NeighbourhoodParams {
            epsilon,
            point,
            brute_force_size: self.brute_force_size,
        };

        Self::find_neighbourhood_by_index_recursive(
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
    fn dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
        full_data: &[[T; N]],
        subtree1: &[usize],
        subtree2: &[usize],
        split_point: &[T; N],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<usize>,
        row: usize,
    ) {
        let next_row = (row + 1) % N;

        Self::find_neighbourhood_by_index_recursive(
            full_data,
            subtree1,
            params,
            subtree_distance,
            result,
            next_row,
        );

        let row_value = subtree_distance[row];
        subtree_distance[row] = params.point[row] - split_point[row];
        if norm(subtree_distance) <= params.epsilon {
            Self::find_neighbourhood_by_index_recursive(
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

    fn find_neighbourhood_by_index_recursive(
        full_data: &[[T; N]],
        subtree: &[usize],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<usize>,
        row: usize,
    ) {
        if subtree.len() <= params.brute_force_size.max(1) {
            for index in subtree {
                let node_point = &full_data[*index];
                if distance(node_point, params.point) <= params.epsilon {
                    result.push(*index);
                }
            }
        } else {
            let split_index = subtree.len() / 2;
            let split_node_index = subtree[split_index];
            let split_node = &full_data[split_node_index];

            if distance(split_node, params.point) <= params.epsilon {
                result.push(split_node_index);
            }

            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_node[row] {
                Self::dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
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
                Self::dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
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
        }
    }
}
