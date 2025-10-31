use super::KdTree;
use crate::{distance, norm, NeighbourhoodParams};
use alloc::{vec, vec::Vec};
use num_traits::Float;

impl<T: Float + Clone, const N: usize> KdTree<T, N> {
    pub fn neighbourhood_by_index(&self, point: &[T; N], epsilon: T) -> Vec<usize> {
        let mut subtree_distance = [T::zero(); N];
        let mut result = vec![];

        let params = NeighbourhoodParams {
            point,
            epsilon,
            brute_force_size: self.brute_force_size,
        };

        Self::find_neighbourhood_by_index_recursive(
            0,
            self.data.as_slice(),
            &params,
            &mut subtree_distance,
            &mut result,
            0,
        );
        result
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn dispatch_find_neighbourhood_by_index_recursive_on_subtrees<'a>(
        subtree1_offset: usize,
        subtree1: &'a [[T; N]],
        subtree2_offset: usize,
        subtree2: &'a [[T; N]],
        split_point: &[T; N],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<usize>,
        row: usize,
    ) {
        let next_row = (row + 1) % N;

        Self::find_neighbourhood_by_index_recursive(
            subtree1_offset,
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
                subtree2_offset,
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
        subtree_offset: usize,
        subtree: &[[T; N]],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<usize>,
        row: usize,
    ) {
        if subtree.len() <= params.brute_force_size.max(1) {
            for (index, pt) in subtree.iter().enumerate() {
                if distance(params.point, pt) <= params.epsilon {
                    result.push(subtree_offset + index);
                }
            }
        } else {
            let split_index = subtree.len() / 2;
            let split_point = &subtree[split_index];

            if distance(split_point, params.point) <= params.epsilon {
                result.push(subtree_offset + split_index);
            }

            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_point[row] {
                Self::dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
                    subtree_offset,
                    subtree1,
                    subtree_offset + split_index + 1,
                    subtree2,
                    split_point,
                    params,
                    subtree_distance,
                    result,
                    row,
                );
            } else if params.point[row] > split_point[row] {
                Self::dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
                    subtree_offset + split_index + 1,
                    subtree2,
                    subtree_offset,
                    subtree1,
                    split_point,
                    params,
                    subtree_distance,
                    result,
                    row,
                );
            }
        }
    }
}
