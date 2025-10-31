use super::KdTree;
use crate::{distance, norm, NeighbourhoodParams};
use alloc::{vec, vec::Vec};
use num_traits::Float;

impl<T: Float + Clone, const N: usize> KdTree<T, N> {
    /// Returns a list of references to points with a distance less than or equals to
    /// epsilon from p.
    pub fn neighbourhood<'a>(&'a self, point: &[T; N], epsilon: T) -> Vec<&'a [T; N]> {
        let mut subtree_distance = [T::zero(); N];
        let mut result = vec![];

        let params = NeighbourhoodParams {
            point,
            epsilon,
            brute_force_size: self.brute_force_size,
        };

        Self::find_neighbourhood_recursive(
            self.data.as_slice(),
            &params,
            &mut subtree_distance,
            &mut result,
            0,
        );
        result
    }

    #[inline]
    fn dispatch_find_neighbourhood_recursive_on_subtrees<'a>(
        subtree_offset1: &'a [[T; N]],
        subtree_offset2: &'a [[T; N]],
        split_point: &[T; N],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<&'a [T; N]>,
        row: usize,
    ) {
        let next_row = (row + 1) % N;

        Self::find_neighbourhood_recursive(
            subtree_offset1,
            params,
            subtree_distance,
            result,
            next_row,
        );

        let row_value = subtree_distance[row];
        subtree_distance[row] = params.point[row] - split_point[row];
        if norm(subtree_distance) <= params.epsilon {
            Self::find_neighbourhood_recursive(
                subtree_offset2,
                params,
                subtree_distance,
                result,
                next_row,
            );
        }
        subtree_distance[row] = row_value;
    }

    fn find_neighbourhood_recursive<'a>(
        subtree: &'a [[T; N]],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<&'a [T; N]>,
        row: usize,
    ) {
        if subtree.len() <= params.brute_force_size.max(1) {
            for pt in subtree.iter() {
                if distance(params.point, pt) <= params.epsilon {
                    result.push(pt);
                }
            }
        } else {
            let split_index = subtree.len() / 2;
            let split_point = &subtree[split_index];

            if distance(split_point, params.point) <= params.epsilon {
                result.push(split_point);
            }

            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_point[row] {
                Self::dispatch_find_neighbourhood_recursive_on_subtrees(
                    subtree1,
                    subtree2,
                    split_point,
                    params,
                    subtree_distance,
                    result,
                    row,
                );
            } else if params.point[row] > split_point[row] {
                Self::dispatch_find_neighbourhood_recursive_on_subtrees(
                    subtree2,
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
