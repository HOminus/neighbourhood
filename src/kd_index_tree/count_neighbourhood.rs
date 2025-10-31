use super::KdIndexTree;
use crate::{distance, norm, NeighbourhoodParams};
use num_traits::Float;

impl<'a, T: Float + Clone, const N: usize> KdIndexTree<'a, T, N> {
    pub fn count_neighbourhood(&self, point: &[T; N], epsilon: T) -> usize {
        let mut subtree_distance = [T::zero(); N];

        let params = NeighbourhoodParams {
            epsilon,
            point,
            brute_force_size: self.brute_force_size,
        };

        Self::count_neighbourhood_recursive(
            self.data,
            &self.indices,
            &params,
            &mut subtree_distance,
            0,
        )
    }

    #[inline]
    fn dispatch_count_neighbourhood_recursive_on_subtrees(
        full_data: &[[T; N]],
        subtree1: &[usize],
        subtree2: &[usize],
        split_point: &[T; N],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        row: usize,
    ) -> usize {
        let mut result = 0;
        let next_row = (row + 1) % N;

        result += Self::count_neighbourhood_recursive(
            full_data,
            subtree1,
            params,
            subtree_distance,
            next_row,
        );

        let row_value = subtree_distance[row];
        subtree_distance[row] = params.point[row] - split_point[row];
        if norm(subtree_distance) <= params.epsilon {
            result += Self::count_neighbourhood_recursive(
                full_data,
                subtree2,
                params,
                subtree_distance,
                next_row,
            );
        }
        subtree_distance[row] = row_value;

        result
    }

    fn count_neighbourhood_recursive(
        full_data: &[[T; N]],
        subtree: &[usize],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        row: usize,
    ) -> usize {
        let mut count = 0;
        if subtree.len() <= params.brute_force_size.max(1) {
            for index in subtree {
                let node_point = &full_data[*index];
                if distance(node_point, params.point) <= params.epsilon {
                    count += 1;
                }
            }
        } else {
            let split_index = subtree.len() / 2;
            let split_node_index = subtree[split_index];
            let split_node = &full_data[split_node_index];

            if distance(split_node, params.point) <= params.epsilon {
                count += 1;
            }

            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_node[row] {
                count += Self::dispatch_count_neighbourhood_recursive_on_subtrees(
                    full_data,
                    subtree1,
                    subtree2,
                    split_node,
                    params,
                    subtree_distance,
                    row,
                );
            } else if params.point[row] > split_node[row] {
                count += Self::dispatch_count_neighbourhood_recursive_on_subtrees(
                    full_data,
                    subtree2,
                    subtree1,
                    split_node,
                    params,
                    subtree_distance,
                    row,
                );
            }
        }
        count
    }
}
