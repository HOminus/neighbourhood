use super::KdTree;
use crate::{distance, norm, NeighbourhoodParams};
use num_traits::Float;

impl<T: Float + Clone, const N: usize> KdTree<T, N> {
    pub fn count_neighbourhood(&self, point: &[T; N], epsilon: T) -> usize {
        let mut subtree_distance = [T::zero(); N];
        let params = NeighbourhoodParams {
            point,
            epsilon,
            brute_force_size: self.brute_force_size,
        };

        Self::count_neighbourhood_recursive(&self.data, &params, &mut subtree_distance, 0)
    }

    #[inline]
    fn dispatch_count_neighbourhood_recursive_on_subtrees<'a>(
        subtree1: &'a [[T; N]],
        subtree2: &'a [[T; N]],
        split_point: &[T; N],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        row: usize,
    ) -> usize {
        let mut result = 0;
        let next_row = (row + 1) % N;

        result += Self::count_neighbourhood_recursive(subtree1, params, subtree_distance, next_row);

        let row_value = subtree_distance[row];
        subtree_distance[row] = params.point[row] - split_point[row];
        if norm(subtree_distance) <= params.epsilon {
            result +=
                Self::count_neighbourhood_recursive(subtree2, params, subtree_distance, next_row);
        }
        subtree_distance[row] = row_value;

        result
    }

    fn count_neighbourhood_recursive(
        subtree: &[[T; N]],
        params: &NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        row: usize,
    ) -> usize {
        let mut count = 0;
        if subtree.len() <= params.brute_force_size.max(1) {
            for pt in subtree.iter() {
                if distance(params.point, pt) <= params.epsilon {
                    count += 1;
                }
            }
        } else {
            let split_index = subtree.len() / 2;
            let split_point = &subtree[split_index];

            if distance(split_point, params.point) <= params.epsilon {
                count += 1;
            }

            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_point[row] {
                count += Self::dispatch_count_neighbourhood_recursive_on_subtrees(
                    subtree1,
                    subtree2,
                    split_point,
                    params,
                    subtree_distance,
                    row,
                );
            } else if params.point[row] > split_point[row] {
                count += Self::dispatch_count_neighbourhood_recursive_on_subtrees(
                    subtree2,
                    subtree1,
                    split_point,
                    params,
                    subtree_distance,
                    row,
                );
            }
        }
        count
    }
}
