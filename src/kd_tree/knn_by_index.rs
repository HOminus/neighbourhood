use super::KdTree;
use crate::{distance, norm, KnnParams};
use alloc::{vec, vec::Vec};
use num_traits::Float;

impl<T: Float + Clone, const N: usize> KdTree<T, N> {
    pub fn knn_by_index(&self, point: &[T; N], k: usize) -> Vec<(T, usize)> {
        if k == 0 {
            return vec![];
        }
        let mut subtree_distance = [T::zero(); N];
        let mut result = Vec::with_capacity(k);

        let params = KnnParams {
            point,
            k: core::num::NonZero::new(k).unwrap(),
            brute_force_size: self.brute_force_size,
        };
        Self::find_knn_by_index_recursive(
            0,
            &self.data,
            &params,
            &mut subtree_distance,
            &mut result,
            0,
        );

        result
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn dispatch_find_knn_by_index_recursive_on_subtrees<'a>(
        subtree1_offset: usize,
        subtree1: &'a [[T; N]],
        subtree2_offset: usize,
        subtree2: &'a [[T; N]],
        split_point: &[T; N],
        params: &KnnParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<(T, usize)>,
        row: usize,
    ) {
        let next_row = (row + 1) % N;
        Self::find_knn_by_index_recursive(
            subtree1_offset,
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
            Self::find_knn_by_index_recursive(
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

    fn find_knn_by_index_recursive(
        subtree_offset: usize,
        subtree: &[[T; N]],
        params: &KnnParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<(T, usize)>,
        row: usize,
    ) {
        if subtree.len() <= params.brute_force_size.max(1) {
            for (index, pt) in subtree.iter().enumerate() {
                Self::knn_try_insert_index(params, result, pt, subtree_offset + index);
            }
        } else {
            let split_index = subtree.len() / 2;
            let split_point = &subtree[split_index];

            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_point[row] {
                Self::dispatch_find_knn_by_index_recursive_on_subtrees(
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
                Self::dispatch_find_knn_by_index_recursive_on_subtrees(
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

            Self::knn_try_insert_index(params, result, split_point, subtree_offset + split_index);
        }
    }

    #[inline]
    fn knn_try_insert_index(
        params: &KnnParams<T, N>,
        result: &mut Vec<(T, usize)>,
        point: &[T; N],
        index: usize,
    ) {
        let dst = distance(point, params.point);
        if result.is_empty() {
            result.push((dst, index));
        } else if result.len() < params.k.get() {
            let pos = result
                .iter()
                .position(|p| dst < p.0)
                .unwrap_or(result.len());
            result.insert(pos, (dst, index));
        } else if params.k.get() <= 20 {
            let pos = result
                .iter()
                .position(|p| dst < p.0)
                .unwrap_or(result.len());
            result.insert(pos, (dst, index));
            let _ = result.pop();
        } else {
            let pos = result
                .binary_search_by(|(lhs, _)| lhs.partial_cmp(&dst).unwrap())
                .unwrap_or_else(|i| i);

            result.insert(pos, (dst, index));
            let _ = result.pop();
        }
    }
}
