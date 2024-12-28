use alloc::{vec, vec::Vec};
use num_traits::Float;

use crate::{distance, norm, KnnParams, NeighbourhoodParams};

pub struct KdTree<T, const N: usize> {
    data: Vec<[T; N]>,

    /// Determines the size at which the KdIndexTree will switch
    /// to a brute force approach instead of further recursing
    /// the tree.
    pub brute_force_size: usize,
}

impl<T: Float + Clone, const N: usize> KdTree<T, N> {
    pub const DEFAULT_BRUTE_FORCE_SIZE: usize = if core::mem::size_of::<T>() >= 64 {
        25
    } else {
        34
    };

    fn select_median_with_row_recursive(slice: &mut [[T; N]], row: usize) {
        let split_index = slice.len() / 2;
        slice.select_nth_unstable_by(split_index, |lhs, rhs| {
            lhs[row].partial_cmp(&rhs[row]).unwrap()
        });

        if slice.len() > 3 {
            let (slice1, slice2) = slice.split_at_mut(split_index);

            let row = (row + 1) % N;
            if slice1.len() > 1 {
                Self::select_median_with_row_recursive(slice1, row);
            }

            if slice2.len() > 2 {
                let slice2 = &mut slice2[1..];
                Self::select_median_with_row_recursive(slice2, row);
            }
        }
    }

    /// Create a new K-d Tree.
    pub fn new(mut data: Vec<[T; N]>) -> Self {
        Self::select_median_with_row_recursive(&mut data, 0);
        Self {
            data,
            brute_force_size: Self::DEFAULT_BRUTE_FORCE_SIZE,
        }
    }

    /// Create a new K-d Tree and sets the `brute_force_size`.
    pub fn with_brute_force_size(data: Vec<[T; N]>, brute_force_size: usize) -> Self {
        let mut self_ = Self::new(data);
        self_.brute_force_size = brute_force_size;
        self_
    }

    /// Number of points in the KdTree.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the Kd-tree is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a read-only reference to the data.
    pub fn data(&self) -> &[[T; N]] {
        self.data.as_slice()
    }

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

        Self::find_neighbourhood_recursive(
            subtree_offset1,
            params,
            subtree_distance,
            result,
            next_row,
        );
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

        let row_value = subtree_distance[row];
        subtree_distance[row] = params.point[row] - split_point[row];
        if norm(subtree_distance) <= params.epsilon {
            result +=
                Self::count_neighbourhood_recursive(subtree2, params, subtree_distance, next_row);
        }
        subtree_distance[row] = row_value;

        result += Self::count_neighbourhood_recursive(subtree1, params, subtree_distance, next_row);
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

    pub fn knn<'a>(&'a self, point: &[T; N], k: usize) -> Vec<(T, &'a [T; N])> {
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
        Self::find_knn_recursive(&self.data, &params, &mut subtree_distance, &mut result, 0);

        result
    }

    #[inline]
    fn dispatch_find_knn_recursive_on_subtrees<'a>(
        subtree1: &'a [[T; N]],
        subtree2: &'a [[T; N]],
        split_point: &[T; N],
        params: &KnnParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<(T, &'a [T; N])>,
        row: usize,
    ) {
        let next_row = (row + 1) % N;
        Self::find_knn_recursive(subtree1, params, subtree_distance, result, next_row);

        let row_value = subtree_distance[row];
        subtree_distance[row] = params.point[row] - split_point[row];
        let dst = norm(subtree_distance);
        if result.len() < params.k.get() || dst < result.last().unwrap().0 {
            Self::find_knn_recursive(subtree2, params, subtree_distance, result, next_row);
        }
        subtree_distance[row] = row_value;
    }

    fn find_knn_recursive<'a>(
        subtree: &'a [[T; N]],
        params: &KnnParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<(T, &'a [T; N])>,
        row: usize,
    ) {
        if subtree.len() <= params.brute_force_size.max(1) {
            for pt in subtree.iter() {
                Self::knn_try_insert(params, result, pt);
            }
        } else {
            let split_index = subtree.len() / 2;
            let split_point = &subtree[split_index];

            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_point[row] {
                Self::dispatch_find_knn_recursive_on_subtrees(
                    subtree1,
                    subtree2,
                    split_point,
                    params,
                    subtree_distance,
                    result,
                    row,
                );
            } else if params.point[row] > split_point[row] {
                Self::dispatch_find_knn_recursive_on_subtrees(
                    subtree2,
                    subtree1,
                    split_point,
                    params,
                    subtree_distance,
                    result,
                    row,
                );
            }

            Self::knn_try_insert(params, result, split_point);
        }
    }

    #[inline]
    fn knn_try_insert<'a>(
        params: &KnnParams<T, N>,
        result: &mut Vec<(T, &'a [T; N])>,
        point: &'a [T; N],
    ) {
        let dst = distance(point, params.point);
        if result.is_empty() {
            result.push((dst, point));
        } else if result.len() < params.k.get() {
            let pos = result
                .iter()
                .position(|p| dst < p.0)
                .unwrap_or(result.len());
            result.insert(pos, (dst, point));
        } else if params.k.get() <= 20 {
            let pos = result
                .iter()
                .position(|p| dst < p.0)
                .unwrap_or(result.len());
            result.insert(pos, (dst, point));
            let _ = result.pop();
        } else {
            let pos = result
                .binary_search_by(|(lhs, _)| lhs.partial_cmp(&dst).unwrap())
                .unwrap_or_else(|i| i);

            result.insert(pos, (dst, point));
            let _ = result.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::KdTree;
    use crate::distance;

    #[test]
    fn simple_neighbourhood_query_test() {
        let mut data = vec![];

        let line = [-2.0, -1.0, 0.0, 1.0, 2.0];
        for x in line {
            for y in line {
                for z in line {
                    data.push([x, y, z]);
                }
            }
        }

        let kd_tree = KdTree::new(data);

        let eps = 1.2;
        let point = [0.0, 0.0, 0.0];
        let neighbourhood = kd_tree.neighbourhood(&point, eps);
        assert_eq!(neighbourhood.len(), 7);
        for pt in neighbourhood {
            assert!(distance(&point, pt) <= eps);
        }
    }
}
