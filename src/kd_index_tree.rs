use num_traits::Float;

use alloc::{vec, vec::Vec};

use crate::{distance, norm, NeighbourhoodParams};

pub struct KdIndexTree<'a, T, const N: usize> {
    indices: Vec<usize>,

    /// Reference to the points indexed in the KdIndexTree.
    pub data: &'a [[T; N]],

    /// Determines the size at which the KdIndexTree will switch
    /// to a brute force approach instead of further recursing
    /// the tree.
    pub brute_force_size: usize,
}

impl<'a, T: Float + Clone, const N: usize> KdIndexTree<'a, T, N> {
    fn select_median_with_respect_to_row_recursive(
        slice: &mut [usize],
        full_data: &[[T; N]],
        row: usize,
    ) {
        let split_index = slice.len() / 2;
        slice.select_nth_unstable_by(split_index, |lhs, rhs| {
            full_data[*lhs][row]
                .partial_cmp(&full_data[*rhs][row])
                .unwrap()
        });

        if slice.len() > 3 {
            let (slice1, slice2) = slice.split_at_mut(split_index);

            let row = (row + 1) % N;
            if slice1.len() > 1 {
                Self::select_median_with_respect_to_row_recursive(slice1, full_data, row);
            }

            if slice2.len() > 2 {
                let slice2 = &mut slice2[1..];
                Self::select_median_with_respect_to_row_recursive(slice2, full_data, row);
            }
        }
    }

    /// Create a new K-d Index Tree.
    pub fn new(data: &'a [[T; N]]) -> Self {
        let mut indices: Vec<_> = (0..data.len()).collect();

        Self::select_median_with_respect_to_row_recursive(&mut indices[..], data, 0);
        Self {
            indices,
            data,
            brute_force_size: 0,
        }
    }

    /// Create a new K-d Index Tree and sets the `brute_force_size`.
    pub fn with_brute_force_size(data: &'a [[T; N]], brute_force_size: usize) -> Self {
        let mut self_ = Self::new(data);
        self_.brute_force_size = brute_force_size;
        self_
    }

    /// Returns true id the KdIndexTree is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of points in the KdIndexTree.
    pub fn len(&self) -> usize {
        self.data.len()
    }

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

        Self::find_neighbourhood_by_index_recursive(
            full_data,
            subtree1,
            params,
            subtree_distance,
            result,
            next_row,
        );
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

        result += Self::count_neighbourhood_recursive(
            full_data,
            subtree1,
            params,
            subtree_distance,
            next_row,
        );
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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::KdIndexTree;
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

        let kd_index_tree = KdIndexTree::new(&data);

        let eps = 1.2;
        let point = [0.0, 0.0, 0.0];
        let neighbourhood = kd_index_tree.neighbourhood_by_index(&point, eps);
        assert_eq!(neighbourhood.len(), 7);
        for index in neighbourhood {
            assert!(distance(&point, &data[index]) < eps);
        }
    }
}
