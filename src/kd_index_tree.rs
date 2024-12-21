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

    /// Create a new KdIndexTree.
    pub fn new(data: &'a [[T; N]]) -> Self {
        let mut indices: Vec<_> = (0..data.len()).collect();

        Self::select_median_with_respect_to_row_recursive(&mut indices[..], data, 0);
        Self {
            indices,
            data,
            brute_force_size: 0,
        }
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
            row: 0,
        };

        self.find_neighbourhood_by_index_recursive(
            &self.indices,
            params,
            &mut subtree_distance,
            &mut result,
        );
        result
    }

    #[inline]
    fn dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
        &self,
        subtree1: &[usize],
        subtree2: &[usize],
        split_point: &[T; N],
        mut params: NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<usize>,
    ) {
        let row = params.row;
        params.next_row();

        let row_value = subtree_distance[row];
        subtree_distance[row] = Float::abs(params.point[row] - split_point[row]);
        if norm(subtree_distance) <= params.epsilon {
            self.find_neighbourhood_by_index_recursive(subtree2, params, subtree_distance, result);
        }
        subtree_distance[row] = row_value;

        self.find_neighbourhood_by_index_recursive(subtree1, params, subtree_distance, result);
    }

    fn find_neighbourhood_by_index_recursive(
        &self,
        subtree: &[usize],
        params: NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<usize>,
    ) {
        if subtree.len() <= self.brute_force_size.max(1) {
            for index in subtree {
                let node_point = &self.data[*index];
                if distance(node_point, params.point) <= params.epsilon {
                    result.push(*index);
                }
            }
        } else {
            let row = params.row;
            let split_index = subtree.len() / 2;
            let split_node_index = subtree[split_index];
            let split_node = &self.data[split_node_index];

            if distance(split_node, params.point) <= params.epsilon {
                result.push(split_node_index);
            }

            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_node[row] {
                self.dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
                    subtree1,
                    subtree2,
                    split_node,
                    params,
                    subtree_distance,
                    result,
                );
            } else if params.point[row] > split_node[row] {
                self.dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
                    subtree2,
                    subtree1,
                    split_node,
                    params,
                    subtree_distance,
                    result,
                );
            }
        }
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
