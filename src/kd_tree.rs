use alloc::{vec, vec::Vec};
use num_traits::Float;

use crate::{distance, norm, NeighbourhoodParams};

pub struct KdTree<T, const N: usize> {
    data: Vec<[T; N]>,

    /// Determines the size at which the KdIndexTree will switch
    /// to a brute force approach instead of further recursing
    /// the tree.
    pub brute_force_size: usize,
}

impl<T: Float + Clone, const N: usize> KdTree<T, N> {
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

    /// Create a new KdTree.
    pub fn new(mut data: Vec<[T; N]>) -> Self {
        Self::select_median_with_row_recursive(&mut data, 0);
        Self {
            data,
            brute_force_size: 0,
        }
    }

    /// Number of points in the KdTree.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the Kd-tree is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a read-only reference to the data. This can be used together
    /// with the indices returned by [Self::neighbourhood_by_index] to get the actual
    /// points.
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
            row: 0,
        };
        self.find_neighbourhood_recursive(
            self.data.as_slice(),
            params,
            &mut subtree_distance,
            &mut result,
        );
        result
    }

    #[inline]
    fn dispatch_find_neighbourhood_recursive_on_subtrees<'a>(
        &'a self,
        subtree_offset1: &'a [[T; N]],
        subtree_offset2: &'a [[T; N]],
        split_point: &[T; N],
        mut params: NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<&'a [T; N]>,
    ) {
        let row = params.row;
        params.next_row();

        let row_value = subtree_distance[row];
        subtree_distance[row] = Float::abs(params.point[row] - split_point[row]);
        if norm(subtree_distance) <= params.epsilon {
            self.find_neighbourhood_recursive(
                subtree_offset2,
                params,
                subtree_distance,
                result,
            );
        }
        subtree_distance[row] = row_value;

        self.find_neighbourhood_recursive(
            subtree_offset1,
            params,
            subtree_distance,
            result,
        );
    }

    fn find_neighbourhood_recursive<'a>(
        &'a self,
        subtree: &'a [[T; N]],
        params: NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<&'a [T; N]>,
    ) {
        if subtree.len() <= self.brute_force_size.max(1) {
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

            let row = params.row;
            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_point[row] {
                self.dispatch_find_neighbourhood_recursive_on_subtrees(
                    subtree1,
                    subtree2,
                    split_point,
                    params,
                    subtree_distance,
                    result,
                );
            } else if params.point[row] > split_point[row] {
                self.dispatch_find_neighbourhood_recursive_on_subtrees(
                    subtree2,
                    subtree1,
                    split_point,
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
