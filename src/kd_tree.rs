use alloc::{vec, vec::Vec};
use num_traits::Float;

use crate::{distance, norm, NeighbourhoodParams};

#[derive(Copy, Clone)]
struct OffsetSubtree<'a, T, const N: usize>(usize, &'a [[T; N]]);

pub struct KdTree<T, const N: usize> {
    data: Vec<[T; N]>,
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

    pub fn new(mut data: Vec<[T; N]>) -> Self {
        Self::select_median_with_row_recursive(&mut data, 0);
        Self {
            data,
            brute_force_size: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[[T; N]] {
        self.data.as_slice()
    }

    pub fn neighbourhood_by_index(&self, point: &[T; N], epsilon: T) -> Vec<usize> {
        let mut subtree_distance = [T::zero(); N];
        let mut result = vec![];

        let params = NeighbourhoodParams {
            point,
            epsilon,
            row: 0,
        };
        let subtree_offset = OffsetSubtree(0, self.data.as_slice());
        self.find_neighbourhood_by_index_recursive(
            subtree_offset,
            params,
            &mut subtree_distance,
            &mut result,
        );
        result
    }

    #[inline]
    fn dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
        &self,
        subtree_offset1: OffsetSubtree<T, N>,
        subtree_offset2: OffsetSubtree<T, N>,
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
            self.find_neighbourhood_by_index_recursive(
                subtree_offset2,
                params,
                subtree_distance,
                result,
            );
        }
        subtree_distance[row] = row_value;

        self.find_neighbourhood_by_index_recursive(
            subtree_offset1,
            params,
            subtree_distance,
            result,
        );
    }

    fn find_neighbourhood_by_index_recursive(
        &self,
        subtree_offset: OffsetSubtree<T, N>,
        params: NeighbourhoodParams<T, N>,
        subtree_distance: &mut [T; N],
        result: &mut Vec<usize>,
    ) {
        let OffsetSubtree(subtree_offset, subtree) = subtree_offset;
        if subtree.len() <= self.brute_force_size.max(1) {
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

            let row = params.row;
            let subtree1 = &subtree[..split_index];
            let subtree2 = &subtree[(split_index + 1)..];
            if params.point[row] <= split_point[row] {
                self.dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
                    OffsetSubtree(subtree_offset, subtree1),
                    OffsetSubtree(subtree_offset + split_index + 1, subtree2),
                    split_point,
                    params,
                    subtree_distance,
                    result,
                );
            } else if params.point[row] > split_point[row] {
                self.dispatch_find_neighbourhood_by_index_recursive_on_subtrees(
                    OffsetSubtree(subtree_offset + split_index + 1, subtree2),
                    OffsetSubtree(subtree_offset, subtree1),
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
        let neighbourhood = kd_tree.neighbourhood_by_index(&point, eps);
        assert_eq!(neighbourhood.len(), 7);
        for index in neighbourhood {
            assert!(distance(&point, &kd_tree.data()[index]) <= eps);
        }
    }
}
