use alloc::vec::Vec;
use num_traits::Float;

pub mod count_neighbourhood;
pub mod filtered_knn_by_index;
pub mod knn_by_index;
pub mod neighbourhood_by_index;

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
