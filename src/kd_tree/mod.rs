use alloc::vec::Vec;
use num_traits::Float;

pub mod count_neighbourhood;
pub mod knn;
pub mod knn_by_index;
pub mod neighbourhood;
pub mod neighbourhood_by_index;

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
