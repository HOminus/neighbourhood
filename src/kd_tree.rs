use alloc::{vec, vec::Vec};
use num_traits::Float;


pub struct KdTree<T, const N: usize> {
    data: Vec<[T; N]>,
    pub brute_force_size: usize,
}

impl<T: Float + Clone, const N: usize> KdTree<T, N> {
    fn select_median_with_row_recursive(
        slice: &mut [[T; N]],
        row: usize, 
    ) {
        let split_index = slice.len() / 2;
        slice.select_nth_unstable_by(split_index,
            |lhs, rhs| lhs[row].partial_cmp(&rhs[row]).unwrap());

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
            brute_force_size: 0
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
}
