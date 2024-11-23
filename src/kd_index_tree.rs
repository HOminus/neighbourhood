use num_traits::Float;

use alloc::vec::Vec;

pub struct KdIndexTree<'a, T, const N: usize> {
    indices: Vec<usize>,
    pub data: &'a [[T; N]],
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

    pub fn new(data: &'a [[T; N]]) -> Self {
        let mut indices: Vec<_> = (0..data.len()).collect();

        Self::select_median_with_respect_to_row_recursive(&mut indices[..], data, 0);
        Self {
            indices,
            data,
            brute_force_size: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}
