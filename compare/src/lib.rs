use rand::{
    distributions::{
        uniform::{SampleBorrow, SampleUniform},
        Distribution, Uniform,
    },
    SeedableRng,
};
use std::fmt::Debug;

pub fn make_points<
    S: SampleUniform + Default + Copy,
    const N: usize,
    B: SampleBorrow<S> + Sized,
>(
    num_points: usize,
    l: B,
    h: B,
    seed: u64,
) -> Vec<[S; N]> {
    let mut points = Vec::with_capacity(num_points);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    let distr = Uniform::new_inclusive(l, h);
    for _ in 0..num_points {
        let mut p: [S; N] = [S::default(); N];
        for item in p.iter_mut() {
            *item = distr.sample(&mut rng);
        }
        points.push(p);
    }
    points
}

pub fn distance<T: num_traits::Float, const N: usize>(p1: &[T; N], p2: &[T; N]) -> T {
    let dst = p1
        .iter()
        .zip(p2.iter())
        .fold(T::zero(), |acc, x| acc + (*x.0 - *x.1).powi(2));
    dst.sqrt()
}

pub fn print_mismatch_founds<T: num_traits::Float + Debug, const N: usize>(
    p: &[T; N],
    found1: &[[T; N]],
    found2: &[[T; N]],
) {
    for i in 0..usize::max(found1.len(), found2.len()) {
        let p1 = found1.get(i);
        let p2 = found2.get(i);
        let d1 = p1.map(|p1| distance(p, p1));
        let d2 = p2.map(|p2| distance(p, p2));

        match (p1, p2) {
            (Some(p1), Some(p2)) => {
                println!(
                    "P1: {:?} {:?} P2: {:?} {:?}",
                    p1,
                    d1.unwrap(),
                    p2,
                    d2.unwrap()
                );
            }
            (Some(p1), None) => {
                println!("P1: {:?} {:?}", p1, d1.unwrap());
            }
            (None, Some(p2)) => {
                println!("P2: {:?} {:?}", p2, d2.unwrap());
            }
            _ => unreachable!(),
        }
    }
}

pub fn sort_query_result<T: num_traits::Float, const N: usize>(p: &[T; N], pts: &mut [[T; N]]) {
    pts.sort_by(|p1, p2| {
        let d1 = distance(p, p1);
        let d2 = distance(p, p2);
        d1.partial_cmp(&d2).unwrap()
    });
}

// A unified KdTree Api. Useful for testing.
pub trait UnifiedKdTreeTestApi<T: num_traits::Float, const N: usize> {
    //fn new(data: &'a [[T; N]]) -> Self;

    fn query_within(&self, p: &[T; N], eps: T, points: &[[T; N]]) -> Vec<[T; N]>;

    fn count_within(&self, p: &[T; N], eps: T) -> usize;

}

pub mod nh {
    pub struct KdTree<T: num_traits::Float, const N: usize>(neighbourhood::KdTree<T, N>);

    impl<T: num_traits::Float, const N: usize> KdTree<T, N> {
        pub fn new(data: &[[T; N]]) -> Self {
            Self(neighbourhood::KdTree::with_brute_force_size(
                data.to_vec(),
                0,
            ))
        }
    }

    impl<T: num_traits::Float, const N: usize> crate::UnifiedKdTreeTestApi<T, N> for KdTree<T, N> {
        fn query_within(&self, p: &[T; N], eps: T, _: &[[T; N]]) -> Vec<[T; N]> {
            let result = self.0.neighbourhood(p, eps);
            let mut points: Vec<_> = result.into_iter().cloned().collect();
            crate::sort_query_result(p, &mut points);
            points
        }

        fn count_within(&self, p: &[T; N], eps: T) -> usize {
            self.0.count_neighbourhood(p, eps)
        }
    }

    pub struct KdIndexTree<'a, T: num_traits::Float, const N: usize>(
        neighbourhood::KdIndexTree<'a, T, N>,
    );

    impl<'a, T: num_traits::Float, const N: usize> KdIndexTree<'a, T, N> {
        pub fn new(data: &'a [[T; N]]) -> Self {
            Self(neighbourhood::KdIndexTree::with_brute_force_size(data, 0))
        }
    }

    impl<T: num_traits::Float, const N: usize> crate::UnifiedKdTreeTestApi<T, N>
        for KdIndexTree<'_, T, N>
    {
        fn query_within(&self, p: &[T; N], eps: T, _: &[[T; N]]) -> Vec<[T; N]> {
            let result = self.0.neighbourhood_by_index(p, eps);
            let mut points: Vec<_> = result.into_iter().map(|i| self.0.data[i]).collect();
            crate::sort_query_result(p, &mut points);
            points
        }

        fn count_within(&self, p: &[T; N], eps: T) -> usize {
            self.0.count_neighbourhood(p, eps)
        }
    }
}

pub mod kiddo {
    use kiddo::immutable::float::kdtree::ImmutableKdTree;

    pub struct KdTree<T, const N: usize>(ImmutableKdTree<T, u64, N, 32>)
    where
        T: Default
            + std::fmt::Debug
            + Sync
            + Send
            + num_traits::Float
            + num_traits::float::FloatCore
            + num_traits::NumAssign
            + kiddo::float_leaf_slice::leaf_slice::LeafSliceFloat<u64>
            + kiddo::float_leaf_slice::leaf_slice::LeafSliceFloatChunk<u64, N>;

    impl<T, const N: usize> KdTree<T, N>
    where
        T: Default
            + std::fmt::Debug
            + Sync
            + Send
            + num_traits::Float
            + num_traits::float::FloatCore
            + num_traits::NumAssign
            + kiddo::float_leaf_slice::leaf_slice::LeafSliceFloat<u64>
            + kiddo::float_leaf_slice::leaf_slice::LeafSliceFloatChunk<u64, N>,
    {
        pub fn new(data: &[[T; N]]) -> Self {
            Self(ImmutableKdTree::new_from_slice(data))
        }
    }

    impl<T, const N: usize> crate::UnifiedKdTreeTestApi<T, N> for KdTree<T, N>
    where
        T: Default
            + std::fmt::Debug
            + Sync
            + Send
            + num_traits::Float
            + num_traits::float::FloatCore
            + num_traits::NumAssign
            + kiddo::float_leaf_slice::leaf_slice::LeafSliceFloat<u64>
            + kiddo::float_leaf_slice::leaf_slice::LeafSliceFloatChunk<u64, N>,
    {
        fn query_within(&self, p: &[T; N], eps: T, points: &[[T; N]]) -> Vec<[T; N]> {
            let result = self.0.within_unsorted::<kiddo::SquaredEuclidean>(p, eps * eps);
            let mut points: Vec<_> = result
                .into_iter()
                .map(|n| points[n.item as usize])
                .collect();
            crate::sort_query_result(p, &mut points);
            points
        }

        fn count_within(&self, p: &[T; N], eps: T) -> usize {
            let result = self.0.within_unsorted::<kiddo::SquaredEuclidean>(p, eps * eps);
            result.len()
        }
    }
}

pub mod kdtree {
    pub struct KdTree<T: num_traits::Float, const N: usize>(kdtree::KdTree<T, usize, [T; N]>);

    impl<T: num_traits::Float, const N: usize> KdTree<T, N> {
        pub fn new(data: &[[T; N]]) -> Self {
            let mut kdtree = kdtree::KdTree::with_capacity(N, data.len());
            for (index, p) in data.iter().enumerate() {
                kdtree.add(*p, index).unwrap();
            }
            Self(kdtree)
        }
    }

    impl<T: num_traits::Float, const N: usize> crate::UnifiedKdTreeTestApi<T, N> for KdTree<T, N> {
        fn query_within(&self, p: &[T; N], eps: T, points: &[[T; N]]) -> Vec<[T; N]> {
            let result = self
                .0
                .within(p, eps * eps, &kdtree::distance::squared_euclidean)
                .unwrap();
            let mut points: Vec<_> = result.into_iter().map(|(_, v)| points[*v]).collect();
            crate::sort_query_result(p, &mut points);
            points
        }

        fn count_within(&self, p: &[T; N], eps: T) -> usize {
            let result = self
                .0
                .within(p, eps * eps, &kdtree::distance::squared_euclidean)
                .unwrap();
            result.len()
        }
    }
}

pub mod kd_tree {
    //kd_tree::KdTree is not generic over N
    pub enum KdTree<T>
    where
        T: num_traits::Float + num_traits::NumAssign,
    {
        One(kd_tree::KdTree<[T; 1]>),
        Two(kd_tree::KdTree<[T; 2]>),
        Three(kd_tree::KdTree<[T; 3]>),
        Four(kd_tree::KdTree<[T; 4]>),
        Five(kd_tree::KdTree<[T; 5]>),
        Six(kd_tree::KdTree<[T; 6]>),
        Seven(kd_tree::KdTree<[T; 7]>),
        Eight(kd_tree::KdTree<[T; 8]>),
        Nine(kd_tree::KdTree<[T; 9]>),
        Ten(kd_tree::KdTree<[T; 10]>),
        Eleven(kd_tree::KdTree<[T; 11]>),
        Twelve(kd_tree::KdTree<[T; 12]>),
        Thirteen(kd_tree::KdTree<[T; 13]>),
        Fourteen(kd_tree::KdTree<[T; 14]>),
        Fifteen(kd_tree::KdTree<[T; 15]>),
        Sixteen(kd_tree::KdTree<[T; 16]>),
    }

    #[allow(clippy::missing_transmute_annotations)]
    impl<T> KdTree<T>
    where
        T: num_traits::Float + num_traits::float::FloatCore + num_traits::NumAssign,
    {
        pub fn new<const N: usize>(data: &[[T; N]]) -> Self {
            unsafe {
                match N {
                    1 => Self::One(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 1]]>(data).to_vec(),
                    )),
                    2 => Self::Two(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 2]]>(data).to_vec(),
                    )),
                    3 => Self::Three(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 3]]>(data).to_vec(),
                    )),
                    4 => Self::Four(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 4]]>(data).to_vec(),
                    )),
                    5 => Self::Five(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 5]]>(data).to_vec(),
                    )),
                    6 => Self::Six(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 6]]>(data).to_vec(),
                    )),
                    7 => Self::Seven(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 7]]>(data).to_vec(),
                    )),
                    8 => Self::Eight(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 8]]>(data).to_vec(),
                    )),
                    9 => Self::Nine(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 9]]>(data).to_vec(),
                    )),
                    10 => Self::Ten(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 10]]>(data).to_vec(),
                    )),
                    11 => Self::Eleven(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 11]]>(data).to_vec(),
                    )),
                    12 => Self::Twelve(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 12]]>(data).to_vec(),
                    )),
                    13 => Self::Thirteen(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 13]]>(data).to_vec(),
                    )),
                    14 => Self::Fourteen(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 14]]>(data).to_vec(),
                    )),
                    15 => Self::Fifteen(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 15]]>(data).to_vec(),
                    )),
                    16 => Self::Sixteen(kd_tree::KdTree::build_by_ordered_float(
                        std::mem::transmute::<_, &[[T; 16]]>(data).to_vec(),
                    )),
                    _ => panic!("kd_tree::KdTree does not support more than sixteen dimensions."),
                }
            }
        }
    }

    #[allow(clippy::missing_transmute_annotations)]
    impl<T, const N: usize> crate::UnifiedKdTreeTestApi<T, N> for KdTree<T>
    where
        T: num_traits::Float + num_traits::float::FloatCore + num_traits::NumAssign,
    {
        fn query_within(&self, p: &[T; N], eps: T, _: &[[T; N]]) -> Vec<[T; N]> {
            let mut points = unsafe {
                match self {
                    Self::One(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 1]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Two(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 2]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Three(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 3]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Four(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 4]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Five(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 5]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Six(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 6]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Seven(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 7]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Eight(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 8]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Nine(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 9]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Ten(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 10]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Eleven(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 11]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Twelve(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 12]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Thirteen(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 13]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Fourteen(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 14]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Fifteen(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 15]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                    Self::Sixteen(kdtree) => {
                        let result =
                            kdtree.within_radius(std::mem::transmute::<_, &[T; 16]>(p), eps);
                        let points: Vec<_> = result
                            .into_iter()
                            .map(|v| *std::mem::transmute::<_, &[T; N]>(v))
                            .collect();
                        points
                    }
                }
            };
            crate::sort_query_result(p, &mut points);
            points
        }

        fn count_within(&self, p: &[T; N], eps: T) -> usize {
            self.query_within(p, eps, &[]).len()
        }
    }
}
