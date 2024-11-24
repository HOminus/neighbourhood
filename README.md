# Neighbourhood
Fast fixed size index based Kd-Tree.

## KdIndexTree
Takes a shared reference to a point-cloud and provides a KdTreeApi.
```rust
let point_cloud: Vec<[f32; 3]> = ...;
let kd_tree = KdIndexTree::new(&point_cloud);

// Find all points in point_cloud with an euclidean distance
// less or equal than two from [1.0, -2.0, 3.0]
for point_index in kd_tree.neighbourhood_by_index(&[1.0, -2.0, 3.0], 2.0) {
  println!("Found point {}.", point_cloud[point_index]);
}
```
