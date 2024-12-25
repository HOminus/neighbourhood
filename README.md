# Neighbourhood
Super fast fixed size K-d Trees for extremely large datasets.

## KdTree
Data structure for super fast neighbourhood queries.
```rust,ignore
let point_cloud: Vec<[f32; 3]> = ...;
let kd_tree = KdIndexTree::new(&point_cloud);

// Find all points in point_cloud with an euclidean distance <= 2 from [1, -2, 3]
for point_index in kd_tree.neighbourhood_by_index(&[1.0, -2.0, 3.0], 2.0) {
  println!("Found point {:?}.", point_cloud[point_index]);
}
```

## KdIndexTree
Takes a shared reference to a point-cloud and provides a K-d Tree Api.
```rust,ignore
let point_cloud: Vec<[f32; 3]> = ...;
let kd_tree = KdIndexTree::new(&point_cloud);

// Find all points in point_cloud with an euclidean distance <= 2 from [1, -2, 3]
for point_index in kd_tree.neighbourhood_by_index(&[1.0, -2.0, 3.0], 2.0) {
  println!("Found point {:?}.", point_cloud[point_index]);
}
```

## Benchmarks
On large datasets neighbourhoods K-d tree typically outperforms other implementations.

### Build K-d Tree
Time measurments in seconds.
```text
Points      | neighbourhood |  kd-tree   |      kiddo      |
            |    KdTree     |            | ImmutableKdTree |
------------------------------------------------------------
    100'000 |   0.01041     |  0.01015   |    0.00881      |
------------------------------------------------------------
  1'000'000 |   0.12776     |  0.12511   |    0.27243      |
------------------------------------------------------------
 10'000'000 |   1.52024     |  1.49897   |    4.53685      |
------------------------------------------------------------
100'000'000 |  18.01505     | 17.77733   |   63.02797      |
------------------------------------------------------------
```

### Query all points within a range
100'000'000 points in a cube with edge length 20. Unit: microseconds per lookup.
```text
epsilon | neighbourhood |  kd-tree  |      kiddo      |
        |     KdTree    |           | ImmutableKdTree |
-------------------------------------------------------
  0.02  |     1.296     |   1.375   |      1.561      |
-------------------------------------------------------
  0.05  |     2.159     |   3.138   |      2.705      |
-------------------------------------------------------
  0.1   |     4.743     |   9.060   |      5.631      |
-------------------------------------------------------
  0.2   |    15.022     |  34.329   |     17.450      |
-------------------------------------------------------

```

## Performance
For optimal performance it is crucial to have a good `brute_force_size` parameter. The `brute_force_size` can always be changed, even multiple times after construction of the KdTree. By default the value is chosen, s.t. 3 dimensional points will perform very well. But benchmarks showed that even with a non-optimal `brute_force_size`, Neighbourhoods K-d Trees do perform very well. An optimal `brute_force_size` value depends on the query parameters. For maximum performance case by case benchmarking is strongly recommended.

## Correctness
Neighbourhoods K-d Trees are validated by running extensive tests against other K-d tree implementations. At this point in time, no bugs are known.

## Contributing
Contributions are welcome. All contributions submitted to this project are assumed to be dual-licensed as "MIT OR Apache-2.0" unless explicitly stated otherwise.
