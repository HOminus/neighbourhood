// Use neighbourhodd::KdTree to compute the correlation dimension
// https://en.wikipedia.org/wiki/Correlation_dimension of the
// Lorenz Attractor.

fn lorenz(dt: f64, xyz: [f64; 3]) -> [f64; 3] {
    let sigma = 10.;
    let beta = 8. / 3.;
    let rho = 28.;

    let [x, y, z] = xyz;

    let x1 = dt * (sigma * (y - x));
    let y1 = dt * (x * (rho - z) - y);
    let z1 = dt * (x * y - beta * z);
    [x + x1, y + y1, z + z1]
}

fn main() {
    const DATA_POINTS: usize = 100_000;
    let mut coordinates = Vec::with_capacity(DATA_POINTS);
    let mut point = [1., 2., 3.];
    for index in 0..(DATA_POINTS + 1000) {
        point = lorenz(0.02, point);
        if index > 1000 {
            coordinates.push(point);
        }
    }

    let kd_tree = neighbourhood::KdTree::new(coordinates);

    let mut counts = [(1.0, 0), (4.0, 0)];
    for data_point in kd_tree.data() {
        for count in counts.iter_mut() {
            count.1 += kd_tree.count_neighbourhood(data_point, count.0);
        }
    }

    let count_1 = counts[0].1 as f64 / DATA_POINTS as f64;
    let count_4 = counts[1].1 as f64 / DATA_POINTS as f64;
    let corrdim = (count_1.ln() - count_4.ln()) / (counts[0].0.ln() - counts[1].0.ln());

    println!("Correlation dimension: {corrdim}");
}
