use potts_core::graph::Graph;
use potts_core::temperature::{extract_clusters, run, RunConfig};
use potts_graph::interaction_strength;
use std::collections::HashSet;

const IRIS: &str = include_str!("data/iris.tsv");
const TIE_TOLERANCE: f64 = 1e-9;

fn main() {
    let (points, species) = read_iris();
    let n = points.len();
    let graph = Graph::new(n, interaction_strength(n, mutual_knn_with_ties(&points, 5)));
    let n_states = 20;

    let n_edges = graph.edges.len();
    let k_hat = 2.0 * n_edges as f64 / n as f64;
    let mean_scaled_d2 =
        graph.edges.iter().map(|&(_, _, w)| -(k_hat * w).ln()).sum::<f64>() / n_edges as f64;
    let transition_estimate =
        (-mean_scaled_d2).exp() / (4.0 * (1.0 + (n_states as f64).sqrt()).ln());
    let components = extract_clusters(&graph, &vec![1.0; n_edges], n_states)
        .into_iter()
        .collect::<HashSet<_>>()
        .len();
    eprintln!(
        "iris: {n} points, {n_edges} edges, K̂ = {k_hat:.2}, {components} components, eq. 10 T ≈ {transition_estimate:.4}"
    );

    let config = RunConfig {
        temperatures: (1..=60).map(|i| 0.0025 * i as f64).collect(),
        n_sweeps: 1000,
        n_states,
        base_seed: 1997,
    };
    let stats = run(&graph, &config);

    println!("temp\tsusceptibility\tpoint\tspecies\tlabel");
    for s in &stats {
        let labels = extract_clusters(&graph, &s.edge_correlations, n_states);
        for (i, label) in labels.iter().enumerate() {
            println!("{:.4}\t{:.6}\t{}\t{}\t{}", s.temp, s.susceptibility, i + 1, species[i], label);
        }
    }
}

/// Mutual K-nearest neighbours where every point within TIE_TOLERANCE of the
/// K-th nearest distance also counts as a K-th nearest neighbour.
fn mutual_knn_with_ties(points: &[[f64; 4]], k: usize) -> Vec<(usize, usize, f64)> {
    let n = points.len();
    let dist = |i: usize, j: usize| {
        (0..4).map(|c| (points[i][c] - points[j][c]).powi(2)).sum::<f64>().sqrt()
    };
    let neighbours: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut sorted: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| dist(i, j)).collect();
            sorted.sort_by(f64::total_cmp);
            let kth = sorted[k - 1];
            (0..n).filter(|&j| j != i && dist(i, j) <= kth + TIE_TOLERANCE).collect()
        })
        .collect();

    let mut pairs = Vec::new();
    for i in 0..n {
        for &j in &neighbours[i] {
            if i < j && neighbours[j].contains(&i) {
                pairs.push((i, j, dist(i, j)));
            }
        }
    }
    pairs
}

fn read_iris() -> (Vec<[f64; 4]>, Vec<&'static str>) {
    IRIS.lines()
        .skip(1)
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            ([0, 1, 2, 3].map(|c| fields[c].parse().unwrap()), fields[4])
        })
        .unzip()
}
