use potts_core::graph::Graph;
use std::collections::BTreeMap;

/// Build a k-NN graph with Gaussian coupling weights.
///
/// `dissimilarity(i, j)` should be symmetric and non-negative; smaller means
/// more similar. Points should be distinct.
pub fn knn_graph<F: Fn(usize, usize) -> f64>(n: usize, k: usize, dissimilarity: F) -> Graph {
    build(n, k, false, dissimilarity)
}

/// k-NN graph augmented with Prim's MST edges for guaranteed connectivity.
pub fn knn_mst_graph<F: Fn(usize, usize) -> f64>(n: usize, k: usize, dissimilarity: F) -> Graph {
    build(n, k, true, dissimilarity)
}

fn build<F: Fn(usize, usize) -> f64>(
    n: usize,
    k: usize,
    augment_mst: bool,
    dissimilarity: F,
) -> Graph {
    assert!(n >= 2, "need at least 2 points");
    let k = k.min(n - 1);

    // for each point, k nearest neighbours with their dissimilarities
    let knn: Vec<Vec<(f64, usize)>> = (0..n)
        .map(|i| {
            let mut nbrs: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (dissimilarity(i, j), j))
                .collect();
            nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            nbrs.truncate(k);
            nbrs
        })
        .collect();

    // local scale: mean k-NN dissimilarity per point
    let sigma: Vec<f64> = (0..n)
        .map(|i| knn[i].iter().map(|(d, _)| d).sum::<f64>() / k as f64)
        .collect();

    // unique undirected edges from k-NN, keyed (min, max)
    let mut edge_map: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for i in 0..n {
        for &(d, j) in &knn[i] {
            let key = if i < j { (i, j) } else { (j, i) };
            edge_map.entry(key).or_insert(d);
        }
    }

    if augment_mst {
        for (i, j, d) in prim_mst(n, &dissimilarity) {
            edge_map.entry((i, j)).or_insert(d);
        }
    }

    let edges = edge_map
        .into_iter()
        .map(|((i, j), d)| (i, j, gaussian_coupling(d, sigma[i], sigma[j])))
        .collect();

    Graph::new(n, edges)
}

/// Blatt et al. 1997 eq. — local-density-normalised Gaussian coupling.
fn gaussian_coupling(d: f64, sigma_i: f64, sigma_j: f64) -> f64 {
    (-d * d / (2.0 * sigma_i * sigma_j)).exp()
}

/// O(n²) Prim's MST on the full pairwise dissimilarity graph.
/// Returns edges as (i, j, d) with i < j.
fn prim_mst<F: Fn(usize, usize) -> f64>(n: usize, dissimilarity: &F) -> Vec<(usize, usize, f64)> {
    let mut in_tree = vec![false; n];
    let mut min_dist = vec![f64::INFINITY; n];
    let mut parent = vec![0usize; n];
    min_dist[0] = 0.0;

    let mut edges = Vec::with_capacity(n - 1);

    for _ in 0..n {
        let u = (0..n)
            .filter(|&i| !in_tree[i])
            .min_by(|&a, &b| {
                min_dist[a]
                    .partial_cmp(&min_dist[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        in_tree[u] = true;

        // skip the seed node (min_dist == 0.0 by construction); it has no parent
        if min_dist[u] > 0.0 {
            let p = parent[u];
            let (i, j) = if p < u { (p, u) } else { (u, p) };
            edges.push((i, j, min_dist[u]));
        }

        for v in 0..n {
            if !in_tree[v] {
                let d = dissimilarity(u, v);
                if d < min_dist[v] {
                    min_dist[v] = d;
                    parent[v] = u;
                }
            }
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    // four collinear points at 0, 1, 3, 4
    // k=1 gives two disconnected edges: (0,1) and (2,3)
    fn line_dist(i: usize, j: usize) -> f64 {
        let pos = [0.0_f64, 1.0, 3.0, 4.0];
        (pos[i] - pos[j]).abs()
    }

    #[test]
    fn knn_graph_edges() {
        let g = knn_graph(4, 1, line_dist);
        assert_eq!(g.n_nodes, 4);
        assert_eq!(g.edges.len(), 2);
        let keys: Vec<(usize, usize)> = g.edges.iter().map(|&(i, j, _)| (i, j)).collect();
        assert!(keys.contains(&(0, 1)));
        assert!(keys.contains(&(2, 3)));
    }

    #[test]
    fn knn_mst_adds_bridge() {
        let g = knn_mst_graph(4, 1, line_dist);
        assert_eq!(g.n_nodes, 4);
        assert_eq!(g.edges.len(), 3);
        let keys: Vec<(usize, usize)> = g.edges.iter().map(|&(i, j, _)| (i, j)).collect();
        assert!(keys.contains(&(0, 1)));
        assert!(keys.contains(&(1, 2)));
        assert!(keys.contains(&(2, 3)));
    }

    #[test]
    fn weights_in_range() {
        let g = knn_mst_graph(4, 1, line_dist);
        for &(_, _, w) in &g.edges {
            assert!(w > 0.0 && w <= 1.0, "weight out of range: {w}");
        }
    }

    #[test]
    fn knn_already_connected_mst_adds_nothing() {
        let g_knn = knn_graph(2, 1, |_, _| 1.0);
        let g_mst = knn_mst_graph(2, 1, |_, _| 1.0);
        assert_eq!(g_knn.edges.len(), g_mst.edges.len());
    }

    #[test]
    fn k_clamped_to_n_minus_1() {
        // k=100 on 3 points should not panic; yields complete graph (3 edges)
        let g = knn_graph(3, 100, |_, _| 1.0);
        assert_eq!(g.n_nodes, 3);
        assert_eq!(g.edges.len(), 3);
    }
}
