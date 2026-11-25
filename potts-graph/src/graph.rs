use potts_core::graph::Graph;

/// Mutual k-nearest-neighbour graph with the Gaussian coupling of
/// Blatt, Wiseman & Domany 1997, §4.1.2–4.1.3.
///
/// Points i and j are neighbours iff i is among the k nearest neighbours of j
/// *and* j is among the k nearest neighbours of i (the "mutual neighbourhood
/// value" of §4.1.2). The graph is not guaranteed to be connected; the paper
/// chooses k large enough that it is.
///
/// Edge weights follow eq. (9):
///
///   J_ij = (1/K̂) exp(−d_ij² / 2a²)
///
/// where `a` is the mean of d_ij over all neighbouring pairs and
/// K̂ = 2|E|/N is the mean number of neighbours.
pub fn mutual_knn_graph<P: AsRef<[f64]>>(points: &[P], k: usize) -> Graph {
    let n = points.len();
    assert!(n >= 2, "need at least 2 points");
    let k = k.min(n - 1);

    let dist = |i: usize, j: usize| euclidean(points[i].as_ref(), points[j].as_ref());

    // for each point, the indices of its k nearest neighbours
    let knn: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut nbrs: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (dist(i, j), j))
                .collect();
            nbrs.sort_by(|a, b| a.0.total_cmp(&b.0));
            nbrs.truncate(k);
            nbrs.into_iter().map(|(_, j)| j).collect()
        })
        .collect();

    // keep (i, j) only if each is in the other's k-NN list
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for &j in &knn[i] {
            if i < j && knn[j].contains(&i) {
                pairs.push((i, j, dist(i, j)));
            }
        }
    }
    pairs.sort_by_key(|&(i, j, _)| (i, j));

    Graph::new(n, interaction_strength(n, pairs))
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

fn interaction_strength(n: usize, pairs: Vec<(usize, usize, f64)>) -> Vec<(usize, usize, f64)> {
    let n_edges = pairs.len();
    assert!(n_edges > 0, "graph has no edges");
    let a = pairs.iter().map(|&(_, _, d)| d).sum::<f64>() / n_edges as f64;
    assert!(a > 0.0, "all neighbour distances are zero");
    let k_hat = 2.0 * n_edges as f64 / n as f64;

    pairs
        .into_iter()
        .map(|(i, j, d)| (i, j, (-d * d / (2.0 * a * a)).exp() / k_hat))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // four collinear points at 0, 1, 3, 4
    fn line_points() -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0], vec![3.0], vec![4.0]]
    }

    fn keys(g: &Graph) -> Vec<(usize, usize)> {
        g.edges.iter().map(|&(i, j, _)| (i, j)).collect()
    }

    /// k=1: 0↔1 and 2↔3 are mutual nearest neighbours → two components.
    /// |E| = 2, K̂ = 2·2/4 = 1, a = 1, so every J = exp(−1/2).
    #[test]
    fn mutual_k1_two_components() {
        let g = mutual_knn_graph(&line_points(), 1);
        assert_eq!(g.n_nodes, 4);
        assert_eq!(keys(&g), vec![(0, 1), (2, 3)]);
        for &(_, _, w) in &g.edges {
            assert!((w - (-0.5_f64).exp()).abs() < 1e-12);
        }
    }

    /// k=2 lists: 0→{1,2}, 1→{0,2}, 2→{3,1}, 3→{2,1}.
    /// One-sided k-NN would also give (0,2) and (1,3); mutual drops them.
    /// Edges (0,1) d=1, (1,2) d=2, (2,3) d=1 → K̂ = 6/4 = 1.5, a = 4/3.
    #[test]
    fn mutual_k2_drops_one_sided_pairs_and_matches_eq9() {
        let g = mutual_knn_graph(&line_points(), 2);
        assert_eq!(keys(&g), vec![(0, 1), (1, 2), (2, 3)]);
        let a = 4.0 / 3.0_f64;
        let k_hat = 1.5;
        let j = |d: f64| (-d * d / (2.0 * a * a)).exp() / k_hat;
        let expected = [j(1.0), j(2.0), j(1.0)];
        for (&(_, _, w), e) in g.edges.iter().zip(expected) {
            assert!((w - e).abs() < 1e-12, "got {w}, expected {e}");
        }
    }

    #[test]
    fn k_clamped_to_n_minus_1() {
        let pts = vec![vec![0.0], vec![1.0], vec![2.0]];
        let g = mutual_knn_graph(&pts, 100);
        assert_eq!(g.n_nodes, 3);
        assert_eq!(g.edges.len(), 3);
    }
}
