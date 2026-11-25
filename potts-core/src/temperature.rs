use crate::graph::Graph;
use crate::sweep::{activate_bonds, assign_spins};
use crate::union_find::UnionFind;

pub struct RunConfig {
    pub temperatures: Vec<f64>,
    pub n_sweeps: usize,
    pub n_states: u32,
    pub base_seed: u64,
}

pub struct TempStats {
    pub temp: f64,
    /// χ = (N/T)(⟨m²⟩ − ⟨m⟩²), eq. 6 of Blatt, Wiseman & Domany (1997), with m from eq. 4.
    pub susceptibility: f64,
    /// Fraction of measurement sweeps in which each edge's endpoints were
    /// in the same SW cluster. Indexed parallel to `graph.edges`.
    pub edge_correlations: Vec<f64>,
}

/// Heats the system through `config.temperatures`, which must increase. Every
/// spin starts in the same state (§3) and the spins carry over from one
/// temperature to the next.
pub fn run(graph: &Graph, config: &RunConfig) -> Vec<TempStats> {
    debug_assert!(config.n_sweeps > 0, "n_sweeps must be > 0");
    assert!(
        config.temperatures.windows(2).all(|w| w[0] < w[1]),
        "temperatures must increase"
    );

    let n = graph.n_nodes;
    let n_edges = graph.edges.len();
    let mut spins = vec![0u32; n];

    config
        .temperatures
        .iter()
        .enumerate()
        .map(|(t_idx, &temp)| {
            let bond_probs: Vec<f64> = graph
                .edges
                .iter()
                .map(|&(_, _, w)| 1.0 - (-w / temp).exp())
                .collect();

            let base_sweep = (t_idx * config.n_sweeps) as u64;

            let mut edge_co_count = vec![0u64; n_edges];
            let mut sum_m = 0.0_f64;
            let mut sum_m_sq = 0.0_f64;

            for s in 0..config.n_sweeps {
                let sweep_seed = base_sweep + s as u64;

                let mut uf =
                    activate_bonds(graph, &bond_probs, &spins, config.base_seed, sweep_seed);

                for (e, &(i, j, _)) in graph.edges.iter().enumerate() {
                    if uf.same(i, j) {
                        edge_co_count[e] += 1;
                    }
                }

                assign_spins(&mut uf, &mut spins, config.n_states, config.base_seed, sweep_seed);

                let m = magnetization(&spins, config.n_states);
                sum_m += m;
                sum_m_sq += m * m;
            }

            let n_sweeps_f = config.n_sweeps as f64;
            let edge_correlations = edge_co_count
                .iter()
                .map(|&c| c as f64 / n_sweeps_f)
                .collect();

            let mean_m = sum_m / n_sweeps_f;
            let mean_m_sq = sum_m_sq / n_sweeps_f;
            let susceptibility = (n as f64 / temp) * (mean_m_sq - mean_m * mean_m);

            TempStats {
                temp,
                susceptibility,
                edge_correlations,
            }
        })
        .collect()
}

/// Eq. 4 of Blatt, Wiseman & Domany (1997).
fn magnetization(spins: &[u32], n_states: u32) -> f64 {
    let mut counts = vec![0usize; n_states as usize];
    for &s in spins {
        counts[s as usize] += 1;
    }
    let n_max = *counts.iter().max().unwrap() as f64;
    let n = spins.len() as f64;
    let q = n_states as f64;
    (q * n_max - n) / ((q - 1.0) * n)
}

/// Data clusters from the edge correlations at one temperature
/// (Blatt, Wiseman & Domany 1997, §4.3.2).
///
/// 1. Core: link i and j if their spin–spin correlation G_ij > 0.5, where
///    G_ij = ((q − 1) C_ij + 1) / q (eq. 11) and C_ij is `edge_correlations`.
/// 2. Periphery: link every point to its neighbour of maximal correlation.
/// 3. Clusters are the connected components of the links from 1 and 2.
///
/// Labels are numbered 0, 1, 2, … in order of each cluster's first point.
/// A point with no edges stays a singleton.
pub fn extract_clusters(graph: &Graph, edge_correlations: &[f64], n_states: u32) -> Vec<usize> {
    let n = graph.n_nodes;
    let q = n_states as f64;
    let mut uf = UnionFind::new(n);

    // step 1: core links
    for (e, &(i, j, _)) in graph.edges.iter().enumerate() {
        let g = ((q - 1.0) * edge_correlations[e] + 1.0) / q;
        if g > 0.5 {
            uf.union(i, j);
        }
    }

    // step 2: link each point to its neighbour of maximal correlation;
    // ties go to the lowest edge index
    let mut best: Vec<Option<(f64, usize)>> = vec![None; n];
    for (e, &(i, j, _)) in graph.edges.iter().enumerate() {
        let c = edge_correlations[e];
        for (p, other) in [(i, j), (j, i)] {
            if best[p].map_or(true, |(bc, _)| c > bc) {
                best[p] = Some((c, other));
            }
        }
    }
    for (p, b) in best.iter().enumerate() {
        if let Some((_, other)) = *b {
            uf.union(p, other);
        }
    }

    // step 3: label connected components
    let mut root_to_label = vec![usize::MAX; n];
    let mut next_label = 0usize;
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let root = uf.find(i);
        if root_to_label[root] == usize::MAX {
            root_to_label[root] = next_label;
            next_label += 1;
        }
        labels[i] = root_to_label[root];
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    fn triangle_graph() -> Graph {
        Graph::new(3, vec![(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)])
    }

    /// At very low T a triangle is almost always one SW cluster, so all
    /// edge correlations should be close to 1.
    #[test]
    fn low_temp_high_correlation() {
        let graph = triangle_graph();
        let config = RunConfig {
            temperatures: vec![0.01],
            n_sweeps: 500,
            n_states: 2, // small q → fast equilibration
            base_seed: 0xdeadbeef,
        };
        let stats = run(&graph, &config);
        for &c in &stats[0].edge_correlations {
            assert!(c > 0.9, "expected high correlation at low T, got {c}");
        }
    }

    /// At very high T no bonds activate, so SW co-clustering is ≈ 0.
    #[test]
    fn high_temp_low_correlation() {
        let graph = triangle_graph();
        let config = RunConfig {
            temperatures: vec![1e6],
            n_sweeps: 200,
            n_states: 2,
            base_seed: 0xcafebabe,
        };
        let stats = run(&graph, &config);
        for &c in &stats[0].edge_correlations {
            assert!(c < 0.05, "expected near-zero correlation at high T, got {c}");
        }
    }

    /// extract_clusters: all edges above threshold → one cluster.
    #[test]
    fn extract_all_above_threshold() {
        let graph = triangle_graph();
        let corr = vec![1.0, 1.0, 1.0];
        let labels = extract_clusters(&graph, &corr, 20);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
    }

    /// extract_clusters: no core links, but step 2 links every point to its
    /// best neighbour (ties → lowest edge index) → one cluster.
    #[test]
    fn extract_all_zero_still_linked_by_step2() {
        let graph = triangle_graph();
        let corr = vec![0.0, 0.0, 0.0];
        let labels = extract_clusters(&graph, &corr, 20);
        assert_eq!(labels, vec![0, 0, 0]);
    }

    /// extract_clusters: point 4 is below the core threshold to both cores
    /// (C=0.3 → G=0.335 with q=20) but joins core A, its best neighbour.
    /// The cores stay separate.
    #[test]
    fn extract_periphery_joins_best_core() {
        let edges = vec![(0, 1, 1.0), (2, 3, 1.0), (1, 4, 1.0), (2, 4, 1.0)];
        let graph = Graph::new(5, edges);
        let corr = vec![0.9, 0.9, 0.3, 0.1];
        let labels = extract_clusters(&graph, &corr, 20);
        assert_eq!(labels, vec![0, 0, 1, 1, 0]);
    }

    /// extract_clusters: a node with no edges stays a singleton.
    #[test]
    fn extract_isolated_node_singleton() {
        let graph = Graph::new(3, vec![(0, 1, 1.0)]);
        let labels = extract_clusters(&graph, &[0.0], 20);
        assert_eq!(labels, vec![0, 0, 1]);
    }

    /// extract_clusters: two disconnected groups with high internal correlation.
    #[test]
    fn extract_two_clusters() {
        // nodes 0-2: clique A, nodes 3-4: edge B, no cross edges
        let edges = vec![
            (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0),
            (3, 4, 1.0),
        ];
        let graph = Graph::new(5, edges);
        let corr = vec![0.9, 0.9, 0.9, 0.9];
        let labels = extract_clusters(&graph, &corr, 20);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn magnetization_ordered_and_uniform() {
        assert_eq!(magnetization(&[3, 3, 3, 3], 4), 1.0);
        assert_eq!(magnetization(&[0, 1, 2, 3], 4), 0.0);
    }

    #[test]
    #[should_panic(expected = "temperatures must increase")]
    fn temperatures_must_increase() {
        let config = RunConfig { temperatures: vec![0.2, 0.1], n_sweeps: 1, n_states: 2, base_seed: 0 };
        run(&triangle_graph(), &config);
    }
}
