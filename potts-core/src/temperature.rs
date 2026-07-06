use crate::graph::Graph;
use crate::rng::{edge_seed, splitmix64};
use crate::sweep::{activate_bonds, assign_spins, sweep};
use crate::union_find::UnionFind;

pub struct RunConfig {
    pub temperatures: Vec<f64>,
    pub n_warmup: usize,
    pub n_sweeps: usize,
    pub n_states: u32,
    pub base_seed: u64,
}

pub struct TempStats {
    pub temp: f64,
    pub susceptibility: f64,
    /// Fraction of measurement sweeps in which each edge's endpoints were
    /// in the same SW cluster. Indexed parallel to `graph.edges`.
    pub edge_correlations: Vec<f64>,
}

pub fn run(graph: &Graph, config: &RunConfig) -> Vec<TempStats> {
    debug_assert!(config.n_sweeps > 0, "n_sweeps must be > 0");

    let n = graph.n_nodes;
    let n_edges = graph.edges.len();
    let sweeps_per_temp = (config.n_warmup + config.n_sweeps) as u64;

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

            // fresh random spins for each temperature
            let init_seed =
                splitmix64(config.base_seed ^ (t_idx as u64).wrapping_mul(0x9e3779b97f4a7c15));
            let mut spins: Vec<u32> = (0..n)
                .map(|i| {
                    (splitmix64(edge_seed(init_seed, 0, i as u64)) % config.n_states as u64) as u32
                })
                .collect();

            let base_sweep = (t_idx as u64) * sweeps_per_temp;

            for s in 0..config.n_warmup {
                sweep(
                    graph,
                    &bond_probs,
                    &mut spins,
                    config.n_states,
                    config.base_seed,
                    base_sweep + s as u64,
                );
            }

            let mut edge_co_count = vec![0u64; n_edges];
            let mut sum_sq_sizes = 0.0_f64;

            for s in 0..config.n_sweeps {
                let sweep_seed = base_sweep + config.n_warmup as u64 + s as u64;

                let mut uf =
                    activate_bonds(graph, &bond_probs, &spins, config.base_seed, sweep_seed);

                for (e, &(i, j, _)) in graph.edges.iter().enumerate() {
                    if uf.same(i, j) {
                        edge_co_count[e] += 1;
                    }
                }

                let mut cluster_sizes = vec![0u32; n];
                for i in 0..n {
                    cluster_sizes[uf.find(i)] += 1;
                }
                sum_sq_sizes +=
                    cluster_sizes.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>();

                assign_spins(&mut uf, &mut spins, config.n_states, config.base_seed, sweep_seed);
            }

            let n_sweeps_f = config.n_sweeps as f64;
            let edge_correlations = edge_co_count
                .iter()
                .map(|&c| c as f64 / n_sweeps_f)
                .collect();

            // NOTE: susceptibility estimator χ = (1/NT) * <Σ_k n_k²>.
            // The paper (Blatt et al. 1997) identifies the superparamagnetic phase via a peak
            // in χ(T) but does not prescribe this estimator — this follows standard Potts MC
            // practice. The peak location is correct; the absolute scale is not calibrated.
            let susceptibility = sum_sq_sizes / (n_sweeps_f * n as f64 * temp);

            TempStats {
                temp,
                susceptibility,
                edge_correlations,
            }
        })
        .collect()
}

/// Threshold edge correlations and return contiguous cluster labels (one per node).
/// Nodes connected only by below-threshold edges form singleton clusters.
pub fn extract_clusters(
    graph: &Graph,
    edge_correlations: &[f64],
    threshold: f64,
) -> Vec<usize> {
    let n = graph.n_nodes;
    let mut uf = UnionFind::new(n);
    for (e, &(i, j, _)) in graph.edges.iter().enumerate() {
        if edge_correlations[e] >= threshold {
            uf.union(i, j);
        }
    }
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
            n_warmup: 50,
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
            n_warmup: 10,
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
        let labels = extract_clusters(&graph, &corr, 0.5);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
    }

    /// extract_clusters: all edges below threshold → three singletons.
    #[test]
    fn extract_all_below_threshold() {
        let graph = triangle_graph();
        let corr = vec![0.0, 0.0, 0.0];
        let labels = extract_clusters(&graph, &corr, 0.5);
        assert_ne!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[1], labels[2]);
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
        let labels = extract_clusters(&graph, &corr, 0.5);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }
}
