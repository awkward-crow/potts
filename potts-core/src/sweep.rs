use crate::graph::Graph;
use crate::rng::{edge_seed, splitmix64, to_f64};
use crate::union_find::UnionFind;

pub fn sweep(
    graph: &Graph,
    bond_probs: &[f64],
    spins: &mut [u32],
    n_states: u32,
    base_seed: u64,
    sweep_seed: u64,
) {
    debug_assert_eq!(bond_probs.len(), graph.edges.len());
    debug_assert_eq!(spins.len(), graph.n_nodes);

    let n = graph.n_nodes;
    let mut uf = UnionFind::new(n);

    for (e, &(i, j, _)) in graph.edges.iter().enumerate() {
        if spins[i] == spins[j] {
            let u = to_f64(splitmix64(edge_seed(base_seed, sweep_seed << 1, e as u64)));
            if u < bond_probs[e] {
                uf.union(i, j);
            }
        }
    }

    // spin assignment seeds use the odd lane to avoid colliding with bond seeds
    let spin_sweep = (sweep_seed << 1) | 1;
    let mut component_spin = vec![0u32; n];
    for i in 0..n {
        if uf.find(i) == i {
            let raw = splitmix64(edge_seed(base_seed, spin_sweep, i as u64));
            component_spin[i] = (raw % n_states as u64) as u32;
        }
    }
    for i in 0..n {
        spins[i] = component_spin[uf.find(i)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bond_probs(graph: &Graph, temp: f64) -> Vec<f64> {
        graph
            .edges
            .iter()
            .map(|&(_, _, w)| 1.0 - (-w / temp).exp())
            .collect()
    }

    /// Two disconnected triangles. At T→0 each triangle collapses to one spin.
    #[test]
    fn two_cliques_coalesce() {
        // nodes 0-2: clique A, nodes 3-5: clique B, no cross edges
        let edges = vec![
            (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0),
            (3, 4, 1.0), (3, 5, 1.0), (4, 5, 1.0),
        ];
        let graph = Graph::new(6, edges);
        let bp = bond_probs(&graph, 0.01); // prob ≈ 1

        let mut spins = vec![0u32; 6]; // all same spin so bonds can activate
        sweep(&graph, &bp, &mut spins, 20, 12345, 0);

        assert_eq!(spins[0], spins[1]);
        assert_eq!(spins[0], spins[2]);
        assert_eq!(spins[3], spins[4]);
        assert_eq!(spins[3], spins[5]);
    }

    /// At T→∞ bond probs ≈ 0, so no bonds activate and each node gets an
    /// independent new spin. With n_states=2 and 100 nodes, it's astronomically
    /// unlikely all end up the same.
    #[test]
    fn high_temp_decorrelates() {
        let n = 100;
        let edges: Vec<_> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        let graph = Graph::new(n, edges);
        let bp = bond_probs(&graph, 1e9); // prob ≈ 0

        let mut spins = vec![0u32; n];
        sweep(&graph, &bp, &mut spins, 2, 99999, 0);

        let all_same = spins.windows(2).all(|w| w[0] == w[1]);
        assert!(!all_same);
    }

    /// Same inputs must always produce identical outputs.
    #[test]
    fn reproducible() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let graph = Graph::new(4, edges);
        let bp = bond_probs(&graph, 1.0);

        let mut spins_a = vec![0u32, 1, 0, 1];
        let mut spins_b = spins_a.clone();

        sweep(&graph, &bp, &mut spins_a, 20, 0xabad1dea, 7);
        sweep(&graph, &bp, &mut spins_b, 20, 0xabad1dea, 7);

        assert_eq!(spins_a, spins_b);
    }
}
