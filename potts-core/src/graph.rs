/// Weighted undirected graph passed into the Potts model.
///
/// Edges are stored as `(i, j, weight)` triples with `i < j` by convention
/// (not enforced, but callers should normalise). Weights are the coupling
/// strengths J_ij used directly in the bond activation probability
/// `1 - exp(-J_ij / T)`.
#[derive(Debug, Clone)]
pub struct Graph {
    pub n_nodes: usize,
    pub edges: Vec<(usize, usize, f64)>,
}

impl Graph {
    pub fn new(n_nodes: usize, edges: Vec<(usize, usize, f64)>) -> Self {
        debug_assert!(
            edges.iter().all(|&(i, j, _)| i < n_nodes && j < n_nodes),
            "edge index out of range"
        );
        Graph { n_nodes, edges }
    }

    pub fn max_weight(&self) -> f64 {
        self.edges.iter().map(|&(_, _, w)| w).fold(0.0_f64, f64::max)
    }

    pub fn min_weight(&self) -> f64 {
        self.edges
            .iter()
            .map(|&(_, _, w)| w)
            .fold(f64::INFINITY, f64::min)
    }
}
