use potts_core::temperature::{extract_clusters, run, RunConfig};
use potts_graph::knn_mst_graph;
use std::collections::HashSet;

fn main() {
    let pts = buckyball_vertices();
    let graph = knn_mst_graph(&pts, 5);

    eprintln!("C60 buckyball: {} vertices, {} edges", pts.len(), graph.edges.len());

    // log-spaced temperatures from high to low
    let n_temps = 30;
    let t_min = 0.2f64;
    let t_max = 1.0_f64;
    let temperatures: Vec<f64> = (0..n_temps)
        .map(|i| t_max * (t_min / t_max).powf(i as f64 / (n_temps - 1) as f64))
        .collect();

    // n_sweeps=1000 is enough to locate the transition but susceptibility_connected
    // may show small zig-zags just below the transition temperature due to critical
    // slowing down — the chain mixes slowly there and occasional small clusters break
    // off the ordered phase, inflating the estimator. Increase n_sweeps (try 5000+)
    // if smoother curves are needed. Changing n_temps shifts the per-temperature seeds
    // and can also affect noise levels, but that is luck rather than a fix.
    let config = RunConfig {
        temperatures,
        n_warmup: 300,
        n_sweeps: 1000,
        n_states: 50,
        base_seed: 0xC60C60C60C60C60,
    };

    let stats = run(&graph, &config);

    println!("temp\tsusceptibility\tsusceptibility_connected\tn_clusters");
    for s in &stats {
        let labels = extract_clusters(&graph, &s.edge_correlations, 0.5);
        let n_clusters = labels.iter().cloned().collect::<HashSet<_>>().len();
        println!(
            "{:.6}\t{:.6}\t{:.6}\t{}",
            s.temp, s.susceptibility, s.susceptibility_connected, n_clusters
        );
    }
}

/// Vertices of C60 (truncated icosahedron) on the unit sphere.
/// All even (cyclic) permutations of three families, normalised to unit radius.
fn buckyball_vertices() -> Vec<[f64; 3]> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(60);

    // family 1: even permutations of (0, ±1, ±3φ)  →  12 points
    for &s1 in &[1.0_f64, -1.0] {
        for &s2 in &[1.0_f64, -1.0] {
            pts.push([0.0, s1, s2 * 3.0 * phi]);
            pts.push([s1, s2 * 3.0 * phi, 0.0]);
            pts.push([s2 * 3.0 * phi, 0.0, s1]);
        }
    }

    // family 2: even permutations of (±1, ±(2+φ), ±2φ)  →  24 points
    for &s1 in &[1.0_f64, -1.0] {
        for &s2 in &[1.0_f64, -1.0] {
            for &s3 in &[1.0_f64, -1.0] {
                pts.push([s1, s2 * (2.0 + phi), s3 * 2.0 * phi]);
                pts.push([s2 * (2.0 + phi), s3 * 2.0 * phi, s1]);
                pts.push([s3 * 2.0 * phi, s1, s2 * (2.0 + phi)]);
            }
        }
    }

    // family 3: even permutations of (±2, ±(1+2φ), ±φ)  →  24 points
    for &s1 in &[1.0_f64, -1.0] {
        for &s2 in &[1.0_f64, -1.0] {
            for &s3 in &[1.0_f64, -1.0] {
                pts.push([2.0 * s1, s2 * (1.0 + 2.0 * phi), s3 * phi]);
                pts.push([s2 * (1.0 + 2.0 * phi), s3 * phi, 2.0 * s1]);
                pts.push([s3 * phi, 2.0 * s1, s2 * (1.0 + 2.0 * phi)]);
            }
        }
    }

    assert_eq!(pts.len(), 60);

    // normalise to unit sphere (all three families share radius r² = 10 + 9φ)
    for p in &mut pts {
        let r = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
        p[0] /= r;
        p[1] /= r;
        p[2] /= r;
    }

    pts
}
