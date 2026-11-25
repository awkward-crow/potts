// Two-Gaussian separation experiment from Blatt et al. 1997.
//
// Generates two isotropic Gaussian clusters in 2D and runs a temperature sweep.
// Output has two tab-separated sections divided by a blank line:
//   1. temperature sweep (temp, susceptibility, susceptibility_connected, n_clusters)
//   2. point assignments at the peak chi_c temperature (x, y, true_cluster, pred_cluster)
//
// Usage:
//   cargo run --example two_gaussians > out.tsv
//
// In R:
//   lines  <- readLines("out.tsv")
//   blank  <- which(lines == "")
//   sweep  <- read.table(text = paste(lines[1:(blank-1)], collapse="\n"), header=TRUE, sep="\t")
//   pts    <- read.table(text = paste(lines[(blank+1):length(lines)], collapse="\n"), header=TRUE, sep="\t")

use potts_core::rng::{splitmix64, to_f64};
use potts_core::temperature::{extract_clusters, run, RunConfig};
use potts_graph::knn_mst_graph;
use std::collections::HashSet;
use std::f64::consts::PI;

const N_PER_CLUSTER: usize = 100;
const SIGMA: f64 = 1.0;
const SEPARATION: f64 = 3.0;
const K: usize = 10;

fn main() {
    let pts = generate_points();
    let n = pts.len();
    let graph = knn_mst_graph(&pts, K);

    eprintln!(
        "two Gaussians: {} points, {} edges, separation={SEPARATION}, sigma={SIGMA}",
        n,
        graph.edges.len(),
    );

    let n_temps = 30;
    let t_min = 0.1_f64;
    let t_max = 2.0_f64;
    let temperatures: Vec<f64> = (0..n_temps)
        .map(|i| t_max * (t_min / t_max).powf(i as f64 / (n_temps - 1) as f64))
        .collect();

    let config = RunConfig {
        temperatures,
        n_warmup: 300,
        n_sweeps: 2000,
        n_states: 20,
        base_seed: 0xDEADBEEFCAFE,
    };

    let stats = run(&graph, &config);

    // section 1: temperature sweep
    println!("temp\tsusceptibility\tsusceptibility_connected\tn_clusters");
    let mut peak_idx = 0;
    let mut peak_chi_c = 0.0_f64;
    for (i, s) in stats.iter().enumerate() {
        let labels = extract_clusters(&graph, &s.edge_correlations, 0.5);
        let n_clusters = labels.iter().cloned().collect::<HashSet<_>>().len();
        println!(
            "{:.6}\t{:.6}\t{:.6}\t{}",
            s.temp, s.susceptibility, s.susceptibility_connected, n_clusters
        );
        if s.susceptibility_connected > peak_chi_c {
            peak_chi_c = s.susceptibility_connected;
            peak_idx = i;
        }
    }

    // section 2: cluster assignments at the peak chi_c temperature
    let peak_stats = &stats[peak_idx];
    let labels = extract_clusters(&graph, &peak_stats.edge_correlations, 0.5);

    eprintln!("peak chi_c at T={:.4}", peak_stats.temp);

    println!();
    println!("x\ty\ttrue_cluster\tpred_cluster");
    for i in 0..n {
        let true_label = if i < N_PER_CLUSTER { 0 } else { 1 };
        println!(
            "{:.6}\t{:.6}\t{}\t{}",
            pts[i][0], pts[i][1], true_label, labels[i]
        );
    }
}

fn generate_points() -> Vec<[f64; 2]> {
    let mut seed = 0xDEADBEEFCAFE_u64;
    let mut pts = Vec::with_capacity(2 * N_PER_CLUSTER);

    for _ in 0..N_PER_CLUSTER {
        pts.push([
            standard_normal(&mut seed) * SIGMA - SEPARATION / 2.0,
            standard_normal(&mut seed) * SIGMA,
        ]);
    }
    for _ in 0..N_PER_CLUSTER {
        pts.push([
            standard_normal(&mut seed) * SIGMA + SEPARATION / 2.0,
            standard_normal(&mut seed) * SIGMA,
        ]);
    }

    pts
}

fn standard_normal(seed: &mut u64) -> f64 {
    *seed = splitmix64(*seed);
    let u1 = to_f64(*seed).max(1e-10); // guard against ln(0)
    *seed = splitmix64(*seed);
    let u2 = to_f64(*seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
