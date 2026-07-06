# potts

Rust implementation of Superparamagnetic Clustering (SPC), based on:

> Blatt, M., Wiseman, S., & Domany, E. (1997). *Data Clustering Using a Model of a Granular Magnet.* Neural Computation 9(8), 1805–1842. Preprint: [cond-mat/9702072](https://arxiv.org/abs/cond-mat/9702072)

## crates

- `potts-core` — Potts model dynamics: Swendsen-Wang Monte Carlo, temperature sweep, susceptibility tracking, and cluster extraction. Input: a weighted graph.
- `potts-graph` — builds that graph from data via a dissimilarity closure; supports k-NN and k-NN + minimum spanning tree topologies with Gaussian coupling weights.

## example

```sh
cargo run --example buckyball
```

Runs a temperature sweep on the 60 vertices of a C60 buckyball (unit sphere). Output is tab-separated (temp, susceptibility, n_clusters) and can be redirected to a file for plotting. The transition from 60 singleton clusters to a single cluster is sharp and occurs around T ≈ 0.41.

## license

MIT
