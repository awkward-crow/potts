# potts

Rust implementation of Superparamagnetic Clustering (SPC), based on:

> Blatt, M., Wiseman, S., & Domany, E. (1997). *Data Clustering Using a Model of a Granular Magnet.* Neural Computation 9(8), 1805–1842. Preprint: [cond-mat/9702072](https://arxiv.org/abs/cond-mat/9702072)

## crates

- `potts-core` — Potts model dynamics: Swendsen-Wang Monte Carlo, temperature sweep, susceptibility tracking, and cluster extraction. Input: a weighted graph.
- `potts-graph` — builds that graph from Euclidean point data: mutual k-nearest neighbours with the Gaussian coupling of Blatt et al. 1997, eq. (9).

## example

```sh
cargo run --example buckyball
```

Runs a temperature sweep on the 60 vertices of a C60 buckyball (unit sphere). Output is tab-separated (temp, susceptibility, n_clusters) and can be redirected to a file for plotting. The vertices form a single cluster at low temperature; the susceptibility peaks sharply at T ≈ 0.09, close to the estimate from eq. 10 of Blatt et al. (1997), and above it the cluster breaks up.

## license

MIT

## acknowledgement

Developed with the assistance of [Claude Code](https://claude.com/claude-code), Anthropic's AI coding tool.
