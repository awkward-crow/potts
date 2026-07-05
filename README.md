# potts

Rust implementation of Superparamagnetic Clustering (SPC), based on:

> Blatt, M., Wiseman, S., & Domany, E. (1997). *Data Clustering Using a Model of a Granular Magnet.* Neural Computation 9(8), 1805–1842. Preprint: [cond-mat/9702072](https://arxiv.org/abs/cond-mat/9702072)

## Crates

- `potts-core` — Potts model dynamics: Swendsen-Wang Monte Carlo, temperature sweep, spin-spin correlation accumulation. Input: a weighted graph.
- `potts-graph` — builds that graph from data via a `Clusterable` trait; supports k-NN and k-NN + minimum spanning tree topologies.

## Status

Early development.

## License

MIT
