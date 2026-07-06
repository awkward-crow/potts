/// splitmix64 finalizer applied once to a counter value.
/// Used counter-based: derive a seed from (base, sweep, edge), call once, get a u64.
pub fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e3779b97f4a7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Derive a seed for a specific (sweep, edge) pair.
pub fn edge_seed(base: u64, sweep: u64, edge: u64) -> u64 {
    base ^ sweep.wrapping_mul(0x9e3779b97f4a7c15) ^ edge.wrapping_mul(0x6c62272e07bb0142)
}

/// Map a u64 to a uniform float in [0, 1).
pub fn to_f64(x: u64) -> f64 {
    // use the upper 53 bits for the mantissa
    (x >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn different_edges_differ() {
        let base = 0xdeadbeefcafebabe;
        let s0 = splitmix64(edge_seed(base, 0, 0));
        let s1 = splitmix64(edge_seed(base, 0, 1));
        let s2 = splitmix64(edge_seed(base, 1, 0));
        assert_ne!(s0, s1);
        assert_ne!(s0, s2);
        assert_ne!(s1, s2);
    }

    #[test]
    fn same_inputs_same_output() {
        let base = 0x0123456789abcdef;
        assert_eq!(
            splitmix64(edge_seed(base, 42, 7)),
            splitmix64(edge_seed(base, 42, 7)),
        );
    }

    #[test]
    fn different_base_differs() {
        assert_ne!(
            splitmix64(edge_seed(1, 0, 0)),
            splitmix64(edge_seed(2, 0, 0)),
        );
    }

    #[test]
    fn to_f64_in_range() {
        let cases = [0u64, 1, u64::MAX, 0x9e3779b97f4a7c15, 0xffffffff00000000];
        for x in cases {
            let f = to_f64(x);
            assert!(f >= 0.0 && f < 1.0, "out of range: {f}");
        }
    }

    #[test]
    fn to_f64_not_all_same() {
        let vals: Vec<f64> = (0u64..8).map(|i| to_f64(splitmix64(i))).collect();
        let first = vals[0];
        assert!(vals.iter().any(|&v| v != first));
    }
}
