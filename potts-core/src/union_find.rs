pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        // path halving: same amortized complexity as full compression, no recursion
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }

    pub fn same(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_singleton() {
        let mut uf = UnionFind::new(5);
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
        }
    }

    #[test]
    fn union_connects_pair() {
        let mut uf = UnionFind::new(4);
        uf.union(1, 2);
        assert!(uf.same(1, 2));
        assert!(!uf.same(0, 1));
        assert!(!uf.same(2, 3));
    }

    #[test]
    fn union_is_transitive() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);
        assert!(uf.same(0, 2));
        assert!(uf.same(3, 4));
        assert!(!uf.same(0, 3));
    }

    #[test]
    fn chain_merges_to_one_component() {
        let n = 8;
        let mut uf = UnionFind::new(n);
        for i in 0..n - 1 {
            uf.union(i, i + 1);
        }
        let root = uf.find(0);
        for i in 1..n {
            assert_eq!(uf.find(i), root);
        }
    }

    #[test]
    fn union_self_is_noop() {
        let mut uf = UnionFind::new(3);
        uf.union(1, 1);
        assert!(!uf.same(0, 1));
        assert!(!uf.same(1, 2));
    }
}
