use super::BasisSpace;
use bitbasis::BitInt;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FullSpace
// ---------------------------------------------------------------------------

/// The full Hilbert space of `dim` basis states.
///
/// States are the integers `0 .. dim` stored in **descending** order
/// (matching the C++ `space::state_at` / `index` convention):
/// `state_at(i) = dim - i - 1` and `index(s) = dim - s - 1`.
///
/// No storage is needed beyond `dim`; `state_at` and `index` are O(1).
///
/// Mirrors `space<bitset_t>` from `space.hpp`, restricted to `u32`/`u64`
/// (multi-word integers are never practical for a full space).
#[derive(Clone, Debug)]
pub struct FullSpace<B: BitInt> {
    dim: usize,
    _marker: std::marker::PhantomData<B>,
}

impl<B: BitInt> FullSpace<B> {
    pub fn new(dim: usize) -> Self {
        FullSpace {
            dim,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: BitInt> BasisSpace<B> for FullSpace<B> {
    #[inline]
    fn size(&self) -> usize {
        self.dim
    }

    /// `state_at(i) = dim - i - 1` (descending order).
    #[inline]
    fn state_at(&self, i: usize) -> B {
        B::from_u64((self.dim - i - 1) as u64)
    }

    /// `index(s) = dim - s - 1`.
    #[inline]
    fn index(&self, state: B) -> Option<usize> {
        let s = state.to_usize();
        if s < self.dim {
            Some(self.dim - s - 1)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Subspace
// ---------------------------------------------------------------------------

/// A filtered subspace of basis states, stored sorted in ascending order.
///
/// States are discovered via a DFS/BFS reachability walk starting from seed
/// states, then sorted.  A `HashMap` provides O(1) average-case index lookup.
///
/// Mirrors `subspace<bitset_t>` from `space.hpp`.
#[derive(Debug)]
pub struct Subspace<B: BitInt> {
    states: Vec<B>,
    index_map: HashMap<B, usize>,
}

impl<B: BitInt> Subspace<B> {
    /// Create an empty subspace.
    pub fn new() -> Self {
        Subspace {
            states: Vec::new(),
            index_map: HashMap::new(),
        }
    }

    /// Build the subspace reachable from `seed` under the action of `op`.
    ///
    /// `op(state)` must return an iterator of `(amplitude, new_state, cindex)`
    /// triples — matching the signature of `PauliHamiltonian::apply`.
    ///
    /// Uses iterative DFS (matching the C++ `subspace::build` stack-based
    /// implementation).  After the walk, states are sorted ascending and the
    /// index map is rebuilt.
    ///
    /// Mirrors `subspace::build` from `space.hpp`.
    pub fn build<Op, I, Iter>(&mut self, seed: B, op: Op)
    where
        Op: Fn(B) -> Iter,
        Iter: IntoIterator<Item = (num_complex::Complex<f64>, B, I)>,
    {
        let mut stack: Vec<B> = Vec::new();

        if !self.index_map.contains_key(&seed) {
            self.index_map.insert(seed, self.states.len());
            self.states.push(seed);
            stack.push(seed);
        }

        while let Some(state) = stack.pop() {
            for (_amp, next_state, _cindex) in op(state) {
                if next_state != state && !self.index_map.contains_key(&next_state) {
                    self.index_map.insert(next_state, self.states.len());
                    self.states.push(next_state);
                    stack.push(next_state);
                }
            }
        }

        self.sort();
    }

    fn sort(&mut self) {
        self.states.sort_unstable();
        self.index_map.clear();
        for (i, &s) in self.states.iter().enumerate() {
            self.index_map.insert(s, i);
        }
    }
}

impl<B: BitInt> Default for Subspace<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: BitInt> BasisSpace<B> for Subspace<B> {
    #[inline]
    fn size(&self) -> usize {
        self.states.len()
    }

    #[inline]
    fn state_at(&self, i: usize) -> B {
        self.states[i]
    }

    #[inline]
    fn index(&self, state: B) -> Option<usize> {
        self.index_map.get(&state).copied()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    // --- FullSpace ---

    #[test]
    fn full_space_state_at() {
        let fs = FullSpace::<u32>::new(4);
        assert_eq!(fs.state_at(0), 3u32); // dim-0-1 = 3
        assert_eq!(fs.state_at(1), 2u32);
        assert_eq!(fs.state_at(2), 1u32);
        assert_eq!(fs.state_at(3), 0u32);
    }

    #[test]
    fn full_space_index_roundtrip() {
        let fs = FullSpace::<u64>::new(8);
        for i in 0..8 {
            let s = fs.state_at(i);
            assert_eq!(fs.index(s), Some(i));
        }
    }

    #[test]
    fn full_space_out_of_range() {
        let fs = FullSpace::<u32>::new(4);
        assert_eq!(fs.index(4u32), None);
        assert_eq!(fs.index(100u32), None);
    }

    // --- Subspace build via X-only hopping (connects all 2^N states) ---

    fn x_op_all_sites(n_sites: u32) -> impl Fn(u32) -> Vec<(Complex<f64>, u32, u8)> {
        move |state: u32| {
            (0..n_sites)
                .map(|loc| {
                    let ns = state ^ (1u32 << loc);
                    (Complex::new(1.0, 0.0), ns, 0u8)
                })
                .collect()
        }
    }

    #[test]
    fn subspace_build_full_connectivity() {
        // X on every site connects all 2^3 = 8 states from seed 0
        let mut sub = Subspace::<u32>::new();
        sub.build(0u32, x_op_all_sites(3));
        assert_eq!(sub.size(), 8);
    }

    #[test]
    fn subspace_build_sorted_ascending() {
        let mut sub = Subspace::<u32>::new();
        sub.build(0u32, x_op_all_sites(3));
        for i in 0..sub.size() {
            assert_eq!(sub.state_at(i), i as u32);
        }
    }

    #[test]
    fn subspace_index_roundtrip() {
        let mut sub = Subspace::<u32>::new();
        sub.build(0u32, x_op_all_sites(3));
        for i in 0..sub.size() {
            let s = sub.state_at(i);
            assert_eq!(sub.index(s), Some(i));
        }
    }

    #[test]
    fn subspace_build_constrained() {
        // Hopping that only connects states with the same particle number
        // (XX + YY preserves particle number) — simulate with a ZZ-like op
        // that only connects states which differ by swapping adjacent 01↔10.
        let hop_op = |state: u32| {
            let mut results = vec![];
            // swap site i and i+1 if they differ
            for i in 0..2u32 {
                let si = (state >> i) & 1;
                let sj = (state >> (i + 1)) & 1;
                if si != sj {
                    let ns = state ^ (1 << i) ^ (1 << (i + 1));
                    results.push((Complex::new(1.0, 0.0), ns, 0u8));
                }
            }
            results
        };
        // Start from |01⟩ = 1 (1 particle): should reach |01⟩=1 and |10⟩=2 in 3 sites
        let mut sub = Subspace::<u32>::new();
        sub.build(0b001u32, hop_op);
        // 3-site, 1-particle sector: {001, 010, 100} = {1, 2, 4}
        assert_eq!(sub.size(), 3);
        assert!(sub.index(0b001u32).is_some());
        assert!(sub.index(0b010u32).is_some());
        assert!(sub.index(0b100u32).is_some());
    }
}
