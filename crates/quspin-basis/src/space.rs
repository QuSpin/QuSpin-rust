use super::BasisSpace;
use super::bfs::bfs_wave;
use quspin_bitbasis::{
    BitInt, StateTransitions,
    manip::{DitManip, DynamicDitManip},
};
use quspin_types::QuSpinError;
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
/// `manip` is cached at construction so `state_at`/`index` never rebuild it.
/// For `lhss=2,3,4` the hot paths dispatch to `DitManip::<N>` which bakes
/// `bits` and `mask` as compile-time constants.
///
/// Mirrors `space<bitset_t>` from `space.hpp`, restricted to `u32`/`u64`
/// (multi-word integers are never practical for a full space).
#[derive(Clone, Debug)]
pub struct FullSpace<B: BitInt> {
    n_sites: usize,
    dim: usize,
    manip: DynamicDitManip,
    fermionic: bool,
    _marker: std::marker::PhantomData<B>,
}

impl<B: BitInt> FullSpace<B> {
    pub fn new(lhss: usize, n_sites: usize, fermionic: bool) -> Self {
        let dim = lhss.saturating_pow(n_sites as u32);
        FullSpace {
            n_sites,
            dim,
            manip: DynamicDitManip::new(lhss),
            fermionic,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn lhss(&self) -> usize {
        self.manip.lhss
    }
}

impl<B: BitInt> BasisSpace<B> for FullSpace<B> {
    #[inline]
    fn n_sites(&self) -> usize {
        self.n_sites
    }

    #[inline]
    fn lhss(&self) -> usize {
        self.manip.lhss
    }

    #[inline]
    fn fermionic(&self) -> bool {
        self.fermionic
    }

    #[inline]
    fn size(&self) -> usize {
        self.dim
    }

    /// Returns the `i`-th basis state in descending dense order.
    ///
    /// For lhss=2 the state integer equals the dense index, so this is O(1).
    /// For lhss=3,4 `DitManip::<N>` is used with compile-time constant
    /// `bits`/`mask`.  Other values fall back to the cached `DynamicDitManip`.
    #[inline]
    fn state_at(&self, i: usize) -> B {
        let dense = self.dim - i - 1;
        match self.manip.lhss {
            2 => B::from_u64(dense as u64),
            3 => DitManip::<3>::state_from_dense(dense, self.n_sites),
            4 => DitManip::<4>::state_from_dense(dense, self.n_sites),
            _ => self.manip.state_from_dense(dense, self.n_sites),
        }
    }

    /// Returns the dense index of `state`, or `None` if it is not a valid
    /// state in this full space.
    ///
    /// For lhss=2 the state integer is already a dense index (O(1)).
    /// For lhss=3,4 `DitManip::<N>` is used with compile-time constant
    /// `bits`/`mask`.  Other values fall back to the cached `DynamicDitManip`.
    #[inline]
    fn index(&self, state: B) -> Option<usize> {
        match self.manip.lhss {
            2 => {
                let s = state.to_usize();
                if s < self.dim {
                    Some(self.dim - s - 1)
                } else {
                    None
                }
            }
            3 => DitManip::<3>::dense_from_state(state, self.n_sites).map(|d| self.dim - d - 1),
            4 => DitManip::<4>::dense_from_state(state, self.n_sites).map(|d| self.dim - d - 1),
            _ => self
                .manip
                .dense_from_state(state, self.n_sites)
                .map(|d| self.dim - d - 1),
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
    lhss: usize,
    n_sites: usize,
    fermionic: bool,
    states: Vec<B>,
    index_map: HashMap<B, usize>,
    built: bool,
}

impl<B: BitInt> Subspace<B> {
    /// Create an empty subspace for a system with `lhss` local states and `n_sites` lattice sites.
    pub fn new(lhss: usize, n_sites: usize, fermionic: bool) -> Self {
        Subspace {
            lhss,
            n_sites,
            fermionic,
            states: Vec::new(),
            index_map: HashMap::new(),
            built: false,
        }
    }

    /// Construct an empty subspace (API uniformity with [`SymBasis::new_empty`]).
    pub fn new_empty(lhss: usize, n_sites: usize, fermionic: bool) -> Self {
        Self::new(lhss, n_sites, fermionic)
    }

    pub fn lhss(&self) -> usize {
        self.lhss
    }

    /// Returns `true` once [`build`](Self::build) has been called.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Build the subspace reachable from `seed` under the connectivity
    /// described by `graph`.
    ///
    /// Uses level-synchronous BFS: each wave processes the current frontier
    /// in parallel (when large enough), discovers new states, and forms the
    /// next frontier.  After the walk, states are sorted ascending and the
    /// index map is rebuilt.
    ///
    /// # Errors
    /// - `graph.lhss() != self.lhss()`
    pub fn build<G: StateTransitions>(&mut self, seed: B, graph: &G) -> Result<(), QuSpinError> {
        if graph.lhss() != self.lhss {
            return Err(QuSpinError::ValueError(format!(
                "graph.lhss()={} does not match basis lhss={}",
                graph.lhss(),
                self.lhss,
            )));
        }
        self.built = true;

        let is_new =
            if let std::collections::hash_map::Entry::Vacant(e) = self.index_map.entry(seed) {
                e.insert(self.states.len());
                self.states.push(seed);
                true
            } else {
                false
            };

        if !is_new {
            return Ok(());
        }

        let max_size = self.lhss.saturating_pow(self.n_sites as u32);
        let mut frontier: Vec<B> = vec![seed];

        while !frontier.is_empty() && self.states.len() < max_size {
            let discovered = bfs_wave(&frontier, graph);

            // Register new states and form the next frontier.
            frontier.clear();
            for next_state in discovered {
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.index_map.entry(next_state)
                {
                    e.insert(self.states.len());
                    self.states.push(next_state);
                    frontier.push(next_state);
                }
            }
        }

        self.sort();
        Ok(())
    }

    fn sort(&mut self) {
        self.states.sort_unstable();
        self.index_map.clear();
        for (i, &s) in self.states.iter().enumerate() {
            self.index_map.insert(s, i);
        }
    }
}

impl<B: BitInt> BasisSpace<B> for Subspace<B> {
    #[inline]
    fn n_sites(&self) -> usize {
        self.n_sites
    }

    #[inline]
    fn lhss(&self) -> usize {
        self.lhss
    }

    #[inline]
    fn fermionic(&self) -> bool {
        self.fermionic
    }

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
        let fs = FullSpace::<u32>::new(2, 2, false);
        assert_eq!(fs.state_at(0), 3u32); // dim-0-1 = 3
        assert_eq!(fs.state_at(1), 2u32);
        assert_eq!(fs.state_at(2), 1u32);
        assert_eq!(fs.state_at(3), 0u32);
    }

    #[test]
    fn full_space_index_roundtrip() {
        let fs = FullSpace::<u64>::new(2, 3, false);
        for i in 0..8 {
            let s = fs.state_at(i);
            assert_eq!(fs.index(s), Some(i));
        }
    }

    #[test]
    fn full_space_out_of_range() {
        let fs = FullSpace::<u32>::new(2, 2, false);
        assert_eq!(fs.index(4u32), None);
        assert_eq!(fs.index(100u32), None);
    }

    // --- Subspace build via X-only hopping (connects all 2^N states) ---

    use quspin_bitbasis::test_graphs::{NearestNeighborSwap, XAllSites};

    #[test]
    fn subspace_build_full_connectivity() {
        // X on every site connects all 2^3 = 8 states from seed 0
        let mut sub = Subspace::<u32>::new(2, 3, false);
        sub.build(0u32, &XAllSites::new(3));
        assert_eq!(sub.size(), 8);
    }

    #[test]
    fn subspace_build_sorted_ascending() {
        let mut sub = Subspace::<u32>::new(2, 3, false);
        sub.build(0u32, &XAllSites::new(3));
        for i in 0..sub.size() {
            assert_eq!(sub.state_at(i), i as u32);
        }
    }

    #[test]
    fn subspace_index_roundtrip() {
        let mut sub = Subspace::<u32>::new(2, 3, false);
        sub.build(0u32, &XAllSites::new(3));
        for i in 0..sub.size() {
            let s = sub.state_at(i);
            assert_eq!(sub.index(s), Some(i));
        }
    }

    /// Binomial coefficient C(n, k) — used to compute exact sector dimensions.
    fn binom(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        (0..k).fold(1usize, |acc, i| acc * (n - i) / (i + 1))
    }

    #[test]
    fn subspace_xx_yy_conserves_particle_number() {
        // H = Σ_i (X_i X_{i+1} + Y_i Y_{i+1}) is the XX+YY Hamiltonian.
        // Symbolically XX+YY = 2(σ⁺σ⁻ + σ⁻σ⁺), which conserves particle number.
        // The individual XX and YY terms connect |00>↔|11> with amplitudes +1
        // and -1 respectively — they cancel in the sum, so no state should be
        // added to the subspace when the net amplitude is zero.
        // Starting from a seed with k ones the subspace dimension must equal
        // C(n_sites, k) for all k in 0..=n_sites.
        use quspin_operator::pauli::{HardcoreOp, HardcoreOperator, OpEntry};
        use smallvec::smallvec;

        let n_sites: usize = 6;

        let mut terms: Vec<OpEntry<u8>> = Vec::new();
        for i in 0..(n_sites - 1) as u32 {
            terms.push(OpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(HardcoreOp::X, i), (HardcoreOp::X, i + 1)],
            ));
            terms.push(OpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(HardcoreOp::Y, i), (HardcoreOp::Y, i + 1)],
            ));
        }
        let ham = HardcoreOperator::new(terms);

        for k in 0..=n_sites {
            // Seed: lowest k bits set (e.g. k=2, n=6 → 0b000011)
            let seed = if k == 0 { 0u32 } else { (1u32 << k) - 1 };
            let mut sub = Subspace::<u32>::new(2, n_sites, false);
            sub.build(seed, &ham);

            let expected = binom(n_sites, k);
            assert_eq!(
                sub.size(),
                expected,
                "k={k}: expected C({n_sites},{k})={expected}, got {}",
                sub.size()
            );
        }
    }

    /// Regression test for issue #12: `lhss.pow(n_sites)` overflows `usize`
    /// when n_sites >= 64 with lhss=2. The DFS safety bound must use
    /// `saturating_pow` to avoid the panic.
    #[test]
    fn subspace_build_large_n_sites_no_overflow() {
        use ruint::aliases::U128;

        // 65 sites with lhss=2 requires 65 bits → B = U128.
        // Single-particle hopping: seed has 1 particle, should find exactly 65 states.
        let n_sites: usize = 65;

        let seed = U128::from(1); // single particle at site 0
        let mut sub = Subspace::<U128>::new(2, n_sites, false);
        sub.build(seed, &NearestNeighborSwap::new(n_sites as u32));
        assert_eq!(sub.size(), n_sites);
    }

    // --- Parallel path tests (frontier > PARALLEL_FRONTIER_THRESHOLD) ---

    #[test]
    fn subspace_parallel_full_connectivity() {
        // 12 sites, X on all → 2^12 = 4096 states.
        // After the first BFS wave the frontier has 12 states,
        // after the second it has ~66, after the third ~220, etc.
        // By wave 4+ the frontier exceeds the threshold.
        let n_sites = 12u32;
        let mut sub = Subspace::<u32>::new(2, n_sites as usize, false);
        sub.build(0u32, &XAllSites::new(n_sites));
        assert_eq!(sub.size(), 1 << n_sites);
        // Verify sorted ascending
        for i in 1..sub.size() {
            assert!(sub.state_at(i) > sub.state_at(i - 1));
        }
        // Verify index roundtrip
        for i in 0..sub.size() {
            assert_eq!(sub.index(sub.state_at(i)), Some(i));
        }
    }

    #[test]
    fn subspace_parallel_xx_yy_conserves_particle_number() {
        // Same as the small test but at 12 sites — frontier grows large enough
        // to trigger parallel BFS.  Amplitude cancellation must still be exact.
        use quspin_operator::pauli::{HardcoreOp, HardcoreOperator, OpEntry};
        use smallvec::smallvec;

        let n_sites: usize = 12;
        let mut terms: Vec<OpEntry<u8>> = Vec::new();
        for i in 0..(n_sites - 1) as u32 {
            terms.push(OpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(HardcoreOp::X, i), (HardcoreOp::X, i + 1)],
            ));
            terms.push(OpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(HardcoreOp::Y, i), (HardcoreOp::Y, i + 1)],
            ));
        }
        let ham = HardcoreOperator::new(terms);

        // Half-filling sector: k = n_sites/2
        let k = n_sites / 2;
        let seed = (1u32 << k) - 1;
        let mut sub = Subspace::<u32>::new(2, n_sites, false);
        sub.build(seed, &ham);
        assert_eq!(sub.size(), binom(n_sites, k));
    }

    #[test]
    fn subspace_build_constrained() {
        // NN-swap on 3 sites from |01⟩ = 1 (1 particle): should reach
        // {|01⟩=1, |10⟩=2, |100⟩=4} = 1-particle sector.
        let mut sub = Subspace::<u32>::new(2, 3, false);
        sub.build(0b001u32, &NearestNeighborSwap::new(3));
        // 3-site, 1-particle sector: {001, 010, 100} = {1, 2, 4}
        assert_eq!(sub.size(), 3);
        assert!(sub.index(0b001u32).is_some());
        assert!(sub.index(0b010u32).is_some());
        assert!(sub.index(0b100u32).is_some());
    }
}
