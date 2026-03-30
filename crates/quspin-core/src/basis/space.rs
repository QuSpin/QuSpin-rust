use super::BasisSpace;
use crate::bitbasis::{
    BitInt,
    manip::{DitManip, DynamicDitManip},
};
use std::collections::HashMap;

/// A net amplitude is treated as zero when its magnitude is below
/// `sum_of_input_magnitudes * AMP_CANCEL_TOL`.
///
/// This is a relative tolerance: it scales with the coupling strengths in
/// the Hamiltonian, so it catches exact symbolic cancellations (e.g. the
/// `+1` from `XX` and `−1` from `YY` on `|00⟩`) without requiring a
/// hardcoded absolute threshold.  The factor of 4 covers the two complex
/// multiplications in a typical single-site operator application.
const AMP_CANCEL_TOL: f64 = 4.0 * f64::EPSILON;

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
        let dim = lhss.pow(n_sites as u32);
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
        self.built = true;
        let mut stack: Vec<B> = Vec::new();

        if let std::collections::hash_map::Entry::Vacant(e) = self.index_map.entry(seed) {
            e.insert(self.states.len());
            self.states.push(seed);
            stack.push(seed);
        }
        let max_size = self.lhss.pow(self.n_sites as u32);
        while let Some(state) = stack.pop()
            && stack.len() < max_size
        {
            // Accumulate (net_amp, sum_of_magnitudes) per output state.
            // The sum of magnitudes sets the scale for the cancellation check.
            let mut contributions: HashMap<B, (num_complex::Complex<f64>, f64)> = HashMap::new();
            for (amp, next_state, _cindex) in op(state) {
                if next_state != state {
                    let e = contributions.entry(next_state).or_default();
                    e.0 += amp;
                    e.1 += amp.norm();
                }
            }
            for (next_state, (net_amp, scale)) in contributions {
                if net_amp.norm() > scale * AMP_CANCEL_TOL
                    && let std::collections::hash_map::Entry::Vacant(e) =
                        self.index_map.entry(next_state)
                {
                    e.insert(self.states.len());
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
        let mut sub = Subspace::<u32>::new(2, 3, false);
        sub.build(0u32, x_op_all_sites(3));
        assert_eq!(sub.size(), 8);
    }

    #[test]
    fn subspace_build_sorted_ascending() {
        let mut sub = Subspace::<u32>::new(2, 3, false);
        sub.build(0u32, x_op_all_sites(3));
        for i in 0..sub.size() {
            assert_eq!(sub.state_at(i), i as u32);
        }
    }

    #[test]
    fn subspace_index_roundtrip() {
        let mut sub = Subspace::<u32>::new(2, 3, false);
        sub.build(0u32, x_op_all_sites(3));
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
        use crate::hamiltonian::pauli::{HardcoreOp, HardcoreOperator, OpEntry};
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
            sub.build(seed, |s| ham.apply_smallvec(s).into_iter());

            let expected = binom(n_sites, k);
            assert_eq!(
                sub.size(),
                expected,
                "k={k}: expected C({n_sites},{k})={expected}, got {}",
                sub.size()
            );
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
        let mut sub = Subspace::<u32>::new(2, 3, false);
        sub.build(0b001u32, hop_op);
        // 3-site, 1-particle sector: {001, 010, 100} = {1, 2, 4}
        assert_eq!(sub.size(), 3);
        assert!(sub.index(0b001u32).is_some());
        assert!(sub.index(0b010u32).is_some());
        assert!(sub.index(0b100u32).is_some());
    }
}
