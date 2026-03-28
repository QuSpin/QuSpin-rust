use super::{BasisSpace, traits::SymGrp};
use num_complex::Complex;
use std::collections::HashMap;

/// See `space::AMP_CANCEL_TOL` for the rationale.
const AMP_CANCEL_TOL: f64 = 4.0 * f64::EPSILON;

// ---------------------------------------------------------------------------
// NormInt
// ---------------------------------------------------------------------------

/// Storage type for orbit norms inside [`SymmetricSubspace`].
///
/// The orbit norm is a small non-negative integer bounded by the symmetry
/// group order.  Using a narrower integer than `f64` reduces memory by up to
/// 8×; the `f64` value is only materialised at the [`entry`](SymmetricSubspace::entry)
/// API boundary.
///
/// Choose based on `n_sites`:
/// - `n_sites <= 32` → `u8`  (max group order 256 fits in u8)
/// - `n_sites <= 64` → `u16`
/// - `n_sites >  64` → `u32`
///
/// [`SymmetricSubspaceInner`] performs this dispatch automatically.
pub trait NormInt: Copy + Send + Sync + 'static {
    fn from_norm(norm: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl NormInt for u8 {
    #[inline]
    fn from_norm(norm: f64) -> Self {
        norm as u8
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl NormInt for u16 {
    #[inline]
    fn from_norm(norm: f64) -> Self {
        norm as u16
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl NormInt for u32 {
    #[inline]
    fn from_norm(norm: f64) -> Self {
        norm as u32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

// ---------------------------------------------------------------------------
// SymmetricSubspace
// ---------------------------------------------------------------------------

/// A symmetry-reduced subspace of basis states.
///
/// Each entry stores the representative state (largest in its orbit) and its
/// orbit norm (number of distinct group images).  States are sorted in
/// ascending order by representative; a `HashMap` provides O(1) lookup.
///
/// The type parameter `N` controls how norms are stored in memory; see
/// [`NormInt`] for the recommended choice.  [`SymmetricSubspaceInner`]
/// dispatches `N` automatically based on `n_sites`.
///
/// Construction is currently sequential (matching the C++ single-thread
/// branch).  Rayon-parallel BFS is a deferred optimisation.
///
/// Mirrors `symmetric_subspace<grp_t, bitset_t, norm_t>` from `space.hpp`.
pub struct SymmetricSubspace<G: SymGrp, N: NormInt = u32> {
    /// `(representative_state, orbit_norm)` pairs, sorted by state ascending.
    states: Vec<(G::State, N)>,
    /// Maps representative state → index in `states`.
    index_map: HashMap<G::State, usize>,
    grp: G,
}

impl<G: SymGrp, N: NormInt> SymmetricSubspace<G, N> {
    pub fn new(grp: G) -> Self {
        SymmetricSubspace {
            states: Vec::new(),
            index_map: HashMap::new(),
            grp,
        }
    }

    /// Build the symmetric subspace reachable from `seed` under `op`, using
    /// the stored symmetry group to identify representative states.
    ///
    /// A state is included if and only if it is its own representative (i.e.
    /// no group image is larger) and its norm > 0 (non-zero orbit).
    ///
    /// Mirrors `symmetric_subspace::build` from `space.hpp` (single-thread branch).
    pub fn build<Op, I, Iter>(&mut self, seed: G::State, op: Op)
    where
        Op: Fn(G::State) -> Iter,
        Iter: IntoIterator<Item = (Complex<f64>, G::State, I)>,
    {
        // Initialise from seed.
        let (ref_seed, _coeff) = self.grp.get_refstate(seed);
        let (_ref2, norm_seed) = self.grp.check_refstate(ref_seed);

        if norm_seed > 0.0
            && let std::collections::hash_map::Entry::Vacant(e) = self.index_map.entry(ref_seed)
        {
            e.insert(self.states.len());
            self.states.push((ref_seed, N::from_norm(norm_seed)));
        }

        let mut stack: Vec<G::State> = vec![ref_seed];

        while let Some(state) = stack.pop() {
            // Accumulate (net_amp, sum_of_magnitudes) per output state.
            let mut contributions: HashMap<G::State, (Complex<f64>, f64)> = HashMap::new();
            for (amp, next_state, _cindex) in op(state) {
                if next_state != state {
                    let e = contributions.entry(next_state).or_default();
                    e.0 += amp;
                    e.1 += amp.norm();
                }
            }

            // Collect non-cancelled candidates.
            let candidates: Vec<G::State> = contributions
                .iter()
                .filter(|(_, (net_amp, scale))| net_amp.norm() > scale * AMP_CANCEL_TOL)
                .map(|(&s, _)| s)
                .collect();

            if candidates.is_empty() {
                continue;
            }

            // Batch-find the representative for each candidate, then process.
            let mut batch_out: Vec<(G::State, f64)> = vec![(candidates[0], 0.0); candidates.len()];
            self.grp.check_refstate_batch(&candidates, &mut batch_out);

            for (next_ref, _) in batch_out {
                let (_, next_norm) = self.grp.check_refstate(next_ref);

                if next_norm > 0.0
                    && let std::collections::hash_map::Entry::Vacant(e) =
                        self.index_map.entry(next_ref)
                {
                    e.insert(self.states.len());
                    self.states.push((next_ref, N::from_norm(next_norm)));
                    stack.push(next_ref);
                }
            }
        }

        self.sort();
    }

    fn sort(&mut self) {
        self.states.sort_unstable_by_key(|&(s, _)| s);
        self.index_map.clear();
        for (i, &(s, _)) in self.states.iter().enumerate() {
            self.index_map.insert(s, i);
        }
    }

    /// Return the representative state and orbit norm for index `i`.
    ///
    /// The norm is cast to `f64` at this boundary; internally it is stored as
    /// `N` (typically `u8`, `u16`, or `u32`) to reduce memory usage.
    #[inline]
    pub fn entry(&self, i: usize) -> (G::State, f64) {
        let (s, n) = self.states[i];
        (s, n.to_f64())
    }

    /// Find the representative and accumulated coefficient for `state`.
    pub fn get_refstate(&self, state: G::State) -> (G::State, Complex<f64>) {
        self.grp.get_refstate(state)
    }

    /// Batch variant of [`get_refstate`].
    pub fn get_refstate_batch(&self, states: &[G::State], out: &mut [(G::State, Complex<f64>)]) {
        self.grp.get_refstate_batch(states, out);
    }

    /// Check whether `state` is a representative and return its norm.
    pub fn check_refstate(&self, state: G::State) -> (G::State, f64) {
        self.grp.check_refstate(state)
    }
}

impl<G: SymGrp, N: NormInt> BasisSpace<G::State> for SymmetricSubspace<G, N> {
    #[inline]
    fn n_sites(&self) -> usize {
        self.grp.n_sites()
    }

    #[inline]
    fn size(&self) -> usize {
        self.states.len()
    }

    /// The `i`-th representative state (ascending order).
    #[inline]
    fn state_at(&self, i: usize) -> G::State {
        self.states[i].0
    }

    #[inline]
    fn index(&self, state: G::State) -> Option<usize> {
        self.index_map.get(&state).copied()
    }
}

// ---------------------------------------------------------------------------
// SymmetricSubspaceInner — dispatch N based on n_sites
// ---------------------------------------------------------------------------

/// A [`SymmetricSubspace`] with the norm storage type selected automatically
/// from `n_sites` at construction time:
///
/// | `n_sites` | `N`  | bytes/state |
/// |-----------|------|-------------|
/// | ≤ 32      | `u8` | `size_of(B) + 1` |
/// | ≤ 64      | `u16`| `size_of(B) + 2` |
/// | > 64      | `u32`| `size_of(B) + 4` |
pub enum SymmetricSubspaceInner<G: SymGrp> {
    U8(SymmetricSubspace<G, u8>),
    U16(SymmetricSubspace<G, u16>),
    U32(SymmetricSubspace<G, u32>),
}

impl<G: SymGrp> SymmetricSubspaceInner<G> {
    /// Construct a new empty subspace, choosing the norm storage type based on
    /// `grp.n_sites()`.
    pub fn new(grp: G) -> Self {
        match grp.n_sites() {
            n if n <= 32 => Self::U8(SymmetricSubspace::new(grp)),
            n if n <= 64 => Self::U16(SymmetricSubspace::new(grp)),
            _ => Self::U32(SymmetricSubspace::new(grp)),
        }
    }

    pub fn build<Op, I, Iter>(&mut self, seed: G::State, op: Op)
    where
        Op: Fn(G::State) -> Iter,
        Iter: IntoIterator<Item = (Complex<f64>, G::State, I)>,
    {
        match self {
            Self::U8(inner) => inner.build(seed, op),
            Self::U16(inner) => inner.build(seed, op),
            Self::U32(inner) => inner.build(seed, op),
        }
    }

    #[inline]
    pub fn entry(&self, i: usize) -> (G::State, f64) {
        match self {
            Self::U8(inner) => inner.entry(i),
            Self::U16(inner) => inner.entry(i),
            Self::U32(inner) => inner.entry(i),
        }
    }

    pub fn get_refstate(&self, state: G::State) -> (G::State, Complex<f64>) {
        match self {
            Self::U8(inner) => inner.get_refstate(state),
            Self::U16(inner) => inner.get_refstate(state),
            Self::U32(inner) => inner.get_refstate(state),
        }
    }

    pub fn get_refstate_batch(&self, states: &[G::State], out: &mut [(G::State, Complex<f64>)]) {
        match self {
            Self::U8(inner) => inner.get_refstate_batch(states, out),
            Self::U16(inner) => inner.get_refstate_batch(states, out),
            Self::U32(inner) => inner.get_refstate_batch(states, out),
        }
    }

    pub fn check_refstate(&self, state: G::State) -> (G::State, f64) {
        match self {
            Self::U8(inner) => inner.check_refstate(state),
            Self::U16(inner) => inner.check_refstate(state),
            Self::U32(inner) => inner.check_refstate(state),
        }
    }
}

impl<G: SymGrp> BasisSpace<G::State> for SymmetricSubspaceInner<G> {
    #[inline]
    fn n_sites(&self) -> usize {
        match self {
            Self::U8(inner) => inner.n_sites(),
            Self::U16(inner) => inner.n_sites(),
            Self::U32(inner) => inner.n_sites(),
        }
    }

    #[inline]
    fn size(&self) -> usize {
        match self {
            Self::U8(inner) => inner.size(),
            Self::U16(inner) => inner.size(),
            Self::U32(inner) => inner.size(),
        }
    }

    #[inline]
    fn state_at(&self, i: usize) -> G::State {
        match self {
            Self::U8(inner) => inner.state_at(i),
            Self::U16(inner) => inner.state_at(i),
            Self::U32(inner) => inner.state_at(i),
        }
    }

    #[inline]
    fn index(&self, state: G::State) -> Option<usize> {
        match self {
            Self::U8(inner) => inner.index(state),
            Self::U16(inner) => inner.index(state),
            Self::U32(inner) => inner.index(state),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::sym_grp::SymGrpInner;
    use num_complex::Complex;

    /// X operator on all sites of an N-site chain.
    fn x_op(n_sites: u32) -> impl Fn(u32) -> Vec<(Complex<f64>, u32, u8)> {
        move |state: u32| {
            (0..n_sites)
                .map(|loc| (Complex::new(1.0, 0.0), state ^ (1 << loc), 0u8))
                .collect()
        }
    }

    /// Add an identity lattice element (all sites map to themselves) to a group.
    fn push_id_lattice(grp: &mut SymGrpInner<u32>, n_sites: usize) {
        let perm: Vec<usize> = (0..n_sites).collect();
        grp.push_lattice(Complex::new(1.0, 0.0), &perm);
    }

    /// Z₂ bit-flip group on the full N-site chain.
    fn bitflip_grp(n_sites: u32) -> SymGrpInner<u32> {
        let n = n_sites as usize;
        let mut grp = SymGrpInner::<u32>::new_empty(2, n, false);
        push_id_lattice(&mut grp, n);
        grp.push_inverse(Complex::new(1.0, 0.0), &(0..n).collect::<Vec<_>>());
        grp
    }

    #[test]
    fn symmetric_subspace_bitflip_2site() {
        // 2-site chain, Z₂ bitflip symmetry.
        // Orbits: {0↔3}, {1↔2}. Representatives (largest): 3, 2.
        let mut grp = SymGrpInner::<u32>::new_empty(2, 2, false);
        push_id_lattice(&mut grp, 2);
        grp.push_inverse(Complex::new(1.0, 0.0), &[0, 1]);
        let mut sym = SymmetricSubspace::<_, u32>::new(grp);
        sym.build(0u32, x_op(2));

        assert_eq!(sym.size(), 2);
        assert!(sym.index(3u32).is_some());
        assert!(sym.index(2u32).is_some());
    }

    #[test]
    fn symmetric_subspace_no_symmetry_matches_subspace() {
        // With a trivial group (identity lattice only, no local ops), every
        // state is its own representative with norm = 1.
        let mut grp = SymGrpInner::<u32>::new_empty(2, 3, false);
        push_id_lattice(&mut grp, 3);
        let mut sym = SymmetricSubspace::<_, u32>::new(grp);
        sym.build(0u32, x_op(3));

        // All 2^3 = 8 states should be present.
        assert_eq!(sym.size(), 8);
    }

    #[test]
    fn symmetric_subspace_sorted_ascending() {
        let mut grp = SymGrpInner::<u32>::new_empty(2, 3, false);
        push_id_lattice(&mut grp, 3);
        let mut sym = SymmetricSubspace::<_, u32>::new(grp);
        sym.build(0u32, x_op(3));

        for i in 1..sym.size() {
            assert!(sym.state_at(i) > sym.state_at(i - 1));
        }
    }

    #[test]
    fn symmetric_subspace_bitflip_grp_helper() {
        let _ = bitflip_grp(3); // verify construction doesn't panic
    }

    #[test]
    fn norm_int_dispatch_u8() {
        // n_sites=2 → U8 variant
        let mut grp = SymGrpInner::<u32>::new_empty(2, 2, false);
        push_id_lattice(&mut grp, 2);
        grp.push_inverse(Complex::new(1.0, 0.0), &[0, 1]);
        let mut sym = SymmetricSubspaceInner::new(grp);
        assert!(matches!(sym, SymmetricSubspaceInner::U8(_)));
        sym.build(0u32, x_op(2));
        assert_eq!(sym.size(), 2);
        assert!(sym.index(3u32).is_some());
    }

    #[test]
    fn norm_int_roundtrip() {
        // entry() must return the same norm as check_refstate would compute.
        // With 1 lattice element (identity) and 1 local op (full flip), only
        // the identity maps each representative back to itself, so norm = 1.
        let mut grp = SymGrpInner::<u32>::new_empty(2, 2, false);
        push_id_lattice(&mut grp, 2);
        grp.push_inverse(Complex::new(1.0, 0.0), &[0, 1]);
        let mut sym = SymmetricSubspace::<_, u8>::new(grp);
        sym.build(0u32, x_op(2));
        for i in 0..sym.size() {
            let (_, norm) = sym.entry(i);
            assert_eq!(norm, 1.0);
        }
    }
}
