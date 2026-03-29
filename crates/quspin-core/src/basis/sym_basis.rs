/// Merged symmetry-group + basis type: [`SymBasis<B, L, N>`].
///
/// Replaces the two-type hierarchy (`SymGrpBase<B,L>` + `SymmetricSubspace<G,N>`)
/// with a single flat struct. `N` is an explicit type parameter — the B→N
/// pairing is encoded in `BasisInner` variant definitions, not a runtime enum.
use super::lattice::BenesLatticeElement;
use super::sym_grp::SymGrpBase;
use super::traits::BasisSpace;
use crate::bitbasis::{BenesPermDitLocations, BitInt, BitStateOp};
use num_complex::Complex;
use std::collections::HashMap;

/// See `space::AMP_CANCEL_TOL` for the rationale.
const AMP_CANCEL_TOL: f64 = 4.0 * f64::EPSILON;

// ---------------------------------------------------------------------------
// NormInt
// ---------------------------------------------------------------------------

/// Storage type for orbit norms inside [`SymBasis`].
///
/// The orbit norm is a small non-negative integer bounded by the symmetry
/// group order. Using a narrower integer than `f64` reduces memory by up to
/// 8×; the `f64` value is only materialised at the [`entry`](SymBasis::entry)
/// API boundary.
///
/// The B→N pairing policy is:
/// - B = u32  → N = u8  (n_sites ≤ 32, max group order fits in u8)
/// - B = u64  → N = u16
/// - B = Uint<128..8192> → N = u32
///
/// This pairing is enforced statically via [`BasisInner`](crate::basis::dispatch::BasisInner)
/// variant definitions and the `with_sym_grp!` macro.
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
// SymBasis
// ---------------------------------------------------------------------------

/// Merged symmetry group + symmetric subspace.
///
/// Combines the group data previously in `SymGrpBase<B, L>` (lattice ops,
/// local ops, metadata) with the basis data previously in
/// `SymmetricSubspace<G, N>` (representative states, index map) into a single
/// flat struct. `N` is an explicit type parameter — no runtime enum dispatch
/// occurs in `get_refstate_batch` or `check_refstate_batch`.
///
/// # Type parameters
/// - `B` — basis integer type ([`BitInt`])
/// - `L` — local-op type: [`PermDitMask<B>`] for LHSS=2, [`DynamicPermDitValues`] for LHSS≥3
/// - `N` — norm storage type ([`NormInt`]): `u8`, `u16`, or `u32`
///
/// [`PermDitMask<B>`]: crate::bitbasis::PermDitMask
/// [`DynamicPermDitValues`]: crate::bitbasis::DynamicPermDitValues
pub struct SymBasis<B: BitInt, L, N: NormInt> {
    // Group data (was SymGrpBase<B, L>)
    pub(crate) lhss: usize,
    pub(crate) fermionic: bool,
    pub(crate) n_sites: usize,
    pub(crate) lattice: Vec<BenesLatticeElement<B>>,
    pub(crate) local: Vec<(Complex<f64>, L)>,
    // Basis data (was SymmetricSubspace<G, N>)
    /// `(representative_state, orbit_norm)` pairs, sorted ascending by state.
    states: Vec<(B, N)>,
    /// Maps representative state → index in `states`.
    index_map: HashMap<B, usize>,
    /// Set to `true` at the start of `build()`, never reset.
    /// A seed whose orbit norm is zero produces no entries in `states`, so
    /// `states.is_empty()` is not a reliable built indicator.
    built: bool,
}

// ---------------------------------------------------------------------------
// SymBasis: construction and non-orbit methods (no L bound)
// ---------------------------------------------------------------------------

impl<B: BitInt, L, N: NormInt> SymBasis<B, L, N> {
    /// Construct from a consumed group builder, starting with empty basis data.
    ///
    /// The group fields (lattice, local ops, metadata) are moved in from `grp`.
    /// Call [`build`](Self::build) afterwards to populate the basis states.
    pub fn from_grp(grp: SymGrpBase<B, L>) -> Self {
        SymBasis {
            lhss: grp.lhss,
            fermionic: grp.fermionic,
            n_sites: grp.n_sites,
            lattice: grp.lattice,
            local: grp.local,
            states: Vec::new(),
            index_map: HashMap::new(),
            built: false,
        }
    }

    /// Construct an empty basis with no group elements and no states.
    ///
    /// Call [`push_lattice`](Self::push_lattice) / [`push_local`](Self::push_local)
    /// to add symmetry elements, then [`build`](Self::build) to populate states.
    pub fn new_empty(lhss: usize, n_sites: usize, fermionic: bool) -> Self {
        SymBasis {
            lhss,
            fermionic,
            n_sites,
            lattice: Vec::new(),
            local: Vec::new(),
            states: Vec::new(),
            index_map: HashMap::new(),
            built: false,
        }
    }

    /// Add a lattice (site-permutation) symmetry element. Valid before [`build`](Self::build).
    ///
    /// `perm[src] = dst` maps source site `src` to destination `dst`.
    pub fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        let op = BenesPermDitLocations::<B>::new(self.lhss, perm, self.fermionic);
        self.lattice
            .push(BenesLatticeElement::new(grp_char, op, self.n_sites));
    }

    /// Add a local symmetry element. Valid before [`build`](Self::build).
    pub fn push_local(&mut self, grp_char: Complex<f64>, local_op: L) {
        self.local.push((grp_char, local_op));
    }

    /// Returns `true` once [`build`](Self::build) has been called.
    ///
    /// Set at the very start of `build()` so a seed with zero orbit norm
    /// (which produces no states) still marks the basis as built.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Local Hilbert-space size.
    pub fn lhss(&self) -> usize {
        self.lhss
    }

    /// Whether Jordan-Wigner signs are tracked.
    pub fn fermionic(&self) -> bool {
        self.fermionic
    }

    /// Return the representative state and orbit norm for index `i`.
    ///
    /// The norm is cast to `f64` at this boundary; internally it is stored as
    /// `N` (typically `u8`, `u16`, or `u32`) to reduce memory usage.
    #[inline]
    pub fn entry(&self, i: usize) -> (B, f64) {
        let (s, n) = self.states[i];
        (s, n.to_f64())
    }
}

// ---------------------------------------------------------------------------
// SymBasis: orbit methods and basis construction (require L: BitStateOp<B>)
// ---------------------------------------------------------------------------

impl<B: BitInt, L: BitStateOp<B>, N: NormInt> SymBasis<B, L, N> {
    /// Return the representative state (largest orbit element) and the
    /// accumulated group character for `state`.
    pub fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        super::orbit::get_refstate(&self.lattice, &self.local, state)
    }

    /// Return the representative state and the orbit norm.
    pub fn check_refstate(&self, state: B) -> (B, f64) {
        super::orbit::check_refstate(&self.lattice, &self.local, state)
    }

    /// Batch variant of [`get_refstate`](Self::get_refstate).
    ///
    /// Calls `orbit::get_refstate_batch` directly — zero N dispatch.
    ///
    /// # Panics
    ///
    /// Panics if `states.len() != out.len()`.
    pub fn get_refstate_batch(&self, states: &[B], out: &mut [(B, Complex<f64>)]) {
        super::orbit::get_refstate_batch(&self.lattice, &self.local, states, out);
    }

    /// Batch variant of [`check_refstate`](Self::check_refstate).
    ///
    /// # Panics
    ///
    /// Panics if `states.len() != out.len()`.
    pub fn check_refstate_batch(&self, states: &[B], out: &mut [(B, f64)]) {
        super::orbit::check_refstate_batch(&self.lattice, &self.local, states, out);
    }

    /// Build the symmetric subspace reachable from `seed` under `op`.
    ///
    /// Mirrors `symmetric_subspace::build` from `space.hpp` (single-thread branch).
    pub fn build<Op, I, Iter>(&mut self, seed: B, op: Op)
    where
        Op: Fn(B) -> Iter,
        Iter: IntoIterator<Item = (Complex<f64>, B, I)>,
    {
        self.built = true;
        let (ref_seed, _coeff) = self.get_refstate(seed);
        let (_ref2, norm_seed) = self.check_refstate(ref_seed);

        if norm_seed > 0.0
            && let std::collections::hash_map::Entry::Vacant(e) = self.index_map.entry(ref_seed)
        {
            e.insert(self.states.len());
            self.states.push((ref_seed, N::from_norm(norm_seed)));
        }

        let mut stack: Vec<B> = vec![ref_seed];

        while let Some(state) = stack.pop() {
            let mut contributions: HashMap<B, (Complex<f64>, f64)> = HashMap::new();
            for (amp, next_state, _cindex) in op(state) {
                if next_state != state {
                    let e = contributions.entry(next_state).or_default();
                    e.0 += amp;
                    e.1 += amp.norm();
                }
            }

            let candidates: Vec<B> = contributions
                .iter()
                .filter(|(_, (net_amp, scale))| net_amp.norm() > scale * AMP_CANCEL_TOL)
                .map(|(&s, _)| s)
                .collect();

            if candidates.is_empty() {
                continue;
            }

            let mut batch_out: Vec<(B, f64)> = vec![(candidates[0], 0.0); candidates.len()];
            self.check_refstate_batch(&candidates, &mut batch_out);

            for (next_ref, _) in batch_out {
                let (_, next_norm) = self.check_refstate(next_ref);

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
}

// ---------------------------------------------------------------------------
// BasisSpace impl (no L bound needed — no orbit computation)
// ---------------------------------------------------------------------------

impl<B: BitInt, L, N: NormInt> BasisSpace<B> for SymBasis<B, L, N> {
    #[inline]
    fn n_sites(&self) -> usize {
        self.n_sites
    }

    #[inline]
    fn size(&self) -> usize {
        self.states.len()
    }

    #[inline]
    fn state_at(&self, i: usize) -> B {
        self.states[i].0
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
    use crate::basis::sym_grp::SymGrpInner;
    use num_complex::Complex;

    fn x_op(n_sites: u32) -> impl Fn(u32) -> Vec<(Complex<f64>, u32, u8)> {
        move |state: u32| {
            (0..n_sites)
                .map(|loc| (Complex::new(1.0, 0.0), state ^ (1 << loc), 0u8))
                .collect()
        }
    }

    fn push_id_lattice(grp: &mut SymGrpInner<u32>, n_sites: usize) {
        let perm: Vec<usize> = (0..n_sites).collect();
        grp.push_lattice(Complex::new(1.0, 0.0), &perm);
    }

    #[test]
    fn sym_basis_bitflip_2site() {
        let mut grp = SymGrpInner::<u32>::new_empty(2, 2, false);
        push_id_lattice(&mut grp, 2);
        grp.push_inverse(Complex::new(1.0, 0.0), &[0, 1]);
        let mut basis = SymBasis::<u32, _, u32>::from_grp(grp);
        basis.build(0u32, x_op(2));

        assert_eq!(basis.size(), 2);
        assert!(basis.index(3u32).is_some());
        assert!(basis.index(2u32).is_some());
    }

    #[test]
    fn sym_basis_no_symmetry_matches_subspace() {
        let mut grp = SymGrpInner::<u32>::new_empty(2, 3, false);
        push_id_lattice(&mut grp, 3);
        let mut basis = SymBasis::<u32, _, u32>::from_grp(grp);
        basis.build(0u32, x_op(3));
        assert_eq!(basis.size(), 8);
    }

    #[test]
    fn sym_basis_sorted_ascending() {
        let mut grp = SymGrpInner::<u32>::new_empty(2, 3, false);
        push_id_lattice(&mut grp, 3);
        let mut basis = SymBasis::<u32, _, u32>::from_grp(grp);
        basis.build(0u32, x_op(3));
        for i in 1..basis.size() {
            assert!(basis.state_at(i) > basis.state_at(i - 1));
        }
    }

    #[test]
    fn sym_basis_u8_norm() {
        let mut grp = SymGrpInner::<u32>::new_empty(2, 2, false);
        push_id_lattice(&mut grp, 2);
        grp.push_inverse(Complex::new(1.0, 0.0), &[0, 1]);
        let mut basis = SymBasis::<u32, _, u8>::from_grp(grp);
        basis.build(0u32, x_op(2));
        assert_eq!(basis.size(), 2);
        for i in 0..basis.size() {
            let (_, norm) = basis.entry(i);
            assert_eq!(norm, 1.0);
        }
    }

    #[test]
    fn sym_basis_entry_roundtrip() {
        let mut grp = SymGrpInner::<u32>::new_empty(2, 2, false);
        push_id_lattice(&mut grp, 2);
        grp.push_inverse(Complex::new(1.0, 0.0), &[0, 1]);
        let mut basis = SymBasis::<u32, _, u8>::from_grp(grp);
        basis.build(0u32, x_op(2));
        for i in 0..basis.size() {
            let (_, norm) = basis.entry(i);
            assert_eq!(norm, 1.0);
        }
    }
}
