use super::benes::{BenesNetwork, benes_fwd, gen_benes};
use super::manip::{DitManip, DynamicDitManip};
use quspin_types::BitInt;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// BitStateOp — the shared trait
// ---------------------------------------------------------------------------

/// A bijective transformation on basis-state integers.
///
/// All symmetry operations in this module (site permutations, value
/// permutations, spin inversion, XOR masks) implement this trait.
pub trait BitStateOp<I: BitInt> {
    /// Apply the operation: `state → T(state)`.
    fn apply(&self, state: I) -> I;
}

/// Extension of [`BitStateOp`] that reports a fermion sign contributed by
/// the operation when applied on a fermionic basis.
///
/// The default implementation returns `1.0` — no sign contribution —
/// which is correct for every symmetry that acts only by reordering /
/// bit-flipping without crossing fermion creation operators. Types that
/// genuinely pick up a state-dependent fermion sign (e.g.
/// [`SignedPermDitMask`] for particle-hole) override `fermion_sign`.
///
/// The symmetry walker consults this method **only** when the enclosing
/// basis's `fermionic` flag is `true`; on spin-1/2 / bosonic paths the
/// call is dead code and is eliminated by LLVM.
pub trait FermionicBitStateOp<I: BitInt>: BitStateOp<I> {
    /// Fermion sign contributed when applied to `state`. Default: `1.0`.
    fn fermion_sign(&self, _state: I) -> f64 {
        1.0
    }
}

/// Group-composition trait for local operators.
///
/// `a.compose(b)` returns the element representing `a · b` — the operator
/// that, when applied, first applies `b` and then `a`. Required for types
/// used with [`SymElement::compose`](../../quspin_basis/struct.SymElement.html)
/// and the `add_cyclic` helper on `SymBasis`.
///
/// Not implemented for [`SignedPermDitMask`]: composition of two signed
/// bit-flip operators produces a state-independent prefactor that
/// belongs on the character (not the operator), so it does not round-trip
/// cleanly to a new `SignedPermDitMask`. Callers who need cyclic groups
/// of signed elements should enumerate the group explicitly via
/// `add_symmetry` instead of using `add_cyclic`.
pub trait Compose: Clone + Sized {
    fn compose(&self, other: &Self) -> Self;
}

// ---------------------------------------------------------------------------
// PermDitLocations — permute which site goes where
// ---------------------------------------------------------------------------

/// Permutation of dit **locations** (sites).
///
/// Convention: `forward[src] = dst` means the dit currently at site `src` is
/// mapped to site `dst`.
///
/// Mirrors `perm_dit_locations` from `dit_perm.hpp`.
#[derive(Clone, Debug)]
pub struct PermDitLocations {
    /// forward_perm[src] = dst
    forward: Vec<usize>,
    manip: DynamicDitManip,
}

impl PermDitLocations {
    /// Construct from a `lhss` value and a forward permutation `perm` where
    /// `perm[src] = dst`.
    ///
    /// # Panics
    /// Panics if `perm` contains out-of-range destinations or is not a valid
    /// permutation.
    pub fn new(lhss: usize, perm: &[usize]) -> Self {
        let n = perm.len();
        for (src, &dst) in perm.iter().enumerate() {
            assert!(dst < n, "perm[{src}]={dst} is out of range 0..{n}");
        }
        PermDitLocations {
            forward: perm.to_vec(),
            manip: DynamicDitManip::new(lhss),
        }
    }
}

impl<I: BitInt> BitStateOp<I> for PermDitLocations {
    /// Apply the forward permutation: the dit at site `src` moves to site
    /// `forward[src]`.
    #[inline]
    fn apply(&self, state: I) -> I {
        let mut out = I::from_u64(0);
        for (src, &dst) in self.forward.iter().enumerate() {
            let val = self.manip.get_dit(state, src);
            out = self.manip.set_dit(out, val, dst);
        }
        out
    }
}

impl<I: BitInt> FermionicBitStateOp<I> for PermDitLocations {}

// ---------------------------------------------------------------------------
// PermDitValues — permute the dit values at given locations
// ---------------------------------------------------------------------------

/// Compile-time dit-value permutation for `LHSS`-valued dits.
///
/// For each site in `locs`, maps the dit value `v` → `perm[v]`.
///
/// Mirrors `perm_dit_values<bitset_t, lhss>` from `dit_perm.hpp`.
#[derive(Clone, Debug)]
pub struct PermDitValues<const LHSS: usize> {
    /// Maps dit value v → perm[v].  Length must equal LHSS.
    perm: [u8; LHSS],
    /// Sites to which the value permutation is applied.
    locs: Vec<usize>,
}

impl<const LHSS: usize> PermDitValues<LHSS> {
    pub fn new(perm: [u8; LHSS], locs: Vec<usize>) -> Self {
        PermDitValues { perm, locs }
    }
}

impl<const LHSS: usize, I: BitInt> BitStateOp<I> for PermDitValues<LHSS> {
    #[inline]
    fn apply(&self, state: I) -> I {
        let mut out = state;
        for &loc in &self.locs {
            let v = DitManip::<LHSS>::get_dit(state, loc);
            out = DitManip::<LHSS>::set_dit(out, self.perm[v] as usize, loc);
        }
        out
    }
}

impl<const LHSS: usize, I: BitInt> FermionicBitStateOp<I> for PermDitValues<LHSS> {}

impl<const LHSS: usize> Compose for PermDitValues<LHSS> {
    /// `(a ∘ b)[v] = a[b[v]]`. Requires identical `locs`; panics otherwise.
    fn compose(&self, other: &Self) -> Self {
        assert_eq!(
            self.locs, other.locs,
            "Compose on PermDitValues requires identical `locs`"
        );
        let mut combined = [0u8; LHSS];
        for (v, slot) in combined.iter_mut().enumerate() {
            *slot = self.perm[other.perm[v] as usize];
        }
        PermDitValues::new(combined, self.locs.clone())
    }
}

// ---------------------------------------------------------------------------
// DynamicPermDitValues — runtime LHSS variant
// ---------------------------------------------------------------------------

/// Runtime dit-value permutation.
///
/// Mirrors `dynamic_perm_dit_values` from `dit_perm.hpp`.
#[derive(Clone, Debug)]
pub struct DynamicPermDitValues {
    perm: Vec<u8>,
    locs: Vec<usize>,
    manip: DynamicDitManip,
}

impl DynamicPermDitValues {
    pub fn new(lhss: usize, perm: Vec<u8>, locs: Vec<usize>) -> Self {
        assert_eq!(
            perm.len(),
            lhss,
            "perm must have exactly lhss={lhss} entries"
        );
        DynamicPermDitValues {
            perm,
            locs,
            manip: DynamicDitManip::new(lhss),
        }
    }
}

impl<I: BitInt> BitStateOp<I> for DynamicPermDitValues {
    #[inline]
    fn apply(&self, state: I) -> I {
        let mut out = state;
        for &loc in &self.locs {
            let v = self.manip.get_dit(state, loc);
            out = self.manip.set_dit(out, self.perm[v] as usize, loc);
        }
        out
    }
}

impl<I: BitInt> FermionicBitStateOp<I> for DynamicPermDitValues {}

impl Compose for DynamicPermDitValues {
    /// `(a ∘ b)[v] = a[b[v]]`. Requires identical `lhss` and `locs`; panics otherwise.
    fn compose(&self, other: &Self) -> Self {
        assert_eq!(
            self.perm.len(),
            other.perm.len(),
            "Compose on DynamicPermDitValues requires identical `lhss`"
        );
        assert_eq!(
            self.locs, other.locs,
            "Compose on DynamicPermDitValues requires identical `locs`"
        );
        let combined: Vec<u8> = (0..self.perm.len())
            .map(|v| self.perm[other.perm[v] as usize])
            .collect();
        DynamicPermDitValues::new(self.perm.len(), combined, self.locs.clone())
    }
}

// ---------------------------------------------------------------------------
// PermDitMask — XOR with a fixed mask (parity-like operations)
// ---------------------------------------------------------------------------

/// Applies a bit-flip symmetry via XOR: `s → s ^ mask`.
///
/// Mirrors `perm_dit_mask` from `dit_perm.hpp`.
#[derive(Clone, Copy, Debug)]
pub struct PermDitMask<I: BitInt> {
    mask: I,
}

impl<I: BitInt> PermDitMask<I> {
    pub fn new(mask: I) -> Self {
        PermDitMask { mask }
    }
}

impl<I: BitInt> BitStateOp<I> for PermDitMask<I> {
    #[inline]
    fn apply(&self, state: I) -> I {
        state ^ self.mask
    }
}

impl<I: BitInt> FermionicBitStateOp<I> for PermDitMask<I> {}

impl<I: BitInt> Compose for PermDitMask<I> {
    /// XOR masks compose by XOR: `(a ∘ b)(state) = a(b(state)) = state ^ b.mask ^ a.mask`.
    #[inline]
    fn compose(&self, other: &Self) -> Self {
        PermDitMask::new(self.mask ^ other.mask)
    }
}

// ---------------------------------------------------------------------------
// SignedPermDitMask — bit-flip with per-site fermion signs (particle-hole)
// ---------------------------------------------------------------------------

/// Bit-flip at every set bit in `mask`, with a per-site fermion sign.
///
/// Used for the particle-hole transformation on fermionic bases. When
/// applied to `state`:
///
/// - [`apply`]: returns `state ^ mask` (same as [`PermDitMask`]).
/// - [`fermion_sign`]: returns `∏_{i: bit i of state is set} sign[i]` —
///   a product over the sites that are *occupied* in the input state.
///
/// The per-site `sign` array has length `n_sites`. For a 1D fermionic
/// chain the canonical choice is `sign[i] = (-1)^i`; in higher
/// dimensions or for non-uniform lattices the sign structure depends
/// on the model, so the user must supply it explicitly. See
/// [QuSpin/QuSpin#390](https://github.com/QuSpin/QuSpin/issues/390) for
/// the Python-side history of this requirement.
///
/// [`apply`]: BitStateOp::apply
/// [`fermion_sign`]: FermionicBitStateOp::fermion_sign
#[derive(Clone, Debug)]
pub struct SignedPermDitMask<I: BitInt> {
    mask: I,
    sign: Vec<f64>,
}

impl<I: BitInt> SignedPermDitMask<I> {
    /// Construct with an explicit per-site sign array.
    ///
    /// # Panics
    /// Panics if `sign` is empty (the caller must provide at least one
    /// site's sign for the type to be meaningful).
    pub fn new(mask: I, sign: Vec<f64>) -> Self {
        assert!(!sign.is_empty(), "sign must be non-empty");
        SignedPermDitMask { mask, sign }
    }

    /// Convenience: `(-1)^i` per-site sign for a 1D fermionic chain of
    /// length `n_sites`.
    pub fn default_1d(mask: I, n_sites: usize) -> Self {
        let sign = (0..n_sites)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        Self::new(mask, sign)
    }

    /// Borrow the per-site sign array.
    pub fn sign(&self) -> &[f64] {
        &self.sign
    }

    /// Borrow the bit-flip mask.
    pub fn mask(&self) -> I {
        self.mask
    }
}

impl<I: BitInt> BitStateOp<I> for SignedPermDitMask<I> {
    #[inline]
    fn apply(&self, state: I) -> I {
        state ^ self.mask
    }
}

impl<I: BitInt> FermionicBitStateOp<I> for SignedPermDitMask<I> {
    #[inline]
    fn fermion_sign(&self, state: I) -> f64 {
        let mut s = 1.0;
        for (i, &si) in self.sign.iter().enumerate() {
            if (state & (I::from_u64(1) << i)) != I::from_u64(0) {
                s *= si;
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// BenesPermDitLocations — hardware-fast site permutation via Benes network
// ---------------------------------------------------------------------------

/// Site-permutation backed by a Benes permutation network.
///
/// For LHSS = 2 (hardcore bosons / fermions) this replaces the naive O(n)
/// loop in [`PermDitLocations`] with an O(log n) butterfly evaluation.
///
/// When `fermionic = true` and `lhss = 2`, `sign_masks` holds precomputed
/// Jordan-Wigner sign masks for efficient fermionic sign computation.
#[derive(Clone)]
pub struct BenesPermDitLocations<B: BitInt> {
    benes: BenesNetwork<B>,
    /// `sign_masks[j]` has bit `i` set iff `i < j` AND `perm[i] > perm[j]`.
    /// Empty if `fermionic = false` or `lhss != 2`.
    /// Inline capacity covers u32 (≤32 sites) and u64 (≤64 sites) without heap allocation.
    sign_masks: SmallVec<[B; 64]>,
    n_sites: usize,
}

impl<B: BitInt> BenesPermDitLocations<B> {
    /// Construct a Benes-backed site permutation.
    ///
    /// `lhss`: local Hilbert-space size (bits per site = ⌈log₂(lhss)⌉).
    /// `perm[src] = dst` maps site `src` to site `dst`.
    /// `fermionic`: if `true` AND `lhss == 2`, precompute fermionic sign masks.
    pub fn new(lhss: usize, perm: &[usize], fermionic: bool) -> Self {
        let n_sites = perm.len();

        // bits_per_dit = ceil(log2(lhss))
        let bits_per_dit = if lhss <= 1 {
            0
        } else {
            (usize::BITS - (lhss - 1).leading_zeros()) as usize
        };

        let bits = B::BITS as usize;

        // Build bit-level target permutation.
        // Convention: c_tgt[dst_bit] = src_bit (output bit dst comes from input bit src).
        // perm[src_site] = dst_site: bits of src_site appear at dst_site in output.
        let mut c_tgt: Vec<Option<usize>> = vec![None; bits];
        for (src, &dst) in perm.iter().enumerate() {
            for j in 0..bits_per_dit {
                let src_bit = src * bits_per_dit + j;
                let dst_bit = dst * bits_per_dit + j;
                c_tgt[dst_bit] = Some(src_bit);
            }
        }
        // Fill remaining positions with identity.
        for (i, entry) in c_tgt.iter_mut().enumerate().take(bits) {
            if entry.is_none() {
                *entry = Some(i);
            }
        }

        let benes = gen_benes::<B>(&c_tgt);

        let sign_masks = if fermionic && lhss == 2 {
            compute_sign_masks::<B>(perm)
        } else {
            SmallVec::new()
        };

        BenesPermDitLocations {
            benes,
            sign_masks,
            n_sites,
        }
    }

    /// Compute the fermionic Jordan-Wigner sign for a given state.
    ///
    /// Returns `+1.0` or `-1.0`. Always returns `1.0` if `fermionic = false`.
    ///
    /// The sign is `(-1)^(number of inversions in the permutation restricted
    /// to the occupied sites of `state`)`.
    pub fn fermionic_sign(&self, state: B) -> f64 {
        if self.sign_masks.is_empty() {
            return 1.0;
        }
        let mut parity = 0u32;
        for (j, &mask) in self.sign_masks.iter().enumerate() {
            let occupied_j = ((state >> j) & B::from_u64(1)).to_usize() as u32 & 1;
            let count = (state & mask).count_ones() & 1;
            parity ^= count * occupied_j;
        }
        if parity == 0 { 1.0 } else { -1.0 }
    }

    /// Number of sites.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }
}

/// Precompute Jordan-Wigner sign masks from a site permutation.
///
/// `masks[j]` has bit `i` set iff `i < j` AND `perm[i] > perm[j]`.
fn compute_sign_masks<B: BitInt>(perm: &[usize]) -> SmallVec<[B; 64]> {
    let n = perm.len();
    let mut masks: SmallVec<[B; 64]> = smallvec::smallvec![B::from_u64(0); n];
    for j in 0..n {
        for i in 0..j {
            if perm[i] > perm[j] {
                masks[j] = masks[j] | (B::from_u64(1) << i);
            }
        }
    }
    masks
}

impl<B: BitInt> BitStateOp<B> for BenesPermDitLocations<B> {
    #[inline]
    fn apply(&self, state: B) -> B {
        benes_fwd(&self.benes, state)
    }
}

impl<B: BitInt> FermionicBitStateOp<B> for BenesPermDitLocations<B> {
    /// Jordan-Wigner sign from the permutation's inversion count restricted
    /// to occupied sites of `state`. Matches the existing
    /// [`fermionic_sign`](BenesPermDitLocations::fermionic_sign) method.
    #[inline]
    fn fermion_sign(&self, state: B) -> f64 {
        self.fermionic_sign(state)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- PermDitLocations ---

    #[test]
    fn perm_dit_locations_identity() {
        let perm = PermDitLocations::new(2, &[0, 1, 2, 3]);
        let s: u32 = 0b1011;
        assert_eq!(perm.apply(s), s);
    }

    #[test]
    fn perm_dit_locations_swap_lhss2() {
        // perm = [1, 0, 2, 3]: site 0 → site 1, site 1 → site 0
        let perm = PermDitLocations::new(2, &[1, 0, 2, 3]);
        let s: u32 = 0b0001; // site 0=1, rest 0
        let expected: u32 = 0b0010; // site 1=1, rest 0
        assert_eq!(perm.apply(s), expected);
    }

    #[test]
    fn perm_dit_locations_cycle_lhss3() {
        // lhss=3 (2 bits/site), cycle [0,1,2]: 0→1, 1→2, 2→0
        let perm = PermDitLocations::new(3, &[1, 2, 0]);
        let manip = DynamicDitManip::new(3);
        let mut s: u64 = 0;
        s = manip.set_dit(s, 1, 0); // site 0 = 1
        s = manip.set_dit(s, 2, 1); // site 1 = 2
        s = manip.set_dit(s, 0, 2); // site 2 = 0

        let out = perm.apply(s);
        // after 0→1, 1→2, 2→0: site 1=1, site 2=2, site 0=0
        assert_eq!(manip.get_dit(out, 0), 0);
        assert_eq!(manip.get_dit(out, 1), 1);
        assert_eq!(manip.get_dit(out, 2), 2);
    }

    // --- PermDitValues ---

    #[test]
    fn perm_dit_values_swap_lhss3() {
        let manip = DynamicDitManip::new(3);
        let mut s: u64 = 0;
        s = manip.set_dit(s, 0, 0);
        s = manip.set_dit(s, 1, 1);
        s = manip.set_dit(s, 2, 2);

        let perm = PermDitValues::<3>::new([0, 2, 1], vec![0, 1, 2]);
        let out = perm.apply(s);
        assert_eq!(manip.get_dit(out, 0), 0); // 0→0
        assert_eq!(manip.get_dit(out, 1), 2); // 1→2
        assert_eq!(manip.get_dit(out, 2), 1); // 2→1
    }

    // --- DynamicPermDitValues ---

    #[test]
    fn dynamic_perm_dit_values_matches_static() {
        let manip = DynamicDitManip::new(3);
        let mut s: u64 = 0;
        s = manip.set_dit(s, 0, 0);
        s = manip.set_dit(s, 1, 1);
        s = manip.set_dit(s, 2, 2);

        let static_perm = PermDitValues::<3>::new([0, 2, 1], vec![0, 1, 2]);
        let dynamic_perm = DynamicPermDitValues::new(3, vec![0, 2, 1], vec![0, 1, 2]);

        assert_eq!(static_perm.apply(s), dynamic_perm.apply(s));
    }

    // --- PermDitMask ---

    #[test]
    fn perm_dit_mask_xor() {
        let mask = PermDitMask::new(0b1010u32);
        let s: u32 = 0b1100;
        assert_eq!(mask.apply(s), 0b0110);
        // involution: apply twice = identity
        assert_eq!(mask.apply(mask.apply(s)), s);
    }

    // --- BenesPermDitLocations ---

    #[test]
    fn benes_perm_dit_locations_identity() {
        let perm = BenesPermDitLocations::<u32>::new(2, &[0, 1, 2, 3], false);
        let s: u32 = 0b1011;
        assert_eq!(perm.apply(s), s);
    }

    #[test]
    fn benes_perm_dit_locations_swap_lhss2() {
        // perm = [1, 0, 2, 3]: site 0 → site 1, site 1 → site 0
        let benes_perm = BenesPermDitLocations::<u32>::new(2, &[1, 0, 2, 3], false);
        let naive_perm = PermDitLocations::new(2, &[1, 0, 2, 3]);
        // Test all 4-bit states
        for s in 0u32..16u32 {
            assert_eq!(
                benes_perm.apply(s),
                naive_perm.apply(s),
                "mismatch for s={s:#06b}"
            );
        }
    }

    #[test]
    fn benes_perm_dit_locations_matches_naive_4sites() {
        // Translation: perm = [1,2,3,0]
        let perm = &[1, 2, 3, 0];
        let benes = BenesPermDitLocations::<u32>::new(2, perm, false);
        let naive = PermDitLocations::new(2, perm);
        for s in 0u32..16u32 {
            assert_eq!(benes.apply(s), naive.apply(s), "mismatch at s={s:#06b}");
        }
    }

    #[test]
    fn benes_fermionic_sign_identity_is_plus_one() {
        // Identity permutation: no inversions, sign = +1 for all states.
        let perm = BenesPermDitLocations::<u32>::new(2, &[0, 1, 2, 3], true);
        for s in 0u32..16u32 {
            assert_eq!(
                perm.fermionic_sign(s),
                1.0,
                "sign should be +1 for identity"
            );
        }
    }

    #[test]
    fn benes_fermionic_sign_reverse_perm() {
        // perm = [2, 1, 0] (reverse of 3 sites).
        // Example from the spec: state=0b101 (sites 0,2 occupied) → sign = -1.
        let perm = BenesPermDitLocations::<u32>::new(2, &[2, 1, 0], true);
        // state = 0b101: bit 0 = 1, bit 2 = 1
        let state = 0b101u32;
        assert_eq!(perm.fermionic_sign(state), -1.0);
        // state = 0b110: sites 1,2 occupied. perm=[2,1,0] restricted to {1,2}: (1→1, 2→0). inversion: 2→0 before 1→1, that's 1 inversion. sign=-1.
        let state2 = 0b110u32;
        assert_eq!(perm.fermionic_sign(state2), -1.0);
        // state = 0b011: sites 0,1 occupied. perm restricted to {0,1}: (0→2, 1→1). inversion: 2>1 so perm[0]=2 > perm[1]=1. sign=-1.
        let state3 = 0b011u32;
        assert_eq!(perm.fermionic_sign(state3), -1.0);
    }

    #[test]
    fn benes_fermionic_sign_no_fermionic_flag() {
        // Without fermionic=true, fermionic_sign always returns 1.
        let perm = BenesPermDitLocations::<u32>::new(2, &[2, 1, 0], false);
        for s in 0u32..8u32 {
            assert_eq!(perm.fermionic_sign(s), 1.0);
        }
    }

    // --- SignedPermDitMask ---

    #[test]
    fn signed_mask_apply_matches_unsigned() {
        let mask = 0b1111u32;
        let unsigned = PermDitMask::new(mask);
        let signed = SignedPermDitMask::default_1d(mask, 4);
        for s in 0u32..16u32 {
            assert_eq!(
                BitStateOp::apply(&unsigned, s),
                BitStateOp::apply(&signed, s)
            );
        }
    }

    #[test]
    fn signed_mask_sign_on_vacuum_is_one() {
        let op = SignedPermDitMask::default_1d(0b1111u32, 4);
        // No occupied sites → empty product → +1.
        assert_eq!(op.fermion_sign(0u32), 1.0);
    }

    #[test]
    fn signed_mask_sign_products() {
        // sign = [1, -1, 1, -1]
        let op = SignedPermDitMask::new(0b1111u32, vec![1.0, -1.0, 1.0, -1.0]);

        // |0001⟩: only site 0 occupied → sign = 1.
        assert_eq!(op.fermion_sign(0b0001u32), 1.0);

        // |0010⟩: only site 1 occupied → sign = -1.
        assert_eq!(op.fermion_sign(0b0010u32), -1.0);

        // |1010⟩: sites 1 and 3 occupied → sign = -1 · -1 = 1.
        assert_eq!(op.fermion_sign(0b1010u32), 1.0);

        // |1100⟩: sites 2 and 3 occupied → sign = 1 · -1 = -1.
        assert_eq!(op.fermion_sign(0b1100u32), -1.0);

        // |1111⟩: all four sites → sign = 1 · -1 · 1 · -1 = 1.
        assert_eq!(op.fermion_sign(0b1111u32), 1.0);
    }

    #[test]
    fn signed_mask_default_1d_alternates() {
        let op = SignedPermDitMask::default_1d(0b1111u32, 6);
        assert_eq!(op.sign(), &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    #[should_panic(expected = "sign must be non-empty")]
    fn signed_mask_rejects_empty_sign() {
        let _ = SignedPermDitMask::new(0u32, vec![]);
    }
}
