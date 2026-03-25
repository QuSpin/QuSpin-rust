use super::int::BitInt;
use super::manip::{DitManip, DynamicDitManip};

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

// ---------------------------------------------------------------------------
// HigherSpinInv — spin inversion s → lhss - s - 1
// ---------------------------------------------------------------------------

/// Compile-time spin inversion for `LHSS`-valued dits.
///
/// Maps each dit value `v → LHSS - v - 1` at the specified sites.
///
/// Mirrors `higher_spin_inv<bitset_t, lhss>` from `dit_perm.hpp`.
#[derive(Clone, Debug)]
pub struct HigherSpinInv<const LHSS: usize> {
    locs: Vec<usize>,
}

impl<const LHSS: usize> HigherSpinInv<LHSS> {
    pub fn new(locs: Vec<usize>) -> Self {
        HigherSpinInv { locs }
    }
}

impl<const LHSS: usize, I: BitInt> BitStateOp<I> for HigherSpinInv<LHSS> {
    #[inline]
    fn apply(&self, state: I) -> I {
        let mut out = state;
        for &loc in &self.locs {
            let v = DitManip::<LHSS>::get_dit(state, loc);
            out = DitManip::<LHSS>::set_dit(out, LHSS - v - 1, loc);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// DynamicHigherSpinInv — runtime LHSS variant
// ---------------------------------------------------------------------------

/// Runtime spin inversion.
///
/// Mirrors `dynamic_higher_spin_inv` from `dit_perm.hpp`.
#[derive(Clone, Debug)]
pub struct DynamicHigherSpinInv {
    locs: Vec<usize>,
    manip: DynamicDitManip,
}

impl DynamicHigherSpinInv {
    pub fn new(lhss: usize, locs: Vec<usize>) -> Self {
        DynamicHigherSpinInv {
            locs,
            manip: DynamicDitManip::new(lhss),
        }
    }
}

impl<I: BitInt> BitStateOp<I> for DynamicHigherSpinInv {
    #[inline]
    fn apply(&self, state: I) -> I {
        let lhss = self.manip.lhss;
        let mut out = state;
        for &loc in &self.locs {
            let v = self.manip.get_dit(state, loc);
            out = self.manip.set_dit(out, lhss - v - 1, loc);
        }
        out
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

    // --- HigherSpinInv ---

    #[test]
    fn higher_spin_inv_lhss3_involution() {
        let inv = HigherSpinInv::<3>::new(vec![0, 1, 2]);
        let manip = DynamicDitManip::new(3);
        let mut s: u64 = 0;
        s = manip.set_dit(s, 0, 0);
        s = manip.set_dit(s, 1, 1);
        s = manip.set_dit(s, 2, 2);

        let out = inv.apply(s);
        assert_eq!(manip.get_dit(out, 0), 2); // 2-0=2
        assert_eq!(manip.get_dit(out, 1), 1); // 2-1=1
        assert_eq!(manip.get_dit(out, 2), 0); // 2-2=0

        assert_eq!(inv.apply(out), s); // involution
    }

    #[test]
    fn higher_spin_inv_lhss2_spin_flip() {
        let inv = HigherSpinInv::<2>::new(vec![0, 1, 2, 3]);
        let s: u32 = 0b1010;
        let out = inv.apply(s);
        assert_eq!(out, 0b0101);
        assert_eq!(inv.apply(out), s);
    }

    // --- DynamicHigherSpinInv ---

    #[test]
    fn dynamic_higher_spin_inv_matches_static() {
        let static_inv = HigherSpinInv::<3>::new(vec![0, 1, 2]);
        let dynamic_inv = DynamicHigherSpinInv::new(3, vec![0, 1, 2]);

        let manip = DynamicDitManip::new(3);
        let mut s: u64 = 0;
        s = manip.set_dit(s, 0, 0);
        s = manip.set_dit(s, 1, 1);
        s = manip.set_dit(s, 2, 2);

        assert_eq!(static_inv.apply(s), dynamic_inv.apply(s));
    }

    #[test]
    fn dynamic_higher_spin_inv_partial_sites() {
        let manip = DynamicDitManip::new(4);
        let inv = DynamicHigherSpinInv::new(4, vec![1, 3]);

        let mut s: u64 = 0;
        s = manip.set_dit(s, 1, 0); // site 0 = 1 (untouched)
        s = manip.set_dit(s, 2, 1); // site 1 = 2 → 4-2-1 = 1
        s = manip.set_dit(s, 3, 2); // site 2 = 3 (untouched)
        s = manip.set_dit(s, 0, 3); // site 3 = 0 → 4-0-1 = 3

        let out = inv.apply(s);
        assert_eq!(manip.get_dit(out, 1), 1); // 4-2-1
        assert_eq!(manip.get_dit(out, 3), 3); // 4-0-1
    }
}
