//! Shared orbit helpers used by [`SymBasis`](super::SymBasis).
//!
//! The walker iterates three type-homogeneous loops, one per storage
//! vector on `SymBasis`:
//!
//! - `lattice_only`: pure site permutations.
//! - `local_only`: pure local operators.
//! - `composite`: atomic site-permutation + local-op elements.
//!
//! The implicit identity element is injected by every function's
//! initialisation (self-image with character 1.0); the user never adds
//! it explicitly to any of the three vectors.
//!
//! When the enclosing basis's `fermionic` flag is true, the walker
//! multiplies [`FermionicBitStateOp::fermion_sign`] of the local-op
//! part into the running character; on non-fermionic bases the branch
//! is constant-`false` and LLVM eliminates the sign multiplication.

use super::lattice::{BenesLatticeElement, LatEl};
use num_complex::Complex;
// `LatEl` brings `apply_state` / `grp_char_for` on `BenesLatticeElement`.
// `FermionicBitStateOp: BitStateOp` so `apply` and `fermion_sign` are both
// reachable via a single trait import.
use quspin_bitbasis::{BitInt, FermionicBitStateOp};
use smallvec::SmallVec;

/// Inline capacity for the orbit image buffer.
///
/// Covers the common cases without heap allocation (e.g. 32-site
/// translation × 1 local op = 64 images).
const ORBIT_INLINE_CAP: usize = 512;

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

/// Enumerate all group-orbit images of `state`.
///
/// Total image count is
/// `1 + lattice_only.len() + local_only.len() + composite.len()`
/// (the `1` is the implicit identity).
///
/// Returns a stack-allocated [`SmallVec`] that avoids heap allocation
/// for orbits up to [`ORBIT_INLINE_CAP`] elements.
pub(crate) fn iter_images<B, L>(
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[(Complex<f64>, L)],
    composite: &[(BenesLatticeElement<B>, L)],
    fermionic: bool,
    state: B,
) -> SmallVec<[(B, Complex<f64>); ORBIT_INLINE_CAP]>
where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let mut images: SmallVec<[(B, Complex<f64>); ORBIT_INLINE_CAP]> = SmallVec::new();

    // Implicit identity.
    images.push((state, Complex::new(1.0, 0.0)));

    // Pure lattice.
    for lat in lattice_only {
        let s = lat.apply_state(state);
        let c = lat.grp_char_for(state);
        images.push((s, c));
    }

    // Pure local.
    for (chi, loc) in local_only {
        let s = loc.apply(state);
        let mut c = *chi;
        if fermionic {
            c *= Complex::new(loc.fermion_sign(state), 0.0);
        }
        images.push((s, c));
    }

    // Composite: perm first, then local. Character lives on the
    // BenesLatticeElement and is evaluated against the pre-permutation
    // state (so its Jordan-Wigner sign matches the permutation's action).
    for (lat, loc) in composite {
        let perm_state = lat.apply_state(state);
        let s = loc.apply(perm_state);
        let mut c = lat.grp_char_for(state);
        if fermionic {
            c *= Complex::new(loc.fermion_sign(perm_state), 0.0);
        }
        images.push((s, c));
    }

    images
}

/// Return the orbit representative (largest state in the orbit) and the
/// group-character coefficient accumulated to reach it.
pub(crate) fn get_refstate<B, L>(
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[(Complex<f64>, L)],
    composite: &[(BenesLatticeElement<B>, L)],
    fermionic: bool,
    state: B,
) -> (B, Complex<f64>)
where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let mut best = state;
    let mut best_coeff = Complex::new(1.0, 0.0);
    for (s, c) in iter_images(lattice_only, local_only, composite, fermionic, state) {
        let cond = s > best;
        best = best.max(s);
        best_coeff = if cond { c } else { best_coeff };
    }
    (best, best_coeff)
}

/// Return the orbit representative and the orbit norm.
///
/// The norm is the number of orbit images that equal `state`.
pub(crate) fn check_refstate<B, L>(
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[(Complex<f64>, L)],
    composite: &[(BenesLatticeElement<B>, L)],
    fermionic: bool,
    state: B,
) -> (B, f64)
where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let mut ref_state = state;
    let mut norm = 0u32;
    for (s, _) in iter_images(lattice_only, local_only, composite, fermionic, state) {
        ref_state = ref_state.max(s);
        norm += (s == state) as u32;
    }
    (ref_state, norm as f64)
}

// ---------------------------------------------------------------------------
// Batch helpers — rearranged so outer loops are group elements, inner
// loops are the batch (amortises orbit traversal, enables SIMD).
// ---------------------------------------------------------------------------

/// Batch variant of [`get_refstate`].
///
/// # Panics
///
/// Panics if `states.len() != out.len()`.
pub(crate) fn get_refstate_batch<B, L>(
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[(Complex<f64>, L)],
    composite: &[(BenesLatticeElement<B>, L)],
    fermionic: bool,
    states: &[B],
    out: &mut [(B, Complex<f64>)],
) where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let n = states.len();
    assert_eq!(n, out.len());

    // Init: each state is its own representative with identity character.
    for (state, o) in states.iter().zip(out.iter_mut()) {
        o.0 = *state;
        o.1 = Complex::new(1.0, 0.0);
    }

    // Pure lattice.
    for lat in lattice_only {
        for (state, o) in states.iter().zip(out.iter_mut()) {
            let s = lat.apply_state(*state);
            let c = lat.grp_char_for(*state);
            let cond = s > o.0;
            o.0 = o.0.max(s);
            o.1 = if cond { c } else { o.1 };
        }
    }

    // Pure local.
    for (chi, loc) in local_only {
        for (state, o) in states.iter().zip(out.iter_mut()) {
            let s = loc.apply(*state);
            let mut c = *chi;
            if fermionic {
                c *= Complex::new(loc.fermion_sign(*state), 0.0);
            }
            let cond = s > o.0;
            o.0 = o.0.max(s);
            o.1 = if cond { c } else { o.1 };
        }
    }

    // Composite: perm then local, single character per element.
    for (lat, loc) in composite {
        for (state, o) in states.iter().zip(out.iter_mut()) {
            let perm_state = lat.apply_state(*state);
            let s = loc.apply(perm_state);
            let mut c = lat.grp_char_for(*state);
            if fermionic {
                c *= Complex::new(loc.fermion_sign(perm_state), 0.0);
            }
            let cond = s > o.0;
            o.0 = o.0.max(s);
            o.1 = if cond { c } else { o.1 };
        }
    }
}

/// Batch variant of [`check_refstate`].
///
/// # Panics
///
/// Panics if `states.len() != out.len()`.
pub(crate) fn check_refstate_batch<B, L>(
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[(Complex<f64>, L)],
    composite: &[(BenesLatticeElement<B>, L)],
    _fermionic: bool,
    states: &[B],
    out: &mut [(B, f64)],
) where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let n = states.len();
    assert_eq!(n, out.len());

    // Init representatives; norms start at 1 (self-image from implicit identity).
    for (state, o) in states.iter().zip(out.iter_mut()) {
        o.0 = *state;
    }
    let mut norms = vec![1u32; n]; // identity contributes +1 to every state's norm

    // Pure lattice.
    for lat in lattice_only {
        for ((state, o), norm) in states.iter().zip(out.iter_mut()).zip(norms.iter_mut()) {
            let s = lat.apply_state(*state);
            o.0 = o.0.max(s);
            *norm += (s == *state) as u32;
        }
    }

    // Pure local.
    for (_, loc) in local_only {
        for ((state, o), norm) in states.iter().zip(out.iter_mut()).zip(norms.iter_mut()) {
            let s = loc.apply(*state);
            o.0 = o.0.max(s);
            *norm += (s == *state) as u32;
        }
    }

    // Composite: perm then local.
    for (lat, loc) in composite {
        for ((state, o), norm) in states.iter().zip(out.iter_mut()).zip(norms.iter_mut()) {
            let s = loc.apply(lat.apply_state(*state));
            o.0 = o.0.max(s);
            *norm += (s == *state) as u32;
        }
    }

    // Write norms as f64.
    for (o, norm) in out.iter_mut().zip(norms.iter()) {
        o.1 = *norm as f64;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use quspin_bitbasis::{BenesPermDitLocations, PermDitMask};

    fn one() -> Complex<f64> {
        Complex::new(1.0, 0.0)
    }

    fn minus_one() -> Complex<f64> {
        Complex::new(-1.0, 0.0)
    }

    /// Build a `BenesLatticeElement` that applies a site permutation with
    /// character `char_` over `n_sites` sites of LHSS 2.
    fn lat(char_: Complex<f64>, perm: &[usize]) -> BenesLatticeElement<u32> {
        let n_sites = perm.len();
        let op = BenesPermDitLocations::<u32>::new(2, perm, false);
        BenesLatticeElement::new(char_, op, n_sites)
    }

    /// XOR bit-flip local op with its character.
    fn local_xor(char_: Complex<f64>, mask: u32) -> (Complex<f64>, PermDitMask<u32>) {
        (char_, PermDitMask::new(mask))
    }

    // --- iter_images ---------------------------------------------------------

    #[test]
    fn iter_images_empty_group_is_identity_only() {
        let images = iter_images::<u32, PermDitMask<u32>>(&[], &[], &[], false, 0b01u32);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].0, 0b01u32);
        assert_eq!(images[0].1, one());
    }

    #[test]
    fn iter_images_lattice_only() {
        let lattice = vec![lat(one(), &[1, 0])];
        let images = iter_images::<u32, PermDitMask<u32>>(&lattice, &[], &[], false, 0b01u32);
        assert_eq!(images.len(), 2);
        let states: Vec<u32> = images.iter().map(|(s, _)| *s).collect();
        assert!(states.contains(&0b01u32)); // identity
        assert!(states.contains(&0b10u32)); // swap
    }

    #[test]
    fn iter_images_local_only() {
        let local = vec![local_xor(minus_one(), 0b11u32)];
        let images = iter_images(&[], &local, &[], false, 0b01u32);
        assert_eq!(images.len(), 2);
        assert!(images.iter().any(|(s, c)| *s == 0b01u32 && *c == one()));
        assert!(
            images
                .iter()
                .any(|(s, c)| *s == 0b10u32 && *c == minus_one())
        );
    }

    #[test]
    fn iter_images_composite_pz_example() {
        // The motivating case: reflection P = [1, 0] composed with spin-flip
        // Z = XOR 0b11 as a single composite element. Group is {I, PZ};
        // images of 0b01 are {0b01, 0b01} — same state twice! (P flips the
        // bits position, Z flips the values, so PZ on 0b01 = swap-then-flip.
        // Actually: state 0b01 → swap [1,0] → 0b10 → XOR 11 → 0b01.)
        let lattice = vec![lat(one(), &[1, 0])];
        let local = vec![local_xor(one(), 0b11u32)];
        let composite = vec![(lat(one(), &[1, 0]), PermDitMask::new(0b11u32))];

        // Three separate vectors produce three separate images (+ identity).
        // Group {I, P, Z, PZ} = 4 elements.
        let images = iter_images(&lattice, &local, &composite, false, 0b01u32);
        assert_eq!(images.len(), 4);

        // Composite-only — user's actual case: no P, no Z alone, just PZ.
        let images_pz_only = iter_images::<u32, PermDitMask<u32>>(
            &[],
            &[],
            &composite, // just (P, Z) atomic
            false,
            0b01u32,
        );
        assert_eq!(images_pz_only.len(), 2); // {I, PZ}
    }

    // --- get_refstate / check_refstate --------------------------------------

    #[test]
    fn get_refstate_returns_max_image() {
        let lattice = vec![lat(one(), &[1, 0])];
        let (ref_s, _) = get_refstate::<u32, PermDitMask<u32>>(&lattice, &[], &[], false, 0b01u32);
        assert_eq!(ref_s, 0b10u32);
    }

    #[test]
    fn get_refstate_character_from_winning_image() {
        let lattice = vec![lat(minus_one(), &[1, 0])];
        let (ref_s, c) = get_refstate::<u32, PermDitMask<u32>>(&lattice, &[], &[], false, 0b01u32);
        assert_eq!(ref_s, 0b10u32);
        assert_eq!(c, minus_one());
    }

    #[test]
    fn check_refstate_identity_contributes_to_norm() {
        // Empty group → one self-image → norm = 1.
        let (ref_s, norm) = check_refstate::<u32, PermDitMask<u32>>(&[], &[], &[], false, 0b01u32);
        assert_eq!(ref_s, 0b01u32);
        assert_eq!(norm, 1.0);
    }

    #[test]
    fn check_refstate_z2_orbit_max_has_norm_2() {
        // P = swap; state 0b11 maps to itself under P.
        // Group: {I, P}. Images of 0b11: {0b11, 0b11} → norm 2.
        let lattice = vec![lat(one(), &[1, 0])];
        let (ref_s, norm) =
            check_refstate::<u32, PermDitMask<u32>>(&lattice, &[], &[], false, 0b11u32);
        assert_eq!(ref_s, 0b11u32);
        assert_eq!(norm, 2.0);
    }

    // --- batch matches scalar -----------------------------------------------

    #[test]
    fn batch_matches_scalar_lattice_only() {
        let lattice = vec![lat(one(), &[0, 1]), lat(one(), &[1, 0])];
        let states: Vec<u32> = (0u32..4).collect();
        let mut out = vec![(0u32, 0.0); states.len()];
        check_refstate_batch::<u32, PermDitMask<u32>>(&lattice, &[], &[], false, &states, &mut out);
        for (state, (batch_ref, batch_norm)) in states.iter().zip(out.iter()) {
            let (scalar_ref, scalar_norm) =
                check_refstate::<u32, PermDitMask<u32>>(&lattice, &[], &[], false, *state);
            assert_eq!(*batch_ref, scalar_ref);
            assert_eq!(*batch_norm, scalar_norm);
        }
    }

    #[test]
    fn batch_matches_scalar_with_local_and_composite() {
        let lattice = vec![lat(one(), &[1, 0])];
        let local = vec![local_xor(one(), 0b11u32)];
        let composite = vec![(lat(one(), &[1, 0]), PermDitMask::new(0b11u32))];
        let states: Vec<u32> = (0u32..4).collect();
        let mut out = vec![(0u32, 0.0); states.len()];
        check_refstate_batch(&lattice, &local, &composite, false, &states, &mut out);
        for (state, (batch_ref, batch_norm)) in states.iter().zip(out.iter()) {
            let (scalar_ref, scalar_norm) =
                check_refstate(&lattice, &local, &composite, false, *state);
            assert_eq!(*batch_ref, scalar_ref);
            assert_eq!(*batch_norm, scalar_norm);
        }
    }

    #[test]
    fn batch_empty_input_is_noop() {
        let lattice = vec![lat(one(), &[0, 1])];
        let mut out: Vec<(u32, f64)> = vec![];
        check_refstate_batch::<u32, PermDitMask<u32>>(&lattice, &[], &[], false, &[], &mut out); // must not panic
    }
}
