//! Shared orbit helpers used by [`SymBasis`](super::SymBasis).
//!
//! The walker iterates three type-homogeneous loops, one per storage
//! vector on `SymBasis`:
//!
//! - `lattice_only`: pure site permutations.
//! - `local_only`: pure local operators.
//! - `composite`: atomic site-permutation + local-op elements.
//!
//! Every storage shape implements [`OrbitImage::apply`](super::lattice::OrbitImage::apply)
//! returning `(new_state, char)`, so each loop is identical:
//! `images.push(el.apply(state))`. Fermion signs are folded into `char`
//! by `apply`; the walker has no `fermionic` flag.
//!
//! The implicit identity element is injected by every function's
//! initialisation (self-image with character 1.0); the user never adds
//! it explicitly to any of the three vectors.

use super::lattice::{BenesLatticeElement, CompositeElement, LocalElement, OrbitImage};
use num_complex::Complex;
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
    local_only: &[LocalElement<L>],
    composite: &[CompositeElement<B, L>],
    state: B,
) -> SmallVec<[(B, Complex<f64>); ORBIT_INLINE_CAP]>
where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let mut images: SmallVec<[(B, Complex<f64>); ORBIT_INLINE_CAP]> = SmallVec::new();

    // Implicit identity.
    images.push((state, Complex::new(1.0, 0.0)));

    for el in lattice_only {
        images.push(el.apply(state));
    }
    for el in local_only {
        images.push(el.apply(state));
    }
    for el in composite {
        images.push(el.apply(state));
    }

    images
}

/// Return the orbit representative (largest state in the orbit) and the
/// group-character coefficient accumulated to reach it.
pub(crate) fn get_refstate<B, L>(
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[LocalElement<L>],
    composite: &[CompositeElement<B, L>],
    state: B,
) -> (B, Complex<f64>)
where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let mut best = state;
    let mut best_coeff = Complex::new(1.0, 0.0);
    for (s, c) in iter_images(lattice_only, local_only, composite, state) {
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
    local_only: &[LocalElement<L>],
    composite: &[CompositeElement<B, L>],
    state: B,
) -> (B, f64)
where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let mut ref_state = state;
    let mut norm = 0u32;
    for (s, _) in iter_images(lattice_only, local_only, composite, state) {
        ref_state = ref_state.max(s);
        norm += (s == state) as u32;
    }
    (ref_state, norm as f64)
}

// ---------------------------------------------------------------------------
// Batch helpers — outer loop over group elements, inner loop over the
// batch (amortises orbit traversal, enables auto-vectorisation).
// ---------------------------------------------------------------------------

/// Apply one element to the whole batch, updating `(best_state, best_coeff)`
/// in place via the `s > best` / max bookkeeping.
#[inline]
fn batch_update_best<B, E>(states: &[B], out: &mut [(B, Complex<f64>)], el: &E)
where
    B: BitInt,
    E: OrbitImage<B>,
{
    for (state, o) in states.iter().zip(out.iter_mut()) {
        let (s, c) = el.apply(*state);
        let cond = s > o.0;
        o.0 = o.0.max(s);
        o.1 = if cond { c } else { o.1 };
    }
}

/// Apply one element to the whole batch, accumulating self-image counts
/// into `norms` and tracking the running max representative.
#[inline]
fn batch_update_count<B, E>(states: &[B], out: &mut [(B, f64)], norms: &mut [u32], el: &E)
where
    B: BitInt,
    E: OrbitImage<B>,
{
    for ((state, o), norm) in states.iter().zip(out.iter_mut()).zip(norms.iter_mut()) {
        let (s, _) = el.apply(*state);
        o.0 = o.0.max(s);
        *norm += (s == *state) as u32;
    }
}

/// Batch variant of [`get_refstate`].
///
/// # Panics
///
/// Panics if `states.len() != out.len()`.
pub(crate) fn get_refstate_batch<B, L>(
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[LocalElement<L>],
    composite: &[CompositeElement<B, L>],
    states: &[B],
    out: &mut [(B, Complex<f64>)],
) where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    assert_eq!(states.len(), out.len());

    // Init: implicit identity image (state itself, character 1).
    for (state, o) in states.iter().zip(out.iter_mut()) {
        o.0 = *state;
        o.1 = Complex::new(1.0, 0.0);
    }

    for el in lattice_only {
        batch_update_best(states, out, el);
    }
    for el in local_only {
        batch_update_best(states, out, el);
    }
    for el in composite {
        batch_update_best(states, out, el);
    }
}

/// Batch variant of [`check_refstate`].
///
/// # Panics
///
/// Panics if `states.len() != out.len()`.
pub(crate) fn check_refstate_batch<B, L>(
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[LocalElement<L>],
    composite: &[CompositeElement<B, L>],
    states: &[B],
    out: &mut [(B, f64)],
) where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let n = states.len();
    assert_eq!(n, out.len());

    // Init representatives; norms start at 1 (the implicit identity is
    // every state's first self-image).
    for (state, o) in states.iter().zip(out.iter_mut()) {
        o.0 = *state;
    }
    let mut norms = vec![1u32; n];

    for el in lattice_only {
        batch_update_count(states, out, &mut norms, el);
    }
    for el in local_only {
        batch_update_count(states, out, &mut norms, el);
    }
    for el in composite {
        batch_update_count(states, out, &mut norms, el);
    }

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
    fn local_xor(char_: Complex<f64>, mask: u32) -> LocalElement<PermDitMask<u32>> {
        LocalElement::new(char_, PermDitMask::new(mask))
    }

    fn composite_lat_local(
        char_: Complex<f64>,
        perm: &[usize],
        mask: u32,
    ) -> CompositeElement<u32, PermDitMask<u32>> {
        CompositeElement::new(lat(char_, perm), PermDitMask::new(mask))
    }

    // --- iter_images ---------------------------------------------------------

    #[test]
    fn iter_images_empty_group_is_identity_only() {
        let images = iter_images::<u32, PermDitMask<u32>>(&[], &[], &[], 0b01u32);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].0, 0b01u32);
        assert_eq!(images[0].1, one());
    }

    #[test]
    fn iter_images_lattice_only() {
        let lattice = vec![lat(one(), &[1, 0])];
        let images = iter_images::<u32, PermDitMask<u32>>(&lattice, &[], &[], 0b01u32);
        assert_eq!(images.len(), 2);
        let states: Vec<u32> = images.iter().map(|(s, _)| *s).collect();
        assert!(states.contains(&0b01u32)); // identity
        assert!(states.contains(&0b10u32)); // swap
    }

    #[test]
    fn iter_images_local_only() {
        let local = vec![local_xor(minus_one(), 0b11u32)];
        let images = iter_images::<u32, PermDitMask<u32>>(&[], &local, &[], 0b01u32);
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
        // composite-only walker yields 2 images.
        let lattice = vec![lat(one(), &[1, 0])];
        let local = vec![local_xor(one(), 0b11u32)];
        let composite = vec![composite_lat_local(one(), &[1, 0], 0b11u32)];

        // Three separate vectors produce three separate images (+ identity).
        let images = iter_images(&lattice, &local, &composite, 0b01u32);
        assert_eq!(images.len(), 4);

        // Composite-only: just the {I, PZ} group of order 2.
        let images_pz_only = iter_images::<u32, PermDitMask<u32>>(&[], &[], &composite, 0b01u32);
        assert_eq!(images_pz_only.len(), 2);
    }

    // --- get_refstate / check_refstate --------------------------------------

    #[test]
    fn get_refstate_returns_max_image() {
        let lattice = vec![lat(one(), &[1, 0])];
        let (ref_s, _) = get_refstate::<u32, PermDitMask<u32>>(&lattice, &[], &[], 0b01u32);
        assert_eq!(ref_s, 0b10u32);
    }

    #[test]
    fn get_refstate_character_from_winning_image() {
        let lattice = vec![lat(minus_one(), &[1, 0])];
        let (ref_s, c) = get_refstate::<u32, PermDitMask<u32>>(&lattice, &[], &[], 0b01u32);
        assert_eq!(ref_s, 0b10u32);
        assert_eq!(c, minus_one());
    }

    #[test]
    fn check_refstate_identity_contributes_to_norm() {
        // Empty group → one self-image → norm = 1.
        let (ref_s, norm) = check_refstate::<u32, PermDitMask<u32>>(&[], &[], &[], 0b01u32);
        assert_eq!(ref_s, 0b01u32);
        assert_eq!(norm, 1.0);
    }

    #[test]
    fn check_refstate_z2_orbit_max_has_norm_2() {
        // P = swap; state 0b11 maps to itself under P.
        // Group: {I, P}. Images of 0b11: {0b11, 0b11} → norm 2.
        let lattice = vec![lat(one(), &[1, 0])];
        let (ref_s, norm) = check_refstate::<u32, PermDitMask<u32>>(&lattice, &[], &[], 0b11u32);
        assert_eq!(ref_s, 0b11u32);
        assert_eq!(norm, 2.0);
    }

    // --- batch matches scalar -----------------------------------------------

    #[test]
    fn batch_matches_scalar_lattice_only() {
        let lattice = vec![lat(one(), &[0, 1]), lat(one(), &[1, 0])];
        let states: Vec<u32> = (0u32..4).collect();
        let mut out = vec![(0u32, 0.0); states.len()];
        check_refstate_batch::<u32, PermDitMask<u32>>(&lattice, &[], &[], &states, &mut out);
        for (state, (batch_ref, batch_norm)) in states.iter().zip(out.iter()) {
            let (scalar_ref, scalar_norm) =
                check_refstate::<u32, PermDitMask<u32>>(&lattice, &[], &[], *state);
            assert_eq!(*batch_ref, scalar_ref);
            assert_eq!(*batch_norm, scalar_norm);
        }
    }

    #[test]
    fn batch_matches_scalar_with_local_and_composite() {
        let lattice = vec![lat(one(), &[1, 0])];
        let local = vec![local_xor(one(), 0b11u32)];
        let composite = vec![composite_lat_local(one(), &[1, 0], 0b11u32)];
        let states: Vec<u32> = (0u32..4).collect();
        let mut out = vec![(0u32, 0.0); states.len()];
        check_refstate_batch(&lattice, &local, &composite, &states, &mut out);
        for (state, (batch_ref, batch_norm)) in states.iter().zip(out.iter()) {
            let (scalar_ref, scalar_norm) = check_refstate(&lattice, &local, &composite, *state);
            assert_eq!(*batch_ref, scalar_ref);
            assert_eq!(*batch_norm, scalar_norm);
        }
    }

    #[test]
    fn batch_empty_input_is_noop() {
        let lattice = vec![lat(one(), &[0, 1])];
        let mut out: Vec<(u32, f64)> = vec![];
        check_refstate_batch::<u32, PermDitMask<u32>>(&lattice, &[], &[], &[], &mut out);
        // must not panic
    }
}
