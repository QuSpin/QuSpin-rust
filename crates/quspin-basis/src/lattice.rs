//! Lattice / local / composite group-element types used by the orbit
//! computation.
//!
//! All three element shapes implement the [`OrbitImage`] trait with a
//! uniform `apply(state) -> (new_state, char)` interface, mirroring the
//! pre-refactor design. The character returned already includes any
//! state-dependent fermion sign (Jordan-Wigner, particle-hole, …) so
//! the walker never needs to consult a `fermionic` flag — when the
//! basis is non-fermionic the underlying `fermionic_sign` /
//! `fermion_sign` methods return `1.0` and LLVM eliminates the
//! multiplication.

use num_complex::Complex;
use quspin_bitbasis::{BenesPermDitLocations, BitInt, BitStateOp, FermionicBitStateOp};

// ---------------------------------------------------------------------------
// OrbitImage
// ---------------------------------------------------------------------------

/// A symmetry-group element acting on a basis state.
///
/// `apply(state)` returns the orbit image `(g · state, χ_g · sign(state))`
/// — the mapped state plus the full coefficient (1D-representation
/// character times any state-dependent fermion sign). The walker
/// iterates each storage vector and calls `apply` uniformly; storage
/// stays type-homogeneous so the hot loop has zero variant-branching.
pub(crate) trait OrbitImage<B: BitInt> {
    fn apply(&self, state: B) -> (B, Complex<f64>);
}

// ---------------------------------------------------------------------------
// BenesLatticeElement — pure site permutation
// ---------------------------------------------------------------------------

/// A lattice (site-permutation) group element backed by a Benes
/// permutation network.
///
/// Supports both bosonic (`fermionic = false`) and fermionic
/// (`fermionic = true`) symmetry. When fermionic, the Jordan-Wigner
/// sign returned by [`BenesPermDitLocations::fermionic_sign`] is folded
/// into the character at [`apply`](OrbitImage::apply) time.
#[derive(Clone)]
pub struct BenesLatticeElement<B: BitInt> {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub op: BenesPermDitLocations<B>,
}

impl<B: BitInt> BenesLatticeElement<B> {
    pub fn new(grp_char: Complex<f64>, op: BenesPermDitLocations<B>, n_sites: usize) -> Self {
        BenesLatticeElement {
            grp_char,
            n_sites,
            op,
        }
    }

    #[allow(dead_code)]
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }
}

impl<B: BitInt> OrbitImage<B> for BenesLatticeElement<B> {
    #[inline]
    fn apply(&self, state: B) -> (B, Complex<f64>) {
        let new_state = self.op.apply(state);
        let char = self.grp_char * Complex::new(self.op.fermionic_sign(state), 0.0);
        (new_state, char)
    }
}

// ---------------------------------------------------------------------------
// LocalElement — pure local op (no site permutation)
// ---------------------------------------------------------------------------

/// A pure-local group element: character + local op, no site
/// permutation.
///
/// `L: FermionicBitStateOp<B>` so [`apply`](OrbitImage::apply) can fold
/// in any state-dependent sign (default `1.0` for non-fermionic local
/// ops like `PermDitMask`).
#[derive(Clone)]
pub struct LocalElement<L> {
    pub grp_char: Complex<f64>,
    pub op: L,
}

impl<L> LocalElement<L> {
    pub fn new(grp_char: Complex<f64>, op: L) -> Self {
        LocalElement { grp_char, op }
    }
}

impl<B: BitInt, L: FermionicBitStateOp<B>> OrbitImage<B> for LocalElement<L> {
    #[inline]
    fn apply(&self, state: B) -> (B, Complex<f64>) {
        let new_state = self.op.apply(state);
        let char = self.grp_char * Complex::new(self.op.fermion_sign(state), 0.0);
        (new_state, char)
    }
}

// ---------------------------------------------------------------------------
// CompositeElement — atomic site-permutation + local op
// ---------------------------------------------------------------------------

/// An atomic composite group element: site permutation followed by a
/// local op, treated as one element with a single character.
///
/// The character is stored on the lattice component; the local
/// component contributes only its state-dependent fermion sign (if
/// any). The two components commute (orthogonal degrees of freedom),
/// so "perm first, local second" and "local first, perm second" are
/// equivalent — the walker uses the former.
#[derive(Clone)]
pub struct CompositeElement<B: BitInt, L> {
    pub lat: BenesLatticeElement<B>,
    pub loc: L,
}

impl<B: BitInt, L> CompositeElement<B, L> {
    pub fn new(lat: BenesLatticeElement<B>, loc: L) -> Self {
        CompositeElement { lat, loc }
    }
}

impl<B: BitInt, L: FermionicBitStateOp<B>> OrbitImage<B> for CompositeElement<B, L> {
    #[inline]
    fn apply(&self, state: B) -> (B, Complex<f64>) {
        let perm_state = self.lat.op.apply(state);
        let new_state = self.loc.apply(perm_state);
        let char = self.lat.grp_char
            * Complex::new(self.lat.op.fermionic_sign(state), 0.0)
            * Complex::new(self.loc.fermion_sign(perm_state), 0.0);
        (new_state, char)
    }
}
