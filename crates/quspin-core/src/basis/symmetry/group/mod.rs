/// Symmetry group types.
///
/// - [`LatticeElement`]: shared site-permutation element used by both group kinds.
/// - [`traits::LocalOpItem`]: trait abstracting local symmetry operations.
/// - [`orbit`]: shared orbit helpers (`iter_images`, `get_refstate`, `check_refstate`).
/// - [`SpinSymGrp`]: lattice + spin-inversion / bit-flip operations.
/// - [`DitSymGrp`]: lattice + local value-permutation operations (LHSS ≥ 3).
pub mod dispatch;
pub mod dit;
pub mod fermion_grp;
pub(crate) mod orbit;
pub mod spin;
pub(crate) mod traits;

pub use dispatch::SymmetryGrpInner;
pub use dit::DitSymGrp;
pub use fermion_grp::FermionicSymGrp;
pub use spin::{HardcoreGrpElement, HardcoreSymmetryGrp, SpinSymGrp};
// BenesLatticeElement is declared below in this file — no re-export alias needed.

pub(crate) use orbit::{check_refstate, get_refstate};
pub(crate) use traits::{LatEl, LocalOpItem};

use crate::bitbasis::{BenesPermDitLocations, BitInt, BitStateOp, PermDitLocations};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// LatticeElement — shared by both SpinSymGrp and DitSymGrp
// ---------------------------------------------------------------------------

/// A site-permutation symmetry element with an associated group character.
///
/// Used by both [`SpinSymGrp`] and [`DitSymGrp`].
#[derive(Clone)]
pub struct LatticeElement {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub op: PermDitLocations,
}

impl LatticeElement {
    pub fn new(grp_char: Complex<f64>, op: PermDitLocations, n_sites: usize) -> Self {
        LatticeElement {
            grp_char,
            n_sites,
            op,
        }
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Apply the permutation and accumulate the group character into `coeff`.
    #[inline]
    pub fn apply<B: BitInt>(&self, state: B, coeff: Complex<f64>) -> (B, Complex<f64>) {
        (self.op.apply(state), coeff * self.grp_char)
    }

    /// Apply only the site permutation, discarding the group character.
    ///
    /// Used by the batch orbit helpers where the character is not needed
    /// (e.g. [`check_refstate_batch`](super::orbit::check_refstate_batch)).
    #[inline]
    pub fn apply_state<B: BitInt>(&self, state: B) -> B {
        self.op.apply(state)
    }
}

// ---------------------------------------------------------------------------
// LatEl impls for LatticeElement
// ---------------------------------------------------------------------------

impl<B: BitInt> LatEl<B> for LatticeElement {
    #[inline]
    fn apply_state(&self, state: B) -> B {
        self.op.apply(state)
    }

    #[inline]
    fn grp_char_for(&self, _state: B) -> Complex<f64> {
        self.grp_char
    }
}

// ---------------------------------------------------------------------------
// BenesLatticeElement — Benes-backed lattice element (bosonic or fermionic)
// ---------------------------------------------------------------------------

/// A lattice element backed by a Benes permutation network.
///
/// Supports both bosonic (`fermionic=false`) and fermionic (`fermionic=true`)
/// symmetry. When fermionic, [`grp_char_for`](LatEl::grp_char_for) multiplies
/// in the Jordan-Wigner permutation sign computed from the pre-image state.
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

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }
}

impl<B: BitInt> LatEl<B> for BenesLatticeElement<B> {
    #[inline]
    fn apply_state(&self, state: B) -> B {
        self.op.apply(state)
    }

    #[inline]
    fn grp_char_for(&self, state: B) -> Complex<f64> {
        self.grp_char * Complex::new(self.op.fermionic_sign(state), 0.0)
    }
}
