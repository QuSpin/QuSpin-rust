/// Symmetry group types.
///
/// - [`LatticeElement`]: shared site-permutation element used by both group kinds.
/// - [`traits::LocalOpItem`]: trait abstracting local symmetry operations.
/// - [`orbit`]: shared orbit helpers (`iter_images`, `get_refstate`, `check_refstate`).
/// - [`SpinSymGrp`]: lattice + spin-inversion / bit-flip operations.
/// - [`DitSymGrp`]: lattice + local value-permutation operations (LHSS ≥ 3).
pub mod dispatch;
pub mod dit;
pub(crate) mod orbit;
pub mod spin;
pub(crate) mod traits;

pub use dispatch::SymmetryGrpInner;
pub use dit::DitSymGrp;
pub use spin::{HardcoreGrpElement, HardcoreSymmetryGrp, SpinSymGrp};

pub(crate) use orbit::{check_refstate, get_refstate};
pub(crate) use traits::LocalOpItem;

use crate::bitbasis::{BitInt, BitStateOp, PermDitLocations};
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
