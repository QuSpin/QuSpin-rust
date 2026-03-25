/// Symmetry group types.
///
/// - [`LatticeElement`]: shared site-permutation element used by both group kinds.
/// - [`SpinSymGrp`]: lattice + spin-inversion / bit-flip operations.
/// - [`ValuePermSymGrp`]: lattice + local value-permutation operations (LHSS ≥ 3).
pub mod spin;
pub mod value_perm;

pub use spin::{HardcoreGrpElement, HardcoreSymmetryGrp, SpinSymGrp, SymmetryGrpInner};
pub use value_perm::ValuePermSymGrp;

use crate::bitbasis::{BitInt, BitStateOp, PermDitLocations};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// LatticeElement — shared by both SpinSymGrp and ValuePermSymGrp
// ---------------------------------------------------------------------------

/// A site-permutation symmetry element with an associated group character.
///
/// Used by both [`SpinSymGrp`] and [`ValuePermSymGrp`].
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

    #[inline]
    pub fn apply<B: BitInt>(&self, state: B, coeff: Complex<f64>) -> (B, Complex<f64>) {
        (self.op.apply(state), coeff * self.grp_char)
    }
}
