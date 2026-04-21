use num_complex::Complex;
/// Lattice element types used by the orbit computation.
use quspin_bitbasis::{BenesPermDitLocations, BitInt, BitStateOp};

// ---------------------------------------------------------------------------
// LatEl
// ---------------------------------------------------------------------------

/// A lattice element with state-dependent group character.
///
/// For bosonic elements: `grp_char_for` returns a constant.
/// For fermionic elements: `grp_char_for` includes the Jordan-Wigner
/// permutation sign computed from the current state.
pub(crate) trait LatEl<B: BitInt> {
    /// Apply the site permutation, returning the new state.
    fn apply_state(&self, state: B) -> B;

    /// Return the group character for the given pre-image state.
    ///
    /// For bosonic elements this is a constant independent of `state`.
    /// For fermionic elements this includes the Jordan-Wigner sign.
    fn grp_char_for(&self, state: B) -> Complex<f64>;
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

    #[allow(dead_code)]
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
