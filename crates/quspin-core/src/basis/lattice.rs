use num_complex::Complex;
/// Lattice element types and local-op traits used by the orbit computation.
use quspin_bitbasis::{BenesPermDitLocations, BitInt, BitStateOp, PermDitLocations};

// ---------------------------------------------------------------------------
// LocalOpItem
// ---------------------------------------------------------------------------

/// A single local symmetry operation that maps a basis state to a new state
/// and returns the associated group character.
///
/// Implemented for:
/// - [`HardcoreGrpElement<B>`](super::spin::HardcoreGrpElement) — XOR
///   bit-flip, stores `grp_char` inside the element.
/// - `(Complex<f64>, Op)` where `Op: BitStateOp<B>` — all dit op types
///   (`HigherSpinInv`, `DynamicHigherSpinInv`, `PermDitValues`, …).
pub(crate) trait LocalOpItem<B: BitInt> {
    /// Apply the local operation to `state`, returning `(new_state, grp_char)`.
    fn apply_local(&self, state: B) -> (B, Complex<f64>);
}

/// Blanket impl: a `(grp_char, op)` pair where `op` implements [`BitStateOp<B>`].
impl<B: BitInt, Op: BitStateOp<B>> LocalOpItem<B> for (Complex<f64>, Op) {
    #[inline]
    fn apply_local(&self, state: B) -> (B, Complex<f64>) {
        (self.1.apply(state), self.0)
    }
}

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
// LatticeElement — shared by both SpinSymGrp and DitSymGrp
// ---------------------------------------------------------------------------

/// A site-permutation symmetry element with an associated group character.
///
/// Used by both [`SpinSymGrp`] and [`DitSymGrp`].
#[derive(Clone)]
#[allow(dead_code)]
pub struct LatticeElement {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub op: PermDitLocations,
}

#[allow(dead_code)]
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
#[allow(dead_code)]
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
