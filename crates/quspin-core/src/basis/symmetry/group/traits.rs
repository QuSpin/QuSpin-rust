/// Shared traits for local symmetry operations, and `SymGrp` impls for the
/// group types defined in this module.
use crate::bitbasis::{BitInt, BitStateOp};
use num_complex::Complex;

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
