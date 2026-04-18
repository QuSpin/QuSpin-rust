//! Operator type definitions and apply kernels.
//!
//! No basis knowledge. The `apply_and_project_to` extension methods on
//! `*OperatorInner` (which require basis types) live in downstream crates
//! (currently `quspin-core`, moving to `quspin-matrix` in PR 5).

pub mod bond;
pub mod boson;
pub mod fermion;
pub mod monomial;
pub mod pauli;
pub mod spin;

pub use bond::{BondOperator, BondOperatorInner, BondTerm};
pub use boson::{BosonOp, BosonOpEntry, BosonOperator, BosonOperatorInner};
pub use fermion::{FermionOp, FermionOpEntry, FermionOperator, FermionOperatorInner};
pub use monomial::{MonomialOperator, MonomialOperatorInner, MonomialTerm};
pub use pauli::{HardcoreOp, HardcoreOperator, HardcoreOperatorInner, OpEntry};
pub use spin::{SpinOp, SpinOpEntry, SpinOperator, SpinOperatorInner};

use num_complex::Complex;
use quspin_bitbasis::BitInt;
use quspin_types::QuSpinError;

/// Abstraction over operators that can be applied to basis states.
///
/// The `apply` method uses a callback (`emit`) rather than returning a
/// collection, keeping the hot path allocation-free.
pub trait Operator<C> {
    fn max_site(&self) -> usize;
    fn num_cindices(&self) -> usize;

    /// Apply `self` to `state`, calling `emit(cindex, amplitude, new_state)`
    /// for each non-zero contribution.
    fn apply<B: BitInt, F>(&self, state: B, emit: F)
    where
        F: FnMut(C, Complex<f64>, B);
}

impl<C, T: Operator<C> + ?Sized> Operator<C> for &T {
    fn max_site(&self) -> usize {
        (**self).max_site()
    }
    fn num_cindices(&self) -> usize {
        (**self).num_cindices()
    }
    fn apply<B: BitInt, F>(&self, state: B, emit: F)
    where
        F: FnMut(C, Complex<f64>, B),
    {
        (**self).apply(state, emit)
    }
}

/// Operator types that can be parsed from a single ASCII character.
///
/// Implement this trait on an operator enum to plug into the shared
/// `parse_term` / `parse_coupling` / `parse_ops` pipeline in any FFI crate.
pub trait ParseOp: Sized {
    /// Convert a single character to `Self`.
    ///
    /// Returns `QuSpinError::ValueError` for unrecognised characters.  The
    /// error message should name the valid characters for this operator type.
    fn from_char(ch: char) -> Result<Self, QuSpinError>;
}
