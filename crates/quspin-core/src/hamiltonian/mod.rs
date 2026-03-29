pub mod bond;
pub mod boson;
pub mod fermion;
pub mod hardcore;
pub mod spin;

pub use bond::{BondHamiltonian, BondHamiltonianInner, BondTerm};
pub use boson::{BosonHamiltonian, BosonHamiltonianInner, BosonOp, BosonOpEntry};
pub use fermion::{FermionHamiltonian, FermionHamiltonianInner, FermionOp, FermionOpEntry};
pub use hardcore::{HardcoreHamiltonian, HardcoreHamiltonianInner, HardcoreOp, OpEntry};
pub use spin::{SpinHamiltonian, SpinHamiltonianInner, SpinOp, SpinOpEntry};

use crate::bitbasis::BitInt;
use crate::error::QuSpinError;
use num_complex::Complex;

/// Abstraction over Hamiltonians that can be applied to basis states.
///
/// The `apply` method uses a callback (`emit`) rather than returning a
/// collection, keeping the hot path allocation-free.
pub trait Hamiltonian<C> {
    fn max_site(&self) -> usize;
    fn num_cindices(&self) -> usize;

    /// Apply `self` to `state`, calling `emit(cindex, amplitude, new_state)`
    /// for each non-zero contribution.
    fn apply<B: BitInt, F>(&self, state: B, emit: F)
    where
        F: FnMut(C, Complex<f64>, B);
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
