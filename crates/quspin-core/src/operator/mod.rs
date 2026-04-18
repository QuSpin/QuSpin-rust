//! Operator types and `*Inner` dispatch enums.
//!
//! The type definitions (`SpinOperator<C>`, `BondOperator<C>`, etc.) and the
//! `Operator` / `ParseOp` traits live in `quspin-operator`, along with the
//! `*OperatorInner` dispatch enums and their non-basis methods.
//!
//! This crate adds basis-dependent dispatch methods (`apply_and_project_to`,
//! `apply`) via the [`OperatorDispatch`] extension trait, impl'd here for each
//! `*Inner` type. Pulled into scope by `pub use quspin_core::OperatorDispatch;`.
//! The trait will move to `quspin-matrix` in PR 5.

pub mod apply;
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
pub use quspin_operator::{Operator, ParseOp};
pub use spin::{SpinOp, SpinOpEntry, SpinOperator, SpinOperatorInner};

use crate::basis::dispatch::SpaceInner;
use num_complex::Complex;
use quspin_types::QuSpinError;

/// Extension trait adding basis-dependent dispatch methods to the
/// `*OperatorInner` enums. Impl'd in each of `bond::dispatch`,
/// `boson::dispatch`, etc. for the corresponding `*Inner` type.
pub trait OperatorDispatch {
    fn apply_and_project_to(
        &self,
        input: &SpaceInner,
        output: &SpaceInner,
        coeffs: &[Complex<f64>],
        in_vec: &[Complex<f64>],
        out_vec: &mut [Complex<f64>],
        overwrite: bool,
    ) -> Result<(), QuSpinError>;

    fn apply(
        &self,
        space: &SpaceInner,
        coeffs: &[Complex<f64>],
        in_vec: &[Complex<f64>],
        out_vec: &mut [Complex<f64>],
        overwrite: bool,
    ) -> Result<(), QuSpinError>;
}
