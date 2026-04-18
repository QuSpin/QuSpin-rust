//! `OperatorDispatch` extension trait: basis-aware `apply_and_project_to`
//! methods for each `*OperatorInner` enum.
//!
//! The `*Inner` enums themselves live in `quspin-operator`; the extension
//! trait is defined here because its impls need `SpaceInner` (from
//! `quspin-basis`).

use crate::apply::apply_and_project_to;
use num_complex::Complex;
use quspin_basis::dispatch::SpaceInner;
use quspin_operator::{
    BondOperatorInner, BosonOperatorInner, FermionOperatorInner, HardcoreOperatorInner,
    MonomialOperatorInner, SpinOperatorInner,
};
use quspin_types::QuSpinError;

/// Extension trait adding basis-dependent dispatch methods to the
/// `*OperatorInner` enums. Impl'd below for each concrete enum.
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

macro_rules! impl_operator_dispatch {
    ($inner:ty) => {
        impl OperatorDispatch for $inner {
            fn apply_and_project_to(
                &self,
                input: &SpaceInner,
                output: &SpaceInner,
                coeffs: &[Complex<f64>],
                in_vec: &[Complex<f64>],
                out_vec: &mut [Complex<f64>],
                overwrite: bool,
            ) -> Result<(), QuSpinError> {
                match self {
                    Self::Ham8(h) => {
                        apply_and_project_to(h, input, output, coeffs, in_vec, out_vec, overwrite)
                    }
                    Self::Ham16(h) => {
                        apply_and_project_to(h, input, output, coeffs, in_vec, out_vec, overwrite)
                    }
                }
            }

            fn apply(
                &self,
                space: &SpaceInner,
                coeffs: &[Complex<f64>],
                in_vec: &[Complex<f64>],
                out_vec: &mut [Complex<f64>],
                overwrite: bool,
            ) -> Result<(), QuSpinError> {
                match self {
                    Self::Ham8(h) => {
                        apply_and_project_to(h, space, space, coeffs, in_vec, out_vec, overwrite)
                    }
                    Self::Ham16(h) => {
                        apply_and_project_to(h, space, space, coeffs, in_vec, out_vec, overwrite)
                    }
                }
            }
        }
    };
}

impl_operator_dispatch!(BondOperatorInner);
impl_operator_dispatch!(BosonOperatorInner);
impl_operator_dispatch!(FermionOperatorInner);
impl_operator_dispatch!(HardcoreOperatorInner);
impl_operator_dispatch!(MonomialOperatorInner);
impl_operator_dispatch!(SpinOperatorInner);
