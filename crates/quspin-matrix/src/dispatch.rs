//! `OperatorDispatch` extension trait: basis-aware `apply_and_project_to`
//! methods for each `*OperatorInner` enum.
//!
//! The `*Inner` enums themselves live in `quspin-operator`; the extension
//! trait is defined here because its impls need the basis dispatch types
//! (from `quspin-basis`).

use crate::apply::{apply_and_project_to, apply_and_project_to_bit};
use crate::matrix_free as mf;
use num_complex::Complex;
use quspin_basis::dispatch::{BitBasis, GenericBasis};
use quspin_operator::{
    BondOperatorInner, BosonOperatorInner, FermionOperatorInner, HardcoreOperatorInner,
    MonomialOperatorInner, SpinOperatorInner,
};
use quspin_types::QuSpinError;

/// Extension trait adding basis-dependent dispatch methods to the
/// `*OperatorInner` enums. Impl'd below for each concrete enum.
///
/// The `_bit` variants take a `&BitBasis` directly so callers from the
/// fermion path (which carries `BitBasis` without going through
/// `GenericBasis`) don't pull in dit-family monomorphizations.
pub trait OperatorDispatch {
    fn apply_and_project_to(
        &self,
        input: &GenericBasis,
        output: &GenericBasis,
        coeffs: &[Complex<f64>],
        in_vec: &[Complex<f64>],
        out_vec: &mut [Complex<f64>],
        overwrite: bool,
    ) -> Result<(), QuSpinError>;

    fn apply(
        &self,
        space: &GenericBasis,
        coeffs: &[Complex<f64>],
        in_vec: &[Complex<f64>],
        out_vec: &mut [Complex<f64>],
        overwrite: bool,
    ) -> Result<(), QuSpinError>;

    fn apply_and_project_to_bit(
        &self,
        input: &BitBasis,
        output: &BitBasis,
        coeffs: &[Complex<f64>],
        in_vec: &[Complex<f64>],
        out_vec: &mut [Complex<f64>],
        overwrite: bool,
    ) -> Result<(), QuSpinError>;

    fn apply_bit(
        &self,
        space: &BitBasis,
        coeffs: &[Complex<f64>],
        in_vec: &[Complex<f64>],
        out_vec: &mut [Complex<f64>],
        overwrite: bool,
    ) -> Result<(), QuSpinError>;

    /// Number of distinct cindex values — the required `coeffs` length.
    fn num_cindices(&self) -> usize;

    /// Local Hilbert-space size the operator's matrix elements assume.
    fn lhss(&self) -> usize;

    /// Largest site index appearing in any operator string.
    fn max_site(&self) -> usize;

    // -----------------------------------------------------------------
    // Matrix-free `LinearOperator` support
    //
    // These back [`OperatorOnBasis`](crate::OperatorOnBasis), which needs
    // the transpose product and the two spectral quantities without ever
    // assembling a `QMatrix`. Each is a single sweep over the basis.
    // -----------------------------------------------------------------

    /// `out_vec = Aᵀ · in_vec` (or `+=` when `overwrite` is false).
    fn dot_transpose(
        &self,
        space: &GenericBasis,
        coeffs: &[Complex<f64>],
        in_vec: &[Complex<f64>],
        out_vec: &mut [Complex<f64>],
        overwrite: bool,
    ) -> Result<(), QuSpinError>;

    /// `out_vec = Aᵀ · in_vec` over a `BitBasis`.
    fn dot_transpose_bit(
        &self,
        space: &BitBasis,
        coeffs: &[Complex<f64>],
        in_vec: &[Complex<f64>],
        out_vec: &mut [Complex<f64>],
        overwrite: bool,
    ) -> Result<(), QuSpinError>;

    /// `Σ_i A[i, i]`.
    fn trace(
        &self,
        space: &GenericBasis,
        coeffs: &[Complex<f64>],
    ) -> Result<Complex<f64>, QuSpinError>;

    /// `Σ_i A[i, i]` over a `BitBasis`.
    fn trace_bit(
        &self,
        space: &BitBasis,
        coeffs: &[Complex<f64>],
    ) -> Result<Complex<f64>, QuSpinError>;

    /// `‖A − shift·I‖₁`.
    fn onenorm(
        &self,
        space: &GenericBasis,
        coeffs: &[Complex<f64>],
        shift: Complex<f64>,
    ) -> Result<f64, QuSpinError>;

    /// `‖A − shift·I‖₁` over a `BitBasis`.
    fn onenorm_bit(
        &self,
        space: &BitBasis,
        coeffs: &[Complex<f64>],
        shift: Complex<f64>,
    ) -> Result<f64, QuSpinError>;
}

macro_rules! impl_operator_dispatch {
    ($inner:ty) => {
        impl OperatorDispatch for $inner {
            fn apply_and_project_to(
                &self,
                input: &GenericBasis,
                output: &GenericBasis,
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
                space: &GenericBasis,
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

            fn apply_and_project_to_bit(
                &self,
                input: &BitBasis,
                output: &BitBasis,
                coeffs: &[Complex<f64>],
                in_vec: &[Complex<f64>],
                out_vec: &mut [Complex<f64>],
                overwrite: bool,
            ) -> Result<(), QuSpinError> {
                match self {
                    Self::Ham8(h) => apply_and_project_to_bit(
                        h, input, output, coeffs, in_vec, out_vec, overwrite,
                    ),
                    Self::Ham16(h) => apply_and_project_to_bit(
                        h, input, output, coeffs, in_vec, out_vec, overwrite,
                    ),
                }
            }

            fn apply_bit(
                &self,
                space: &BitBasis,
                coeffs: &[Complex<f64>],
                in_vec: &[Complex<f64>],
                out_vec: &mut [Complex<f64>],
                overwrite: bool,
            ) -> Result<(), QuSpinError> {
                match self {
                    Self::Ham8(h) => apply_and_project_to_bit(
                        h, space, space, coeffs, in_vec, out_vec, overwrite,
                    ),
                    Self::Ham16(h) => apply_and_project_to_bit(
                        h, space, space, coeffs, in_vec, out_vec, overwrite,
                    ),
                }
            }

            fn num_cindices(&self) -> usize {
                Self::num_cindices(self)
            }

            fn lhss(&self) -> usize {
                Self::lhss(self)
            }

            fn max_site(&self) -> usize {
                Self::max_site(self)
            }

            fn dot_transpose(
                &self,
                space: &GenericBasis,
                coeffs: &[Complex<f64>],
                in_vec: &[Complex<f64>],
                out_vec: &mut [Complex<f64>],
                overwrite: bool,
            ) -> Result<(), QuSpinError> {
                match self {
                    Self::Ham8(h) => {
                        mf::dot_transpose(h, space, coeffs, in_vec, out_vec, overwrite)
                    }
                    Self::Ham16(h) => {
                        mf::dot_transpose(h, space, coeffs, in_vec, out_vec, overwrite)
                    }
                }
            }

            fn dot_transpose_bit(
                &self,
                space: &BitBasis,
                coeffs: &[Complex<f64>],
                in_vec: &[Complex<f64>],
                out_vec: &mut [Complex<f64>],
                overwrite: bool,
            ) -> Result<(), QuSpinError> {
                match self {
                    Self::Ham8(h) => {
                        mf::dot_transpose_bit(h, space, coeffs, in_vec, out_vec, overwrite)
                    }
                    Self::Ham16(h) => {
                        mf::dot_transpose_bit(h, space, coeffs, in_vec, out_vec, overwrite)
                    }
                }
            }

            fn trace(
                &self,
                space: &GenericBasis,
                coeffs: &[Complex<f64>],
            ) -> Result<Complex<f64>, QuSpinError> {
                match self {
                    Self::Ham8(h) => mf::trace(h, space, coeffs),
                    Self::Ham16(h) => mf::trace(h, space, coeffs),
                }
            }

            fn trace_bit(
                &self,
                space: &BitBasis,
                coeffs: &[Complex<f64>],
            ) -> Result<Complex<f64>, QuSpinError> {
                match self {
                    Self::Ham8(h) => mf::trace_bit(h, space, coeffs),
                    Self::Ham16(h) => mf::trace_bit(h, space, coeffs),
                }
            }

            fn onenorm(
                &self,
                space: &GenericBasis,
                coeffs: &[Complex<f64>],
                shift: Complex<f64>,
            ) -> Result<f64, QuSpinError> {
                match self {
                    Self::Ham8(h) => mf::onenorm(h, space, coeffs, shift),
                    Self::Ham16(h) => mf::onenorm(h, space, coeffs, shift),
                }
            }

            fn onenorm_bit(
                &self,
                space: &BitBasis,
                coeffs: &[Complex<f64>],
                shift: Complex<f64>,
            ) -> Result<f64, QuSpinError> {
                match self {
                    Self::Ham8(h) => mf::onenorm_bit(h, space, coeffs, shift),
                    Self::Ham16(h) => mf::onenorm_bit(h, space, coeffs, shift),
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
