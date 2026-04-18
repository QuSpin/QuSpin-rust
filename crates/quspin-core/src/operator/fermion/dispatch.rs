/// Type-erased `FermionOperatorInner`.
///
/// Selects between `FermionOperator<u8>` (≤ 255 cindices / site indices)
/// and `FermionOperator<u16>` (larger), matching the cindex types supported
/// by `QMatrixInner`.
use super::FermionOperator;

pub enum FermionOperatorInner {
    Ham8(FermionOperator<u8>),
    Ham16(FermionOperator<u16>),
}

impl FermionOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            FermionOperatorInner::Ham8(h) => h.max_site(),
            FermionOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            FermionOperatorInner::Ham8(h) => h.num_cindices(),
            FermionOperatorInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub const fn lhss(&self) -> usize {
        2
    }

    pub fn apply_and_project_to(
        &self,
        input: &crate::basis::dispatch::SpaceInner,
        output: &crate::basis::dispatch::SpaceInner,
        coeffs: &[num_complex::Complex<f64>],
        in_vec: &[num_complex::Complex<f64>],
        out_vec: &mut [num_complex::Complex<f64>],
        overwrite: bool,
    ) -> Result<(), quspin_types::QuSpinError> {
        match self {
            Self::Ham8(h) => super::super::apply::apply_and_project_to(
                h, input, output, coeffs, in_vec, out_vec, overwrite,
            ),
            Self::Ham16(h) => super::super::apply::apply_and_project_to(
                h, input, output, coeffs, in_vec, out_vec, overwrite,
            ),
        }
    }

    pub fn apply(
        &self,
        space: &crate::basis::dispatch::SpaceInner,
        coeffs: &[num_complex::Complex<f64>],
        in_vec: &[num_complex::Complex<f64>],
        out_vec: &mut [num_complex::Complex<f64>],
        overwrite: bool,
    ) -> Result<(), quspin_types::QuSpinError> {
        match self {
            Self::Ham8(h) => super::super::apply::apply_and_project_to(
                h, space, space, coeffs, in_vec, out_vec, overwrite,
            ),
            Self::Ham16(h) => super::super::apply::apply_and_project_to(
                h, space, space, coeffs, in_vec, out_vec, overwrite,
            ),
        }
    }
}
