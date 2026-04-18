use super::BondOperator;
use quspin_operator::Operator;

/// Type-erased `BondOperator`: either u8 or u16 cindex type.
///
/// The cindex type is chosen at construction time based on the number of
/// distinct cindices and the maximum site index.
pub enum BondOperatorInner {
    Ham8(BondOperator<u8>),
    Ham16(BondOperator<u16>),
}

impl BondOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            BondOperatorInner::Ham8(h) => h.max_site(),
            BondOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            BondOperatorInner::Ham8(h) => h.num_cindices(),
            BondOperatorInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub fn lhss(&self) -> usize {
        match self {
            BondOperatorInner::Ham8(h) => h.lhss(),
            BondOperatorInner::Ham16(h) => h.lhss(),
        }
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
