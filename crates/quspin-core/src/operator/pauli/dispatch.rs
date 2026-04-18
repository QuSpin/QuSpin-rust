/// Type-erased `HardcoreOperatorInner`.
///
/// Selects between `HardcoreOperator<u8>` (≤ 255 cindices / site indices) and
/// `HardcoreOperator<u16>` (larger).  The cindex type is chosen at
/// construction time by `PyHardcoreOperator::new`.
use super::HardcoreOperator;

/// Type-erased `HardcoreOperator`: either u8 or u16 cindex type.
pub enum HardcoreOperatorInner {
    Ham8(HardcoreOperator<u8>),
    Ham16(HardcoreOperator<u16>),
}

impl HardcoreOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            HardcoreOperatorInner::Ham8(h) => h.max_site(),
            HardcoreOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            HardcoreOperatorInner::Ham8(h) => h.num_cindices(),
            HardcoreOperatorInner::Ham16(h) => h.num_cindices(),
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
