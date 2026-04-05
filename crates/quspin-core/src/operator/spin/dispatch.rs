use super::SpinOperator;

/// Type-erased `SpinOperator`: either u8 or u16 cindex type.
pub enum SpinOperatorInner {
    Ham8(SpinOperator<u8>),
    Ham16(SpinOperator<u16>),
}

impl SpinOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            SpinOperatorInner::Ham8(h) => h.max_site(),
            SpinOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            SpinOperatorInner::Ham8(h) => h.num_cindices(),
            SpinOperatorInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub fn lhss(&self) -> usize {
        match self {
            SpinOperatorInner::Ham8(h) => h.lhss(),
            SpinOperatorInner::Ham16(h) => h.lhss(),
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
    ) -> Result<(), crate::error::QuSpinError> {
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
    ) -> Result<(), crate::error::QuSpinError> {
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
