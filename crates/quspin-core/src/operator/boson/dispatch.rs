pub use quspin_operator::BosonOperatorInner;

use super::super::apply::apply_and_project_to as apply_free_fn;
use crate::basis::dispatch::SpaceInner;
use num_complex::Complex;
use quspin_types::QuSpinError;

impl super::super::OperatorDispatch for BosonOperatorInner {
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
            Self::Ham8(h) => apply_free_fn(h, input, output, coeffs, in_vec, out_vec, overwrite),
            Self::Ham16(h) => apply_free_fn(h, input, output, coeffs, in_vec, out_vec, overwrite),
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
            Self::Ham8(h) => apply_free_fn(h, space, space, coeffs, in_vec, out_vec, overwrite),
            Self::Ham16(h) => apply_free_fn(h, space, space, coeffs, in_vec, out_vec, overwrite),
        }
    }
}
