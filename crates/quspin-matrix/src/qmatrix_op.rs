//! [`QMatrixOperator`]: [`LinearOperator`] adapter for [`QMatrix`].

use std::ops::Range;

use ndarray::{ArrayView2, ArrayViewMut2};
use num_complex::Complex;

use quspin_types::{AtomicAccum, ExpmComputation, LinearOperator, Primitive, QuSpinError};

use crate::qmatrix::matrix::{CIndex, Index, QMatrix};
use crate::qmatrix_helpers::{compute_trace_c64, onenorm_shifted_c64};

// ---------------------------------------------------------------------------
// QMatrixOperator
// ---------------------------------------------------------------------------

/// [`LinearOperator`] adapter that wraps a `&QMatrix<M, I, C>` with
/// pre-evaluated coefficients.
///
/// Construct via [`QMatrixOperator::new`], which validates that
/// `coeffs.len() == matrix.num_coeff()`.
pub struct QMatrixOperator<'a, M, I, C, V> {
    matrix: &'a QMatrix<M, I, C>,
    coeffs: Vec<V>,
}

impl<'a, M, I, C, V> QMatrixOperator<'a, M, I, C, V>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    V: ExpmComputation,
{
    /// Construct a `QMatrixOperator`.
    ///
    /// # Errors
    /// Returns `ValueError` when `coeffs.len() != matrix.num_coeff()`.
    pub fn new(matrix: &'a QMatrix<M, I, C>, coeffs: Vec<V>) -> Result<Self, QuSpinError> {
        if coeffs.len() != matrix.num_coeff() {
            return Err(QuSpinError::ValueError(format!(
                "coeffs.len()={} must equal matrix.num_coeff()={}",
                coeffs.len(),
                matrix.num_coeff(),
            )));
        }
        Ok(Self { matrix, coeffs })
    }

    /// Borrow the underlying matrix.
    pub fn matrix(&self) -> &QMatrix<M, I, C> {
        self.matrix
    }

    /// Borrow the coefficient slice.
    pub fn coeffs(&self) -> &[V] {
        &self.coeffs
    }

    fn coeffs_c64(&self) -> Vec<Complex<f64>> {
        self.coeffs.iter().map(|c| c.to_complex()).collect()
    }
}

impl<M, I, C, V> LinearOperator<V> for QMatrixOperator<'_, M, I, C, V>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    V: ExpmComputation,
{
    fn dim(&self) -> usize {
        self.matrix.dim()
    }

    fn trace(&self) -> V {
        let coeffs_c64 = self.coeffs_c64();
        let tr = compute_trace_c64(self.matrix, &coeffs_c64);
        V::from_complex(tr)
    }

    fn onenorm(&self, shift: V) -> V::Real {
        let coeffs_c64 = self.coeffs_c64();
        let shift_c64 = shift.to_complex();
        let norm = onenorm_shifted_c64(self.matrix, &coeffs_c64, shift_c64);
        V::real_from_f64(norm)
    }

    fn dot(&self, overwrite: bool, input: &[V], output: &mut [V]) -> Result<(), QuSpinError> {
        self.matrix.dot(overwrite, &self.coeffs, input, output)
    }

    fn dot_transpose(
        &self,
        overwrite: bool,
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        self.matrix
            .dot_transpose(overwrite, &self.coeffs, input, output)
    }

    fn dot_many(
        &self,
        overwrite: bool,
        input: ArrayView2<'_, V>,
        output: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError> {
        self.matrix.dot_many(overwrite, &self.coeffs, input, output)
    }

    fn dot_chunk(
        &self,
        overwrite: bool,
        input: &[V],
        output_chunk: &mut [V],
        row_start: usize,
    ) -> Result<(), QuSpinError> {
        if overwrite {
            output_chunk.iter_mut().for_each(|v| *v = V::default());
        }
        for (k_local, r) in (row_start..row_start + output_chunk.len()).enumerate() {
            for e in self.matrix.row(r) {
                let scale =
                    self.coeffs[e.cindex.as_usize()] * V::from_complex(e.value.to_complex());
                output_chunk[k_local] += scale * input[e.col.as_usize()];
            }
        }
        Ok(())
    }

    fn dot_transpose_chunk(
        &self,
        input: &[V],
        output: &[V::Atomic],
        rows: Range<usize>,
    ) -> Result<(), QuSpinError> {
        for r in rows {
            for e in self.matrix.row(r) {
                let scale =
                    self.coeffs[e.cindex.as_usize()] * V::from_complex(e.value.to_complex());
                output[e.col.as_usize()].fetch_add(scale * input[r]);
            }
        }
        Ok(())
    }
}
