//! [`OwnedQMatrixOperator`]: a `'static`-friendly [`LinearOperator`] adapter
//! around `Arc<QMatrixInner>` plus a fully-evaluated coefficient vector.
//!
//! Like [`QMatrixOperator`], but owns its matrix via `Arc` and carries no
//! lifetime parameter — usable as the `Op` type of `ExpmOp` for indefinite
//! lifetimes (PyO3 wrappers, long-lived caches, …).

use std::ops::Range;
use std::sync::Arc;

use ndarray::{ArrayView2, ArrayViewMut2};
use num_complex::Complex;

use quspin_types::{ExpmComputation, LinearOperator, Primitive, QuSpinError};

use crate::qmatrix::QMatrixInner;
use crate::qmatrix_helpers::{compute_trace_c64, onenorm_shifted_c64};
use crate::qmatrix_op::QMatrixOperator;

/// Owned linear-operator wrapper around `Arc<QMatrixInner>` + coefficients.
pub struct OwnedQMatrixOperator<V> {
    inner: Arc<QMatrixInner>,
    coeffs: Vec<V>,
}

impl<V> OwnedQMatrixOperator<V>
where
    V: Primitive,
{
    /// Construct an `OwnedQMatrixOperator`.
    ///
    /// # Errors
    /// Returns `ValueError` when `coeffs.len() != inner.num_coeff()`.
    pub fn new(inner: Arc<QMatrixInner>, coeffs: Vec<V>) -> Result<Self, QuSpinError> {
        let need = inner.num_coeff();
        if coeffs.len() != need {
            return Err(QuSpinError::ValueError(format!(
                "coeffs.len()={} must equal matrix.num_coeff()={need}",
                coeffs.len(),
            )));
        }
        Ok(Self { inner, coeffs })
    }

    /// Borrow the underlying matrix.
    pub fn matrix(&self) -> &QMatrixInner {
        &self.inner
    }

    /// Borrow the coefficient slice.
    pub fn coeffs(&self) -> &[V] {
        &self.coeffs
    }

    /// Coefficients cast to `Complex<f64>` for the trace / one-norm helpers
    /// (which operate uniformly on `Complex<f64>`).
    fn coeffs_c64(&self) -> Vec<Complex<f64>> {
        self.coeffs.iter().map(|c| c.to_complex()).collect()
    }
}

impl<V> Clone for OwnedQMatrixOperator<V>
where
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            coeffs: self.coeffs.clone(),
        }
    }
}

impl<V> LinearOperator<V> for OwnedQMatrixOperator<V>
where
    V: ExpmComputation,
{
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn trace(&self) -> V {
        let coeffs_c64 = self.coeffs_c64();
        let tr = crate::with_qmatrix!(&*self.inner, _M, _C, mat, {
            compute_trace_c64(mat, &coeffs_c64)
        });
        V::from_complex(tr)
    }

    fn onenorm(&self, shift: V) -> V::Real {
        let coeffs_c64 = self.coeffs_c64();
        let shift_c64 = shift.to_complex();
        let norm = crate::with_qmatrix!(&*self.inner, _M, _C, mat, {
            onenorm_shifted_c64(mat, &coeffs_c64, shift_c64)
        });
        V::real_from_f64(norm)
    }

    /// `QMatrixInner::dot` already parallelises with rayon, so opt out of the
    /// persistent-thread parallel expm path to avoid nested parallelism.
    fn parallel_hint(&self) -> bool {
        false
    }

    fn dot(&self, overwrite: bool, input: &[V], output: &mut [V]) -> Result<(), QuSpinError> {
        self.inner.dot(overwrite, &self.coeffs, input, output)
    }

    fn dot_transpose(
        &self,
        overwrite: bool,
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        self.inner
            .dot_transpose(overwrite, &self.coeffs, input, output)
    }

    fn dot_many(
        &self,
        overwrite: bool,
        input: ArrayView2<'_, V>,
        output: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError> {
        self.inner.dot_many(overwrite, &self.coeffs, input, output)
    }

    /// `parallel_hint = false` keeps the partitioned-Taylor algorithm on the
    /// sequential path, so this method is unreachable in practice.  Provide a
    /// correct fallback for completeness — delegates to a transient
    /// [`QMatrixOperator`] borrowed from `self`.
    fn dot_chunk(
        &self,
        overwrite: bool,
        input: &[V],
        output_chunk: &mut [V],
        row_start: usize,
    ) -> Result<(), QuSpinError> {
        crate::with_qmatrix!(&*self.inner, _M, _C, mat, {
            let qop = QMatrixOperator::new(mat, self.coeffs.clone())?;
            qop.dot_chunk(overwrite, input, output_chunk, row_start)
        })
    }

    fn dot_transpose_chunk(
        &self,
        input: &[V],
        output: &[V::Atomic],
        rows: Range<usize>,
    ) -> Result<(), QuSpinError> {
        crate::with_qmatrix!(&*self.inner, _M, _C, mat, {
            let qop = QMatrixOperator::new(mat, self.coeffs.clone())?;
            qop.dot_transpose_chunk(input, output, rows)
        })
    }
}
