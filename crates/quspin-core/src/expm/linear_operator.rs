//! [`LinearOperator`] trait and [`QMatrixOperator`] adapter.
//!
//! `LinearOperator<V>` is a pure `y = A·x` interface with coefficients baked
//! in at construction.  The `expm` algorithm functions are generic over any
//! implementor, enabling alternative operator representations without touching
//! the Taylor-series core.
//!
//! [`QMatrixOperator`] is the standard adapter: it wraps a `&QMatrix<M, I, C>`
//! and a pre-evaluated coefficient vector `Vec<V>`.

use std::ops::Range;

use ndarray::{ArrayView2, ArrayViewMut2};
use num_complex::Complex;

use crate::error::QuSpinError;
use crate::primitive::Primitive;
use crate::qmatrix::matrix::{CIndex, Index, QMatrix};

use super::compute::{AtomicAccum, ExpmComputation};

// ---------------------------------------------------------------------------
// LinearOperator trait
// ---------------------------------------------------------------------------

/// Pure `y = A·x` interface used by the `expm` Taylor-series algorithm.
///
/// Coefficients are baked into the implementation at construction time; no
/// `coeffs` argument appears on any method.  All operators must be `Send +
/// Sync` so the algorithm can be called from parallel contexts.
///
/// ## Required methods
///
/// - `dim`, `trace`, `onenorm` — matrix metadata used for parameter selection.
/// - `dot`, `dot_transpose`, `dot_many` — whole-vector products; the operator
///   may parallelise internally.
/// - `dot_chunk`, `dot_transpose_chunk` — row-range-limited sequential products
///   for the Phase-2 persistent-thread-pool algorithm (issue #33).  The caller
///   partitions rows across threads; the operator must **not** spawn its own.
pub trait LinearOperator<V: ExpmComputation>: Send + Sync {
    /// Dimension of the square operator.
    fn dim(&self) -> usize;

    /// Trace of the effective matrix: `Σ_i A_eff[i, i]`.
    fn trace(&self) -> V;

    /// Column 1-norm of the shifted matrix: `‖A_eff − shift·I‖₁`.
    fn onenorm(&self, shift: V) -> V::Real;

    /// Whole-vector product: `output = A_eff · input` (or `+=` when
    /// `overwrite = false`).
    fn dot(&self, overwrite: bool, input: &[V], output: &mut [V]) -> Result<(), QuSpinError>;

    /// Whole-vector transpose product: `output = A_eff^T · input`.
    fn dot_transpose(
        &self,
        overwrite: bool,
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError>;

    /// Batch product: `output[[r, k]] = Σ_col A_eff[r, col] · input[[col, k]]`.
    fn dot_many(
        &self,
        overwrite: bool,
        input: ArrayView2<'_, V>,
        output: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError>;

    /// Row-chunk dot product: computes rows `row_start .. row_start + output_chunk.len()`
    /// of `A_eff · input` and writes the results into the caller-supplied
    /// `output_chunk` slice.
    ///
    /// `output_chunk` is the caller's exclusive subslice for these rows; it may
    /// point into a larger contiguous buffer.  The caller is responsible for
    /// ensuring no other thread accesses `output_chunk` concurrently.
    fn dot_chunk(
        &self,
        overwrite: bool,
        input: &[V],
        output_chunk: &mut [V],
        row_start: usize,
    ) -> Result<(), QuSpinError>;

    /// Row-range transpose product: scatter-adds `A_eff^T` contributions from
    /// `input[rows]` into a shared atomic output array.
    ///
    /// Unlike `dot_chunk`, the transpose scatters writes to arbitrary column
    /// indices, so different threads processing disjoint row ranges will
    /// concurrently write to the same output slots.  The output is therefore
    /// typed as `&[V::Atomic]` — a shared, atomically-acumulatable array —
    /// rather than `&mut [V]`.
    ///
    /// The caller initialises `output` to all-zeros (via `V::Atomic::zero()`)
    /// before spawning threads, and reads the final values only after all
    /// threads have finished (the thread-pool join provides the required
    /// synchronisation barrier).
    fn dot_transpose_chunk(
        &self,
        input: &[V],
        output: &[V::Atomic],
        rows: Range<usize>,
    ) -> Result<(), QuSpinError>;
}

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

    // Helpers that convert coefficients to Complex<f64> for norm computations
    // that must always run in full precision regardless of V.
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
        let tr = super::compute_trace_c64(self.matrix, &coeffs_c64);
        V::from_complex(tr)
    }

    fn onenorm(&self, shift: V) -> V::Real {
        let coeffs_c64 = self.coeffs_c64();
        let shift_c64 = shift.to_complex();
        let norm = super::onenorm_shifted_c64(self.matrix, &coeffs_c64, shift_c64);
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
