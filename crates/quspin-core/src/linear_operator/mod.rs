//! Type-erased [`LinearOperator`] trait and concrete implementations.
//!
//! # Implementations
//!
//! | Type | Description |
//! |------|-------------|
//! | [`QMatrixOperator`] | Wraps a `&QMatrix` with pre-evaluated coefficients |
//! | [`FnLinearOperator`] | Wraps user-supplied closures; built via [`FnLinearOperator::builder`] |
//!
//! # Dispatch hint
//!
//! [`LinearOperator::parallel_hint`] signals whether the operator can safely be
//! called from the persistent-thread parallel expm path.  Operators with internal
//! rayon parallelism or GIL-bound callbacks should return `false`; the
//! `expm_multiply_auto` functions will then use the sequential path, which calls
//! [`LinearOperator::dot`] once per Taylor iteration and allows the operator to
//! parallelise internally.

mod fn_op;
mod qmatrix_op;

pub use fn_op::{FnLinearOperator, FnLinearOperatorBuilder};
pub use qmatrix_op::QMatrixOperator;

use std::ops::Range;

use ndarray::{ArrayView2, ArrayViewMut2};

use crate::error::QuSpinError;
use crate::expm::compute::ExpmComputation;

// ---------------------------------------------------------------------------
// LinearOperator trait
// ---------------------------------------------------------------------------

/// Pure `y = A·x` interface used by the `expm` Taylor-series algorithm.
///
/// Coefficients are baked into the implementation at construction time; no
/// `coeffs` argument appears on any method.  All implementations must be
/// `Send + Sync` so the algorithm can be called from parallel contexts.
///
/// ## Required methods
///
/// - `dim`, `trace`, `onenorm` — matrix metadata used for parameter selection.
/// - `dot`, `dot_transpose`, `dot_many` — whole-vector products; the operator
///   may parallelise internally.
/// - `dot_chunk`, `dot_transpose_chunk` — row-range-limited products for the
///   persistent-thread parallel path.  When called from that path the operator
///   **must not** spawn its own threads; use [`parallel_hint`] to opt out.
///
/// ## Parallel dispatch
///
/// Override [`parallel_hint`] to return `false` if any of the following apply:
/// - `dot` / `dot_many` already use rayon internally (nested parallelism).
/// - Closures hold a lock incompatible with concurrent calls (e.g. Python GIL).
///
/// [`parallel_hint`]: LinearOperator::parallel_hint
pub trait LinearOperator<V: ExpmComputation>: Send + Sync {
    /// Dimension of the square operator.
    fn dim(&self) -> usize;

    /// Trace of the effective matrix: `Σ_i A_eff[i, i]`.
    fn trace(&self) -> V;

    /// Column 1-norm of the shifted matrix: `‖A_eff − shift·I‖₁`.
    fn onenorm(&self, shift: V) -> V::Real;

    /// Whether `expm_multiply_auto` may use the persistent-thread parallel path.
    ///
    /// Return `false` to force the sequential path, which calls [`dot`] once per
    /// Taylor iteration.  This is the correct choice when the operator has
    /// internal parallelism or when `dot_chunk` cannot efficiently compute a
    /// partial row range.
    ///
    /// Defaults to `true`.
    ///
    /// [`dot`]: LinearOperator::dot
    fn parallel_hint(&self) -> bool {
        true
    }

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

    /// Batch product over a `(dim, n_vecs)` array.
    fn dot_many(
        &self,
        overwrite: bool,
        input: ArrayView2<'_, V>,
        output: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError>;

    /// Row-chunk dot product: computes rows `row_start .. row_start + output_chunk.len()`
    /// of `A_eff · input` into the caller-supplied slice.
    ///
    /// The caller guarantees exclusive access to `output_chunk`.  When called
    /// from the persistent-thread pool, the operator **must not** spawn rayon
    /// work internally — use [`parallel_hint`] to opt out of that path.
    ///
    /// [`parallel_hint`]: LinearOperator::parallel_hint
    fn dot_chunk(
        &self,
        overwrite: bool,
        input: &[V],
        output_chunk: &mut [V],
        row_start: usize,
    ) -> Result<(), QuSpinError>;

    /// Row-range transpose product with atomic accumulation.
    ///
    /// Scatter-adds the contribution of input rows `rows` into a shared atomic
    /// output array.  The caller initialises `output` to zero and reads the
    /// final values only after all threads have finished.
    fn dot_transpose_chunk(
        &self,
        input: &[V],
        output: &[V::Atomic],
        rows: Range<usize>,
    ) -> Result<(), QuSpinError>;
}

/// Convenience alias for a heap-allocated, type-erased [`LinearOperator`].
pub type DynLinearOperator<V> = Box<dyn LinearOperator<V> + Send + Sync>;
