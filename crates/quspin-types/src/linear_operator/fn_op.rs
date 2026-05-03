//! [`FnLinearOperator`]: closure-backed [`LinearOperator`].

use std::ops::Range;
use std::sync::Arc;

use ndarray::{ArrayView2, ArrayViewMut2};

use crate::compute::{AtomicAccum, ExpmComputation};
use crate::error::QuSpinError;

use super::LinearOperator;

// ---------------------------------------------------------------------------
// Type aliases for the closure fields
// ---------------------------------------------------------------------------

type OnenormFn<V> = Arc<dyn Fn(V) -> <V as ExpmComputation>::Real + Send + Sync>;
type DotFn<V> = Arc<dyn Fn(bool, &[V], &mut [V]) -> Result<(), QuSpinError> + Send + Sync>;
type DotManyFn<V> = Arc<
    dyn Fn(bool, ArrayView2<'_, V>, ArrayViewMut2<'_, V>) -> Result<(), QuSpinError> + Send + Sync,
>;

// ---------------------------------------------------------------------------
// FnLinearOperator
// ---------------------------------------------------------------------------

/// Closure-backed [`LinearOperator`].
///
/// Wraps user-supplied `dot` and `dot_transpose` closures together with
/// metadata (`dim`, `trace`, `onenorm_fn`).  An optional `dot_many` closure
/// can be provided for batch efficiency; if absent, the default implementation
/// loops over columns calling `dot`.
///
/// # Construction
///
/// Use the builder returned by [`FnLinearOperator::builder`]:
///
/// ```rust,ignore
/// let op = FnLinearOperator::builder(dim, trace, onenorm_fn, dot_fn, dot_t_fn)
///     .dot_many(batch_fn)   // optional
///     .build();
/// ```
///
/// # Parallel dispatch
///
/// [`parallel_hint`] always returns `false`.  `ExpmOp` will therefore use the
/// sequential path, which calls `dot` once per Taylor iteration and lets the
/// closure parallelise internally.
///
/// `dot_chunk` and `dot_transpose_chunk` are implemented as fallbacks (compute
/// the full product into a temporary buffer and copy/scatter the relevant
/// portion).  They are correct but O(dim) — they should only be reached if a
/// caller manually invokes `expm_multiply_par` on a `FnLinearOperator`, which
/// is a misuse of the API.
///
/// [`parallel_hint`]: LinearOperator::parallel_hint
pub struct FnLinearOperator<V: ExpmComputation> {
    dim: usize,
    trace_val: V,
    onenorm_fn: OnenormFn<V>,
    dot_fn: DotFn<V>,
    dot_t_fn: DotFn<V>,
    dot_many_fn: Option<DotManyFn<V>>,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for [`FnLinearOperator`].
///
/// Obtain via [`FnLinearOperator::builder`].
pub struct FnLinearOperatorBuilder<V: ExpmComputation> {
    dim: usize,
    trace_val: V,
    onenorm_fn: OnenormFn<V>,
    dot_fn: DotFn<V>,
    dot_t_fn: DotFn<V>,
    dot_many_fn: Option<DotManyFn<V>>,
}

impl<V: ExpmComputation> FnLinearOperator<V> {
    /// Begin building a [`FnLinearOperator`].
    ///
    /// # Arguments
    /// - `dim`        — matrix dimension
    /// - `trace`      — trace of the operator (used for the diagonal shift μ)
    /// - `onenorm_fn` — computes `‖A − shift·I‖₁` for a given `shift`
    /// - `dot_fn`     — computes `output = A·input` (or `+=` when `overwrite=false`);
    ///   both `input` and `output` are caller-owned buffers of length `dim`
    /// - `dot_t_fn`   — same contract, for `A^T`
    ///
    /// # Performance note
    /// A tight `onenorm_fn` bound improves expm parameter selection (fewer Taylor
    /// iterations / scaling steps).  An overestimate is always safe for correctness.
    pub fn builder(
        dim: usize,
        trace: V,
        onenorm_fn: impl Fn(V) -> V::Real + Send + Sync + 'static,
        dot_fn: impl Fn(bool, &[V], &mut [V]) -> Result<(), QuSpinError> + Send + Sync + 'static,
        dot_t_fn: impl Fn(bool, &[V], &mut [V]) -> Result<(), QuSpinError> + Send + Sync + 'static,
    ) -> FnLinearOperatorBuilder<V> {
        FnLinearOperatorBuilder {
            dim,
            trace_val: trace,
            onenorm_fn: Arc::new(onenorm_fn),
            dot_fn: Arc::new(dot_fn),
            dot_t_fn: Arc::new(dot_t_fn),
            dot_many_fn: None,
        }
    }
}

impl<V: ExpmComputation> FnLinearOperatorBuilder<V> {
    /// Provide an optional batch `dot_many` implementation.
    ///
    /// `f(overwrite, input, output)` where `input` and `output` have shape
    /// `(dim, n_vecs)`.  If not provided, the default implementation loops over
    /// columns calling the scalar `dot_fn`.
    pub fn dot_many(
        mut self,
        f: impl Fn(bool, ArrayView2<'_, V>, ArrayViewMut2<'_, V>) -> Result<(), QuSpinError>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.dot_many_fn = Some(Arc::new(f));
        self
    }

    /// Finalise the builder and return a [`FnLinearOperator`].
    pub fn build(self) -> FnLinearOperator<V> {
        FnLinearOperator {
            dim: self.dim,
            trace_val: self.trace_val,
            onenorm_fn: self.onenorm_fn,
            dot_fn: self.dot_fn,
            dot_t_fn: self.dot_t_fn,
            dot_many_fn: self.dot_many_fn,
        }
    }
}

// ---------------------------------------------------------------------------
// LinearOperator impl
// ---------------------------------------------------------------------------

impl<V: ExpmComputation> LinearOperator<V> for FnLinearOperator<V> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn trace(&self) -> V {
        self.trace_val
    }

    fn onenorm(&self, shift: V) -> V::Real {
        (self.onenorm_fn)(shift)
    }

    /// Always `false` — the sequential expm path is used so that `dot` can
    /// parallelise internally without conflicting with the thread pool.
    fn parallel_hint(&self) -> bool {
        false
    }

    fn dot(&self, overwrite: bool, input: &[V], output: &mut [V]) -> Result<(), QuSpinError> {
        (self.dot_fn)(overwrite, input, output)
    }

    fn dot_transpose(
        &self,
        overwrite: bool,
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        (self.dot_t_fn)(overwrite, input, output)
    }

    fn dot_many(
        &self,
        overwrite: bool,
        input: ArrayView2<'_, V>,
        mut output: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError> {
        let n_vecs = input.ncols();
        if input.nrows() != self.dim || output.nrows() != self.dim || output.ncols() != n_vecs {
            return Err(QuSpinError::ValueError(format!(
                "dot_many shape mismatch: expected ({}, n_vecs) for both, \
                 got input=({}, {}), output=({}, {})",
                self.dim,
                input.nrows(),
                input.ncols(),
                output.nrows(),
                output.ncols(),
            )));
        }
        match &self.dot_many_fn {
            Some(f) => f(overwrite, input, output),
            None => {
                let n_vecs = input.ncols();
                if overwrite {
                    output.fill(V::default());
                }
                let mut tmp = vec![V::default(); self.dim];
                for k in 0..n_vecs {
                    // ndarray columns may be strided; collect into contiguous owned vec.
                    let col_in: Vec<V> = input.column(k).iter().copied().collect();
                    (self.dot_fn)(true, &col_in, &mut tmp)?;
                    for r in 0..self.dim {
                        output[[r, k]] += tmp[r];
                    }
                }
                Ok(())
            }
        }
    }

    /// Fallback: compute the full product into a temporary buffer and copy the
    /// requested row range.  O(dim) allocation — only reached if the caller
    /// bypasses `ExpmOp` and invokes `expm_multiply_par` directly.
    fn dot_chunk(
        &self,
        overwrite: bool,
        input: &[V],
        output_chunk: &mut [V],
        row_start: usize,
    ) -> Result<(), QuSpinError> {
        let mut full_out = vec![V::default(); self.dim];
        (self.dot_fn)(true, input, &mut full_out)?;
        let chunk_len = output_chunk.len();
        let src = &full_out[row_start..row_start + chunk_len];
        if overwrite {
            output_chunk.copy_from_slice(src);
        } else {
            for (dst, &s) in output_chunk.iter_mut().zip(src.iter()) {
                *dst += s;
            }
        }
        Ok(())
    }

    /// Fallback: mask the input to `rows`, compute the full transpose product,
    /// then scatter-add all output columns atomically.  O(dim) allocation.
    fn dot_transpose_chunk(
        &self,
        input: &[V],
        output: &[V::Atomic],
        rows: Range<usize>,
    ) -> Result<(), QuSpinError> {
        // Zero out entries outside the requested row range so that the full
        // transpose product equals the partial contribution from `rows`.
        let mut masked = vec![V::default(); self.dim];
        for r in rows {
            masked[r] = input[r];
        }
        let mut full_out = vec![V::default(); self.dim];
        (self.dot_t_fn)(true, &masked, &mut full_out)?;
        for (col, val) in full_out.iter().enumerate() {
            output[col].fetch_add(*val);
        }
        Ok(())
    }
}
