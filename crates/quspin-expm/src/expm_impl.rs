//! Free-function entry points for `exp(a·A) · v`.
//!
//! Thin wrappers over [`ExpmOp`] that build a fresh action per call.  Use
//! these for one-shot evaluations; for repeated calls with the same `(op, a)`
//! pair, construct an [`ExpmOp`] once and reuse it.

use ndarray::{ArrayViewMut1, ArrayViewMut2};

use quspin_types::ExpmComputation;
use quspin_types::LinearOperator;
use quspin_types::QuSpinError;

use crate::expm_op::ExpmOp;

/// Compute `exp(a·A) · f` in-place, deriving μ, m_star, s adaptively.
///
/// The caller supplies the scratch buffer `work` (length ≥ `2 * op.dim()`).
///
/// Repeated calls with the same `(op, a)` should construct an [`ExpmOp`] once
/// and reuse it — this convenience function rebuilds the parameter selection
/// on every call.
pub fn expm_multiply_auto_into<V, Op>(
    op: &Op,
    a: V,
    f: ArrayViewMut1<'_, V>,
    work: &mut [V],
) -> Result<(), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    ExpmOp::new(op, a)?.apply_into(f, work)
}

/// Allocate work internally and call [`expm_multiply_auto_into`].
pub fn expm_multiply_auto<V, Op>(op: &Op, a: V, f: ArrayViewMut1<'_, V>) -> Result<(), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    ExpmOp::new(op, a)?.apply(f)
}

/// Batch variant of [`expm_multiply_auto_into`] for `(dim, n_vecs)` array.
pub fn expm_multiply_many_auto_into<V, Op>(
    op: &Op,
    a: V,
    f: ArrayViewMut2<'_, V>,
    work: ArrayViewMut2<'_, V>,
) -> Result<(), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    ExpmOp::new(op, a)?.apply_many_into(f, work)
}

/// Allocate work internally and call [`expm_multiply_many_auto_into`].
pub fn expm_multiply_many_auto<V, Op>(
    op: &Op,
    a: V,
    f: ArrayViewMut2<'_, V>,
) -> Result<(), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    ExpmOp::new(op, a)?.apply_many(f)
}
