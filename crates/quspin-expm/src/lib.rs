//! Matrix-exponential action: `exp(a·A) · v`.
//!
//! Implements the partitioned Taylor/Padé algorithm of
//! Al-Mohy & Higham (2011), porting `ExpmMultiplyParallel` from
//! [`parallel-sparse-tools`](https://github.com/QuSpin/parallel-sparse-tools).
//!
//! Generic over `impl LinearOperator<V>` (trait in `quspin-types`) — does not
//! depend on `quspin-matrix`. Concrete wrappers like `QMatrixOperator` live in
//! `quspin-matrix`.

pub mod algorithm;
pub mod expm_op;
pub mod norm_est;
pub mod params;
mod shifted_op;

pub use algorithm::PAR_THRESHOLD;
pub use expm_op::ExpmOp;
pub use params::{LazyNormInfo, fragment_3_1};
pub use quspin_types::{
    AtomicAccum, DynLinearOperator, ExpmComputation, FnLinearOperator, LinearOperator,
};

use ndarray::{ArrayViewMut1, ArrayViewMut2};

use quspin_types::QuSpinError;

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
