//! User-facing matrix-exponential operator: a cached `exp(a·A)` action.
//!
//! [`ExpmOp`] bundles the shifted operator `B = a·(A − μI)` with the Taylor
//! partition parameters `(s, m_star, tol)` that the parameter-selection step
//! derives for it, so the `(m*, s, μ, tol)` computation runs only once per
//! `(op, a)` pair.  Construct it once with [`ExpmOp::new`], then call
//! [`apply`](ExpmOp::apply) / [`apply_many`](ExpmOp::apply_many) as many
//! times as needed.
//!
//! `Op` is held by value; pass `&T` to obtain a borrowed shifted view, or
//! an owned/shared type (`Box<T>`, `Arc<T>`, …) for a long-lived one — the
//! relevant `LinearOperator<V>` blanket impls live in `quspin-types`.
//!
//! `ExpmOp` is *not* a `LinearOperator` — applying it requires running the
//! Taylor partition algorithm, which doesn't fit the lightweight matvec
//! contract that other QuSpin consumers (e.g. Krylov) expect.

use std::sync::Arc;

use ndarray::{Array2, ArrayViewMut1, ArrayViewMut2};

use quspin_types::ExpmComputation;
use quspin_types::LinearOperator;
use quspin_types::QuSpinError;

use crate::algorithm::{expm_multiply, expm_multiply_many};
use crate::params::{LazyNormInfo, fragment_3_1};
use crate::shifted_op::{ShiftedOp, TaylorParams};

/// Cached `exp(a·A)` action.  See the [module docs](self) for usage.
pub struct ExpmOp<V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    shift_op: ShiftedOp<V, Op>,
    params: TaylorParams<V::Real>,
}

impl<V, Op> ExpmOp<V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    /// Construct by deriving (μ, m*, s, tol) from `(op, a)` adaptively.
    pub fn new(op: Op, a: V) -> Result<Self, QuSpinError> {
        let (m_star, s, mu_v, tol) = compute_expm_params(&op, a)?;
        Ok(Self {
            shift_op: ShiftedOp::new(op, a, mu_v),
            params: TaylorParams::new(s, m_star, tol),
        })
    }

    /// Construct from caller-supplied parameters, skipping the param-selection step.
    pub fn from_parts(op: Op, a: V, mu: V, s: usize, m_star: usize, tol: V::Real) -> Self {
        Self {
            shift_op: ShiftedOp::new(op, a, mu),
            params: TaylorParams::new(s, m_star, tol),
        }
    }

    /// Operator dimension (rows = cols of `A`).
    pub fn dim(&self) -> usize {
        self.shift_op.dim()
    }

    /// Scalar multiplier `a` such that the action is `exp(a · A)`.
    pub fn a(&self) -> V {
        self.shift_op.a
    }

    /// Diagonal shift `μ` chosen by the parameter-selection step.
    pub fn mu(&self) -> V {
        self.shift_op.mu
    }

    /// Number of partition steps `s` (the matrix is split as `(B/s)^s`).
    pub fn s(&self) -> usize {
        self.params.s
    }

    /// Truncated Taylor order `m*` per partition step.
    pub fn m_star(&self) -> usize {
        self.params.m_star
    }

    /// Convergence tolerance used by the partitioned-Taylor algorithm.
    pub fn tol(&self) -> V::Real {
        self.params.tol
    }

    /// `f ← exp(a·A) · f`.  Allocates a `2 · dim()` scratch buffer.
    pub fn apply(&self, mut f: ArrayViewMut1<'_, V>) -> Result<(), QuSpinError> {
        let mut work = vec![V::default(); 2 * self.dim()];
        self.apply_into(f.view_mut(), &mut work)
    }

    /// `f ← exp(a·A) · f` using caller-supplied scratch (length ≥ `2 · dim()`).
    pub fn apply_into(&self, f: ArrayViewMut1<'_, V>, work: &mut [V]) -> Result<(), QuSpinError> {
        expm_multiply(&self.shift_op, &self.params, f, work)
    }

    /// Batch variant of [`apply`](Self::apply) for shape `(dim, n_vecs)`.
    pub fn apply_many(&self, mut f: ArrayViewMut2<'_, V>) -> Result<(), QuSpinError> {
        let n = self.dim();
        let n_vecs = f.ncols();
        let mut work = Array2::from_elem((2 * n, n_vecs), V::default());
        self.apply_many_into(f.view_mut(), work.view_mut())
    }

    /// Batch variant of [`apply_into`](Self::apply_into) for shape `(dim, n_vecs)`.
    /// `work` must have shape `(>= 2 · dim, >= n_vecs)`.
    pub fn apply_many_into(
        &self,
        f: ArrayViewMut2<'_, V>,
        work: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError> {
        expm_multiply_many(&self.shift_op, &self.params, f, work)
    }
}

// ---------------------------------------------------------------------------
// ExpmWorker (1-D)
// ---------------------------------------------------------------------------

/// Reusable 1-D scratch holder bound to an [`ExpmOp`].
///
/// Holds an `Arc<ExpmOp<V, Op>>` so the worker can be freely moved or stored
/// independently of the originating `ExpmOp`.  Each `apply` reuses the
/// internal `2 · dim()` scratch buffer instead of reallocating.
///
/// In the unified [`ExpmOp::worker`] dispatch this is the `n_vec == 0`
/// branch.  Use [`ExpmWorker2`] when you need 2-D batch application.
pub struct ExpmWorker<V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    expm_op: Arc<ExpmOp<V, Op>>,
    work: Vec<V>,
}

impl<V, Op> ExpmWorker<V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    /// Build a worker that allocates its own `2 · dim()` scratch buffer.
    pub fn new(expm_op: Arc<ExpmOp<V, Op>>) -> Self {
        let dim = expm_op.dim();
        Self {
            expm_op,
            work: vec![V::default(); 2 * dim],
        }
    }

    /// Build a worker from a caller-supplied scratch buffer.
    ///
    /// The buffer is adopted as the worker's backing storage — no copy.
    /// Length must be ≥ `2 · expm_op.dim()`.
    pub fn with_buf(expm_op: Arc<ExpmOp<V, Op>>, work: Vec<V>) -> Result<Self, QuSpinError> {
        let need = 2 * expm_op.dim();
        if work.len() < need {
            return Err(QuSpinError::ValueError(format!(
                "ExpmWorker scratch buffer length {} < required {need}",
                work.len(),
            )));
        }
        Ok(Self { expm_op, work })
    }

    /// Operator dimension (rows = cols of `A`).
    pub fn dim(&self) -> usize {
        self.expm_op.dim()
    }

    /// Borrow the underlying `ExpmOp`.
    pub fn expm_op(&self) -> &ExpmOp<V, Op> {
        &self.expm_op
    }

    /// `f ← exp(a·A) · f`.
    pub fn apply(&mut self, f: ArrayViewMut1<'_, V>) -> Result<(), QuSpinError> {
        self.expm_op.apply_into(f, &mut self.work)
    }
}

// ---------------------------------------------------------------------------
// ExpmWorker2 (2-D batch)
// ---------------------------------------------------------------------------

/// Reusable 2-D batch scratch holder bound to an [`ExpmOp`].
///
/// Holds an `Arc<ExpmOp<V, Op>>` plus a `(2 · dim, n_vecs)` scratch buffer
/// reused across `apply` calls.
///
/// In the unified [`ExpmOp::worker`] dispatch this is the `n_vec > 0`
/// branch.  Use [`ExpmWorker`] for single-vector application.
pub struct ExpmWorker2<V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    expm_op: Arc<ExpmOp<V, Op>>,
    work: Array2<V>,
}

impl<V, Op> ExpmWorker2<V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    /// Build a worker that allocates its own `(2 · dim, n_vecs)` scratch.
    pub fn new(expm_op: Arc<ExpmOp<V, Op>>, n_vecs: usize) -> Self {
        let dim = expm_op.dim();
        Self {
            expm_op,
            work: Array2::from_elem((2 * dim, n_vecs), V::default()),
        }
    }

    /// Build a worker from a caller-supplied scratch buffer.
    ///
    /// `work.nrows()` must be ≥ `2 · expm_op.dim()`; the worker's batch
    /// capacity is `work.ncols()`.
    pub fn with_buf(expm_op: Arc<ExpmOp<V, Op>>, work: Array2<V>) -> Result<Self, QuSpinError> {
        let need = 2 * expm_op.dim();
        if work.nrows() < need {
            return Err(QuSpinError::ValueError(format!(
                "ExpmWorker2 scratch first dim {} < required {need}",
                work.nrows(),
            )));
        }
        Ok(Self { expm_op, work })
    }

    /// Operator dimension (rows = cols of `A`).
    pub fn dim(&self) -> usize {
        self.expm_op.dim()
    }

    /// Batch capacity: maximum number of column vectors per `apply` call,
    /// determined by the underlying scratch buffer's column count.
    pub fn n_vecs(&self) -> usize {
        self.work.ncols()
    }

    /// Borrow the underlying `ExpmOp`.
    pub fn expm_op(&self) -> &ExpmOp<V, Op> {
        &self.expm_op
    }

    /// `F ← exp(a·A) · F` for `F` of shape `(dim, k)`.  `f.ncols()` must
    /// not exceed [`n_vecs`](Self::n_vecs).
    pub fn apply(&mut self, f: ArrayViewMut2<'_, V>) -> Result<(), QuSpinError> {
        let want = f.ncols();
        let have = self.work.ncols();
        if want > have {
            return Err(QuSpinError::ValueError(format!(
                "ExpmWorker2: input has {want} columns, worker capacity is {have}"
            )));
        }
        let work_view = self.work.slice_mut(ndarray::s![.., ..want]);
        self.expm_op.apply_many_into(f, work_view)
    }
}

// ---------------------------------------------------------------------------
// AnyExpmWorker — runtime-dispatched 1-D / 2-D worker
// ---------------------------------------------------------------------------

/// Runtime-dispatched worker returned by [`ExpmOp::worker`].
///
/// `Single` wraps a 1-D [`ExpmWorker`] (selected when `n_vec == 0`); `Batch`
/// wraps a 2-D [`ExpmWorker2`] (selected when `n_vec > 0`).  This mirrors
/// the Python `expm_op.worker(n_vec) -> ExpmWorker | ExpmWorker2` dispatch.
pub enum AnyExpmWorker<V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    /// 1-D worker — `n_vec == 0`.  `apply` takes an `ArrayViewMut1`.
    Single(ExpmWorker<V, Op>),
    /// 2-D batch worker — `n_vec > 0`.  `apply` takes an `ArrayViewMut2`
    /// of shape `(dim, k)` with `k <= n_vec`.
    Batch(ExpmWorker2<V, Op>),
}

impl<V, Op> ExpmOp<V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    /// Build a worker bound to `self` (which must already be in an `Arc`).
    ///
    /// `n_vec == 0` produces a 1-D [`ExpmWorker`] for single-vector
    /// application; `n_vec > 0` produces a 2-D [`ExpmWorker2`] whose
    /// `apply` accepts shape `(dim, k)` with `k <= n_vec`.
    ///
    /// For Rust callers that want a typed worker directly, see
    /// [`ExpmWorker::new`] / [`ExpmWorker2::new`].
    pub fn worker(self: &Arc<Self>, n_vec: usize) -> AnyExpmWorker<V, Op> {
        if n_vec == 0 {
            AnyExpmWorker::Single(ExpmWorker::new(Arc::clone(self)))
        } else {
            AnyExpmWorker::Batch(ExpmWorker2::new(Arc::clone(self), n_vec))
        }
    }
}

// ---------------------------------------------------------------------------
// Parameter selection
// ---------------------------------------------------------------------------

/// Compute `(m_star, s, mu_v, tol)` for `exp(a·A)` via [`fragment_3_1`].
///
/// `mu_v` is the diagonal shift cast to type `V`;
/// `tol`  is `V::machine_eps()`.
pub fn compute_expm_params<V, Op>(op: &Op, a: V) -> Result<(usize, usize, V, V::Real), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    let n = op.dim();
    let tol = V::machine_eps();

    if n == 0 {
        return Ok((0, 1, V::default(), tol));
    }

    // μ = trace(A) / n  (diagonal shift)
    let mu_v = op.trace() * V::from_real(V::real_from_f64(1.0 / n as f64));

    // onenorm_exact = |a| * ||A - μI||_1  (in f64)
    let a_norm = a.to_complex().norm();
    let a_1_norm_shifted = op.onenorm(mu_v);
    let a_1_norm_f64 = V::from_real(a_1_norm_shifted).to_complex().re;
    let onenorm_exact = a_norm * a_1_norm_f64;

    if onenorm_exact == 0.0 {
        return Ok((0, 1, mu_v, tol));
    }

    let mut norm_info = LazyNormInfo::new(op, a, mu_v, onenorm_exact, 2);
    let (m_star, s) = fragment_3_1(&mut norm_info, 1, f64::EPSILON / 2.0, 55);

    Ok((m_star, s, mu_v, tol))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex;
    use quspin_types::FnLinearOperator;

    /// Build a 2x2 diagonal `LinearOperator` for `H = diag(d0, d1)`.
    fn diag2(d0: Complex<f64>, d1: Complex<f64>) -> FnLinearOperator<Complex<f64>> {
        let dvec = [d0, d1];
        FnLinearOperator::builder(
            2,
            d0 + d1,
            move |shift| (d0 - shift).norm().max((d1 - shift).norm()),
            move |overwrite, input, output| {
                for i in 0..2 {
                    let v = dvec[i] * input[i];
                    if overwrite {
                        output[i] = v;
                    } else {
                        output[i] += v;
                    }
                }
                Ok(())
            },
            move |overwrite, input, output| {
                for i in 0..2 {
                    let v = dvec[i] * input[i];
                    if overwrite {
                        output[i] = v;
                    } else {
                        output[i] += v;
                    }
                }
                Ok(())
            },
        )
        .build()
    }

    #[test]
    fn worker_apply_matches_apply() {
        let op = diag2(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
        let a = Complex::new(0.0, -std::f64::consts::PI / 4.0);
        let expm_op = ExpmOp::new(&op, a).unwrap();
        let mut f1 = ndarray::array![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];
        let mut f2 = f1.clone();
        expm_op.apply(f1.view_mut()).unwrap();
        let arc = Arc::new(expm_op);
        ExpmWorker::new(Arc::clone(&arc))
            .apply(f2.view_mut())
            .unwrap();
        for (a, b) in f1.iter().zip(f2.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn worker_reuse_across_calls() {
        // Apply twice with the same worker — second call must not see stale state.
        let op = diag2(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
        let a = Complex::new(0.0, -0.5);
        let expm_op = Arc::new(ExpmOp::new(&op, a).unwrap());
        let mut worker = ExpmWorker::new(expm_op);

        let mut f = ndarray::array![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        worker.apply(f.view_mut()).unwrap();
        let after_first = f.clone();

        // Reset and apply again — should produce the same result.
        f[0] = Complex::new(1.0, 0.0);
        f[1] = Complex::new(0.0, 0.0);
        worker.apply(f.view_mut()).unwrap();
        for (a, b) in f.iter().zip(after_first.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn worker_many_apply_matches_apply_many() {
        let op = diag2(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
        let a = Complex::new(0.0, -std::f64::consts::PI / 6.0);
        let expm_op = ExpmOp::new(&op, a).unwrap();
        let f = Array2::from_shape_vec(
            (2, 3),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.5, 0.0),
                Complex::new(0.25, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.75, 0.0),
            ],
        )
        .unwrap();
        let mut f1 = f.clone();
        let mut f2 = f.clone();
        expm_op.apply_many(f1.view_mut()).unwrap();
        let arc = Arc::new(expm_op);
        ExpmWorker2::new(arc, 3).apply(f2.view_mut()).unwrap();
        for (a, b) in f1.iter().zip(f2.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn worker_many_capacity_exceeded_errors() {
        let op = diag2(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
        let expm_op = Arc::new(ExpmOp::new(&op, Complex::new(0.0, -0.5)).unwrap());
        let mut worker = ExpmWorker2::new(expm_op, 2);
        // input has 3 cols but worker capacity is 2 -> error
        let mut f = Array2::from_elem((2, 3), Complex::new(1.0, 0.0));
        assert!(worker.apply(f.view_mut()).is_err());
    }

    #[test]
    fn worker_with_buf_short_errors() {
        let op = diag2(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
        let expm_op = Arc::new(ExpmOp::new(&op, Complex::new(0.0, -0.5)).unwrap());
        let buf = vec![Complex::new(0.0, 0.0); 1]; // need 2*dim = 4
        assert!(ExpmWorker::with_buf(expm_op, buf).is_err());
    }

    /// `Arc<ExpmOp>::worker(0)` returns the 1-D variant.
    #[test]
    fn dispatch_worker_n_vec_zero_is_single() {
        let op = diag2(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
        let expm_op = Arc::new(ExpmOp::new(&op, Complex::new(0.0, -0.5)).unwrap());
        match expm_op.worker(0) {
            AnyExpmWorker::Single(_) => {}
            AnyExpmWorker::Batch(_) => panic!("n_vec == 0 must return Single"),
        }
    }

    /// `Arc<ExpmOp>::worker(n)` for n > 0 returns the 2-D variant.
    #[test]
    fn dispatch_worker_n_vec_positive_is_batch() {
        let op = diag2(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
        let expm_op = Arc::new(ExpmOp::new(&op, Complex::new(0.0, -0.5)).unwrap());
        match expm_op.worker(3) {
            AnyExpmWorker::Single(_) => panic!("n_vec > 0 must return Batch"),
            AnyExpmWorker::Batch(w) => assert_eq!(w.n_vecs(), 3),
        }
    }
}
