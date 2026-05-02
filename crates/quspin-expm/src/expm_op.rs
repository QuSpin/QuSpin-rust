//! User-facing matrix-exponential operator: a cached `exp(a·A)` action.
//!
//! [`ExpmOp`] bundles the shifted operator `B = a·(A − μI)` with the Taylor
//! partition parameters `(s, m_star, tol)` that the parameter-selection step
//! derives for it, so the `(m*, s, μ, tol)` computation runs only once per
//! `(op, a)` pair.  Construct it once with [`ExpmOp::new`], then call
//! [`apply`](ExpmOp::apply) / [`apply_many`](ExpmOp::apply_many) as many
//! times as needed.
//!
//! `ExpmOp` is *not* a `LinearOperator` — applying it requires running the
//! Taylor partition algorithm, which doesn't fit the lightweight matvec
//! contract that other QuSpin consumers (e.g. Krylov) expect.

use ndarray::{Array2, ArrayViewMut1, ArrayViewMut2};

use quspin_types::ExpmComputation;
use quspin_types::LinearOperator;
use quspin_types::QuSpinError;

use crate::algorithm::{expm_multiply, expm_multiply_many};
use crate::params::{LazyNormInfo, fragment_3_1};
use crate::shifted_op::{ShiftedOp, TaylorParams};

/// Cached `exp(a·A)` action over a borrowed linear operator.
///
/// See the [module docs](self) for usage.
pub struct ExpmOp<'a, V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    shift_op: ShiftedOp<'a, V, Op>,
    params: TaylorParams<V::Real>,
}

impl<'a, V, Op> ExpmOp<'a, V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    /// Construct by deriving (μ, m*, s, tol) from `(op, a)` adaptively.
    pub fn new(op: &'a Op, a: V) -> Result<Self, QuSpinError> {
        let (m_star, s, mu_v, tol) = compute_expm_params(op, a)?;
        Ok(Self {
            shift_op: ShiftedOp::new(op, a, mu_v),
            params: TaylorParams::new(s, m_star, tol),
        })
    }

    /// Construct from caller-supplied parameters, skipping the param-selection step.
    pub fn from_parts(op: &'a Op, a: V, mu: V, s: usize, m_star: usize, tol: V::Real) -> Self {
        Self {
            shift_op: ShiftedOp::new(op, a, mu),
            params: TaylorParams::new(s, m_star, tol),
        }
    }

    /// Operator dimension (rows = cols of `A`).
    pub fn dim(&self) -> usize {
        self.shift_op.dim()
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
// Parameter selection
// ---------------------------------------------------------------------------

/// Compute `(m_star, s, mu_v, tol)` for `exp(a·A)` via [`fragment_3_1`].
///
/// `mu_v` is the diagonal shift cast to type `V`;
/// `tol`  is `V::machine_eps()`.
fn compute_expm_params<V, Op>(op: &Op, a: V) -> Result<(usize, usize, V, V::Real), QuSpinError>
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
