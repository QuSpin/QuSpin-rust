//! Internal abstraction for the shifted operator `B = a·(A − μI)`.
//!
//! Bundles the `(op, a, mu)` triple shared by every kernel in this crate
//! (norm estimation and Taylor partition), and exposes the small helpers
//! that operate on it: in-place powers `B^p · x` / `(B*)^p · x`, the
//! per-partition phase factor `η = exp(a·μ/s)`, and the per-Taylor-step
//! scale `a / (j·s)`.

use quspin_types::ExpmComputation;
use quspin_types::LinearOperator;

/// Borrowed view of the shifted operator `B = a·(A − μI)`.
pub(crate) struct ShiftedOp<'a, V, Op> {
    pub op: &'a Op,
    pub a: V,
    pub mu: V,
}

impl<'a, V, Op> ShiftedOp<'a, V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    pub fn new(op: &'a Op, a: V, mu: V) -> Self {
        Self { op, a, mu }
    }

    pub fn dim(&self) -> usize {
        self.op.dim()
    }

    /// Per-partition phase factor: `η = exp(a·μ / s)`.
    pub fn eta(&self, s: usize) -> V {
        let inv_s = V::real_from_f64(1.0 / s as f64);
        (self.a * self.mu * V::from_real(inv_s)).exp_val()
    }

    /// Per-Taylor-step scale: `a / (j·s)`.
    pub fn taylor_scale(&self, j: usize, s: usize) -> V {
        self.a * V::from_real(V::real_from_f64(1.0 / (j * s) as f64))
    }

    /// In-place: `x ← B^p · x`. `work` is scratch of length ≥ `dim()`.
    pub fn apply_pow_in_place(&self, p: usize, x: &mut [V], work: &mut [V]) {
        debug_assert_eq!(x.len(), self.dim());
        debug_assert!(work.len() >= x.len());
        for _ in 0..p {
            self.op
                .dot(true, x, work)
                .expect("ShiftedOp::apply_pow_in_place: dot failed");
            for k in 0..x.len() {
                work[k] = self.a * (work[k] - self.mu * x[k]);
            }
            x.copy_from_slice(work);
        }
    }

    /// In-place: `x ← (B^T)^p · x`.  Uses `op.dot_transpose` and the same
    /// scalars as the forward operator — pure transpose, no conjugation.
    ///
    /// This is what the Higham–Tisseur 1-norm estimator wants: the duality
    /// `‖A‖_1 = ‖A^T‖_∞` holds for any complex matrix without conjugation.
    pub fn apply_pow_in_place_transpose(&self, p: usize, x: &mut [V], work: &mut [V]) {
        debug_assert_eq!(x.len(), self.dim());
        debug_assert!(work.len() >= x.len());
        for _ in 0..p {
            self.op
                .dot_transpose(true, x, work)
                .expect("ShiftedOp::apply_pow_in_place_transpose: dot_transpose failed");
            for k in 0..x.len() {
                work[k] = self.a * (work[k] - self.mu * x[k]);
            }
            x.copy_from_slice(work);
        }
    }
}

/// Taylor-partition parameters for `expm_multiply`.
pub(crate) struct TaylorParams<R> {
    pub s: usize,
    pub m_star: usize,
    pub tol: R,
}

impl<R> TaylorParams<R> {
    pub fn new(s: usize, m_star: usize, tol: R) -> Self {
        Self { s, m_star, tol }
    }
}
