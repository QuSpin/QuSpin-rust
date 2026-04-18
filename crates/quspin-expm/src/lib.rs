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
pub mod norm_est;
pub mod params;

pub use algorithm::{PAR_THRESHOLD, expm_multiply, expm_multiply_many, expm_multiply_par};
pub use params::{LazyNormInfo, fragment_3_1};
pub use quspin_types::{
    AtomicAccum, DynLinearOperator, ExpmComputation, FnLinearOperator, LinearOperator,
};

use ndarray::{Array2, ArrayViewMut1, ArrayViewMut2};

use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// Internal: compute adaptive parameters (μ, m_star, s) for a given operator
// ---------------------------------------------------------------------------

/// Compute `(m_star, s, mu_v, tol)` for `exp(a·A_eff)` via [`fragment_3_1`].
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

    // μ = trace(A_eff) / n  (diagonal shift)
    let mu_v = op.trace() * V::from_real(V::real_from_f64(1.0 / n as f64));

    // onenorm_exact = |a| * ||A_eff - μI||_1  (in f64)
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

/// Compute `exp(a·A) · f` in-place, deriving μ, m_star, s adaptively.
///
/// The caller supplies the scratch buffer `work` (length ≥ `2 * op.dim()`).
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
    let n = op.dim();
    if f.len() != n {
        return Err(QuSpinError::ValueError(format!(
            "f.len()={} must equal op.dim()={}",
            f.len(),
            n
        )));
    }
    let (m_star, s, mu_v, tol) = compute_expm_params(op, a)?;
    expm_multiply(op, a, mu_v, s, m_star, tol, f, work)
}

/// Allocate work internally and call [`expm_multiply_auto_into`].
pub fn expm_multiply_auto<V, Op>(op: &Op, a: V, f: ArrayViewMut1<'_, V>) -> Result<(), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    let mut work = vec![V::default(); 2 * op.dim()];
    expm_multiply_auto_into(op, a, f, &mut work)
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
    let n = op.dim();
    if f.nrows() != n {
        return Err(QuSpinError::ValueError(format!(
            "f.nrows()={} must equal op.dim()={}",
            f.nrows(),
            n
        )));
    }
    let (m_star, s, mu_v, tol) = compute_expm_params(op, a)?;
    expm_multiply_many(op, a, mu_v, s, m_star, tol, f, work)
}

/// Allocate work internally and call [`expm_multiply_many_auto_into`].
pub fn expm_multiply_many_auto<V, Op>(
    op: &Op,
    a: V,
    mut f: ArrayViewMut2<'_, V>,
) -> Result<(), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    let n = op.dim();
    let n_vecs = f.ncols();
    let mut work_buf = Array2::from_elem((2 * n, n_vecs), V::default());
    expm_multiply_many_auto_into(op, a, f.view_mut(), work_buf.view_mut())
}
