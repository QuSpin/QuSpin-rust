//! Core Taylor-series partition algorithm for computing `exp(a·A)·f`.
//!
//! Ports the C++ `expm_multiply` / `expm_multiply_batch` kernels from
//! `parallel-sparse-tools`, which implement the partitioned Taylor method of
//! Al-Mohy & Higham (2011).

use ndarray::ArrayViewMut2;

use crate::error::QuSpinError;
use crate::primitive::Primitive;
use crate::qmatrix::matrix::{CIndex, Index, QMatrix};

use super::compute::ExpmComputation;

// ---------------------------------------------------------------------------
// Scalar variant
// ---------------------------------------------------------------------------

/// Compute `exp(a·A) · f` in-place (single vector).
///
/// Uses the partitioned Taylor expansion:
///
/// ```text
/// exp(a·A) = exp(a·μ) · [exp(a·(A−μI)/s)]^s
///          ≈ exp(a·μ) · [Σ_{j=0}^{m_star} (a·(A−μI)/(j·s))^j]^s
/// ```
///
/// # Arguments
/// - `matrix`  — sparse matrix A
/// - `coeffs`  — operator coefficients (`len = matrix.num_coeff()`)
/// - `a`       — global scalar factor
/// - `mu`      — diagonal shift μ (usually `trace(A)/n`)
/// - `s`       — partition count (scaling factor)
/// - `m_star`  — Taylor truncation order per partition
/// - `tol`     — convergence tolerance (typically `V::machine_eps()`)
/// - `f`       — input/output vector, length = `matrix.dim()`
/// - `work`    — scratch buffer, length ≥ `2 * matrix.dim()`
///
/// # Errors
/// Returns `ValueError` if buffer lengths are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn expm_multiply<M, I, C, V>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[V],
    a: V,
    mu: V,
    s: usize,
    m_star: usize,
    tol: V::Real,
    f: &mut [V],
    work: &mut [V],
) -> Result<(), QuSpinError>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    V: ExpmComputation,
{
    let n = matrix.dim();

    if f.len() != n {
        return Err(QuSpinError::ValueError(format!(
            "f.len()={} must equal matrix.dim()={}",
            f.len(),
            n
        )));
    }
    if work.len() < 2 * n {
        return Err(QuSpinError::ValueError(format!(
            "work.len()={} must be >= 2 * matrix.dim()={}",
            work.len(),
            2 * n
        )));
    }

    if n == 0 || s == 0 {
        return Ok(());
    }

    // Split scratch buffer: b1 = current Taylor term, tmp = matvec output.
    let (b1, tmp) = work[..2 * n].split_at_mut(n);

    // η = exp(a·μ / s) — applied once per outer partition.
    let inv_s = V::real_from_f64(1.0 / s as f64);
    let eta = (a * mu * V::from_real(inv_s)).exp_val();

    // B = F = f  (initial: Taylor term B1 = accumulated sum F)
    b1.copy_from_slice(f);

    for _i in 0..s {
        // c1 = ‖B‖_∞  (inf-norm of the current Taylor term / starting vector)
        let mut c1 = inf_norm(b1);

        'taylor: for j in 1..=m_star {
            // tmp = A · B  (overwrite=true zeros tmp first)
            matrix.dot(true, coeffs, b1, tmp)?;

            // scale = a / (j · s)
            let scale = a * V::from_real(V::real_from_f64(1.0 / (j * s) as f64));

            let mut c2 = V::Real::default();
            let mut c3 = V::Real::default();

            // tmp[k] = scale · (tmp[k] − μ · B[k])      (new Taylor term)
            // f[k]  += tmp[k]
            for k in 0..n {
                tmp[k] = scale * (tmp[k] - mu * b1[k]);
                f[k] += tmp[k];
                let abs_tmp = tmp[k].abs_val();
                let abs_f = f[k].abs_val();
                if abs_tmp > c2 {
                    c2 = abs_tmp;
                }
                if abs_f > c3 {
                    c3 = abs_f;
                }
            }

            // Convergence: latest term negligible relative to accumulated sum.
            if c1 + c2 <= tol * c3 {
                break 'taylor;
            }

            c1 = c2;
            // B = tmp (advance to next Taylor term)
            b1.copy_from_slice(tmp);
        }

        // F *= η  then  B = F  (reset starting point for next partition)
        for fk in f.iter_mut() {
            *fk *= eta;
        }
        b1.copy_from_slice(f);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Batch variant
// ---------------------------------------------------------------------------

/// Compute `exp(a·A) · F` in-place for multiple column vectors simultaneously.
///
/// `f` and `work` have shape `(dim, n_vecs)`.  Parameters `a`, `mu`, `s`,
/// `m_star`, `tol` are the same for all columns; only `f` differs.
///
/// The convergence check aggregates norms across **all** columns (joint
/// termination: the Taylor loop stops only when every column has converged).
///
/// # Errors
/// Returns `ValueError` if array shapes are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn expm_multiply_many<M, I, C, V>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[V],
    a: V,
    mu: V,
    s: usize,
    m_star: usize,
    tol: V::Real,
    mut f: ArrayViewMut2<'_, V>,
    mut work: ArrayViewMut2<'_, V>,
) -> Result<(), QuSpinError>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    V: ExpmComputation,
{
    let n = matrix.dim();
    let n_vecs = f.ncols();

    if f.nrows() != n {
        return Err(QuSpinError::ValueError(format!(
            "f.nrows()={} must equal matrix.dim()={}",
            f.nrows(),
            n
        )));
    }
    if work.nrows() < 2 * n || work.ncols() < n_vecs {
        return Err(QuSpinError::ValueError(format!(
            "work shape ({},{}) must be (>= 2*{}, >= {})",
            work.nrows(),
            work.ncols(),
            n,
            n_vecs
        )));
    }

    if n == 0 || n_vecs == 0 || s == 0 {
        return Ok(());
    }

    let inv_s = V::real_from_f64(1.0 / s as f64);
    let eta = (a * mu * V::from_real(inv_s)).exp_val();

    // Views into the work array: b1 = work[0..n, :], tmp = work[n..2n, :]
    let (mut b1_view, mut tmp_view) = work.view_mut().split_at(ndarray::Axis(0), n);

    // B = F (copy f into b1)
    b1_view.assign(&f);

    for _i in 0..s {
        // c1 = max over all (k, col) of |B[k, col]|
        let mut c1 = inf_norm_2d(b1_view.view());

        'taylor: for j in 1..=m_star {
            // tmp = A · B  (batch matvec)
            matrix.dot_many(true, coeffs, b1_view.view(), tmp_view.view_mut())?;

            let scale = a * V::from_real(V::real_from_f64(1.0 / (j * s) as f64));

            let mut c2 = V::Real::default();
            let mut c3 = V::Real::default();

            for k in 0..n {
                for col in 0..n_vecs {
                    let b = b1_view[[k, col]];
                    let t = scale * (tmp_view[[k, col]] - mu * b);
                    tmp_view[[k, col]] = t;
                    let fval = f[[k, col]] + t;
                    f[[k, col]] = fval;
                    let abs_t = t.abs_val();
                    let abs_f = fval.abs_val();
                    if abs_t > c2 {
                        c2 = abs_t;
                    }
                    if abs_f > c3 {
                        c3 = abs_f;
                    }
                }
            }

            if c1 + c2 <= tol * c3 {
                break 'taylor;
            }

            c1 = c2;
            // B = tmp
            b1_view.assign(&tmp_view);
        }

        // F *= η;  B = F
        for fk in f.iter_mut() {
            *fk *= eta;
        }
        b1_view.assign(&f);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Vector infinity norm: `max_k |v[k]|`.
fn inf_norm<V: ExpmComputation>(v: &[V]) -> V::Real {
    v.iter()
        .map(|x| x.abs_val())
        .fold(V::Real::default(), |acc, x| if x > acc { x } else { acc })
}

/// Matrix infinity norm over all elements: `max_{k,col} |M[k,col]|`.
fn inf_norm_2d<V: ExpmComputation>(m: ndarray::ArrayView2<'_, V>) -> V::Real {
    m.iter()
        .map(|x| x.abs_val())
        .fold(V::Real::default(), |acc, x| if x > acc { x } else { acc })
}
