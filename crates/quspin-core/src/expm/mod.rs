//! Matrix-exponential action: `exp(a·A) · v`.
//!
//! Implements the partitioned Taylor/Padé algorithm of
//! Al-Mohy & Higham (2011), porting `ExpmMultiplyParallel` from
//! [`parallel-sparse-tools`](https://github.com/QuSpin/parallel-sparse-tools).
//!
//! # Module layout
//!
//! | file | contents |
//! |------|----------|
//! | `compute`    | [`ExpmComputation`] trait + impls for `f32`/`f64`/`Complex<_>` |
//! | `norm_est`   | Randomised 1-norm estimator (Higham–Tisseur 2000) |
//! | `params`     | Adaptive parameter selection: [`LazyNormInfo`], [`fragment_3_1`] |
//! | `algorithm`  | Core Taylor loop: [`expm_multiply`], [`expm_multiply_many`] |
//!
//! # High-level API
//!
//! | function | description |
//! |----------|-------------|
//! | [`expm_multiply_auto_into`]      | `exp(a·A)·f` in-place, caller supplies work buffer |
//! | [`expm_multiply_auto`]           | same, allocates work internally |
//! | [`expm_multiply_many_auto_into`] | batch variant, caller supplies work |
//! | [`expm_multiply_many_auto`]      | batch variant, allocates work |

pub mod algorithm;
pub mod compute;
pub mod norm_est;
pub mod params;

pub use algorithm::{expm_multiply, expm_multiply_many};
pub use compute::ExpmComputation;
pub use params::{LazyNormInfo, fragment_3_1};

use ndarray::{Array2, ArrayViewMut2};
use num_complex::Complex;

use crate::error::QuSpinError;
use crate::primitive::Primitive;
use crate::qmatrix::matrix::{CIndex, Index, QMatrix};

// ---------------------------------------------------------------------------
// Internal helpers: trace and 1-norm of the effective shifted matrix
// ---------------------------------------------------------------------------

/// Compute `trace(A_eff)` where `A_eff[r,c] = Σ_cindex coeffs[cindex]·A[r,c,cindex]`.
///
/// Returns `Complex<f64>` regardless of the stored element type.
pub(crate) fn compute_trace_c64<M, I, C>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[Complex<f64>],
) -> Complex<f64>
where
    M: Primitive,
    I: Index,
    C: CIndex,
{
    let mut trace = Complex::new(0.0_f64, 0.0);
    for row in 0..matrix.dim() {
        for entry in matrix.row(row) {
            if entry.col.as_usize() == row {
                trace += coeffs[entry.cindex.as_usize()] * entry.value.to_complex();
            }
        }
    }
    trace
}

/// Compute `‖A_eff − μI‖_1` (column 1-norm of the shifted effective matrix).
///
/// The 1-norm is `max_col Σ_row |A_eff[row,col] − μ·δ_{row,col}|`.
pub(crate) fn onenorm_shifted_c64<M, I, C>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[Complex<f64>],
    mu: Complex<f64>,
) -> f64
where
    M: Primitive,
    I: Index,
    C: CIndex,
{
    let n = matrix.dim();
    // col_sums[c]   = Σ_{row≠c} |A_eff[row,c]|  (off-diagonal column sums)
    // diag_acc[c]   = A_eff[c,c]                 (accumulated diagonal value)
    let mut col_sums = vec![0.0_f64; n];
    let mut diag_acc = vec![Complex::new(0.0_f64, 0.0); n];

    for row in 0..n {
        let entries = matrix.row(row);
        let mut idx = 0;

        while idx < entries.len() {
            let cur_col = entries[idx].col;
            let col = cur_col.as_usize();

            // Accumulate all cindex contributions for this (row, col) pair.
            let mut acc = Complex::new(0.0_f64, 0.0);
            while idx < entries.len() && entries[idx].col == cur_col {
                let e = &entries[idx];
                acc += coeffs[e.cindex.as_usize()] * e.value.to_complex();
                idx += 1;
            }

            if col == row {
                diag_acc[col] = acc; // track diagonal separately
            } else {
                col_sums[col] += acc.norm(); // off-diagonal: |acc|
            }
        }
    }

    // Add diagonal contribution: |A_eff[c,c] − μ|.
    // diag_acc[c] is 0 if the diagonal is not stored (structurally zero).
    for col in 0..n {
        col_sums[col] += (diag_acc[col] - mu).norm();
    }

    col_sums.into_iter().fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Internal: compute adaptive parameters (μ, m_star, s) for a given matrix
// ---------------------------------------------------------------------------

/// Compute `(m_star, s, mu_v, tol)` for `exp(a·A_eff)` via [`fragment_3_1`].
///
/// `mu_v` is the diagonal shift cast to type `V`;
/// `tol`  is `V::machine_eps()`.
fn compute_expm_params<M, I, C, V>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[V],
    a: V,
) -> Result<(usize, usize, V, V::Real), QuSpinError>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    V: ExpmComputation,
{
    let n = matrix.dim();
    let tol = V::machine_eps();

    if n == 0 {
        return Ok((0, 1, V::default(), tol));
    }

    // Convert coefficients to Complex<f64> for the norm-estimation machinery.
    let coeffs_c64: Vec<Complex<f64>> = coeffs.iter().map(|c| c.to_complex()).collect();
    let a_c64 = a.to_complex();

    // μ = trace(A_eff) / n  (diagonal shift)
    let mu_c64 = compute_trace_c64(matrix, &coeffs_c64) / n as f64;
    let mu_v = V::from_complex(mu_c64);

    // ||A_eff − μI||₁ and the operator norm |a|·||A_eff − μI||₁
    let a_1_norm = onenorm_shifted_c64(matrix, &coeffs_c64, mu_c64);
    let onenorm_exact = a_c64.norm() * a_1_norm;

    // If the operator is zero-valued, skip parameter search.
    if onenorm_exact == 0.0 {
        return Ok((0, 1, mu_v, tol));
    }

    let mut norm_info = LazyNormInfo::new(
        matrix,
        coeffs_c64,
        a_c64,
        mu_c64,
        onenorm_exact,
        2, // ell = 2 (matches parallel-sparse-tools)
    );

    // tol is unused inside fragment_3_1 (API-compat placeholder); pass eps/2.
    let (m_star, s) = fragment_3_1(&mut norm_info, 1, f64::EPSILON / 2.0, 55);

    Ok((m_star, s, mu_v, tol))
}

// ---------------------------------------------------------------------------
// High-level API — scalar
// ---------------------------------------------------------------------------

/// Compute `exp(a·A) · f` in-place, deriving μ, m_star, s adaptively.
///
/// The caller supplies the scratch buffer `work` (length ≥ `2 * matrix.dim()`).
///
/// # Errors
/// Returns `ValueError` if buffer lengths are inconsistent.
pub fn expm_multiply_auto_into<M, I, C, V>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[V],
    a: V,
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
            "work.len()={} must be >= 2*matrix.dim()={}",
            work.len(),
            2 * n
        )));
    }

    let (m_star, s, mu, tol) = compute_expm_params(matrix, coeffs, a)?;
    expm_multiply(matrix, coeffs, a, mu, s, m_star, tol, f, work)
}

/// Compute `exp(a·A) · f` in-place, allocating the scratch buffer internally.
///
/// Convenience wrapper around [`expm_multiply_auto_into`].
pub fn expm_multiply_auto<M, I, C, V>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[V],
    a: V,
    f: &mut [V],
) -> Result<(), QuSpinError>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    V: ExpmComputation,
{
    let mut work = vec![V::default(); 2 * matrix.dim()];
    expm_multiply_auto_into(matrix, coeffs, a, f, &mut work)
}

// ---------------------------------------------------------------------------
// High-level API — batch
// ---------------------------------------------------------------------------

/// Compute `exp(a·A) · F` in-place for multiple column vectors.
///
/// `f`    has shape `(dim, n_vecs)`.
/// `work` has shape `(2 * dim, n_vecs)`.
///
/// Parameters μ, m_star, s are computed once from the matrix and reused for
/// all columns.
///
/// # Errors
/// Returns `ValueError` if array shapes are inconsistent.
pub fn expm_multiply_many_auto_into<M, I, C, V>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[V],
    a: V,
    f: ArrayViewMut2<'_, V>,
    work: ArrayViewMut2<'_, V>,
) -> Result<(), QuSpinError>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    V: ExpmComputation,
{
    let n = matrix.dim();
    if f.nrows() != n {
        return Err(QuSpinError::ValueError(format!(
            "f.nrows()={} must equal matrix.dim()={}",
            f.nrows(),
            n
        )));
    }

    let (m_star, s, mu, tol) = compute_expm_params(matrix, coeffs, a)?;
    expm_multiply_many(matrix, coeffs, a, mu, s, m_star, tol, f, work)
}

/// Compute `exp(a·A) · F` in-place for multiple column vectors, allocating
/// the scratch buffer internally.
///
/// Convenience wrapper around [`expm_multiply_many_auto_into`].
pub fn expm_multiply_many_auto<M, I, C, V>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[V],
    a: V,
    mut f: ArrayViewMut2<'_, V>,
) -> Result<(), QuSpinError>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    V: ExpmComputation,
{
    let n = matrix.dim();
    let n_vecs = f.ncols();
    let mut work_buf = Array2::from_elem((2 * n, n_vecs), V::default());
    expm_multiply_many_auto_into(matrix, coeffs, a, f.view_mut(), work_buf.view_mut())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qmatrix::matrix::Entry;
    use num_complex::Complex;
    use std::f64::consts::E;

    /// 2×2 diagonal QMatrix with a single cindex.
    fn diag2(a: f64, b: f64) -> QMatrix<f64, i32, u8> {
        let data = vec![Entry::new(a, 0_i32, 0_u8), Entry::new(b, 1_i32, 0_u8)];
        let indptr = vec![0_i32, 1, 2];
        QMatrix::from_csr(indptr, data)
    }

    #[test]
    fn trace_diagonal() {
        let mat = diag2(3.0, 7.0);
        let coeffs = vec![Complex::new(1.0, 0.0)];
        let tr = compute_trace_c64(&mat, &coeffs);
        assert!((tr.re - 10.0).abs() < 1e-14);
        assert!(tr.im.abs() < 1e-14);
    }

    #[test]
    fn onenorm_shifted_identity_shift() {
        // A = diag(1, 2), mu = 1.5 → A - mu*I = diag(-0.5, 0.5)
        // ||A - mu*I||_1 = 0.5
        let mat = diag2(1.0, 2.0);
        let mu = Complex::new(1.5, 0.0);
        let coeffs = vec![Complex::new(1.0, 0.0)];
        let norm = onenorm_shifted_c64(&mat, &coeffs, mu);
        assert!((norm - 0.5).abs() < 1e-14, "got {norm}");
    }

    #[test]
    fn expm_multiply_diagonal_exact() {
        // For diagonal A = diag(λ₀, λ₁), exp(a·A)·e_i = exp(a·λ_i)·e_i.
        // Use a=1, A = diag(1, 2): result should be [e, e²].
        let mat = diag2(1.0, 2.0);
        let coeffs = vec![1.0_f64];
        let a = 1.0_f64;
        let mu = 1.5_f64; // trace/n
        let s = 1;
        let m_star = 20;
        let tol = f64::EPSILON;
        let mut f = vec![1.0_f64, 1.0_f64];
        let mut work = vec![0.0_f64; 4];

        expm_multiply(&mat, &coeffs, a, mu, s, m_star, tol, &mut f, &mut work).unwrap();

        let e1 = E; // exp(1)
        let e2 = E * E; // exp(2)
        assert!((f[0] - e1).abs() < 1e-10, "f[0]={}, expected {}", f[0], e1);
        assert!((f[1] - e2).abs() < 1e-10, "f[1]={}, expected {}", f[1], e2);
    }

    #[test]
    fn expm_multiply_auto_diagonal() {
        // Same as above but using the auto parameter selection.
        let mat = diag2(1.0, 2.0);
        let coeffs = vec![1.0_f64];
        let a = 1.0_f64;
        let mut f = vec![1.0_f64, 1.0_f64];

        expm_multiply_auto(&mat, &coeffs, a, &mut f).unwrap();

        assert!((f[0] - E).abs() < 1e-10, "f[0]={}", f[0]);
        assert!((f[1] - E * E).abs() < 1e-10, "f[1]={}", f[1]);
    }

    #[test]
    fn expm_multiply_auto_complex() {
        // A = diag(i, 2i), a = -i (imaginary time):
        // exp(-i · i · λ) = exp(λ) — should give [e, e²].
        let data = vec![
            Entry::new(Complex::new(0.0, 1.0), 0_i32, 0_u8),
            Entry::new(Complex::new(0.0, 2.0), 1_i32, 0_u8),
        ];
        let indptr = vec![0_i32, 1, 2];
        let mat: QMatrix<Complex<f64>, i32, u8> = QMatrix::from_csr(indptr, data);

        let coeffs = vec![Complex::new(1.0, 0.0)];
        let a = Complex::new(0.0, -1.0); // -i
        let mut f = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];

        expm_multiply_auto(&mat, &coeffs, a, &mut f).unwrap();

        assert!((f[0].re - E).abs() < 1e-10, "f[0].re={}", f[0].re);
        assert!(f[0].im.abs() < 1e-10, "f[0].im={}", f[0].im);
        assert!((f[1].re - E * E).abs() < 1e-10, "f[1].re={}", f[1].re);
        assert!(f[1].im.abs() < 1e-10, "f[1].im={}", f[1].im);
    }

    #[test]
    fn expm_multiply_many_auto_diagonal() {
        // Two identical columns — should give the same result as scalar.
        let mat = diag2(1.0, 2.0);
        let coeffs = vec![1.0_f64];
        let a = 1.0_f64;

        let mut f_arr = Array2::ones((2, 2_usize));
        expm_multiply_many_auto(&mat, &coeffs, a, f_arr.view_mut()).unwrap();

        for col in 0..2 {
            assert!(
                (f_arr[[0, col]] - E).abs() < 1e-10,
                "col {col}: f[0]={} expected {E}",
                f_arr[[0, col]]
            );
            assert!(
                (f_arr[[1, col]] - E * E).abs() < 1e-10,
                "col {col}: f[1]={} expected {}",
                f_arr[[1, col]],
                E * E
            );
        }
    }
}
