use crate::primitive::Primitive;
use crate::qmatrix::matrix::{CIndex, Index, QMatrix};
use num_complex::Complex;

/// Estimate `||(a*(A - μI))^p||_1` using a randomized block 1-norm estimator.
///
/// Implements a simplified variant of the Higham–Tisseur (2000) block algorithm
/// that alternates forward (`B = a*(A - μI)`) and backward (`B* = ā*(A* - μ̄I)`)
/// matrix–vector products to iteratively refine the estimate.
///
/// # Arguments
/// - `matrix`  — the sparse matrix A
/// - `coeffs`  — effective operator coefficients (`len == matrix.num_coeff()`);
///   passed unchanged to [`QMatrix::dot`] / [`QMatrix::dot_transpose`]
/// - `a`       — global scalar factor in `B = a*(A - μI)`
/// - `mu`      — diagonal shift (usually `trace(A)/n`)
/// - `p`       — power to apply
/// - `ell`     — number of probe vectors (typically 2)
///
/// Returns a lower bound on `||B^p||_1`; the true 1-norm satisfies `result ≤ ||B^p||_1`.
pub fn onenorm_matrix_power_nnm<M, I, C>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[Complex<f64>],
    a: Complex<f64>,
    mu: Complex<f64>,
    p: usize,
    ell: usize,
) -> f64
where
    M: Primitive,
    I: Index,
    C: CIndex,
{
    let n = matrix.dim();
    if n == 0 {
        return 0.0;
    }
    if p == 0 {
        // B^0 = I, ||I||_1 = 1
        return 1.0;
    }

    let a_conj = a.conj();
    let mu_conj = mu.conj();
    let mut est = 0.0_f64;

    // -----------------------------------------------------------------------
    // Forward pass: apply B = a*(A - μI) p times to ell probe vectors.
    // Probe vectors use alternating sign patterns (deterministic, no rng dep).
    // -----------------------------------------------------------------------
    let mut ax = vec![Complex::new(0.0, 0.0); n];
    let mut best_col: Option<Vec<Complex<f64>>> = None;

    for j in 0..ell {
        let mut x: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let sign = if (i + j) % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
                Complex::new(sign, 0.0)
            })
            .collect();

        apply_bp(matrix, coeffs, a, mu, p, &mut x, &mut ax);

        let col_norm: f64 = x.iter().map(|v| v.norm()).sum();
        if col_norm > est {
            est = col_norm;
            best_col = Some(x);
        }
    }

    // -----------------------------------------------------------------------
    // Refinement step (one Higham–Tisseur iteration).
    // Use the sign matrix of the best column to form a backward probe, then
    // apply the adjoint operator B* p times and pick the row with the largest
    // ∞-norm as the next forward starting vector.
    // -----------------------------------------------------------------------
    if let Some(y) = best_col {
        // s = sign(y) (element-wise; treat zero as +1)
        let s: Vec<Complex<f64>> = y
            .iter()
            .map(|v| {
                let nrm = v.norm();
                if nrm == 0.0 {
                    Complex::new(1.0, 0.0)
                } else {
                    v / nrm
                }
            })
            .collect();

        // Apply B*^p to s: B* = ā*(A^T - μ̄I)
        let mut z = s;
        let mut az = vec![Complex::new(0.0, 0.0); n];
        apply_bp_adj(matrix, coeffs, a_conj, mu_conj, p, &mut z, &mut az);

        // h[i] = |z[i]|  (row inf-norms of the 1-column Z matrix)
        // Row with the largest h[i] gives the best unit-vector starting point.
        let i_star = z
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut x_unit = vec![Complex::new(0.0, 0.0); n];
        x_unit[i_star] = Complex::new(1.0, 0.0);

        apply_bp(matrix, coeffs, a, mu, p, &mut x_unit, &mut ax);

        let col_norm: f64 = x_unit.iter().map(|v| v.norm()).sum();
        est = est.max(col_norm);
    }

    est
}

// ---------------------------------------------------------------------------
// Helpers: apply B^p and (B*)^p in-place
// ---------------------------------------------------------------------------

/// In-place: `x ← (a*(A - μI))^p * x`.  `work` is scratch space of length `n`.
fn apply_bp<M, I, C>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[Complex<f64>],
    a: Complex<f64>,
    mu: Complex<f64>,
    p: usize,
    x: &mut [Complex<f64>],
    work: &mut [Complex<f64>],
) where
    M: Primitive,
    I: Index,
    C: CIndex,
{
    for _ in 0..p {
        work.fill(Complex::new(0.0, 0.0));
        matrix
            .dot(true, coeffs, x, work)
            .expect("onenorm_matrix_power_nnm: dot failed");
        // x ← a * (work - μ * x)
        for i in 0..x.len() {
            x[i] = a * (work[i] - mu * x[i]);
        }
    }
}

/// In-place: `x ← (ā*(A^T - μ̄I))^p * x`.  `work` is scratch space of length `n`.
fn apply_bp_adj<M, I, C>(
    matrix: &QMatrix<M, I, C>,
    coeffs: &[Complex<f64>],
    a_conj: Complex<f64>,
    mu_conj: Complex<f64>,
    p: usize,
    x: &mut [Complex<f64>],
    work: &mut [Complex<f64>],
) where
    M: Primitive,
    I: Index,
    C: CIndex,
{
    for _ in 0..p {
        work.fill(Complex::new(0.0, 0.0));
        matrix
            .dot_transpose(true, coeffs, x, work)
            .expect("onenorm_matrix_power_nnm: dot_transpose failed");
        // x ← ā * (work - μ̄ * x)
        for i in 0..x.len() {
            x[i] = a_conj * (work[i] - mu_conj * x[i]);
        }
    }
}
