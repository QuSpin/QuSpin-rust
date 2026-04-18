//! Trace and one-norm computations over `QMatrix` with cindex-linear coefficients.
//!
//! Used by `QMatrixOperator` to satisfy the `LinearOperator` trait.

use num_complex::Complex;
use quspin_types::Primitive;

use crate::qmatrix::matrix::{CIndex, Index, QMatrix};

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
    let mut col_sums = vec![0.0_f64; n];
    let mut diag_acc = vec![Complex::new(0.0_f64, 0.0); n];

    for row in 0..n {
        let entries = matrix.row(row);
        let mut idx = 0;

        while idx < entries.len() {
            let cur_col = entries[idx].col;
            let col = cur_col.as_usize();

            let mut acc = Complex::new(0.0_f64, 0.0);
            while idx < entries.len() && entries[idx].col == cur_col {
                let e = &entries[idx];
                acc += coeffs[e.cindex.as_usize()] * e.value.to_complex();
                idx += 1;
            }

            if col == row {
                diag_acc[col] = acc;
            } else {
                col_sums[col] += acc.norm();
            }
        }
    }

    for col in 0..n {
        col_sums[col] += (diag_acc[col] - mu).norm();
    }

    col_sums.into_iter().fold(0.0_f64, f64::max)
}
