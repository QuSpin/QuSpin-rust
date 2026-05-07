//! Row-range CSR materialisation that bypasses `QMatrix`.
//!
//! Each call walks rows `[row_start, row_end)` of the operator + basis pair
//! directly, returning `(indptr, indices, data)` in the layout petsc4py's
//! `Mat.setValuesCSR(I, J, V)` expects.  Memory is bounded by the slab —
//! useful when each MPI rank only needs its locally-owned rows of a matrix
//! that's otherwise too large to materialise globally.

use num_complex::Complex;
use quspin_basis::BasisSpace;
use quspin_bitbasis::BitInt;
use quspin_operator::Operator;
use quspin_types::QuSpinError;
use rayon::prelude::*;

use crate::qmatrix::CIndex;
use crate::qmatrix::matrix::PARALLEL_DIM_THRESHOLD;

/// Drop-zeros tolerance — matches `QMatrix::to_csr_into` / `materialize`.
const ZERO_TOL: f64 = 4.0 * f64::EPSILON;

/// Output triple of [`csr_slab_kernel`] and the public dispatchers.
///
/// - `indptr` length `row_end - row_start + 1`, zero-based, `indptr[0] == 0`,
///   `indptr[-1] == nnz_local`.
/// - `indices` length `nnz_local`, **global** column indices.
/// - `data` length `nnz_local`, accumulated `Σ_c coeffs[c] * stored_value`.
pub type CsrSlab = (Vec<i64>, Vec<i64>, Vec<Complex<f64>>);

/// Row-range CSR kernel — generic over operator/bitint/cindex/basis-space.
///
/// Validates `row_start <= row_end <= basis.size()` and
/// `coeffs.len() == op.num_cindices()`, then walks each row in parallel
/// (when the slab covers `PARALLEL_DIM_THRESHOLD` or more rows) and emits the
/// merged CSR slab.
pub fn csr_slab_kernel<H, B, C, S>(
    op: &H,
    basis: &S,
    coeffs: &[Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError>
where
    H: Operator<C> + Sync,
    B: BitInt,
    C: CIndex + Copy,
    S: BasisSpace<B> + Sync,
{
    if row_start > row_end {
        return Err(QuSpinError::ValueError(format!(
            "row_start ({row_start}) must be <= row_end ({row_end})"
        )));
    }
    let dim = basis.size();
    if row_end > dim {
        return Err(QuSpinError::ValueError(format!(
            "row_end ({row_end}) must be <= basis.size() ({dim})"
        )));
    }
    if coeffs.len() != op.num_cindices() {
        return Err(QuSpinError::ValueError(format!(
            "coeffs length {} must equal op.num_cindices() {}",
            coeffs.len(),
            op.num_cindices()
        )));
    }

    let n_local = row_end - row_start;

    // Per-row work: collect (col, contribution) entries via op.apply, sort + merge.
    let process_row = |r: usize| -> Vec<(i64, Complex<f64>)> {
        let state = basis.state_at(r);
        let mut entries: Vec<(i64, Complex<f64>)> = Vec::new();
        op.apply(state, |cindex, amp, new_state| {
            if let Some(col) = basis.index(new_state) {
                entries.push((col as i64, amp * coeffs[cindex.as_usize()]));
            }
        });
        // Sort by col, merge same-col, drop_zeros (relative tolerance).
        entries.sort_unstable_by_key(|&(c, _)| c);
        let mut merged: Vec<(i64, Complex<f64>)> = Vec::with_capacity(entries.len());
        let mut i = 0;
        while i < entries.len() {
            let col = entries[i].0;
            let mut acc = Complex::new(0.0, 0.0);
            let mut scale = 0.0f64;
            while i < entries.len() && entries[i].0 == col {
                let v = entries[i].1;
                acc += v;
                scale += v.norm();
                i += 1;
            }
            if !drop_zeros || acc.norm() > scale * ZERO_TOL {
                merged.push((col, acc));
            }
        }
        merged
    };

    let row_results: Vec<Vec<(i64, Complex<f64>)>> = if n_local >= PARALLEL_DIM_THRESHOLD {
        (row_start..row_end)
            .into_par_iter()
            .map(process_row)
            .collect()
    } else {
        (row_start..row_end).map(process_row).collect()
    };

    // Flatten into CSR.
    let total_nnz: usize = row_results.iter().map(|r| r.len()).sum();
    let mut indptr: Vec<i64> = Vec::with_capacity(n_local + 1);
    let mut indices: Vec<i64> = Vec::with_capacity(total_nnz);
    let mut data: Vec<Complex<f64>> = Vec::with_capacity(total_nnz);
    indptr.push(0);
    for row in row_results {
        for (col, val) in row {
            indices.push(col);
            data.push(val);
        }
        indptr.push(indices.len() as i64);
    }
    Ok((indptr, indices, data))
}
