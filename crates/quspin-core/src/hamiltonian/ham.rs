use crate::error::QuSpinError;
use crate::primitive::Primitive;
use crate::qmatrix::matrix::{CIndex, Index, QMatrix};
use ndarray::{ArrayView2, ArrayViewMut2};
use num_complex::Complex;
use std::ops::{Add, Sub};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CoeffFn type alias
// ---------------------------------------------------------------------------

pub type CoeffFn = Arc<dyn Fn(f64) -> Complex<f64> + Send + Sync>;

// ---------------------------------------------------------------------------
// Hamiltonian<M, I, C>
// ---------------------------------------------------------------------------

/// Time-dependent Hamiltonian wrapping a [`QMatrix`].
///
/// The matrix is partitioned into `num_coeff` operator strings.  Each string
/// has either:
/// - a **static** coefficient (`None`) — always multiplied by `1.0`, or
/// - a **dynamic** coefficient (`Some(f)`) — multiplied by `f(t)`.
///
/// All output methods are fixed to `Complex<f64>` regardless of the stored
/// element type `M`.
pub struct Hamiltonian<M: Primitive, I: Index, C: CIndex> {
    pub(crate) matrix: QMatrix<M, I, C>,
    /// One entry per operator string (cindex `0..num_coeff`).
    /// `None` = static (coefficient 1.0), `Some(f)` = time-dependent.
    pub(crate) coeff_fns: Vec<Option<CoeffFn>>,
}

impl<M: Primitive, I: Index, C: CIndex> Hamiltonian<M, I, C> {
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    /// Wrap a `QMatrix` and a list of coefficient descriptors.
    ///
    /// Each entry corresponds to one cindex: `None` marks a static term
    /// (coefficient 1.0) and `Some(f)` marks a time-dependent term.
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff_fns.len() != matrix.num_coeff()`.
    pub fn new(
        matrix: QMatrix<M, I, C>,
        coeff_fns: Vec<Option<CoeffFn>>,
    ) -> Result<Self, QuSpinError> {
        let expected = matrix.num_coeff();
        if coeff_fns.len() != expected {
            return Err(QuSpinError::ValueError(format!(
                "coeff_fns.len() = {} but matrix.num_coeff() = {}",
                coeff_fns.len(),
                expected
            )));
        }
        Ok(Hamiltonian { matrix, coeff_fns })
    }

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    pub fn dim(&self) -> usize {
        self.matrix.dim()
    }

    pub fn num_coeff(&self) -> usize {
        self.matrix.num_coeff()
    }

    // ------------------------------------------------------------------
    // Private: evaluate coefficients at time t
    // ------------------------------------------------------------------

    fn eval_coeffs(&self, time: f64) -> Vec<Complex<f64>> {
        self.coeff_fns
            .iter()
            .map(|f| match f {
                Some(f) => f(time),
                None => Complex::new(1.0, 0.0),
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Time-parameterised output methods (output type fixed to Complex<f64>)
    // ------------------------------------------------------------------

    /// Count non-zero entries that `to_csr` would emit at the given `time`.
    pub fn to_csr_nnz(&self, time: f64, drop_zeros: bool) -> Result<usize, QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        self.matrix.to_csr_nnz::<Complex<f64>>(&coeffs, drop_zeros)
    }

    /// Fill pre-allocated CSR buffers with the matrix evaluated at `time`.
    pub fn to_csr_into(
        &self,
        time: f64,
        drop_zeros: bool,
        indptr: &mut [I],
        indices: &mut [I],
        data: &mut [Complex<f64>],
    ) -> Result<(), QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        self.matrix
            .to_csr_into::<Complex<f64>>(&coeffs, drop_zeros, indptr, indices, data)
    }

    /// Allocate and return `(indptr, indices, data)` for the matrix at `time`.
    #[allow(clippy::type_complexity)]
    pub fn to_csr(
        &self,
        time: f64,
        drop_zeros: bool,
    ) -> Result<(Vec<I>, Vec<I>, Vec<Complex<f64>>), QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        self.matrix.to_csr::<Complex<f64>>(&coeffs, drop_zeros)
    }

    /// Fill a pre-allocated dense buffer `output[r * dim + col]` with the matrix at `time`.
    pub fn to_dense_into(&self, time: f64, output: &mut [Complex<f64>]) -> Result<(), QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        self.matrix.to_dense_into::<Complex<f64>>(&coeffs, output)
    }

    /// Allocate and return a flat row-major dense matrix at `time`.
    pub fn to_dense(&self, time: f64) -> Result<Vec<Complex<f64>>, QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        self.matrix.to_dense::<Complex<f64>>(&coeffs)
    }

    /// Matrix-vector product: `output[row] = Σ_c coeff_c(t) * Σ_col M[c,row,col] * input[col]`.
    ///
    /// When `overwrite` is `true`, `output` is zeroed before accumulation.
    pub fn dot(
        &self,
        overwrite: bool,
        time: f64,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
    ) -> Result<(), QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        self.matrix
            .dot::<Complex<f64>>(overwrite, &coeffs, input, output)
    }

    /// Transpose matrix-vector product.
    pub fn dot_transpose(
        &self,
        overwrite: bool,
        time: f64,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
    ) -> Result<(), QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        self.matrix
            .dot_transpose::<Complex<f64>>(overwrite, &coeffs, input, output)
    }

    /// Batch matrix–vector product over a `(dim, n_vecs)` array.
    pub fn dot_many(
        &self,
        overwrite: bool,
        time: f64,
        input: ArrayView2<'_, Complex<f64>>,
        output: ArrayViewMut2<'_, Complex<f64>>,
    ) -> Result<(), QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        self.matrix
            .dot_many::<Complex<f64>>(overwrite, &coeffs, input, output)
    }

    /// Batch transpose matrix–vector product over a `(dim, n_vecs)` array.
    pub fn dot_transpose_many(
        &self,
        overwrite: bool,
        time: f64,
        input: ArrayView2<'_, Complex<f64>>,
        output: ArrayViewMut2<'_, Complex<f64>>,
    ) -> Result<(), QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        self.matrix
            .dot_transpose_many::<Complex<f64>>(overwrite, &coeffs, input, output)
    }

    // ------------------------------------------------------------------
    // Matrix-exponential action
    // ------------------------------------------------------------------

    /// Compute `exp(a · H(time)) · f` in-place (single vector).
    ///
    /// Uses the adaptive partitioned Taylor method of Al-Mohy & Higham (2011).
    /// The diagonal shift μ, scaling factor s, and Taylor order m_star are
    /// chosen automatically based on the operator norm at `time`.
    ///
    /// # Arguments
    /// - `time`  — evaluation time for the time-dependent coefficients
    /// - `a`     — scalar factor (e.g. `-i·dt` for time evolution)
    /// - `f`     — input/output vector (length = `self.dim()`)
    ///
    /// # Errors
    /// Returns `ValueError` if `f.len() != self.dim()`.
    pub fn expm_dot(
        &self,
        time: f64,
        a: Complex<f64>,
        f: &mut [Complex<f64>],
    ) -> Result<(), QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        let op = crate::expm::QMatrixOperator::new(&self.matrix, coeffs)?;
        crate::expm::expm_multiply_auto(&op, a, ndarray::aview_mut1(f))
    }

    /// Compute `exp(a · H(time)) · F` in-place for multiple column vectors.
    ///
    /// `f` has shape `(dim, n_vecs)`.  The matrix-exponential parameters are
    /// computed once from the operator and reused for every column.
    ///
    /// # Errors
    /// Returns `ValueError` if `f.nrows() != self.dim()`.
    pub fn expm_dot_many(
        &self,
        time: f64,
        a: Complex<f64>,
        f: ArrayViewMut2<'_, Complex<f64>>,
    ) -> Result<(), QuSpinError> {
        let coeffs = self.eval_coeffs(time);
        let op = crate::expm::QMatrixOperator::new(&self.matrix, coeffs)?;
        crate::expm::expm_multiply_many_auto(&op, a, f)
    }
}

// ---------------------------------------------------------------------------
// Add / Sub with cindex merging
// ---------------------------------------------------------------------------

impl<M, I, C> Add for Hamiltonian<M, I, C>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    QMatrix<M, I, C>: Add<Output = QMatrix<M, I, C>>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let (merged_fns, lhs_map, rhs_map) =
            build_merged_cindex_maps::<C>(&self.coeff_fns, &rhs.coeff_fns);
        let lhs_mat = self.matrix.remap_cindices(&lhs_map);
        let rhs_mat = rhs.matrix.remap_cindices(&rhs_map);
        let result_mat = lhs_mat + rhs_mat;
        Hamiltonian {
            matrix: result_mat,
            coeff_fns: merged_fns,
        }
    }
}

impl<M, I, C> Sub for Hamiltonian<M, I, C>
where
    M: Primitive,
    I: Index,
    C: CIndex,
    QMatrix<M, I, C>: Sub<Output = QMatrix<M, I, C>>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let (merged_fns, lhs_map, rhs_map) =
            build_merged_cindex_maps::<C>(&self.coeff_fns, &rhs.coeff_fns);
        let lhs_mat = self.matrix.remap_cindices(&lhs_map);
        let rhs_mat = rhs.matrix.remap_cindices(&rhs_map);
        let result_mat = lhs_mat - rhs_mat;
        Hamiltonian {
            matrix: result_mat,
            coeff_fns: merged_fns,
        }
    }
}

/// Build merged `coeff_fns` list and per-operand cindex remapping vectors.
///
/// - All `None` (static) entries are merged to a single cindex (always cindex 0).
/// - Each time-dependent function (`Some(Arc)`) is deduplicated by `Arc::ptr_eq`.
///   Functions that are the same `Arc` share a single cindex in the result.
fn build_merged_cindex_maps<C: CIndex>(
    lhs_fns: &[Option<CoeffFn>],
    rhs_fns: &[Option<CoeffFn>],
) -> (Vec<Option<CoeffFn>>, Vec<C>, Vec<C>) {
    // Slot 0 is always the static (None) slot.
    let mut merged: Vec<Option<CoeffFn>> = vec![None];

    let mut make_map = |fns: &[Option<CoeffFn>]| -> Vec<C> {
        fns.iter()
            .map(|f| match f {
                None => C::from_usize(0), // all static entries share cindex 0
                Some(arc) => {
                    // Search for an existing dynamic entry with the same Arc.
                    let pos = merged.iter().enumerate().skip(1).find_map(|(i, existing)| {
                        if let Some(existing_arc) = existing
                            && Arc::ptr_eq(existing_arc, arc)
                        {
                            return Some(i);
                        }
                        None
                    });
                    match pos {
                        Some(idx) => C::from_usize(idx),
                        None => {
                            merged.push(Some(Arc::clone(arc)));
                            C::from_usize(merged.len() - 1)
                        }
                    }
                }
            })
            .collect()
    };

    let lhs_map = make_map(lhs_fns);
    let rhs_map = make_map(rhs_fns);

    (merged, lhs_map, rhs_map)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::space::FullSpace;
    use crate::operator::pauli::{HardcoreOp, HardcoreOperator, OpEntry};
    use crate::qmatrix::build::build_from_basis;
    use num_complex::Complex;
    use smallvec::smallvec;

    // Helper: build a simple 2-site XX Hamiltonian → QMatrix<f64, i64, u8>
    fn xx_qmatrix() -> QMatrix<f64, i64, u8> {
        let ops = smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)];
        let terms = vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops)];
        let op = HardcoreOperator::new(terms);
        let basis = FullSpace::<u32>::new(2, 2, false);
        build_from_basis::<_, u32, f64, i64, u8, _>(&op, &basis)
    }

    #[test]
    fn eval_coeffs_static_only() {
        let mat = xx_qmatrix();
        // num_coeff=1, single static entry
        let ham = Hamiltonian::new(mat, vec![None]).unwrap();
        let coeffs = ham.eval_coeffs(0.0);
        assert_eq!(coeffs.len(), 1);
        assert_eq!(coeffs[0], Complex::new(1.0, 0.0));
    }

    #[test]
    fn eval_coeffs_with_time_fn() {
        let mat = xx_qmatrix();
        // num_coeff=1, single static entry — time value should not matter
        let ham = Hamiltonian::new(mat, vec![None]).unwrap();
        let c = ham.eval_coeffs(3.1);
        assert_eq!(c, vec![Complex::new(1.0, 0.0)]);
    }

    #[test]
    fn new_wrong_coeff_count_errors() {
        let mat = xx_qmatrix();
        // mat has num_coeff=1, so expects 1 entry; pass 2 → error
        let f: CoeffFn = Arc::new(|_t| Complex::new(1.0, 0.0));
        assert!(Hamiltonian::new(mat, vec![None, Some(f)]).is_err());
    }

    #[test]
    fn dot_at_time_zero_matches_qmatrix_dot() {
        let mat = xx_qmatrix();
        let ham = Hamiltonian::new(mat.clone(), vec![None]).unwrap();

        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let mut out_ham = vec![Complex::default(); 4];
        let mut out_mat = vec![Complex::default(); 4];

        ham.dot(true, 0.0, &input, &mut out_ham).unwrap();
        let coeff = vec![Complex::new(1.0, 0.0)];
        mat.dot::<Complex<f64>>(true, &coeff, &input, &mut out_mat)
            .unwrap();

        for (a, b) in out_ham.iter().zip(out_mat.iter()) {
            assert!((a - b).norm() < 1e-12, "ham={a}, mat={b}");
        }
    }

    #[test]
    fn add_two_static_hamiltonians_doubles_entries() {
        let mat = xx_qmatrix();
        let h1 = Hamiltonian::new(mat.clone(), vec![None]).unwrap();
        let h2 = Hamiltonian::new(mat, vec![None]).unwrap();
        let h = h1 + h2;

        // H + H should give 2*H entries
        let input = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let mut output = vec![Complex::default(); 4];
        h.dot(true, 0.0, &input, &mut output).unwrap();
        // XX|11⟩ = |00⟩, so result at index 0 (state |00⟩) should be 2.0
        assert!(
            (output[0] - Complex::new(2.0, 0.0)).norm() < 1e-12,
            "output[0]={}",
            output[0]
        );
    }

    #[test]
    fn add_shared_arc_merges_to_one_cindex() {
        // Two Hamiltonians sharing the same Arc<fn> should produce a result
        // with only 2 cindices (static + shared fn), not 3.
        let f: CoeffFn = Arc::new(|t: f64| Complex::new(t, 0.0));
        let fns1 = vec![None, Some(Arc::clone(&f))];
        let fns2 = vec![None, Some(Arc::clone(&f))];
        let (merged, map1, map2) = build_merged_cindex_maps::<u8>(&fns1, &fns2);
        // Static → cindex 0, shared Arc → cindex 1
        assert_eq!(merged.len(), 2);
        assert_eq!(map1, vec![0u8, 1u8]);
        assert_eq!(map2, vec![0u8, 1u8]);
    }

    #[test]
    fn add_distinct_arcs_preserves_both_cindices() {
        let f1: CoeffFn = Arc::new(|t: f64| Complex::new(t, 0.0));
        let f2: CoeffFn = Arc::new(|t: f64| Complex::new(t * 2.0, 0.0));
        let fns1 = vec![None, Some(Arc::clone(&f1))];
        let fns2 = vec![None, Some(Arc::clone(&f2))];
        let (merged, map1, map2) = build_merged_cindex_maps::<u8>(&fns1, &fns2);
        // Static → cindex 0, two distinct Arcs → cindices 1 and 2
        assert_eq!(merged.len(), 3);
        assert_eq!(map1, vec![0u8, 1u8]);
        assert_eq!(map2, vec![0u8, 2u8]);
    }

    #[test]
    fn add_multiple_static_entries_merge_to_one() {
        // All None entries should map to cindex 0
        let fns1 = vec![None, None];
        let fns2 = vec![None, None];
        let (merged, map1, map2) = build_merged_cindex_maps::<u8>(&fns1, &fns2);
        assert_eq!(merged.len(), 1);
        assert_eq!(map1, vec![0u8, 0u8]);
        assert_eq!(map2, vec![0u8, 0u8]);
    }
}
