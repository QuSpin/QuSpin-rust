use crate::error::QuSpinError;
use crate::primitive::Primitive;
use ndarray::{ArrayView2, ArrayViewMut2};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// Relative tolerance for treating a materialised matrix entry as zero.
///
/// An accumulated value `acc` is considered negligible when
/// `|acc| ≤ scale * ZERO_TOL`, where `scale` is the sum of magnitudes of the
/// individual contributions to that entry.  This catches floating-point
/// near-cancellations (e.g. `3.0 - 3.0` that rounds to a ULP-level residual)
/// without a hardcoded absolute threshold.
///
/// Matches the tolerance used in `Subspace::build` / `SymmetricSubspace::build`.
const ZERO_TOL: f64 = 4.0 * f64::EPSILON;

// ---------------------------------------------------------------------------
// Index / CIndex sealed traits
// ---------------------------------------------------------------------------

mod private {
    pub trait Sealed {}
}

/// Sealed marker for valid CSR row/column index types (`i32`, `i64`).
pub trait Index:
    private::Sealed
    + Copy
    + Default
    + Debug
    + Send
    + Sync
    + Eq
    + Ord
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
{
    fn as_usize(self) -> usize;
    fn from_usize(v: usize) -> Self;
}

macro_rules! impl_index {
    ($($T:ty),*) => {
        $(
            impl private::Sealed for $T {}
            impl Index for $T {
                #[inline] fn as_usize(self) -> usize { self as usize }
                #[inline] fn from_usize(v: usize) -> Self { v as $T }
            }
        )*
    }
}

impl_index!(i32, i64);

/// Sealed marker for valid operator-string index types (`u8`, `u16`).
pub trait CIndex: private::Sealed + Copy + Default + Debug + Send + Sync + Eq + Ord {
    fn as_usize(self) -> usize;
    fn from_usize(v: usize) -> Self;
}

macro_rules! impl_cindex {
    ($($T:ty),*) => {
        $(
            impl private::Sealed for $T {}
            impl CIndex for $T {
                #[inline] fn as_usize(self) -> usize { self as usize }
                #[inline] fn from_usize(v: usize) -> Self { v as $T }
            }
        )*
    }
}

impl_cindex!(u8, u16);

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// A single non-zero entry in the sparse matrix.
///
/// `M` is the stored element type; it may differ from the computation type
/// `V` used in [`QMatrix::dot`] and related methods.
///
/// Replaces the C++ `std::tuple<T, I, J>` from `qmatrix.hpp`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry<M, I, C> {
    pub value: M,
    pub col: I,
    /// Operator-string index: selects the coefficient from `coeff[cindex]`
    /// in matrix–vector products.
    pub cindex: C,
}

impl<M: Copy, I: Copy + Ord, C: Copy + Ord> Entry<M, I, C> {
    pub fn new(value: M, col: I, cindex: C) -> Self {
        Entry { value, col, cindex }
    }

    /// Comparison for CSR row-sorting: primary on `col`, secondary on `cindex`.
    pub fn lt_col_cindex(&self, other: &Self) -> bool {
        self.col < other.col || (self.col == other.col && self.cindex < other.cindex)
    }
}

// ---------------------------------------------------------------------------
// QMatrix
// ---------------------------------------------------------------------------

/// Sparse quantum matrix in CSR format.
///
/// Parameterised by:
/// - `M: Primitive` — stored element type (may be real even when the
///   physics is complex; compute methods are generic over a separate output
///   type `V`)
/// - `I: Index` — row/column index type (i32 or i64)
/// - `C: CIndex` — operator-string index type (u8 or u16)
///
/// Mirrors `qmatrix<T, I, J>` from `qmatrix.hpp`.
#[derive(Clone)]
pub struct QMatrix<M, I, C> {
    dim: usize,
    /// Number of distinct cindex values (max cindex + 1).
    num_coeff: usize,
    /// Row pointers, length = dim + 1.  `data[indptr[r]..indptr[r+1]]` holds
    /// the entries of row `r`.
    indptr: Vec<I>,
    /// Non-zero entries sorted within each row by (col, cindex).
    data: Vec<Entry<M, I, C>>,
}

impl<M: Primitive, I: Index, C: CIndex> QMatrix<M, I, C> {
    /// Construct from pre-built CSR arrays.  Entries are sorted per-row if
    /// not already in (col, cindex) order.
    pub fn from_csr(indptr: Vec<I>, data: Vec<Entry<M, I, C>>) -> Self {
        let dim = indptr.len().saturating_sub(1);
        let num_coeff = data
            .iter()
            .map(|e| e.cindex.as_usize() + 1)
            .max()
            .unwrap_or(0);
        let mut mat = QMatrix {
            dim,
            num_coeff,
            indptr,
            data,
        };
        if !mat.is_sorted() {
            mat.sort_entries();
        }
        mat
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn num_coeff(&self) -> usize {
        self.num_coeff
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// The slice of entries for row `r`.
    #[inline]
    pub fn row(&self, r: usize) -> &[Entry<M, I, C>] {
        let start = self.indptr[r].as_usize();
        let end = self.indptr[r + 1].as_usize();
        &self.data[start..end]
    }

    fn is_sorted(&self) -> bool {
        (0..self.dim).all(|r| self.row(r).windows(2).all(|w| w[0].lt_col_cindex(&w[1])))
    }

    fn sort_entries(&mut self) {
        for r in 0..self.dim {
            let start = self.indptr[r].as_usize();
            let end = self.indptr[r + 1].as_usize();
            self.data[start..end]
                .sort_unstable_by(|a, b| a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex)));
        }
    }

    // ------------------------------------------------------------------
    // Private length-check helpers (type-erased)
    // ------------------------------------------------------------------

    fn check_coeff_len(&self, len: usize) -> Result<(), QuSpinError> {
        if len != self.num_coeff {
            return Err(QuSpinError::ValueError(format!(
                "coeff length {} != num_coeff {}",
                len, self.num_coeff
            )));
        }
        Ok(())
    }

    fn check_many_lens(
        &self,
        coeff_len: usize,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<(), QuSpinError> {
        self.check_coeff_len(coeff_len)?;
        if input_shape[0] != self.dim {
            return Err(QuSpinError::ValueError(format!(
                "input first axis {} != dim {}",
                input_shape[0], self.dim
            )));
        }
        if output_shape != input_shape {
            return Err(QuSpinError::ValueError(format!(
                "output shape {:?} != input shape {:?}",
                output_shape, input_shape
            )));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // remap_cindices
    // ------------------------------------------------------------------

    /// Remap each entry's `cindex` via `mapping[old] = new`.
    ///
    /// Re-sorts each row by `(col, cindex)` after remapping and recomputes
    /// `num_coeff`.
    ///
    /// # Precondition
    /// The remapping must not create duplicate `(col, cindex)` keys within
    /// any row.
    pub fn remap_cindices(mut self, mapping: &[C]) -> Self {
        for e in self.data.iter_mut() {
            e.cindex = mapping[e.cindex.as_usize()];
        }
        // Re-sort each row and recompute num_coeff.
        self.sort_entries();
        self.num_coeff = self
            .data
            .iter()
            .map(|e| e.cindex.as_usize() + 1)
            .max()
            .unwrap_or(0);
        self
    }

    // ------------------------------------------------------------------
    // to_csr — materialise with coefficients into a plain CSR matrix
    // ------------------------------------------------------------------

    /// Count the number of structurally non-zero entries that
    /// `to_csr_into` / `to_csr` would emit for the given `coeff`.
    ///
    /// The output type `V` may differ from the stored type `M`; entries are
    /// converted via `V::from_complex(entry.value.to_complex())` before
    /// multiplication.
    ///
    /// When `drop_zeros` is `true`, entries whose accumulated value is
    /// negligible relative to the sum of contribution magnitudes
    /// (`|acc| ≤ scale * ZERO_TOL`) are excluded from the count.
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff`.
    pub fn to_csr_nnz<V: Primitive>(
        &self,
        coeff: &[V],
        drop_zeros: bool,
    ) -> Result<usize, QuSpinError> {
        self.check_coeff_len(coeff.len())?;
        let mut count = 0usize;
        for r in 0..self.dim {
            let mut i = 0;
            let row = self.row(r);
            while i < row.len() {
                let col = row[i].col;
                let mut acc = V::default();
                let mut scale = 0.0f64;
                while i < row.len() && row[i].col == col {
                    let entry_as_v = V::from_complex(row[i].value.to_complex());
                    let contrib = coeff[row[i].cindex.as_usize()] * entry_as_v;
                    acc += contrib;
                    scale += contrib.magnitude();
                    i += 1;
                }
                if !drop_zeros || acc.magnitude() > scale * ZERO_TOL {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Fill pre-allocated CSR buffers with the materialised matrix.
    ///
    /// Entries with the same `(row, col)` are summed across cindices:
    /// `M[row, col] = Σ_c coeff[c] * stored_value[c, row, col]`.
    ///
    /// When `drop_zeros` is `true`, entries whose accumulated value is
    /// negligible relative to the sum of contribution magnitudes
    /// (`|acc| ≤ scale * ZERO_TOL`) are omitted from `indices` and `data`.
    ///
    /// # Buffer sizes
    /// - `indptr`: exactly `dim + 1`
    /// - `indices`, `data`: at least as large as `to_csr_nnz(coeff, drop_zeros)`
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff`,
    /// `indptr.len() != dim + 1`, or the output buffers are too small.
    pub fn to_csr_into<V: Primitive>(
        &self,
        coeff: &[V],
        drop_zeros: bool,
        indptr: &mut [I],
        indices: &mut [I],
        data: &mut [V],
    ) -> Result<(), QuSpinError> {
        self.check_coeff_len(coeff.len())?;
        if indptr.len() != self.dim + 1 {
            return Err(QuSpinError::ValueError(format!(
                "indptr length {} != dim + 1 = {}",
                indptr.len(),
                self.dim + 1
            )));
        }
        if indices.len() != data.len() {
            return Err(QuSpinError::ValueError(format!(
                "indices length {} != data length {}",
                indices.len(),
                data.len()
            )));
        }

        let mut pos = 0usize;
        indptr[0] = I::from_usize(0);
        for r in 0..self.dim {
            let mut i = 0;
            let row = self.row(r);
            while i < row.len() {
                let col = row[i].col;
                let mut acc = V::default();
                let mut scale = 0.0f64;
                while i < row.len() && row[i].col == col {
                    let entry_as_v = V::from_complex(row[i].value.to_complex());
                    let contrib = coeff[row[i].cindex.as_usize()] * entry_as_v;
                    acc += contrib;
                    scale += contrib.magnitude();
                    i += 1;
                }
                if !drop_zeros || acc.magnitude() > scale * ZERO_TOL {
                    if pos >= indices.len() {
                        return Err(QuSpinError::ValueError(format!(
                            "output buffer too small: needed more than {} entries",
                            indices.len()
                        )));
                    }
                    indices[pos] = col;
                    data[pos] = acc;
                    pos += 1;
                }
            }
            indptr[r + 1] = I::from_usize(pos);
        }
        Ok(())
    }

    /// Allocate and return `(indptr, indices, data)` for the materialised matrix.
    ///
    /// Convenience wrapper over `to_csr_nnz` + `to_csr_into`.
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff`.
    #[allow(clippy::type_complexity)]
    pub fn to_csr<V: Primitive>(
        &self,
        coeff: &[V],
        drop_zeros: bool,
    ) -> Result<(Vec<I>, Vec<I>, Vec<V>), QuSpinError> {
        let nnz = self.to_csr_nnz(coeff, drop_zeros)?;
        let mut indptr = vec![I::default(); self.dim + 1];
        let mut indices = vec![I::default(); nnz];
        let mut data = vec![V::default(); nnz];
        self.to_csr_into(coeff, drop_zeros, &mut indptr, &mut indices, &mut data)?;
        Ok((indptr, indices, data))
    }

    // ------------------------------------------------------------------
    // to_dense — materialise into a flat row-major dense array
    // ------------------------------------------------------------------

    /// Fill a pre-allocated row-major dense buffer with the materialised matrix.
    ///
    /// `output` must have length `dim * dim`.  It is zeroed before accumulation.
    /// `output[r * dim + col] = Σ_c coeff[c] * stored_value[c, r, col]`.
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff` or
    /// `output.len() != dim * dim`.
    pub fn to_dense_into<V: Primitive>(
        &self,
        coeff: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        self.check_coeff_len(coeff.len())?;
        let expected = self.dim * self.dim;
        if output.len() != expected {
            return Err(QuSpinError::ValueError(format!(
                "output length {} != dim * dim = {}",
                output.len(),
                expected
            )));
        }
        output.fill(V::default());
        for r in 0..self.dim {
            for e in self.row(r) {
                let entry_as_v = V::from_complex(e.value.to_complex());
                output[r * self.dim + e.col.as_usize()] += coeff[e.cindex.as_usize()] * entry_as_v;
            }
        }
        Ok(())
    }

    /// Allocate and return a row-major dense array for the materialised matrix.
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff`.
    pub fn to_dense<V: Primitive>(&self, coeff: &[V]) -> Result<Vec<V>, QuSpinError> {
        let mut output = vec![V::default(); self.dim * self.dim];
        self.to_dense_into(coeff, &mut output)?;
        Ok(output)
    }

    // ------------------------------------------------------------------
    // dot — output[r] = Σ_{col,cindex} coeff[cindex] * value * input[col]
    // ------------------------------------------------------------------

    /// Compute `output[r] = Σ_{col, cindex} coeff[cindex] * value * input[col]`
    /// for each row `r`.
    ///
    /// The output type `V` may differ from the stored type `M`; entries are
    /// converted via `V::from_complex(entry.value.to_complex())`.
    ///
    /// If `overwrite` is `true`, `output` is zeroed before accumulation.
    ///
    /// # Errors
    /// Returns `ValueError` if any array length is inconsistent.
    pub fn dot<V: Primitive>(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        // Safety: from_shape((n, 1), slice) always succeeds when n == slice.len().
        let input_view = ArrayView2::from_shape((input.len(), 1), input).unwrap();
        let output_view = ArrayViewMut2::from_shape((output.len(), 1), output).unwrap();
        self.dot_many(overwrite, coeff, input_view, output_view)
    }

    // ------------------------------------------------------------------
    // dot_transpose
    // ------------------------------------------------------------------

    /// Compute `output[col] += Σ_{r} coeff[cindex] * value * input[r]`
    /// (the transpose matrix–vector product).
    ///
    /// Sequential to avoid data races on `output`.
    pub fn dot_transpose<V: Primitive>(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        let input_view = ArrayView2::from_shape((input.len(), 1), input).unwrap();
        let output_view = ArrayViewMut2::from_shape((output.len(), 1), output).unwrap();
        self.dot_transpose_many(overwrite, coeff, input_view, output_view)
    }

    // ------------------------------------------------------------------
    // dot_many — batch matrix–matrix product over (dim, n_vecs) arrays
    // ------------------------------------------------------------------

    /// Batch matrix–vector product: `output[[r, k]] = Σ_{col} coeff[cindex] * value * input[[col, k]]`.
    ///
    /// Both `input` and `output` must have shape `(dim, n_vecs)`.
    /// If `overwrite` is `true`, `output` is zeroed before accumulation.
    ///
    /// # Errors
    /// Returns `ValueError` if any shape is inconsistent with `dim` or `num_coeff`.
    pub fn dot_many<V: Primitive>(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: ArrayView2<'_, V>,
        mut output: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError> {
        self.check_many_lens(coeff.len(), input.shape(), output.shape())?;

        if overwrite {
            output.fill(V::default());
        }

        let n_vecs = input.ncols();
        for r in 0..self.dim {
            for e in self.row(r) {
                let scale = coeff[e.cindex.as_usize()] * V::from_complex(e.value.to_complex());
                let col = e.col.as_usize();
                for k in 0..n_vecs {
                    output[[r, k]] += scale * input[[col, k]];
                }
            }
        }
        Ok(())
    }

    /// Batch transpose matrix–vector product: `output[[col, k]] += Σ_{r} coeff[cindex] * value * input[[r, k]]`.
    ///
    /// Both `input` and `output` must have shape `(dim, n_vecs)`.
    /// Sequential to avoid data races on `output`.
    ///
    /// # Errors
    /// Returns `ValueError` if any shape is inconsistent with `dim` or `num_coeff`.
    pub fn dot_transpose_many<V: Primitive>(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: ArrayView2<'_, V>,
        mut output: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError> {
        self.check_many_lens(coeff.len(), input.shape(), output.shape())?;

        if overwrite {
            output.fill(V::default());
        }

        let n_vecs = input.ncols();
        for r in 0..self.dim {
            for e in self.row(r) {
                let scale = coeff[e.cindex.as_usize()] * V::from_complex(e.value.to_complex());
                let col = e.col.as_usize();
                for k in 0..n_vecs {
                    output[[col, k]] += scale * input[[r, k]];
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use num_complex::Complex;

    /// Build a 2×2 matrix [[2, 1], [3, 4]] with a single operator string.
    fn mat_2x2() -> QMatrix<f64, i64, u8> {
        let indptr = vec![0i64, 2, 4];
        let data = vec![
            Entry::new(2.0f64, 0i64, 0u8),
            Entry::new(1.0f64, 1i64, 0u8),
            Entry::new(3.0f64, 0i64, 0u8),
            Entry::new(4.0f64, 1i64, 0u8),
        ];
        QMatrix::from_csr(indptr, data)
    }

    #[test]
    fn qmatrix_metadata() {
        let m = mat_2x2();
        assert_eq!(m.dim(), 2);
        assert_eq!(m.nnz(), 4);
        assert_eq!(m.num_coeff(), 1);
    }

    #[test]
    fn qmatrix_row_slices() {
        let m = mat_2x2();
        let r0 = m.row(0);
        assert_eq!(r0.len(), 2);
        assert_eq!(r0[0], Entry::new(2.0, 0i64, 0u8));
        assert_eq!(r0[1], Entry::new(1.0, 1i64, 0u8));
        let r1 = m.row(1);
        assert_eq!(r1[0], Entry::new(3.0, 0i64, 0u8));
        assert_eq!(r1[1], Entry::new(4.0, 1i64, 0u8));
    }

    #[test]
    fn dot_matvec() {
        // [[2,1],[3,4]] * [1,2] = [4, 11], coeff=[1]
        let m = mat_2x2();
        let coeff = vec![1.0f64];
        let input = vec![1.0f64, 2.0];
        let mut output = vec![0.0f64; 2];
        m.dot(true, &coeff, &input, &mut output).unwrap();
        assert!((output[0] - 4.0).abs() < 1e-12);
        assert!((output[1] - 11.0).abs() < 1e-12);
    }

    #[test]
    fn dot_accumulate() {
        // overwrite=false: output += result
        let m = mat_2x2();
        let coeff = vec![1.0f64];
        let input = vec![1.0f64, 2.0];
        let mut output = vec![1.0f64, 1.0];
        m.dot(false, &coeff, &input, &mut output).unwrap();
        assert!((output[0] - 5.0).abs() < 1e-12);
        assert!((output[1] - 12.0).abs() < 1e-12);
    }

    #[test]
    fn dot_with_coeff_scaling() {
        let m = mat_2x2();
        let coeff = vec![2.0f64];
        let input = vec![1.0f64, 2.0];
        let mut output = vec![0.0f64; 2];
        m.dot(true, &coeff, &input, &mut output).unwrap();
        assert!((output[0] - 8.0).abs() < 1e-12);
        assert!((output[1] - 22.0).abs() < 1e-12);
    }

    #[test]
    fn dot_transpose_matvec() {
        let m = mat_2x2();
        let coeff = vec![1.0f64];
        let input = vec![1.0f64, 1.0];
        let mut output = vec![0.0f64; 2];
        m.dot_transpose(true, &coeff, &input, &mut output).unwrap();
        assert!((output[0] - 5.0).abs() < 1e-12);
        assert!((output[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn dot_wrong_coeff_len_errors() {
        let m = mat_2x2();
        let coeff = vec![1.0f64, 1.0];
        let input = vec![1.0f64, 2.0];
        let mut output = vec![0.0f64; 2];
        assert!(m.dot(true, &coeff, &input, &mut output).is_err());
    }

    // --- cross-type dot: M=f64, V=Complex<f64> ---

    #[test]
    fn dot_cross_type_real_stored_complex_output() {
        // QMatrix<f64> can be dotted with Complex<f64> coeff/input/output.
        let m = mat_2x2();
        let coeff = vec![Complex::new(1.0f64, 0.0)];
        let input = vec![Complex::new(1.0f64, 0.0), Complex::new(2.0, 0.0)];
        let mut output = vec![Complex::new(0.0f64, 0.0); 2];
        m.dot(true, &coeff, &input, &mut output).unwrap();
        assert!((output[0] - Complex::new(4.0, 0.0)).norm() < 1e-12);
        assert!((output[1] - Complex::new(11.0, 0.0)).norm() < 1e-12);
    }

    // --- remap_cindices ---

    #[test]
    fn remap_cindices_identity() {
        // Remapping cindex 0→0 should leave the matrix unchanged.
        let m = mat_2x2();
        let mapping = vec![0u8];
        let remapped = m.clone().remap_cindices(&mapping);
        assert_eq!(remapped.num_coeff(), 1);
        for r in 0..2 {
            for (e_orig, e_new) in m.row(r).iter().zip(remapped.row(r).iter()) {
                assert_eq!(e_orig.col, e_new.col);
                assert_eq!(e_new.cindex, 0u8);
            }
        }
    }

    #[test]
    fn remap_cindices_merge_two_to_one() {
        // Matrix with cindices 0 and 1; remap both to 0.
        let indptr = vec![0i64, 2, 2];
        let data = vec![Entry::new(1.0f64, 0i64, 0u8), Entry::new(2.0f64, 0i64, 1u8)];
        let m = QMatrix::from_csr(indptr, data);
        assert_eq!(m.num_coeff(), 2);
        let mapping = vec![0u8, 0u8];
        let remapped = m.remap_cindices(&mapping);
        // After remapping both entries have cindex=0; num_coeff should be 1.
        assert_eq!(remapped.num_coeff(), 1);
    }

    #[test]
    fn remap_cindices_swap() {
        // Two-cindex matrix; swap: 0→1, 1→0.
        let indptr = vec![0i64, 2, 2];
        let data = vec![Entry::new(1.0f64, 0i64, 0u8), Entry::new(2.0f64, 1i64, 1u8)];
        let m = QMatrix::from_csr(indptr, data);
        let mapping = vec![1u8, 0u8];
        let remapped = m.remap_cindices(&mapping);
        assert_eq!(remapped.num_coeff(), 2);
        // Entry at col=0 should now have cindex=1, entry at col=1 cindex=0.
        let row = remapped.row(0);
        // After sort by (col, cindex): (col=0,c=1), (col=1,c=0)
        assert_eq!(row[0].col, 0i64);
        assert_eq!(row[0].cindex, 1u8);
        assert_eq!(row[1].col, 1i64);
        assert_eq!(row[1].cindex, 0u8);
    }

    // --- to_csr ---

    /// Build a 2×2 matrix where two cindices share the same (row, col).
    fn mat_merge() -> QMatrix<f64, i64, u8> {
        let indptr = vec![0i64, 3, 4];
        let data = vec![
            Entry::new(2.0f64, 0i64, 0u8),
            Entry::new(3.0f64, 1i64, 0u8),
            Entry::new(-1.0f64, 1i64, 1u8),
            Entry::new(5.0f64, 0i64, 0u8),
        ];
        QMatrix::from_csr(indptr, data)
    }

    #[test]
    fn to_csr_identity_coeff() {
        let m = mat_2x2();
        let (indptr, indices, data) = m.to_csr(&[1.0f64], false).unwrap();
        assert_eq!(indptr, vec![0i64, 2, 4]);
        assert_eq!(indices, vec![0i64, 1, 0, 1]);
        assert_eq!(data, vec![2.0f64, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn to_csr_coeff_scaling() {
        let m = mat_2x2();
        let (indptr, indices, data) = m.to_csr(&[2.0f64], false).unwrap();
        assert_eq!(data, vec![4.0f64, 2.0, 6.0, 8.0]);
        assert_eq!(indptr, vec![0i64, 2, 4]);
        assert_eq!(indices, vec![0i64, 1, 0, 1]);
    }

    #[test]
    fn to_csr_merges_same_col() {
        let m = mat_merge();
        let (indptr, indices, data) = m.to_csr(&[1.0f64, 1.0], false).unwrap();
        assert_eq!(indptr, vec![0i64, 2, 3]);
        assert_eq!(indices, vec![0i64, 1, 0]);
        assert!((data[0] - 2.0).abs() < 1e-12);
        assert!((data[1] - 2.0).abs() < 1e-12);
        assert!((data[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn to_csr_drop_zeros_removes_cancelled_entry() {
        let m = mat_merge();
        let (indptr, indices, data) = m.to_csr(&[1.0f64, 3.0], true).unwrap();
        assert_eq!(indptr, vec![0i64, 1, 2]);
        assert_eq!(indices, vec![0i64, 0]);
        assert!((data[0] - 2.0).abs() < 1e-12);
        assert!((data[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn to_csr_drop_zeros_false_keeps_zero_entry() {
        let m = mat_merge();
        let (indptr, indices, data) = m.to_csr(&[1.0f64, 3.0], false).unwrap();
        assert_eq!(indptr, vec![0i64, 2, 3]);
        assert_eq!(indices, vec![0i64, 1, 0]);
        assert!((data[0] - 2.0).abs() < 1e-12);
        assert!(data[1].abs() < 1e-12);
        assert!((data[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn to_csr_nnz_matches_to_csr_len() {
        let m = mat_merge();
        for &drop_zeros in &[true, false] {
            for coeff in [vec![1.0f64, 1.0], vec![1.0, 3.0]] {
                let nnz = m.to_csr_nnz(&coeff, drop_zeros).unwrap();
                let (_, indices, _) = m.to_csr(&coeff, drop_zeros).unwrap();
                assert_eq!(nnz, indices.len());
            }
        }
    }

    #[test]
    fn to_csr_into_matches_to_csr() {
        let m = mat_merge();
        let coeff = vec![1.0f64, 1.0];
        let (expected_ip, expected_idx, expected_data) = m.to_csr(&coeff, true).unwrap();
        let nnz = m.to_csr_nnz(&coeff, true).unwrap();
        let mut indptr = vec![0i64; m.dim() + 1];
        let mut indices = vec![0i64; nnz];
        let mut data = vec![0.0f64; nnz];
        m.to_csr_into(&coeff, true, &mut indptr, &mut indices, &mut data)
            .unwrap();
        assert_eq!(indptr, expected_ip);
        assert_eq!(indices, expected_idx);
        assert_eq!(data, expected_data);
    }

    #[test]
    fn to_csr_wrong_coeff_errors() {
        let m = mat_2x2();
        assert!(m.to_csr(&[1.0f64, 2.0], false).is_err());
    }

    // --- to_dense ---

    #[test]
    fn to_dense_identity_coeff() {
        let m = mat_2x2();
        let d = m.to_dense(&[1.0f64]).unwrap();
        assert_eq!(d.len(), 4);
        assert!((d[0] - 2.0).abs() < 1e-12);
        assert!((d[1] - 1.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        assert!((d[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn to_dense_coeff_scaling() {
        let m = mat_2x2();
        let d = m.to_dense(&[2.0f64]).unwrap();
        assert!((d[0] - 4.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
        assert!((d[3] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn to_dense_merges_same_col() {
        let m = mat_merge();
        let d = m.to_dense(&[1.0f64, 1.0]).unwrap();
        assert!((d[0] - 2.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 5.0).abs() < 1e-12);
        assert!(d[3].abs() < 1e-12);
    }

    #[test]
    fn to_dense_into_matches_to_dense() {
        let m = mat_2x2();
        let expected = m.to_dense(&[1.0f64]).unwrap();
        let mut buf = vec![0.0f64; m.dim() * m.dim()];
        m.to_dense_into(&[1.0f64], &mut buf).unwrap();
        assert_eq!(buf, expected);
    }

    #[test]
    fn to_dense_wrong_output_size_errors() {
        let m = mat_2x2();
        let mut buf = vec![0.0f64; 3];
        assert!(m.to_dense_into(&[1.0f64], &mut buf).is_err());
    }

    #[test]
    fn to_dense_wrong_coeff_errors() {
        let m = mat_2x2();
        assert!(m.to_dense(&[1.0f64, 2.0]).is_err());
    }

    // --- dot_many ---

    #[test]
    fn dot_many_single_vec_matches_dot() {
        // dot_many with n_vecs=1 must agree with dot.
        let m = mat_2x2();
        let coeff = vec![1.0f64];
        let input_vec = vec![1.0f64, 2.0];
        let mut output_vec = vec![0.0f64; 2];
        m.dot(true, &coeff, &input_vec, &mut output_vec).unwrap();

        let input_mat = array![[1.0f64], [2.0]]; // shape (2, 1)
        let mut output_mat = ndarray::Array2::zeros((2, 1));
        m.dot_many(true, &coeff, input_mat.view(), output_mat.view_mut())
            .unwrap();

        for r in 0..2 {
            assert!((output_mat[[r, 0]] - output_vec[r]).abs() < 1e-12);
        }
    }

    #[test]
    fn dot_many_two_vecs() {
        // Apply [[2,1],[3,4]] to columns [1,2] and [0,1] independently.
        // [1,2] → [4, 11],  [0,1] → [1, 4]
        let m = mat_2x2();
        let coeff = vec![1.0f64];
        let input = array![[1.0f64, 0.0], [2.0, 1.0]]; // shape (2, 2)
        let mut output = ndarray::Array2::zeros((2, 2));
        m.dot_many(true, &coeff, input.view(), output.view_mut())
            .unwrap();

        assert!((output[[0, 0]] - 4.0).abs() < 1e-12);
        assert!((output[[1, 0]] - 11.0).abs() < 1e-12);
        assert!((output[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((output[[1, 1]] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn dot_many_accumulate() {
        // overwrite=false: result is added to existing output.
        let m = mat_2x2();
        let coeff = vec![1.0f64];
        let input = array![[1.0f64], [2.0]];
        let mut output = ndarray::Array2::from_elem((2, 1), 1.0f64);
        m.dot_many(false, &coeff, input.view(), output.view_mut())
            .unwrap();
        assert!((output[[0, 0]] - 5.0).abs() < 1e-12); // 1 + 4
        assert!((output[[1, 0]] - 12.0).abs() < 1e-12); // 1 + 11
    }

    #[test]
    fn dot_many_wrong_input_nrows_errors() {
        let m = mat_2x2();
        let input = ndarray::Array2::<f64>::zeros((3, 1)); // nrows=3, dim=2
        let mut output = ndarray::Array2::zeros((3, 1));
        assert!(
            m.dot_many(true, &[1.0f64], input.view(), output.view_mut())
                .is_err()
        );
    }

    #[test]
    fn dot_many_mismatched_output_shape_errors() {
        let m = mat_2x2();
        let input = ndarray::Array2::<f64>::zeros((2, 2));
        let mut output = ndarray::Array2::zeros((2, 3)); // ncols differs
        assert!(
            m.dot_many(true, &[1.0f64], input.view(), output.view_mut())
                .is_err()
        );
    }

    // --- dot_transpose_many ---

    #[test]
    fn dot_transpose_many_single_vec_matches_dot_transpose() {
        let m = mat_2x2();
        let coeff = vec![1.0f64];
        let input_vec = vec![1.0f64, 1.0];
        let mut output_vec = vec![0.0f64; 2];
        m.dot_transpose(true, &coeff, &input_vec, &mut output_vec)
            .unwrap();

        let input_mat = array![[1.0f64], [1.0]];
        let mut output_mat = ndarray::Array2::zeros((2, 1));
        m.dot_transpose_many(true, &coeff, input_mat.view(), output_mat.view_mut())
            .unwrap();

        for r in 0..2 {
            assert!((output_mat[[r, 0]] - output_vec[r]).abs() < 1e-12);
        }
    }

    #[test]
    fn dot_transpose_many_two_vecs() {
        // Transpose of [[2,1],[3,4]] applied to [1,1] → [5, 5]
        //                                  and [1,0] → [2, 1]
        let m = mat_2x2();
        let coeff = vec![1.0f64];
        let input = array![[1.0f64, 1.0], [1.0, 0.0]];
        let mut output = ndarray::Array2::zeros((2, 2));
        m.dot_transpose_many(true, &coeff, input.view(), output.view_mut())
            .unwrap();

        assert!((output[[0, 0]] - 5.0).abs() < 1e-12);
        assert!((output[[1, 0]] - 5.0).abs() < 1e-12);
        assert!((output[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((output[[1, 1]] - 1.0).abs() < 1e-12);
    }
}
