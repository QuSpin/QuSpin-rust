use crate::error::QuSpinError;
use crate::primitive::Primitive;
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
/// Replaces the C++ `std::tuple<T, I, J>` from `qmatrix.hpp`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry<V, I, C> {
    pub value: V,
    pub col: I,
    /// Operator-string index: selects the coefficient from `coeff[cindex]`
    /// in matrix–vector products.
    pub cindex: C,
}

impl<V: Copy, I: Copy + Ord, C: Copy + Ord> Entry<V, I, C> {
    pub fn new(value: V, col: I, cindex: C) -> Self {
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
/// - `V: Primitive` — element type
/// - `I: Index` — row/column index type (i32 or i64)
/// - `C: CIndex` — operator-string index type (u8 or u16)
///
/// Mirrors `qmatrix<T, I, J>` from `qmatrix.hpp`.
#[derive(Clone)]
pub struct QMatrix<V, I, C> {
    dim: usize,
    /// Number of distinct cindex values (max cindex + 1).
    num_coeff: usize,
    /// Row pointers, length = dim + 1.  `data[indptr[r]..indptr[r+1]]` holds
    /// the entries of row `r`.
    indptr: Vec<I>,
    /// Non-zero entries sorted within each row by (col, cindex).
    data: Vec<Entry<V, I, C>>,
}

impl<V: Primitive, I: Index, C: CIndex> QMatrix<V, I, C> {
    /// Construct from pre-built CSR arrays.  Entries are sorted per-row if
    /// not already in (col, cindex) order.
    pub fn from_csr(indptr: Vec<I>, data: Vec<Entry<V, I, C>>) -> Self {
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
    pub fn row(&self, r: usize) -> &[Entry<V, I, C>] {
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
    // to_csr — materialise with coefficients into a plain CSR matrix
    // ------------------------------------------------------------------

    /// Count the number of structurally non-zero entries that
    /// `to_csr_into` / `to_csr` would emit for the given `coeff`.
    ///
    /// When `drop_zeros` is `true`, entries whose accumulated value is
    /// negligible relative to the sum of contribution magnitudes
    /// (`|acc| ≤ scale * ZERO_TOL`) are excluded from the count.
    ///
    /// Use this to pre-allocate buffers before calling `to_csr_into` (the
    /// C-FFI two-phase pattern).
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff`.
    pub fn to_csr_nnz(&self, coeff: &[V], drop_zeros: bool) -> Result<usize, QuSpinError> {
        self.check_coeff(coeff)?;
        let mut count = 0usize;
        for r in 0..self.dim {
            let mut i = 0;
            let row = self.row(r);
            while i < row.len() {
                let col = row[i].col;
                let mut acc = V::default();
                let mut scale = 0.0f64;
                while i < row.len() && row[i].col == col {
                    let contrib = coeff[row[i].cindex.as_usize()] * row[i].value;
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
    /// This is the low-level fill primitive intended for C-FFI callers.
    /// Python callers should use the allocating `to_csr` instead.
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff`,
    /// `indptr.len() != dim + 1`, or the output buffers are too small.
    pub fn to_csr_into(
        &self,
        coeff: &[V],
        drop_zeros: bool,
        indptr: &mut [I],
        indices: &mut [I],
        data: &mut [V],
    ) -> Result<(), QuSpinError> {
        self.check_coeff(coeff)?;
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
                    let contrib = coeff[row[i].cindex.as_usize()] * row[i].value;
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
    /// Convenience wrapper over `to_csr_nnz` + `to_csr_into`.  Ownership of
    /// the three `Vec`s is transferred to the caller.
    ///
    /// Python callers receive these as numpy arrays (zero-copy via
    /// `PyArray1::from_vec`).  C-FFI callers should use the two-phase
    /// `to_csr_nnz` / `to_csr_into` API instead to avoid a second pass.
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff`.
    #[allow(clippy::type_complexity)]
    pub fn to_csr(
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
    /// This is the low-level fill primitive intended for C-FFI callers, who
    /// pre-allocate `dim * dim` elements of the matrix dtype and pass a raw
    /// pointer.  The output size is always known — `dim` is available via
    /// `quspin_qmatrix_dim()` — so no separate size-query step is needed.
    ///
    /// Python callers should use the allocating `to_dense` instead.
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff` or
    /// `output.len() != dim * dim`.
    pub fn to_dense_into(&self, coeff: &[V], output: &mut [V]) -> Result<(), QuSpinError> {
        self.check_coeff(coeff)?;
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
                output[r * self.dim + e.col.as_usize()] += coeff[e.cindex.as_usize()] * e.value;
            }
        }
        Ok(())
    }

    /// Allocate and return a row-major dense array for the materialised matrix.
    ///
    /// Returns a `Vec<V>` of length `dim * dim`.  Element `[r * dim + col]`
    /// holds `Σ_c coeff[c] * stored_value[c, r, col]`.
    ///
    /// Ownership is transferred to the caller.  Python callers receive this
    /// as a 2-D numpy array (via `PyArray2`); C-FFI callers should use
    /// `to_dense_into` with a pre-allocated pointer instead.
    ///
    /// # Errors
    /// Returns `ValueError` if `coeff.len() != num_coeff`.
    pub fn to_dense(&self, coeff: &[V]) -> Result<Vec<V>, QuSpinError> {
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
    /// If `overwrite` is `true`, `output` is zeroed before accumulation.
    ///
    /// # Errors
    /// Returns `ValueError` if any array length is inconsistent.
    pub fn dot(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        self.check_dot_args(coeff, input, output)?;

        for (r, out) in output.iter_mut().enumerate() {
            let mut acc = V::default();
            for e in self.row(r) {
                acc += coeff[e.cindex.as_usize()] * e.value * input[e.col.as_usize()];
            }
            *out = if overwrite { acc } else { *out + acc };
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // dot_transpose
    // ------------------------------------------------------------------

    /// Compute `output[col] += Σ_{r} coeff[cindex] * value * input[r]`
    /// (the transpose matrix–vector product).
    ///
    /// Sequential to avoid data races on `output`; Rayon parallelism
    /// (via partial-array reduce) is a deferred optimisation.
    pub fn dot_transpose(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        self.check_dot_args(coeff, input, output)?;

        if overwrite {
            output.fill(V::default());
        }

        for (r, inp) in input.iter().enumerate() {
            for e in self.row(r) {
                output[e.col.as_usize()] += coeff[e.cindex.as_usize()] * e.value * *inp;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn check_coeff(&self, coeff: &[V]) -> Result<(), QuSpinError> {
        if coeff.len() != self.num_coeff {
            return Err(QuSpinError::ValueError(format!(
                "coeff length {} != num_coeff {}",
                coeff.len(),
                self.num_coeff
            )));
        }
        Ok(())
    }

    fn check_dot_args(&self, coeff: &[V], input: &[V], output: &[V]) -> Result<(), QuSpinError> {
        self.check_coeff(coeff)?;
        if input.len() != self.dim {
            return Err(QuSpinError::ValueError(format!(
                "input length {} != dim {}",
                input.len(),
                self.dim
            )));
        }
        if output.len() != self.dim {
            return Err(QuSpinError::ValueError(format!(
                "output length {} != dim {}",
                output.len(),
                self.dim
            )));
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
        let mut output = vec![1.0f64, 1.0]; // pre-existing values
        m.dot(false, &coeff, &input, &mut output).unwrap();
        assert!((output[0] - 5.0).abs() < 1e-12); // 1 + 4
        assert!((output[1] - 12.0).abs() < 1e-12); // 1 + 11
    }

    #[test]
    fn dot_with_coeff_scaling() {
        // coeff = 2.0 → output = 2 * [[2,1],[3,4]] * [1,2] = [8, 22]
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
        // [[2,1],[3,4]]^T * [1,1] = [2+3, 1+4] = [5, 5]
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
        let coeff = vec![1.0f64, 1.0]; // wrong length
        let input = vec![1.0f64, 2.0];
        let mut output = vec![0.0f64; 2];
        assert!(m.dot(true, &coeff, &input, &mut output).is_err());
    }

    // --- to_csr ---

    /// Build a 2×2 matrix where two cindices share the same (row, col):
    /// row 0: (col=1, c=0) with value 3.0 and (col=1, c=1) with value -1.0.
    /// With coeff=[1, 1] → M[0,1] = 2.0 (merged); with coeff=[1, 3] → M[0,1] = 0.0.
    fn mat_merge() -> QMatrix<f64, i64, u8> {
        // Row 0: col=0 c=0 val=2, col=1 c=0 val=3, col=1 c=1 val=-1
        // Row 1: col=0 c=0 val=5
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
        // coeff = [1]: M[r,c] = Σ_c 1 * val = stored values, merging same-col entries.
        let m = mat_2x2();
        let (indptr, indices, data) = m.to_csr(&[1.0f64], false).unwrap();
        assert_eq!(indptr, vec![0i64, 2, 4]);
        assert_eq!(indices, vec![0i64, 1, 0, 1]);
        assert_eq!(data, vec![2.0f64, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn to_csr_coeff_scaling() {
        // coeff = [2]: every value doubles.
        let m = mat_2x2();
        let (indptr, indices, data) = m.to_csr(&[2.0f64], false).unwrap();
        assert_eq!(data, vec![4.0f64, 2.0, 6.0, 8.0]);
        assert_eq!(indptr, vec![0i64, 2, 4]);
        assert_eq!(indices, vec![0i64, 1, 0, 1]);
    }

    #[test]
    fn to_csr_merges_same_col() {
        // mat_merge row 0, col 1: c=0 val=3, c=1 val=-1.
        // coeff=[1, 1] → M[0,1] = 3 + (-1) = 2.
        let m = mat_merge();
        let (indptr, indices, data) = m.to_csr(&[1.0f64, 1.0], false).unwrap();
        assert_eq!(indptr, vec![0i64, 2, 3]);
        assert_eq!(indices, vec![0i64, 1, 0]);
        assert!((data[0] - 2.0).abs() < 1e-12); // row 0 col 0
        assert!((data[1] - 2.0).abs() < 1e-12); // row 0 col 1 (merged)
        assert!((data[2] - 5.0).abs() < 1e-12); // row 1 col 0
    }

    #[test]
    fn to_csr_drop_zeros_removes_cancelled_entry() {
        // coeff=[1, 3] → M[0,1] = 3*1 + (-1)*3 = 0.  With drop_zeros=true it's absent.
        let m = mat_merge();
        let (indptr, indices, data) = m.to_csr(&[1.0f64, 3.0], true).unwrap();
        // Row 0: col=0 → 2*1=2 (kept), col=1 → 0 (dropped).
        // Row 1: col=0 → 5*1=5 (kept).
        assert_eq!(indptr, vec![0i64, 1, 2]);
        assert_eq!(indices, vec![0i64, 0]);
        assert!((data[0] - 2.0).abs() < 1e-12);
        assert!((data[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn to_csr_drop_zeros_false_keeps_zero_entry() {
        // Same as above but drop_zeros=false: the zero entry at (0,1) is kept.
        let m = mat_merge();
        let (indptr, indices, data) = m.to_csr(&[1.0f64, 3.0], false).unwrap();
        assert_eq!(indptr, vec![0i64, 2, 3]);
        assert_eq!(indices, vec![0i64, 1, 0]);
        assert!((data[0] - 2.0).abs() < 1e-12);
        assert!(data[1].abs() < 1e-12); // zero entry retained
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
        // [[2,1],[3,4]] with coeff=[1] → dense [[2,1],[3,4]]
        let m = mat_2x2();
        let d = m.to_dense(&[1.0f64]).unwrap();
        assert_eq!(d.len(), 4);
        assert!((d[0] - 2.0).abs() < 1e-12); // [0,0]
        assert!((d[1] - 1.0).abs() < 1e-12); // [0,1]
        assert!((d[2] - 3.0).abs() < 1e-12); // [1,0]
        assert!((d[3] - 4.0).abs() < 1e-12); // [1,1]
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
        // mat_merge row 0: col=0 val=2, col=1 c=0 val=3, col=1 c=1 val=-1
        // coeff=[1,1] → M[0,0]=2, M[0,1]=2, M[1,0]=5, M[1,1]=0
        let m = mat_merge();
        let d = m.to_dense(&[1.0f64, 1.0]).unwrap();
        assert!((d[0] - 2.0).abs() < 1e-12); // [0,0]
        assert!((d[1] - 2.0).abs() < 1e-12); // [0,1] merged: 3 + (-1) = 2
        assert!((d[2] - 5.0).abs() < 1e-12); // [1,0]
        assert!(d[3].abs() < 1e-12); // [1,1] absent → zero
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
        let mut buf = vec![0.0f64; 3]; // should be 4
        assert!(m.to_dense_into(&[1.0f64], &mut buf).is_err());
    }

    #[test]
    fn to_dense_wrong_coeff_errors() {
        let m = mat_2x2();
        assert!(m.to_dense(&[1.0f64, 2.0]).is_err());
    }
}
