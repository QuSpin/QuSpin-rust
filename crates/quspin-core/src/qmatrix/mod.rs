pub mod build;
pub mod ops;

use crate::error::QuSpinError;
use crate::primitive::Primitive;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Sub, SubAssign};

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

    fn check_dot_args(&self, coeff: &[V], input: &[V], output: &[V]) -> Result<(), QuSpinError> {
        if coeff.len() != self.num_coeff {
            return Err(QuSpinError::ValueError(format!(
                "coeff length {} != num_coeff {}",
                coeff.len(),
                self.num_coeff
            )));
        }
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
}
