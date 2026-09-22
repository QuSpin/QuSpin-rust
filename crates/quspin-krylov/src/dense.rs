//! Dense Hermitian eigensolvers.
//!
//! Full diagonalization of small dense matrices (up to a few thousand rows),
//! e.g. the symmetry blocks of a finite cluster. Input is a flat row-major
//! `n × n` buffer — the layout `QMatrix::to_dense` (in `quspin-matrix`) produces.
//!
//! Backed by nalgebra's `SymmetricEigen` (Householder tridiagonalization +
//! implicit QR), single-threaded. For large sparse problems use the Lanczos
//! routines in this crate instead.

use nalgebra::{ComplexField, DMatrix};
use num_complex::Complex;
use quspin_types::{Primitive, QuSpinError};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// Scalar types with a dense Hermitian eigensolver: `f64` (real symmetric)
/// and `Complex<f64>` (complex Hermitian).
///
/// Sealed; the method is an implementation detail.
pub trait DenseHermitian: Primitive + sealed::Sealed {
    /// Complex conjugate (identity for `f64`).
    fn conjugate(self) -> Self;

    #[doc(hidden)]
    fn eigen_impl(a: &[Self], n: usize, vectors: bool) -> (Vec<f64>, Option<Vec<Self>>);
}

fn eigen_generic<T>(a: &[T], n: usize, vectors: bool) -> (Vec<f64>, Option<Vec<T>>)
where
    T: ComplexField<RealField = f64> + Copy,
{
    let m = DMatrix::<T>::from_row_slice(n, n, a);
    if !vectors {
        let mut vals: Vec<f64> = m.symmetric_eigenvalues().iter().copied().collect();
        vals.sort_by(f64::total_cmp);
        return (vals, None);
    }
    let eig = m.symmetric_eigen();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
    let vals = order.iter().map(|&i| eig.eigenvalues[i]).collect();
    let mut vecs = Vec::with_capacity(n * n);
    for &col in &order {
        vecs.extend(eig.eigenvectors.column(col).iter().copied());
    }
    (vals, Some(vecs))
}

impl DenseHermitian for f64 {
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
    fn eigen_impl(a: &[Self], n: usize, vectors: bool) -> (Vec<f64>, Option<Vec<Self>>) {
        eigen_generic(a, n, vectors)
    }
}

impl DenseHermitian for Complex<f64> {
    #[inline]
    fn conjugate(self) -> Self {
        self.conj()
    }
    fn eigen_impl(a: &[Self], n: usize, vectors: bool) -> (Vec<f64>, Option<Vec<Self>>) {
        eigen_generic(a, n, vectors)
    }
}

/// Eigendecomposition of a dense Hermitian matrix.
#[derive(Clone, Debug)]
pub struct DenseEigen<V> {
    /// Eigenvalues in ascending order (length `n`).
    pub eigenvalues: Vec<f64>,
    /// Orthonormal eigenvectors, column-major: element `(i, j)` is at
    /// `eigenvectors[i + j * n]`, and column `j` belongs to `eigenvalues[j]`.
    pub eigenvectors: Vec<V>,
    /// Matrix dimension.
    pub n: usize,
}

impl<V: DenseHermitian> DenseEigen<V> {
    /// Eigenvector `j` as a contiguous slice of length `n`.
    pub fn eigenvector(&self, j: usize) -> &[V] {
        &self.eigenvectors[j * self.n..(j + 1) * self.n]
    }
}

/// Relative tolerance of the Hermiticity check: `|a_ij − conj(a_ji)|` must not
/// exceed `HERMITIAN_RTOL · max(1, max_ij |a_ij|)`.
pub const HERMITIAN_RTOL: f64 = 1e-10;

fn validate<V: DenseHermitian>(a: &[V], n: usize) -> Result<(), QuSpinError> {
    if a.len() != n * n {
        return Err(QuSpinError::ValueError(format!(
            "dense matrix buffer has length {} but n * n = {}",
            a.len(),
            n * n
        )));
    }
    let mut scale = 1.0_f64;
    for &x in a {
        let m = x.magnitude();
        if !m.is_finite() {
            return Err(QuSpinError::ValueError(
                "dense matrix contains a non-finite entry".into(),
            ));
        }
        scale = scale.max(m);
    }
    let tol = HERMITIAN_RTOL * scale;
    for i in 0..n {
        for j in i..n {
            let d = (a[i * n + j] - a[j * n + i].conjugate()).magnitude();
            if d > tol {
                return Err(QuSpinError::ValueError(format!(
                    "matrix is not Hermitian: |a[{i},{j}] - conj(a[{j},{i}])| = {d:e} > {tol:e}"
                )));
            }
        }
    }
    Ok(())
}

/// Eigenvalues (ascending) of the dense Hermitian `n × n` matrix `a`, given
/// row-major.
///
/// # Errors
/// `ValueError` if `a.len() != n * n`, an entry is non-finite, or `a` is not
/// Hermitian to within [`HERMITIAN_RTOL`].
pub fn eigvalsh<V: DenseHermitian>(a: &[V], n: usize) -> Result<Vec<f64>, QuSpinError> {
    validate(a, n)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    Ok(V::eigen_impl(a, n, false).0)
}

/// Eigenvalues (ascending) and orthonormal eigenvectors of the dense
/// Hermitian `n × n` matrix `a`, given row-major.
///
/// # Errors
/// Same as [`eigvalsh`].
pub fn eigh<V: DenseHermitian>(a: &[V], n: usize) -> Result<DenseEigen<V>, QuSpinError> {
    validate(a, n)?;
    if n == 0 {
        return Ok(DenseEigen {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
            n,
        });
    }
    let (eigenvalues, vecs) = V::eigen_impl(a, n, true);
    Ok(DenseEigen {
        eigenvalues,
        eigenvectors: vecs.expect("eigenvectors requested"),
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type C64 = Complex<f64>;

    /// Open tight-binding chain: eigenvalues 2 cos(π k / (n + 1)).
    fn chain(n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n - 1 {
            a[i * n + i + 1] = 1.0;
            a[(i + 1) * n + i] = 1.0;
        }
        a
    }

    #[test]
    fn eigvalsh_open_chain_matches_analytic() {
        let n = 12;
        let vals = eigvalsh(&chain(n), n).unwrap();
        let mut exact: Vec<f64> = (1..=n)
            .map(|k| 2.0 * (std::f64::consts::PI * k as f64 / (n + 1) as f64).cos())
            .collect();
        exact.sort_by(f64::total_cmp);
        for (v, e) in vals.iter().zip(&exact) {
            assert!((v - e).abs() < 1e-12, "{v} vs {e}");
        }
    }

    #[test]
    fn eigh_real_reconstructs_matrix() {
        let n = 5;
        let a: Vec<f64> = (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                ((i + j) as f64).sin() + if i == j { i as f64 } else { 0.0 }
            })
            .collect();
        let eig = eigh(&a, n).unwrap();
        for w in eig.eigenvalues.windows(2) {
            assert!(w[0] <= w[1]);
        }
        for i in 0..n {
            for j in 0..n {
                let r: f64 = (0..n)
                    .map(|k| eig.eigenvector(k)[i] * eig.eigenvalues[k] * eig.eigenvector(k)[j])
                    .sum();
                assert!((r - a[i * n + j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn eigh_complex_pauli_y() {
        // σ^y has eigenvalues ±1.
        let a = vec![
            C64::new(0.0, 0.0),
            C64::new(0.0, -1.0),
            C64::new(0.0, 1.0),
            C64::new(0.0, 0.0),
        ];
        let eig = eigh(&a, 2).unwrap();
        assert!((eig.eigenvalues[0] + 1.0).abs() < 1e-14);
        assert!((eig.eigenvalues[1] - 1.0).abs() < 1e-14);
        // A v = λ v for each column.
        for k in 0..2 {
            let v = eig.eigenvector(k);
            for i in 0..2 {
                let av: C64 = (0..2).map(|j| a[i * 2 + j] * v[j]).sum();
                assert!((av - v[i] * eig.eigenvalues[k]).norm() < 1e-13);
            }
        }
    }

    #[test]
    fn rejects_non_hermitian_and_bad_length() {
        assert!(eigvalsh(&[0.0, 1.0, 0.0, 0.0], 2).is_err());
        assert!(eigvalsh(&[0.0, 1.0, 1.0], 2).is_err());
        let a = vec![
            C64::new(1.0, 1.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ];
        assert!(eigvalsh(&a, 2).is_err());
        assert!(eigvalsh(&[f64::NAN], 1).is_err());
    }

    #[test]
    fn empty_matrix() {
        assert!(eigvalsh::<f64>(&[], 0).unwrap().is_empty());
        assert!(eigh::<f64>(&[], 0).unwrap().eigenvalues.is_empty());
    }
}
