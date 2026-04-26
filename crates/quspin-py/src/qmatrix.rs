use crate::basis::AsSpaceInner;
use crate::basis::boson::PyBosonBasis;
use crate::basis::fermion::PyFermionBasis;
use crate::basis::generic::PyGenericBasis;
use crate::basis::spin::PySpinBasis;
use crate::dtype::FromPyDescr;
use crate::error::Error;
use crate::operator::bond::PyBondOperator;
use crate::operator::boson::PyBosonOperator;
use crate::operator::fermion::PyFermionOperator;
use crate::operator::monomial::PyMonomialOperator;
use crate::operator::pauli::PyPauliOperator;
use num_complex::Complex;
use numpy::{
    Complex64, PyArrayDescr, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use numpy::{PyArray1, PyArray2, ToPyArray};

/// Return type of `PyQMatrix::to_csr`.
type CsrArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<Complex64>>,
);
use pyo3::prelude::*;
use quspin_core::dtype::ValueDType;
use quspin_core::qmatrix::QMatrixInner;

/// Python-facing sparse quantum matrix.
///
/// Built from an operator + basis pair; stores a type-erased `QMatrixInner`.
/// Matrix-vector products, CSR export, and arithmetic are all supported.
#[pyclass(name = "QMatrix", module = "quspin._rs")]
pub struct PyQMatrix {
    pub inner: QMatrixInner,
}

// ---------------------------------------------------------------------------
// dtype helpers
// ---------------------------------------------------------------------------

fn dtype_from_py<'py>(py: Python<'py>, descr: &Bound<'py, PyArrayDescr>) -> PyResult<ValueDType> {
    Ok(ValueDType::from_descr(py, descr).map_err(Error::from)?)
}

// ---------------------------------------------------------------------------
// PyQMatrix pymethods
// ---------------------------------------------------------------------------

#[pymethods]
impl PyQMatrix {
    // ------------------------------------------------------------------
    // Build methods
    // ------------------------------------------------------------------

    /// Build from a `PauliOperator` and a `SpinBasis` or `FermionBasis`.
    #[staticmethod]
    #[pyo3(signature = (op, basis, dtype))]
    fn build_pauli(
        py: Python<'_>,
        op: &PyPauliOperator,
        basis: &Bound<'_, PyAny>,
        dtype: &Bound<'_, PyArrayDescr>,
    ) -> PyResult<Self> {
        let vdtype = dtype_from_py(py, dtype)?;
        let space = if let Ok(b) = basis.downcast::<PySpinBasis>() {
            QMatrixInner::build_hardcore(&op.inner, b.borrow().as_space_inner(), vdtype)
        } else if let Ok(b) = basis.downcast::<PyFermionBasis>() {
            QMatrixInner::build_hardcore(&op.inner, b.borrow().as_space_inner(), vdtype)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "basis must be SpinBasis or FermionBasis for build_pauli",
            ));
        };
        Ok(PyQMatrix { inner: space })
    }

    /// Build from a `BondOperator` and any basis type.
    #[staticmethod]
    #[pyo3(signature = (op, basis, dtype))]
    fn build_bond(
        py: Python<'_>,
        op: &PyBondOperator,
        basis: &Bound<'_, PyAny>,
        dtype: &Bound<'_, PyArrayDescr>,
    ) -> PyResult<Self> {
        let vdtype = dtype_from_py(py, dtype)?;
        let space = if let Ok(b) = basis.downcast::<PySpinBasis>() {
            QMatrixInner::build_bond(&op.inner, b.borrow().as_space_inner(), vdtype)
        } else if let Ok(b) = basis.downcast::<PyFermionBasis>() {
            QMatrixInner::build_bond(&op.inner, b.borrow().as_space_inner(), vdtype)
        } else if let Ok(b) = basis.downcast::<PyBosonBasis>() {
            QMatrixInner::build_bond(&op.inner, b.borrow().as_space_inner(), vdtype)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "basis must be SpinBasis, FermionBasis, or BosonBasis for build_bond",
            ));
        };
        Ok(PyQMatrix { inner: space })
    }

    /// Build from a `BosonOperator` and a `BosonBasis`.
    #[staticmethod]
    #[pyo3(signature = (op, basis, dtype))]
    fn build_boson(
        py: Python<'_>,
        op: &PyBosonOperator,
        basis: &PyBosonBasis,
        dtype: &Bound<'_, PyArrayDescr>,
    ) -> PyResult<Self> {
        let vdtype = dtype_from_py(py, dtype)?;
        let inner = QMatrixInner::build_boson(&op.inner, basis.as_space_inner(), vdtype);
        Ok(PyQMatrix { inner })
    }

    /// Build from a `FermionOperator` and a `FermionBasis`.
    #[staticmethod]
    #[pyo3(signature = (op, basis, dtype))]
    fn build_fermion(
        py: Python<'_>,
        op: &PyFermionOperator,
        basis: &PyFermionBasis,
        dtype: &Bound<'_, PyArrayDescr>,
    ) -> PyResult<Self> {
        let vdtype = dtype_from_py(py, dtype)?;
        let inner = QMatrixInner::build_fermion(&op.inner, basis.as_space_inner(), vdtype);
        Ok(PyQMatrix { inner })
    }

    /// Build from a `MonomialOperator` and a `GenericBasis`.
    #[staticmethod]
    #[pyo3(signature = (op, basis, dtype))]
    fn build_monomial(
        py: Python<'_>,
        op: &PyMonomialOperator,
        basis: &PyGenericBasis,
        dtype: &Bound<'_, PyArrayDescr>,
    ) -> PyResult<Self> {
        let vdtype = dtype_from_py(py, dtype)?;
        let inner = QMatrixInner::build_monomial(&op.inner, basis.as_space_inner(), vdtype);
        Ok(PyQMatrix { inner })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[getter]
    fn num_coeff(&self) -> usize {
        self.inner.num_coeff()
    }

    #[getter]
    fn dtype(&self) -> &str {
        self.inner.dtype_name()
    }

    fn __repr__(&self) -> String {
        format!(
            "QMatrix(dim={}, nnz={}, num_coeff={}, dtype={})",
            self.inner.dim(),
            self.inner.nnz(),
            self.inner.num_coeff(),
            self.inner.dtype_name(),
        )
    }

    // ------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------

    fn __add__(&self, rhs: &PyQMatrix) -> PyResult<PyQMatrix> {
        let inner = self
            .inner
            .clone()
            .try_add(rhs.inner.clone())
            .map_err(Error::from)?;
        Ok(PyQMatrix { inner })
    }

    fn __sub__(&self, rhs: &PyQMatrix) -> PyResult<PyQMatrix> {
        let inner = self
            .inner
            .clone()
            .try_sub(rhs.inner.clone())
            .map_err(Error::from)?;
        Ok(PyQMatrix { inner })
    }

    // ------------------------------------------------------------------
    // CSR export
    // ------------------------------------------------------------------

    /// Materialise as scipy-compatible CSR arrays.
    ///
    /// Args:
    ///     coeff:      1-D complex128 array of length `num_cindices`; the
    ///                 coefficient for each operator string.
    ///     drop_zeros: if True, near-zero entries are omitted (default True).
    ///
    /// Returns:
    ///     (indptr, indices, data) — three numpy arrays with dtypes
    ///     (int64, int64, complex128).
    #[pyo3(signature = (coeff, drop_zeros = true))]
    fn to_csr<'py>(
        &self,
        py: Python<'py>,
        coeff: PyReadonlyArray1<'py, Complex64>,
        drop_zeros: bool,
    ) -> PyResult<CsrArrays<'py>> {
        let coeff_vec: Vec<Complex<f64>> = coeff
            .as_array()
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
        let (indptr, indices, data) = self
            .inner
            .materialize(&coeff_vec, drop_zeros)
            .map_err(Error::from)?;
        // materialize returns Vec<Complex<f64>>; Complex64 == Complex<f64> in num-complex
        let indptr_arr = indptr.to_pyarray(py);
        let indices_arr = indices.to_pyarray(py);
        let data_arr = data.to_pyarray(py);
        Ok((indptr_arr, indices_arr, data_arr))
    }

    // ------------------------------------------------------------------
    // Matrix-vector products (2-D batch)
    // ------------------------------------------------------------------

    /// Compute `output += coeff * self @ input` column-wise.
    ///
    /// Args:
    ///     coeff:     1-D complex128 array of length `num_cindices`.
    ///     input:     2-D complex128 array of shape `(dim, n_vecs)`.
    ///     output:    2-D complex128 array of shape `(dim, n_vecs)` (in place).
    ///     overwrite: if True, zero `output` before accumulating.
    fn dot_many<'py>(
        &self,
        coeff: PyReadonlyArray1<'py, Complex64>,
        input: PyReadonlyArray2<'py, Complex64>,
        output: &Bound<'py, PyArray2<Complex64>>,
        overwrite: bool,
    ) -> PyResult<()> {
        let n = self.inner.dim();
        if input.shape()[0] != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input first dim must be {n}"
            )));
        }
        let coeff_vec: Vec<Complex<f64>> = coeff
            .as_array()
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
        let n_vecs = input.shape()[1];
        let in_slice = input
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // Complex64 == Complex<f64>; copy to owned buffer for dispatch
        let in_arr: Vec<Complex<f64>> = in_slice.iter().map(|c| Complex::new(c.re, c.im)).collect();
        let in_view = ndarray::ArrayView2::from_shape((n, n_vecs), &in_arr)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let mut out_vec: Vec<Complex<f64>> = {
            let out_arr = unsafe { output.as_array() };
            out_arr.iter().map(|c| Complex::new(c.re, c.im)).collect()
        };
        let mut out_view = ndarray::ArrayViewMut2::from_shape((n, n_vecs), &mut out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner
            .dot_many(overwrite, &coeff_vec, in_view, out_view.view_mut())
            .map_err(Error::from)?;
        // Write back
        unsafe {
            let mut out_arr = output.as_array_mut();
            for ((r, c), v) in out_arr.indexed_iter_mut() {
                *v = Complex64::new(out_vec[r * n_vecs + c].re, out_vec[r * n_vecs + c].im);
            }
        }
        Ok(())
    }

    /// Transpose matrix-vector product (batch).
    fn dot_transpose_many<'py>(
        &self,
        coeff: PyReadonlyArray1<'py, Complex64>,
        input: PyReadonlyArray2<'py, Complex64>,
        output: &Bound<'py, PyArray2<Complex64>>,
        overwrite: bool,
    ) -> PyResult<()> {
        let n = self.inner.dim();
        if input.shape()[0] != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input first dim must be {n}"
            )));
        }
        let coeff_vec: Vec<Complex<f64>> = coeff
            .as_array()
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
        let n_vecs = input.shape()[1];
        let in_slice = input
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let in_arr: Vec<Complex<f64>> = in_slice.iter().map(|c| Complex::new(c.re, c.im)).collect();
        let in_view = ndarray::ArrayView2::from_shape((n, n_vecs), &in_arr)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let mut out_vec: Vec<Complex<f64>> = {
            let out_arr = unsafe { output.as_array() };
            out_arr.iter().map(|c| Complex::new(c.re, c.im)).collect()
        };
        let mut out_view = ndarray::ArrayViewMut2::from_shape((n, n_vecs), &mut out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner
            .dot_transpose_many(overwrite, &coeff_vec, in_view, out_view.view_mut())
            .map_err(Error::from)?;
        unsafe {
            let mut out_arr = output.as_array_mut();
            for ((r, c), v) in out_arr.indexed_iter_mut() {
                *v = Complex64::new(out_vec[r * n_vecs + c].re, out_vec[r * n_vecs + c].im);
            }
        }
        Ok(())
    }
}
