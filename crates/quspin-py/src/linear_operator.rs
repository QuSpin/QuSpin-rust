use std::sync::Arc;

use ndarray::Array2;
use num_complex::Complex;
use numpy::{
    Complex64, PyArray1, PyArray2, PyArrayDescr, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use quspin_core::{LinearOperator, OwnedQMatrixOperator};

use crate::error::Error;

/// Python-facing snapshot of a `QMatrix` + coefficient vector — i.e. a
/// concrete `LinearOperator<Complex<f64>>`.
///
/// Internally an `Arc<OwnedQMatrixOperator<Complex<f64>>>` shared with
/// `PyExpmOp` so the same matrix allocation backs many `apply` calls.
/// Constructed via `QMatrix.as_linearoperator(coeffs)` or
/// `Hamiltonian.as_linearoperator(time)`; not constructible from Python
/// directly.
///
/// Exposes the SciPy `LinearOperator` duck-typed interface (`shape`,
/// `dtype`, `matvec`, `matmat`, `rmatvec`, `rmatmat`, `@`) so instances can
/// be passed directly to `scipy.sparse.linalg` routines.
#[pyclass(name = "QMatrixLinearOperator", module = "quspin_rs._rs", frozen)]
pub struct PyQMatrixLinearOperator {
    pub inner: Arc<OwnedQMatrixOperator<Complex<f64>>>,
}

#[pymethods]
impl PyQMatrixLinearOperator {
    /// Opt out of numpy ufuncs so `np.ndarray @ qop` defers to our
    /// `__rmatmul__` instead of numpy attempting an elementwise op.
    #[classattr]
    fn __array_ufunc__(py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.matrix().dim()
    }

    #[getter]
    fn num_coeff(&self) -> usize {
        self.inner.matrix().num_coeff()
    }

    /// SciPy-compatible ``(dim, dim)`` shape tuple.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        let n = self.inner.matrix().dim();
        (n, n)
    }

    /// SciPy-compatible ``numpy.dtype``.  Always ``complex128`` — matvec is
    /// run in complex128 regardless of the matrix's storage dtype because
    /// `as_linearoperator` accepts complex coefficients.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        numpy::dtype::<Complex64>(py)
    }

    /// ``A @ x`` for a 1-D complex128 input.
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let n = self.inner.matrix().dim();
        if x.len() != n {
            return Err(PyValueError::new_err(format!(
                "x must have length {n}, got {}",
                x.len(),
            )));
        }
        let in_slice = x
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = PyArray1::<Complex64>::zeros(py, [n], false);
        {
            let mut rw = out.try_readwrite()?;
            let mut view = rw.as_array_mut();
            let out_slice = view
                .as_slice_mut()
                .expect("freshly-allocated PyArray1 is contiguous");
            py.detach(|| self.inner.dot(true, in_slice, out_slice))
                .map_err(Error::from)?;
        }
        Ok(out)
    }

    /// ``A @ X`` for a 2-D complex128 input of shape ``(dim, k)``.
    fn matmat<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let n = self.inner.matrix().dim();
        let shape = x.shape();
        if shape[0] != n {
            return Err(PyValueError::new_err(format!(
                "x.shape[0] must be {n}, got {}",
                shape[0],
            )));
        }
        let k = shape[1];
        let in_view = x.as_array();
        let out = PyArray2::<Complex64>::zeros(py, [n, k], false);
        {
            let mut rw = out.try_readwrite()?;
            let out_view = rw.as_array_mut();
            py.detach(|| self.inner.dot_many(true, in_view, out_view))
                .map_err(Error::from)?;
        }
        Ok(out)
    }

    /// ``A^H @ x`` for a 1-D input — SciPy `rmatvec`.  Computed as
    /// ``conj(A^T @ conj(x))`` so it matches the Hermitian adjoint even when
    /// the underlying matrix has complex entries.
    fn rmatvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let n = self.inner.matrix().dim();
        if x.len() != n {
            return Err(PyValueError::new_err(format!(
                "x must have length {n}, got {}",
                x.len(),
            )));
        }
        let in_conj: Vec<Complex<f64>> = x.as_array().iter().map(|c| c.conj()).collect();
        let out = PyArray1::<Complex64>::zeros(py, [n], false);
        {
            let mut rw = out.try_readwrite()?;
            let mut view = rw.as_array_mut();
            let out_slice = view
                .as_slice_mut()
                .expect("freshly-allocated PyArray1 is contiguous");
            py.detach(|| self.inner.dot_transpose(true, &in_conj, out_slice))
                .map_err(Error::from)?;
            for v in out_slice.iter_mut() {
                *v = v.conj();
            }
        }
        Ok(out)
    }

    /// ``A^H @ X`` for a 2-D input of shape ``(dim, k)`` — SciPy `rmatmat`.
    fn rmatmat<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let n = self.inner.matrix().dim();
        let shape = x.shape();
        if shape[0] != n {
            return Err(PyValueError::new_err(format!(
                "x.shape[0] must be {n}, got {}",
                shape[0],
            )));
        }
        let k = shape[1];
        let x_arr = x.as_array();
        let mut in_conj = Array2::<Complex<f64>>::zeros((n, k));
        for ((r, c), v) in x_arr.indexed_iter() {
            in_conj[[r, c]] = v.conj();
        }
        let out = PyArray2::<Complex64>::zeros(py, [n, k], false);
        {
            let mut rw = out.try_readwrite()?;
            let mut view = rw.as_array_mut();
            py.detach(|| {
                self.inner.matrix().dot_transpose_many(
                    true,
                    self.inner.coeffs(),
                    in_conj.view(),
                    view.view_mut(),
                )
            })
            .map_err(Error::from)?;
            view.mapv_inplace(|c| c.conj());
        }
        Ok(out)
    }

    /// ``A @ x`` — dispatches to `matvec` (1-D) or `matmat` (2-D).
    fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(x) = other.extract::<PyReadonlyArray1<'py, Complex64>>() {
            return Ok(self.matvec(py, x)?.into_any());
        }
        if let Ok(x) = other.extract::<PyReadonlyArray2<'py, Complex64>>() {
            return Ok(self.matmat(py, x)?.into_any());
        }
        Err(PyTypeError::new_err(
            "QMatrixLinearOperator @ x: x must be a 1-D or 2-D complex128 ndarray",
        ))
    }

    /// ``x @ A`` — for a 1-D input returns ``A^T @ x``; for a 2-D ``(m, dim)``
    /// input returns ``x @ A`` (shape ``(m, dim)``).  Note this is the plain
    /// transpose (not Hermitian adjoint) to match numpy's ``@`` semantics.
    fn __rmatmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.matrix().dim();
        if let Ok(x) = other.extract::<PyReadonlyArray1<'py, Complex64>>() {
            if x.len() != n {
                return Err(PyValueError::new_err(format!(
                    "x must have length {n}, got {}",
                    x.len(),
                )));
            }
            let in_slice = x
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let out = PyArray1::<Complex64>::zeros(py, [n], false);
            {
                let mut rw = out.try_readwrite()?;
                let mut view = rw.as_array_mut();
                let out_slice = view
                    .as_slice_mut()
                    .expect("freshly-allocated PyArray1 is contiguous");
                py.detach(|| self.inner.dot_transpose(true, in_slice, out_slice))
                    .map_err(Error::from)?;
            }
            return Ok(out.into_any());
        }
        if let Ok(x) = other.extract::<PyReadonlyArray2<'py, Complex64>>() {
            let shape = x.shape();
            if shape[1] != n {
                return Err(PyValueError::new_err(format!(
                    "x.shape[1] must be {n}, got {}",
                    shape[1],
                )));
            }
            let m = shape[0];
            // x @ A == (A^T @ x^T)^T.  Build a (n, m) contiguous copy of x^T,
            // run dot_transpose_many into a (n, m) buffer, then transpose into
            // the output (m, n).
            let x_arr = x.as_array();
            let mut x_t = Array2::<Complex<f64>>::zeros((n, m));
            for ((i, j), v) in x_arr.indexed_iter() {
                x_t[[j, i]] = *v;
            }
            let mut tmp = Array2::<Complex<f64>>::zeros((n, m));
            py.detach(|| {
                self.inner.matrix().dot_transpose_many(
                    true,
                    self.inner.coeffs(),
                    x_t.view(),
                    tmp.view_mut(),
                )
            })
            .map_err(Error::from)?;
            let out = PyArray2::<Complex64>::zeros(py, [m, n], false);
            {
                let mut rw = out.try_readwrite()?;
                let mut view = rw.as_array_mut();
                for ((i, j), v) in tmp.indexed_iter() {
                    view[[j, i]] = *v;
                }
            }
            return Ok(out.into_any());
        }
        Err(PyTypeError::new_err(
            "x @ QMatrixLinearOperator: x must be a 1-D or 2-D complex128 ndarray",
        ))
    }

    fn __repr__(&self) -> String {
        let n = self.inner.matrix().dim();
        format!(
            "QMatrixLinearOperator(shape=({n}, {n}), num_coeff={}, dtype=complex128)",
            self.inner.matrix().num_coeff(),
        )
    }
}
