use crate::error::Error;
use crate::qmatrix::PyQMatrix;
use num_complex::Complex;
use numpy::{Complex64, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use quspin_core::error::QuSpinError;
use quspin_core::hamiltonian::{CoeffFn, HamiltonianInner};
use std::sync::Arc;

/// Return type of `PyHamiltonian::to_csr`.
type CsrArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<Complex64>>,
);

/// Marker type indicating a static (time-independent) coefficient.
///
/// Pass `Static()` in the `coeff_fns` list to mark a cindex as static
/// (coefficient = 1.0) rather than providing a callable.
#[pyclass(name = "Static", module = "quspin._rs", frozen)]
pub struct PyStatic;

#[pymethods]
impl PyStatic {
    #[new]
    fn new() -> Self {
        PyStatic
    }

    fn __repr__(&self) -> &str {
        "Static"
    }
}

/// Python-facing time-dependent Hamiltonian.
///
/// Wraps a `HamiltonianInner` (from `quspin-core`) behind an `Arc` so that
/// `PySchrodingerEq` can share the same allocation without cloning.
///
/// Python coefficient functions are wrapped in closures that call
/// `Python::attach`, making the underlying `HamiltonianInner` fully
/// `Send + Sync`.  This lets the ODE integrator release the GIL for the
/// bulk of its computation while still calling back into Python per step.
#[pyclass(name = "Hamiltonian", module = "quspin._rs")]
pub struct PyHamiltonian {
    pub inner: Arc<HamiltonianInner>,
}

// ---------------------------------------------------------------------------
// PyHamiltonian pymethods
// ---------------------------------------------------------------------------

#[pymethods]
impl PyHamiltonian {
    /// Construct a Hamiltonian.
    ///
    /// Args:
    ///     qmatrix:   A `QMatrix` whose cindices encode the operator terms.
    ///     coeff_fns: A list of length `qmatrix.num_coeff`, one entry per
    ///                cindex.  Each entry is either `Static()` (coefficient
    ///                is always 1.0) or a callable `f(t: float) -> complex`.
    #[new]
    #[pyo3(signature = (qmatrix, coeff_fns))]
    fn new(py: Python<'_>, qmatrix: &PyQMatrix, coeff_fns: Vec<Py<PyAny>>) -> PyResult<Self> {
        let expected = qmatrix.inner.num_coeff();
        if coeff_fns.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coeff_fns has {} entries but need {} (qmatrix.num_coeff)",
                coeff_fns.len(),
                expected,
            )));
        }

        // Convert each entry: Static() → None, callable → Some(CoeffFn).
        let rust_fns: Vec<Option<CoeffFn>> = coeff_fns
            .into_iter()
            .map(|obj| {
                if obj.cast_bound::<PyStatic>(py).is_ok() {
                    None
                } else {
                    let arc_fn: CoeffFn = Arc::new(move |t: f64| {
                        Python::attach(|py| {
                            let result = obj.call1(py, (t,)).expect("coeff_fn call failed");
                            if let Ok(z) = result.extract::<Complex<f64>>(py) {
                                z
                            } else {
                                let re: f64 = result
                                    .extract(py)
                                    .expect("coeff_fn: expected float or complex");
                                Complex::new(re, 0.0)
                            }
                        })
                    });
                    Some(arc_fn)
                }
            })
            .collect();

        let inner = HamiltonianInner::from_qmatrix_inner(qmatrix.inner.clone(), rust_fns)
            .map_err(Error::from)?;

        Ok(PyHamiltonian {
            inner: Arc::new(inner),
        })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
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
            "Hamiltonian(dim={}, num_coeff={}, dtype={})",
            self.inner.dim(),
            self.inner.num_coeff(),
            self.inner.dtype_name(),
        )
    }

    // ------------------------------------------------------------------
    // CSR export
    // ------------------------------------------------------------------

    /// Materialise the Hamiltonian at time `t` as scipy-compatible CSR arrays.
    ///
    /// The GIL is released during the sparse-matrix assembly; Python
    /// coefficient callables re-acquire it briefly as needed.
    ///
    /// Returns:
    ///     (indptr, indices, data) with dtypes (int64, int64, complex128).
    #[pyo3(signature = (time, drop_zeros = true))]
    fn to_csr<'py>(
        &self,
        py: Python<'py>,
        time: f64,
        drop_zeros: bool,
    ) -> PyResult<CsrArrays<'py>> {
        let inner = Arc::clone(&self.inner);
        let result = py.detach(move || inner.to_csr(time, drop_zeros));
        let (indptr, indices, data) = result.map_err(Error::from)?;
        Ok((
            indptr.to_pyarray(py),
            indices.to_pyarray(py),
            data.to_pyarray(py),
        ))
    }

    // ------------------------------------------------------------------
    // Dense export
    // ------------------------------------------------------------------

    /// Materialise the Hamiltonian at time `t` as a dense complex128 matrix.
    ///
    /// Returns a 2-D array of shape `(dim, dim)` in row-major order.
    fn to_dense<'py>(
        &self,
        py: Python<'py>,
        time: f64,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let n = self.inner.dim();
        let inner = Arc::clone(&self.inner);
        let result = py.detach(move || inner.to_dense(time));
        let data = result.map_err(Error::from)?;
        let arr_data: Vec<Complex64> = data.iter().map(|c| Complex64::new(c.re, c.im)).collect();
        let arr = ndarray::Array2::from_shape_vec((n, n), arr_data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.to_pyarray(py))
    }

    // ------------------------------------------------------------------
    // 1-D matrix-vector product
    // ------------------------------------------------------------------

    /// Compute `output = H(t) @ input` (or `+= H(t) @ input` when `overwrite=False`).
    fn dot<'py>(
        &self,
        py: Python<'py>,
        time: f64,
        input: PyReadonlyArray1<'py, Complex64>,
        output: &Bound<'py, PyArray1<Complex64>>,
        overwrite: bool,
    ) -> PyResult<()> {
        let n = self.inner.dim();
        if input.len() != n || unsafe { output.as_array().len() } != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected arrays of length {n}"
            )));
        }
        // Copy arrays to owned Vecs while holding the GIL.
        let in_vec: Vec<Complex<f64>> = input
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
        let mut out_vec: Vec<Complex<f64>> = unsafe { output.as_array() }
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();

        let inner = Arc::clone(&self.inner);
        let result = py.detach(move || -> Result<Vec<Complex<f64>>, QuSpinError> {
            inner.dot(overwrite, time, &in_vec, &mut out_vec)?;
            Ok(out_vec)
        });
        let out_vec = result.map_err(Error::from)?;

        unsafe {
            let mut out_arr = output.as_array_mut();
            for (i, v) in out_arr.iter_mut().enumerate() {
                *v = Complex64::new(out_vec[i].re, out_vec[i].im);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Batch matrix-vector products
    // ------------------------------------------------------------------

    /// Compute `output += H(t) @ input` column-wise (GIL released).
    ///
    /// Args:
    ///     time:      Evaluation time for the coefficient functions.
    ///     input:     2-D complex128 array of shape `(dim, n_vecs)`.
    ///     output:    2-D complex128 array of shape `(dim, n_vecs)` (in place).
    ///     overwrite: if True, zero `output` before accumulating.
    fn dot_many<'py>(
        &self,
        py: Python<'py>,
        time: f64,
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
        let n_vecs = input.shape()[1];
        let in_arr: Vec<Complex<f64>> = input
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
        let mut out_vec: Vec<Complex<f64>> = unsafe { output.as_array() }
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();

        let inner = Arc::clone(&self.inner);
        let result = py.detach(move || -> Result<Vec<Complex<f64>>, QuSpinError> {
            let in_view = ndarray::ArrayView2::from_shape((n, n_vecs), &in_arr)
                .map_err(|e| QuSpinError::ValueError(e.to_string()))?;
            let mut out_view = ndarray::ArrayViewMut2::from_shape((n, n_vecs), &mut out_vec)
                .map_err(|e| QuSpinError::ValueError(e.to_string()))?;
            inner.dot_many(overwrite, time, in_view, out_view.view_mut())?;
            Ok(out_vec)
        });
        let out_vec = result.map_err(Error::from)?;

        unsafe {
            let mut out_arr = output.as_array_mut();
            for ((r, c), v) in out_arr.indexed_iter_mut() {
                *v = Complex64::new(out_vec[r * n_vecs + c].re, out_vec[r * n_vecs + c].im);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Matrix exponential
    // ------------------------------------------------------------------

    /// Compute `exp(a · H(time)) · f` in-place (GIL released).
    fn expm_dot<'py>(
        &self,
        py: Python<'py>,
        time: f64,
        a: Complex<f64>,
        f: &Bound<'py, PyArray1<Complex64>>,
    ) -> PyResult<()> {
        let n = self.inner.dim();
        if unsafe { f.as_array().len() } != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected array of length {n}"
            )));
        }
        let mut f_vec: Vec<Complex<f64>> = unsafe { f.as_array() }
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
        let inner = Arc::clone(&self.inner);
        let result = py.detach(move || -> Result<Vec<Complex<f64>>, QuSpinError> {
            inner.expm_dot(time, a, &mut f_vec)?;
            Ok(f_vec)
        });
        let f_vec = result.map_err(Error::from)?;
        unsafe {
            let mut f_arr = f.as_array_mut();
            for (i, v) in f_arr.iter_mut().enumerate() {
                *v = Complex64::new(f_vec[i].re, f_vec[i].im);
            }
        }
        Ok(())
    }

    /// Compute `exp(a · H(time)) · F` in-place for multiple column vectors (GIL released).
    ///
    /// Args:
    ///     time: Evaluation time for the coefficient functions.
    ///     a:    Scalar multiplier on the Hamiltonian.
    ///     f:    2-D complex128 array of shape `(dim, n_vecs)` (modified in place).
    fn expm_dot_many<'py>(
        &self,
        py: Python<'py>,
        time: f64,
        a: Complex<f64>,
        f: &Bound<'py, PyArray2<Complex64>>,
    ) -> PyResult<()> {
        let n = self.inner.dim();
        if unsafe { f.as_array().shape()[0] } != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "f first dim must be {n}"
            )));
        }
        let n_vecs = unsafe { f.as_array().shape()[1] };
        let mut f_vec: Vec<Complex<f64>> = unsafe { f.as_array() }
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
        let inner = Arc::clone(&self.inner);
        let result = py.detach(move || -> Result<Vec<Complex<f64>>, QuSpinError> {
            let mut f_view = ndarray::ArrayViewMut2::from_shape((n, n_vecs), &mut f_vec)
                .map_err(|e| QuSpinError::ValueError(e.to_string()))?;
            inner.expm_dot_many(time, a, f_view.view_mut())?;
            Ok(f_vec)
        });
        let f_vec = result.map_err(Error::from)?;
        unsafe {
            let mut f_arr = f.as_array_mut();
            for ((r, c), v) in f_arr.indexed_iter_mut() {
                *v = Complex64::new(f_vec[r * n_vecs + c].re, f_vec[r * n_vecs + c].im);
            }
        }
        Ok(())
    }

    /// Transpose matrix-vector product (batch, GIL released).
    fn dot_transpose_many<'py>(
        &self,
        py: Python<'py>,
        time: f64,
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
        let n_vecs = input.shape()[1];
        let in_arr: Vec<Complex<f64>> = input
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
        let mut out_vec: Vec<Complex<f64>> = unsafe { output.as_array() }
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();

        let inner = Arc::clone(&self.inner);
        let result = py.detach(move || -> Result<Vec<Complex<f64>>, QuSpinError> {
            let in_view = ndarray::ArrayView2::from_shape((n, n_vecs), &in_arr)
                .map_err(|e| QuSpinError::ValueError(e.to_string()))?;
            let mut out_view = ndarray::ArrayViewMut2::from_shape((n, n_vecs), &mut out_vec)
                .map_err(|e| QuSpinError::ValueError(e.to_string()))?;
            inner.dot_transpose_many(overwrite, time, in_view, out_view.view_mut())?;
            Ok(out_vec)
        });
        let out_vec = result.map_err(Error::from)?;

        unsafe {
            let mut out_arr = output.as_array_mut();
            for ((r, c), v) in out_arr.indexed_iter_mut() {
                *v = Complex64::new(out_vec[r * n_vecs + c].re, out_vec[r * n_vecs + c].im);
            }
        }
        Ok(())
    }
}
