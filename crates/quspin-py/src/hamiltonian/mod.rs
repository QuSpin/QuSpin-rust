use crate::error::Error;
use crate::qmatrix::PyQMatrix;
use num_complex::Complex;
use numpy::{Complex64, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use quspin_core::qmatrix::QMatrixInner;

/// Return type of `PyHamiltonian::to_csr`.
type CsrArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<Complex64>>,
);

/// Python-facing time-dependent Hamiltonian.
///
/// Wraps a `QMatrixInner` (the sparse structure, with one operator string per
/// cindex) together with a list of Python callables — one per time-dependent
/// coupling constant (cindex 1, 2, ...).  Cindex 0 is the static part and is
/// always multiplied by 1.
///
/// When evaluating at time `t`, the full coefficient vector is assembled as:
///   `coeffs = [1, f1(t), f2(t), ...]`
/// and forwarded to the underlying `QMatrixInner`.
#[pyclass(name = "Hamiltonian", module = "quspin._rs")]
pub struct PyHamiltonian {
    pub matrix: QMatrixInner,
    /// Python callables `f(t: float) -> complex`, one per dynamic cindex.
    coeff_fns: Vec<PyObject>,
}

// ---------------------------------------------------------------------------
// Helper: evaluate Python coefficient functions at time t
// ---------------------------------------------------------------------------

fn eval_coeffs(py: Python<'_>, coeff_fns: &[PyObject], time: f64) -> PyResult<Vec<Complex<f64>>> {
    let mut coeffs = Vec::with_capacity(1 + coeff_fns.len());
    coeffs.push(Complex::new(1.0, 0.0)); // cindex 0: static
    for f in coeff_fns {
        let result = f.call1(py, (time,))?;
        let c: Complex<f64> = if let Ok(z) = result.extract::<Complex<f64>>(py) {
            z
        } else {
            let re: f64 = result.extract(py)?;
            Complex::new(re, 0.0)
        };
        coeffs.push(c);
    }
    Ok(coeffs)
}

// ---------------------------------------------------------------------------
// PyHamiltonian pymethods
// ---------------------------------------------------------------------------

#[pymethods]
impl PyHamiltonian {
    /// Construct a time-dependent Hamiltonian.
    ///
    /// Args:
    ///     qmatrix:   A `QMatrix` whose cindices encode the operator terms.
    ///                Cindex 0 is the static part (coefficient = 1).
    ///                Cindices 1..num_coeff are time-dependent.
    ///     coeff_fns: A list of Python callables `f(t: float) -> complex`,
    ///                one per dynamic cindex.  Length must equal
    ///                `qmatrix.num_coeff - 1` (or 0 for a static Hamiltonian).
    #[new]
    #[pyo3(signature = (qmatrix, coeff_fns))]
    fn new(qmatrix: &PyQMatrix, coeff_fns: Vec<PyObject>) -> PyResult<Self> {
        let expected = qmatrix.inner.num_coeff().saturating_sub(1);
        if coeff_fns.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coeff_fns has {} entries but need {} (qmatrix.num_coeff - 1)",
                coeff_fns.len(),
                expected,
            )));
        }
        Ok(PyHamiltonian {
            matrix: qmatrix.inner.clone(),
            coeff_fns,
        })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    #[getter]
    fn dim(&self) -> usize {
        self.matrix.dim()
    }

    #[getter]
    fn num_coeff(&self) -> usize {
        self.matrix.num_coeff()
    }

    #[getter]
    fn dtype(&self) -> &str {
        self.matrix.dtype_name()
    }

    fn __repr__(&self) -> String {
        format!(
            "Hamiltonian(dim={}, num_coeff={}, dtype={})",
            self.matrix.dim(),
            self.matrix.num_coeff(),
            self.matrix.dtype_name(),
        )
    }

    // ------------------------------------------------------------------
    // CSR export
    // ------------------------------------------------------------------

    /// Materialise the Hamiltonian at time `t` as scipy-compatible CSR arrays.
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
        let coeffs = eval_coeffs(py, &self.coeff_fns, time)?;
        let (indptr, indices, data) = self
            .matrix
            .materialize(&coeffs, drop_zeros)
            .map_err(Error::from)?;
        Ok((
            indptr.to_pyarray(py),
            indices.to_pyarray(py),
            data.to_pyarray(py),
        ))
    }

    // ------------------------------------------------------------------
    // 1-D matrix-vector products
    // ------------------------------------------------------------------

    /// Compute `output += H(t) @ input` (1-D, single vector).
    fn dot<'py>(
        &self,
        py: Python<'py>,
        time: f64,
        input: PyReadonlyArray1<'py, Complex64>,
        output: &Bound<'py, PyArray1<Complex64>>,
        overwrite: bool,
    ) -> PyResult<()> {
        let n = self.matrix.dim();
        if input.len() != n || unsafe { output.as_array().len() } != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected arrays of length {n}"
            )));
        }
        let coeffs = eval_coeffs(py, &self.coeff_fns, time)?;
        let in_slice = input
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let in_vec: Vec<Complex<f64>> = in_slice.iter().map(|c| Complex::new(c.re, c.im)).collect();
        let mut out_vec: Vec<Complex<f64>> = {
            let out_arr = unsafe { output.as_array() };
            out_arr.iter().map(|c| Complex::new(c.re, c.im)).collect()
        };
        self.matrix
            .dot(overwrite, &coeffs, &in_vec, &mut out_vec)
            .map_err(Error::from)?;
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

    /// Compute `output += H(t) @ input` column-wise.
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
        let n = self.matrix.dim();
        if input.shape()[0] != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input first dim must be {n}"
            )));
        }
        let coeffs = eval_coeffs(py, &self.coeff_fns, time)?;
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
        self.matrix
            .dot_many(overwrite, &coeffs, in_view, out_view.view_mut())
            .map_err(Error::from)?;
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
        py: Python<'py>,
        time: f64,
        input: PyReadonlyArray2<'py, Complex64>,
        output: &Bound<'py, PyArray2<Complex64>>,
        overwrite: bool,
    ) -> PyResult<()> {
        let n = self.matrix.dim();
        if input.shape()[0] != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input first dim must be {n}"
            )));
        }
        let coeffs = eval_coeffs(py, &self.coeff_fns, time)?;
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
        self.matrix
            .dot_transpose_many(overwrite, &coeffs, in_view, out_view.view_mut())
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
