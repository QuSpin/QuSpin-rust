use std::sync::Arc;

use num_complex::Complex;
use pyo3::prelude::*;
use quspin_core::OwnedQMatrixOperator;

/// Python-facing snapshot of a `QMatrix` + coefficient vector — i.e. a
/// concrete `LinearOperator<Complex<f64>>`.
///
/// Internally an `Arc<OwnedQMatrixOperator<Complex<f64>>>` shared with
/// `PyExpmOp` so the same matrix allocation backs many `apply` calls.
/// Constructed via `QMatrix.as_linearoperator(coeffs)` or
/// `Hamiltonian.as_linearoperator(time)`; not constructible from Python
/// directly.
#[pyclass(name = "QMatrixLinearOperator", module = "quspin_rs._rs", frozen)]
pub struct PyQMatrixLinearOperator {
    pub inner: Arc<OwnedQMatrixOperator<Complex<f64>>>,
}

#[pymethods]
impl PyQMatrixLinearOperator {
    #[getter]
    fn dim(&self) -> usize {
        self.inner.matrix().dim()
    }

    #[getter]
    fn num_coeff(&self) -> usize {
        self.inner.matrix().num_coeff()
    }

    #[getter]
    fn dtype(&self) -> &str {
        self.inner.matrix().dtype_name()
    }

    fn __repr__(&self) -> String {
        format!(
            "QMatrixLinearOperator(dim={}, num_coeff={}, dtype={})",
            self.inner.matrix().dim(),
            self.inner.matrix().num_coeff(),
            self.inner.matrix().dtype_name(),
        )
    }
}
