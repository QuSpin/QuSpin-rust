//! Python bindings for `ExpmOp` / `ExpmWorker` / `ExpmWorker2`.
//!
//! `PyExpmOp` wraps an `Arc<ExpmOp<Complex<f64>, OwnedQMatrixOperator<...>>>`
//! constructed once at `__init__` time, so the partitioned-Taylor parameter
//! selection runs exactly once per `(qop, a)` pair.  Workers hold an `Arc`
//! clone of that same `ExpmOp`.
//!
//! `expm_op.worker(n_vec, work)` dispatches at construction time:
//! - `n_vec == 0` → `PyExpmWorker` (1-D scratch, accepts 1-D `apply`)
//! - `n_vec  > 0` → `PyExpmWorker2` (2-D scratch of capacity `n_vec`,
//!   accepts 2-D `apply`)

use std::sync::Arc;

use ndarray::Array2;
use num_complex::Complex;
use numpy::{
    Complex64, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use quspin_core::OwnedQMatrixOperator;
use quspin_core::expm::{ExpmOp, ExpmWorker, ExpmWorker2};

use crate::error::Error;
use crate::linear_operator::PyQMatrixLinearOperator;

/// Concrete `ExpmOp` type stored by the Python wrappers.
type PyExpmOpInner = ExpmOp<Complex<f64>, Arc<OwnedQMatrixOperator<Complex<f64>>>>;

// ---------------------------------------------------------------------------
// PyExpmOp
// ---------------------------------------------------------------------------

/// Cached `exp(a · A) · v` action over a `QMatrixLinearOperator`.
///
/// Construction runs the partitioned-Taylor parameter selection once.  Use
/// `worker(...)` to obtain a worker that reuses scratch memory across
/// `apply` calls.
#[pyclass(name = "ExpmOp", module = "quspin_rs._rs", frozen)]
pub struct PyExpmOp {
    inner: Arc<PyExpmOpInner>,
}

#[pymethods]
impl PyExpmOp {
    /// Construct from a `QMatrixLinearOperator` and scalar `a`.
    #[new]
    fn new(qop: &PyQMatrixLinearOperator, a: Complex<f64>) -> PyResult<Self> {
        let op = Arc::clone(&qop.inner);
        let expm_op = ExpmOp::new(op, a).map_err(Error::from)?;
        Ok(Self {
            inner: Arc::new(expm_op),
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn a(&self) -> Complex<f64> {
        self.inner.a()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExpmOp(dim={}, a={}, s={}, m_star={})",
            self.inner.dim(),
            self.inner.a(),
            self.inner.s(),
            self.inner.m_star(),
        )
    }

    /// Build a worker bound to this operator.
    ///
    /// Args:
    ///     n_vec: Output dimensionality / batch capacity.  ``0`` (the default)
    ///         returns a 1-D ``ExpmWorker`` for single-vector application;
    ///         any value ``> 0`` returns a 2-D ``ExpmWorker2`` whose ``apply``
    ///         accepts shape ``(dim, k)`` with ``k <= n_vec``.
    ///     work:  Optional caller-supplied scratch buffer adopted as the
    ///         worker's backing storage (no copy).  For ``n_vec == 0`` a
    ///         1-D ``complex128`` array of length ``>= 2 * dim``; for
    ///         ``n_vec > 0`` a 2-D ``complex128`` array of shape
    ///         ``(>= 2 * dim, >= n_vec)``.  When ``None`` (default), a
    ///         fresh buffer is allocated.
    #[pyo3(signature = (n_vec = 0, work = None))]
    fn worker<'py>(
        &self,
        py: Python<'py>,
        n_vec: usize,
        work: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let dim = self.inner.dim();
        let need = 2 * dim;
        let expm_op = Arc::clone(&self.inner);

        if n_vec == 0 {
            let inner = match work {
                Some(arr) => {
                    let buf = extract_buf_1d(&arr, need)?;
                    ExpmWorker::with_buf(expm_op, buf).map_err(Error::from)?
                }
                None => ExpmWorker::new(expm_op),
            };
            Ok(Py::new(py, PyExpmWorker { inner })?
                .into_bound(py)
                .into_any())
        } else {
            let inner = match work {
                Some(arr) => {
                    let buf = extract_buf_2d(&arr, need, n_vec)?;
                    ExpmWorker2::with_buf(expm_op, buf).map_err(Error::from)?
                }
                None => ExpmWorker2::new(expm_op, n_vec),
            };
            Ok(Py::new(py, PyExpmWorker2 { inner })?
                .into_bound(py)
                .into_any())
        }
    }
}

// ---------------------------------------------------------------------------
// PyExpmWorker (1-D)
// ---------------------------------------------------------------------------

/// 1-D worker bound to an `ExpmOp`.  Holds `2 * dim` complex128 scratch.
#[pyclass(name = "ExpmWorker", module = "quspin_rs._rs")]
pub struct PyExpmWorker {
    inner: ExpmWorker<Complex<f64>, Arc<OwnedQMatrixOperator<Complex<f64>>>>,
}

#[pymethods]
impl PyExpmWorker {
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn __repr__(&self) -> String {
        format!("ExpmWorker(dim={})", self.inner.dim())
    }

    /// Apply ``f ← exp(a · A) · f`` in place.  ``f`` is a 1-D ``complex128``
    /// array of length ``dim``.
    ///
    /// Borrows ``f`` directly — no allocations on the apply hot path.
    fn apply<'py>(&mut self, py: Python<'py>, f: &Bound<'py, PyArray1<Complex64>>) -> PyResult<()> {
        let dim = self.inner.dim();
        let mut rw = f.try_readwrite()?;
        let view = rw.as_array_mut();
        if view.len() != dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected 1-D array of length {dim}"
            )));
        }
        py.detach(|| self.inner.apply(view).map_err(Error::from))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PyExpmWorker2 (2-D batch)
// ---------------------------------------------------------------------------

/// 2-D batch worker bound to an `ExpmOp`.
#[pyclass(name = "ExpmWorker2", module = "quspin_rs._rs")]
pub struct PyExpmWorker2 {
    inner: ExpmWorker2<Complex<f64>, Arc<OwnedQMatrixOperator<Complex<f64>>>>,
}

#[pymethods]
impl PyExpmWorker2 {
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn n_vec(&self) -> usize {
        self.inner.n_vecs()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExpmWorker2(dim={}, n_vec={})",
            self.inner.dim(),
            self.inner.n_vecs(),
        )
    }

    /// Apply ``F ← exp(a · A) · F`` in place.  ``F`` must be a 2-D
    /// ``complex128`` array of shape ``(dim, k)`` with ``k <= n_vec``.
    ///
    /// Borrows ``F`` directly — no allocations on the apply hot path.
    fn apply<'py>(&mut self, py: Python<'py>, f: &Bound<'py, PyArray2<Complex64>>) -> PyResult<()> {
        let dim = self.inner.dim();
        let mut rw = f.try_readwrite()?;
        let view = rw.as_array_mut();
        if view.shape()[0] != dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "f first dim must be {dim}, got {}",
                view.shape()[0]
            )));
        }
        py.detach(|| self.inner.apply(view).map_err(Error::from))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Buffer extraction helpers
// ---------------------------------------------------------------------------

/// Extract a `Vec<Complex<f64>>` of *exactly* `need` elements from a 1-D
/// caller-supplied numpy array.  Validates dtype and length only — the
/// scratch contents are overwritten on first use, so the values copied
/// in are immaterial; we copy the prefix so the resulting Vec has the
/// exact length the worker needs (no oversized allocation).
fn extract_buf_1d(arr: &Bound<'_, PyAny>, need: usize) -> PyResult<Vec<Complex<f64>>> {
    let a1 = arr
        .extract::<PyReadonlyArray1<'_, Complex64>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "`work` for n_vec=1 must be a 1-D complex128 ndarray",
            )
        })?;
    if a1.len() < need {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "`work` length {} < required {need}",
            a1.len()
        )));
    }
    Ok(a1
        .as_array()
        .iter()
        .take(need)
        .map(|c| Complex::new(c.re, c.im))
        .collect())
}

/// Extract a `(need_rows, need_cols)` `Array2<Complex<f64>>` from a 2-D
/// caller-supplied numpy array.  Same scratch-overwrite reasoning as
/// `extract_buf_1d` — we allocate exactly what the worker needs.
fn extract_buf_2d(
    arr: &Bound<'_, PyAny>,
    need_rows: usize,
    need_cols: usize,
) -> PyResult<Array2<Complex<f64>>> {
    let a2 = arr
        .extract::<PyReadonlyArray2<'_, Complex64>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "`work` for n_vec > 1 must be a 2-D complex128 ndarray",
            )
        })?;
    let shape = a2.shape();
    if shape[0] < need_rows || shape[1] < need_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "`work` shape ({}, {}) too small for required ({need_rows}, {need_cols})",
            shape[0], shape[1]
        )));
    }
    let view = a2.as_array();
    let mut out = Array2::from_elem((need_rows, need_cols), Complex::new(0.0, 0.0));
    for r in 0..need_rows {
        for c in 0..need_cols {
            out[[r, c]] = Complex::new(view[[r, c]].re, view[[r, c]].im);
        }
    }
    Ok(out)
}
