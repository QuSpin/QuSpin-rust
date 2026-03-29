/// Symmetry group pyclasses — STUB (quspin-py rewrite pending).
///
/// The old `SpinSymGrp`, `DitSymGrp`, and `FermionicSymGrp` types have been
/// removed from quspin-core. These stubs keep quspin-py compiling until the
/// full rewrite lands.
use pyo3::prelude::*;

#[pyclass(name = "PySpinSymGrp")]
pub struct PySpinSymGrp;

#[pymethods]
impl PySpinSymGrp {
    #[new]
    pub fn new(_lhss: usize, _n_sites: usize) -> PyResult<Self> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "PySpinSymGrp is not available — quspin-py rewrite in progress",
        ))
    }
}

#[pyclass(name = "PyDitSymGrp")]
pub struct PyDitSymGrp;

#[pymethods]
impl PyDitSymGrp {
    #[new]
    pub fn new(_lhss: usize, _n_sites: usize) -> PyResult<Self> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "PyDitSymGrp is not available — quspin-py rewrite in progress",
        ))
    }
}

#[pyclass(name = "PyFermionicSymGrp")]
pub struct PyFermionicSymGrp;

#[pymethods]
impl PyFermionicSymGrp {
    #[new]
    pub fn new(_n_sites: usize) -> PyResult<Self> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "PyFermionicSymGrp is not available — quspin-py rewrite in progress",
        ))
    }
}
