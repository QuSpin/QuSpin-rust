use crate::error::Error;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::seed::seed_from_str;
use quspin_core::basis::{FermionBasis, SpaceKind};

/// Python-facing fermionic basis.
///
/// Fermions are always LHSS=2 (one bit per orbital).
///
/// `subspace` and `symmetric` constructors are added in Step 3 alongside
/// the operator bindings they require.
#[pyclass(name = "FermionBasis", module = "quspin._rs")]
pub struct PyFermionBasis {
    pub inner: FermionBasis,
}

#[pymethods]
impl PyFermionBasis {
    /// Full Hilbert space (no projection, no build step required).
    ///
    /// Args:
    ///     n_sites: number of orbitals / lattice sites (max 64).
    #[classmethod]
    fn full(_cls: &Bound<'_, PyType>, n_sites: usize) -> PyResult<Self> {
        let inner = FermionBasis::new(n_sites, SpaceKind::Full).map_err(Error::from)?;
        Ok(PyFermionBasis { inner })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    #[getter]
    fn n_sites(&self) -> usize {
        self.inner.n_sites
    }

    #[getter]
    fn lhss(&self) -> usize {
        2
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.inner.size()
    }

    #[getter]
    fn is_built(&self) -> bool {
        self.inner.inner.is_built()
    }

    // ------------------------------------------------------------------
    // Methods
    // ------------------------------------------------------------------

    /// Return the `i`-th basis state as a bit-string (`'0'`/`'1'` per site).
    fn state_at(&self, i: usize) -> PyResult<String> {
        if i >= self.inner.inner.size() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {i} out of range for basis of size {}",
                self.inner.inner.size()
            )));
        }
        Ok(self.inner.inner.state_at_str(i))
    }

    /// Return the index of `state_str` in this basis, or `None` if absent.
    fn index(&self, state_str: &str) -> PyResult<Option<usize>> {
        let bytes = seed_from_str(state_str).map_err(Error::from)?;
        Ok(self.inner.inner.index_of_bytes(&bytes))
    }

    fn __repr__(&self) -> String {
        format!(
            "FermionBasis(n_sites={}, size={}, kind={})",
            self.inner.n_sites,
            self.inner.inner.size(),
            self.inner.inner.kind(),
        )
    }
}
