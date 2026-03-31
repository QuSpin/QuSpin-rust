use crate::error::Error;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::seed::seed_from_str;
use quspin_core::basis::{SpaceKind, SpinBasis};

/// Python-facing spin-½ (or spin-S) basis.
///
/// LHSS is always 2 for the `full` constructor on this class.
/// Use [`PyBosonBasis`] for LHSS ≥ 3.
///
/// `subspace` and `symmetric` constructors are added in Step 3 alongside
/// the operator bindings they require.
#[pyclass(name = "SpinBasis", module = "quspin._rs")]
pub struct PySpinBasis {
    pub inner: SpinBasis,
}

#[pymethods]
impl PySpinBasis {
    /// Full Hilbert space (no projection, no build step required).
    ///
    /// Args:
    ///     n_sites: number of lattice sites.
    ///     lhss: local Hilbert-space size (default 2 for spin-½).
    #[classmethod]
    #[pyo3(signature = (n_sites, lhss = 2))]
    fn full(_cls: &Bound<'_, PyType>, n_sites: usize, lhss: usize) -> PyResult<Self> {
        let inner = SpinBasis::new(n_sites, lhss, SpaceKind::Full).map_err(Error::from)?;
        Ok(PySpinBasis { inner })
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
        self.inner.lhss
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
    ///
    /// `state_str` must be a `'0'`/`'1'` string of length `n_sites`.
    fn index(&self, state_str: &str) -> PyResult<Option<usize>> {
        let bytes = seed_from_str(state_str).map_err(Error::from)?;
        Ok(self.inner.inner.index_of_bytes(&bytes))
    }

    fn __repr__(&self) -> String {
        format!(
            "SpinBasis(n_sites={}, lhss={}, size={}, kind={})",
            self.inner.n_sites,
            self.inner.lhss,
            self.inner.inner.size(),
            self.inner.inner.kind(),
        )
    }
}
