use crate::error::Error;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::seed::{dit_seed_from_str, seed_from_str};
use quspin_core::basis::{BosonBasis, SpaceKind};

/// Python-facing bosonic basis.
///
/// `lhss` is the number of on-site Fock states (≥ 2).  Use `lhss = 2` for
/// hard-core bosons (equivalent to spin-½ without Jordan-Wigner signs).
///
/// `subspace` and `symmetric` constructors are added in Step 3 alongside
/// the operator bindings they require.
#[pyclass(name = "BosonBasis", module = "quspin._rs")]
pub struct PyBosonBasis {
    pub inner: BosonBasis,
}

#[pymethods]
impl PyBosonBasis {
    /// Full Hilbert space (no projection, no build step required).
    ///
    /// Args:
    ///     n_sites: number of lattice sites.
    ///     lhss:    on-site Fock-state count (≥ 2).
    #[classmethod]
    fn full(_cls: &Bound<'_, PyType>, n_sites: usize, lhss: usize) -> PyResult<Self> {
        let inner = BosonBasis::new(n_sites, lhss, SpaceKind::Full).map_err(Error::from)?;
        Ok(PyBosonBasis { inner })
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

    /// Return the `i`-th basis state as a string of site occupations.
    ///
    /// For LHSS=2, each character is `'0'` or `'1'`.
    /// For LHSS>2, use `state_at` — the encoding is the same bit-string
    /// representation used internally.
    fn state_at(&self, i: usize) -> PyResult<String> {
        if i >= self.inner.inner.size() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {i} out of range for basis of size {}",
                self.inner.inner.size()
            )));
        }
        Ok(self.inner.inner.state_at_str(i))
    }

    /// Return the index of `state_str`, or `None` if absent.
    ///
    /// For LHSS=2, `state_str` is a `'0'`/`'1'` string.
    /// For LHSS>2, `state_str` is a string of decimal digit characters,
    /// one per site, each in `0..lhss`.
    fn index(&self, state_str: &str) -> PyResult<Option<usize>> {
        let bytes = if self.inner.lhss == 2 {
            seed_from_str(state_str).map_err(Error::from)?
        } else {
            dit_seed_from_str(state_str, self.inner.lhss).map_err(Error::from)?
        };
        Ok(self.inner.inner.index_of_bytes(&bytes))
    }

    fn __repr__(&self) -> String {
        format!(
            "BosonBasis(n_sites={}, lhss={}, size={}, kind={})",
            self.inner.n_sites,
            self.inner.lhss,
            self.inner.inner.size(),
            self.inner.inner.kind(),
        )
    }
}
