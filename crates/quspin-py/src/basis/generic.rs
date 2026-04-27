use crate::basis::{group_n_sites_lhss, parse_seeds, parse_state_str, replay_group_into_generic};
use crate::error::Error;
use crate::operator::monomial::PyMonomialOperator;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::{GenericBasis, SpaceKind};

/// Python-facing generic basis for any on-site Hilbert-space size.
///
/// Supports both lattice (site-permutation) and local (dit-permutation)
/// symmetries.  Paired with `MonomialOperator` for Hamiltonian construction.
#[pyclass(name = "GenericBasis", module = "quspin._rs")]
pub struct PyGenericBasis {
    pub inner: GenericBasis,
}

#[pymethods]
impl PyGenericBasis {
    /// Full Hilbert space (no projection, no build step required).
    ///
    /// Args:
    ///     n_sites: number of lattice sites.
    ///     lhss:    on-site state count (≥ 2).
    #[classmethod]
    fn full(_cls: &Bound<'_, PyType>, n_sites: usize, lhss: usize) -> PyResult<Self> {
        let inner =
            GenericBasis::new(n_sites, lhss, SpaceKind::Full, false).map_err(Error::from)?;
        Ok(PyGenericBasis { inner })
    }

    /// Subspace built by BFS from seed states using a `MonomialOperator`.
    ///
    /// Args:
    ///     n_sites: number of lattice sites.
    ///     lhss:    on-site state count (≥ 2).
    ///     ham:     `MonomialOperator` used for BFS.
    ///     seeds:   list of seed state strings (one digit per site).
    #[classmethod]
    fn subspace(
        _cls: &Bound<'_, PyType>,
        n_sites: usize,
        lhss: usize,
        ham: &PyMonomialOperator,
        seeds: Vec<String>,
    ) -> PyResult<Self> {
        let byte_seeds = parse_seeds(&seeds, lhss)?;
        let mut basis =
            GenericBasis::new(n_sites, lhss, SpaceKind::Sub, false).map_err(Error::from)?;
        basis.build(&ham.inner, &byte_seeds).map_err(Error::from)?;
        Ok(PyGenericBasis { inner: basis })
    }

    /// Symmetry-reduced subspace.
    ///
    /// Args:
    ///     group: a :class:`SymmetryGroup` describing the symmetry group;
    ///            `n_sites` and `lhss` are read from `group.n_sites` /
    ///            `group.lhss`.
    ///     ham:   `MonomialOperator` used for BFS.
    ///     seeds: list of seed state strings.
    #[classmethod]
    #[pyo3(signature = (group, ham, seeds))]
    fn symmetric(
        _cls: &Bound<'_, PyType>,
        group: &Bound<'_, PyAny>,
        ham: &PyMonomialOperator,
        seeds: Vec<String>,
    ) -> PyResult<Self> {
        let (n_sites, lhss) = group_n_sites_lhss(group)?;
        let byte_seeds = parse_seeds(&seeds, lhss)?;
        let mut basis =
            GenericBasis::new(n_sites, lhss, SpaceKind::Symm, false).map_err(Error::from)?;
        replay_group_into_generic(group, &mut basis)?;
        basis.build(&ham.inner, &byte_seeds).map_err(Error::from)?;
        Ok(PyGenericBasis { inner: basis })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    #[getter]
    fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }

    #[getter]
    fn lhss(&self) -> usize {
        self.inner.lhss()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    fn is_built(&self) -> bool {
        self.inner.is_built()
    }

    // ------------------------------------------------------------------
    // Methods
    // ------------------------------------------------------------------

    /// Return the `i`-th basis state as a string of site occupations.
    fn state_at(&self, i: usize) -> PyResult<String> {
        if i >= self.inner.size() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {i} out of range for basis of size {}",
                self.inner.size()
            )));
        }
        Ok(self.inner.state_at_str(i))
    }

    /// Return the index of `state_str`, or `None` if absent.
    fn index(&self, state_str: &str) -> PyResult<Option<usize>> {
        let bytes = parse_state_str(state_str, self.inner.lhss())?;
        Ok(self.inner.index_of_bytes(&bytes))
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "GenericBasis(n_sites={}, lhss={}, size={}, kind={})",
            self.inner.n_sites(),
            self.inner.lhss(),
            self.inner.size(),
            self.inner.kind(),
        )
    }
}
