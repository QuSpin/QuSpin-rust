use crate::basis::state_int::{py_int_to_state_bytes, state_bytes_to_py_int, states_to_pyarray};
use crate::basis::state_vec::{project_generic, reject_sparse};
use crate::basis::{
    group_n_sites_lhss, parse_seeds, parse_state_str, replay_group_into_generic,
    validate_op_max_site,
};
use crate::error::Error;
use crate::operator::monomial::PyMonomialOperator;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::seed::state_to_display_str;
use quspin_core::basis::{GenericBasis, SpaceKind};
use std::sync::Arc;

/// Python-facing generic basis for any on-site Hilbert-space size.
///
/// Supports both lattice (site-permutation) and local (dit-permutation)
/// symmetries.  Paired with `MonomialOperator` for Hamiltonian construction.
#[pyclass(name = "GenericBasis", module = "quspin._rs")]
pub struct PyGenericBasis {
    pub inner: Arc<GenericBasis>,
}

impl PyGenericBasis {
    /// The unrestricted Hilbert space with the same `n_sites` / `lhss`, used
    /// as the other end of `project_to` / `project_from`.
    fn full_basis(&self) -> PyResult<GenericBasis> {
        GenericBasis::new(
            self.inner.n_sites(),
            self.inner.lhss(),
            SpaceKind::Full,
            false,
        )
        .map_err(Error::from)
        .map_err(PyErr::from)
    }
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
        Ok(PyGenericBasis {
            inner: Arc::new(inner),
        })
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
        validate_op_max_site(ham.inner.max_site(), n_sites)?;
        let byte_seeds = parse_seeds(&seeds, n_sites, lhss)?;
        let mut basis =
            GenericBasis::new(n_sites, lhss, SpaceKind::Sub, false).map_err(Error::from)?;
        basis.build(&ham.inner, &byte_seeds).map_err(Error::from)?;
        Ok(PyGenericBasis {
            inner: Arc::new(basis),
        })
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
        validate_op_max_site(ham.inner.max_site(), n_sites)?;
        let byte_seeds = parse_seeds(&seeds, n_sites, lhss)?;
        let mut basis =
            GenericBasis::new(n_sites, lhss, SpaceKind::Symm, false).map_err(Error::from)?;
        replay_group_into_generic(group, &mut basis)?;
        basis.build(&ham.inner, &byte_seeds).map_err(Error::from)?;
        Ok(PyGenericBasis {
            inner: Arc::new(basis),
        })
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
    #[pyo3(name = "Ns")]
    fn n_size_alias(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    fn is_built(&self) -> bool {
        self.inner.is_built()
    }

    /// Integer representation of every basis state, in array-index order.
    ///
    /// Matches the "integer repr." column of `print(basis)`.
    #[getter]
    fn states(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let decimals: Vec<String> = (0..self.inner.size())
            .map(|i| self.inner.state_at_decimal_str(i))
            .collect();
        states_to_pyarray(py, &decimals)
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

    /// Return the integer representation of `state_str`.
    ///
    /// Accepts ket notation (`"|01>"`), a plain per-site string (`"01"`), and
    /// whitespace- or comma-separated occupations (`"0 1"`, `"0,1"`) — the
    /// last form is the only way to write occupations of 10 or more.
    fn state_to_int(&self, py: Python<'_>, state_str: &str) -> PyResult<Py<PyAny>> {
        let bytes = parse_state_str(state_str, self.inner.n_sites(), self.inner.lhss())?;
        state_bytes_to_py_int(py, &bytes, self.inner.lhss())
    }

    /// Return the Fock-state string for an integer-encoded state.
    #[pyo3(signature = (state_int, bracket_notation = true))]
    fn int_to_state(
        &self,
        state_int: &Bound<'_, PyAny>,
        bracket_notation: bool,
    ) -> PyResult<String> {
        let bytes = py_int_to_state_bytes(state_int, self.inner.n_sites(), self.inner.lhss())?;
        Ok(state_to_display_str(&bytes, bracket_notation))
    }

    /// Return the index of `state_str`, or `None` if absent.
    #[pyo3(name = "index_str")]
    fn index_str(&self, state_str: &str) -> PyResult<Option<usize>> {
        let bytes = parse_state_str(state_str, self.inner.n_sites(), self.inner.lhss())?;
        Ok(self.inner.index_of_bytes(&bytes))
    }

    /// Return the index of an integer-encoded basis state, or `None` if absent.
    ///
    /// `state_int` is the raw integer representation (same convention as the
    /// "integer repr." column in `print(basis)`).
    #[pyo3(name = "index")]
    fn index_int_raw(&self, state_int: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let bytes = py_int_to_state_bytes(state_int, self.inner.n_sites(), self.inner.lhss())?;
        Ok(self.inner.index_of_bytes(&bytes))
    }

    /// Project a full-basis vector into this basis.
    ///
    /// `sparse=True` is not implemented yet and raises `NotImplementedError`
    /// rather than silently returning a dense array.
    #[pyo3(signature = (state, sparse = false))]
    fn project_to(
        &self,
        py: Python<'_>,
        state: &Bound<'_, PyAny>,
        sparse: bool,
    ) -> PyResult<Py<PyAny>> {
        reject_sparse(sparse)?;
        let full_basis = self.full_basis()?;
        project_generic(py, &full_basis, &self.inner, state)
    }

    /// Expand a vector in this basis to the full Hilbert-space basis.
    ///
    /// `sparse=True` is not implemented yet and raises `NotImplementedError`
    /// rather than silently returning a dense array.
    #[pyo3(signature = (state, sparse = false))]
    fn project_from(
        &self,
        py: Python<'_>,
        state: &Bound<'_, PyAny>,
        sparse: bool,
    ) -> PyResult<Py<PyAny>> {
        reject_sparse(sparse)?;
        let full_basis = self.full_basis()?;
        project_generic(py, &self.inner, &full_basis, state)
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
