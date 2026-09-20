use crate::basis::state_int::{py_int_to_state_bytes, state_bytes_to_py_int, states_to_pyarray};
use crate::basis::state_vec::{project_generic, reject_sparse};
use crate::basis::{
    group_n_sites_lhss, parse_seeds, parse_state_str, replay_group_into_generic,
    validate_op_max_site,
};
use crate::error::Error;
use crate::operator::bond::PyBondOperator;
use crate::operator::pauli::PyPauliOperator;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::seed::state_to_display_str;
use quspin_core::basis::{GenericBasis, SpaceKind, SpinBasis};

/// Python-facing spin-½ / spin-S basis.
///
/// `subspace` and `symmetric` constructors accept either a `PauliOperator` or
/// a `BondOperator` as the Hamiltonian used for BFS.
#[pyclass(name = "SpinBasis", module = "quspin._rs")]
pub struct PySpinBasis {
    pub inner: SpinBasis,
}

// Helper: dispatch BFS build to the right operator type.
fn build_spin_basis(
    basis: &mut SpinBasis,
    ham: &Bound<'_, PyAny>,
    n_sites: usize,
    byte_seeds: &[Vec<u8>],
) -> PyResult<()> {
    if let Ok(op) = ham.cast::<PyPauliOperator>() {
        let op = op.borrow();
        validate_op_max_site(op.inner.max_site(), n_sites)?;
        basis.build(&op.inner, byte_seeds).map_err(Error::from)?;
    } else if let Ok(op) = ham.cast::<PyBondOperator>() {
        let op = op.borrow();
        validate_op_max_site(op.inner.max_site(), n_sites)?;
        basis.build(&op.inner, byte_seeds).map_err(Error::from)?;
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "ham must be a PauliOperator (lhss=2) or BondOperator",
        ));
    }
    Ok(())
}

impl PySpinBasis {
    /// The unrestricted Hilbert space with the same `n_sites` / `lhss`, used
    /// as the other end of `project_to` / `project_from`.
    fn full_basis(&self) -> PyResult<GenericBasis> {
        GenericBasis::new(
            self.inner.inner.n_sites(),
            self.inner.inner.lhss(),
            SpaceKind::Full,
            false,
        )
        .map_err(Error::from)
        .map_err(PyErr::from)
    }
}

#[pymethods]
impl PySpinBasis {
    /// Full Hilbert space (no projection, no build step required).
    ///
    /// Args:
    ///     n_sites: number of lattice sites.
    ///     lhss:    local Hilbert-space size (default 2 for spin-½).
    #[classmethod]
    #[pyo3(signature = (n_sites, lhss = 2))]
    fn full(_cls: &Bound<'_, PyType>, n_sites: usize, lhss: usize) -> PyResult<Self> {
        let inner = SpinBasis::new(n_sites, lhss, SpaceKind::Full).map_err(Error::from)?;
        Ok(PySpinBasis { inner })
    }

    /// Particle-number (or energy) sector subspace.
    ///
    /// Args:
    ///     n_sites: number of lattice sites.
    ///     ham:     `PauliOperator` or `BondOperator` used for BFS.
    ///     seeds:   list of seed state strings. For `lhss == 2`: one `'0'`/`'1'`
    ///              char per site. For `lhss > 2`: one decimal digit per site in
    ///              the range `0..lhss`.
    ///     lhss:    local Hilbert-space size (default 2).
    #[classmethod]
    #[pyo3(signature = (n_sites, ham, seeds, lhss = 2))]
    fn subspace(
        _cls: &Bound<'_, PyType>,
        n_sites: usize,
        ham: &Bound<'_, PyAny>,
        seeds: Vec<String>,
        lhss: usize,
    ) -> PyResult<Self> {
        let byte_seeds = parse_seeds(&seeds, n_sites, lhss)?;
        let mut basis = SpinBasis::new(n_sites, lhss, SpaceKind::Sub).map_err(Error::from)?;
        build_spin_basis(&mut basis, ham, n_sites, &byte_seeds)?;
        Ok(PySpinBasis { inner: basis })
    }

    /// Symmetry-reduced subspace.
    ///
    /// Args:
    ///     group: a :class:`SymmetryGroup` describing the symmetry group;
    ///            `n_sites` and `lhss` are read from `group.n_sites` /
    ///            `group.lhss`.
    ///     ham:   `PauliOperator` (LHSS=2) or `BondOperator` used for BFS.
    ///     seeds: list of seed state strings.
    #[classmethod]
    #[pyo3(signature = (group, ham, seeds))]
    fn symmetric(
        _cls: &Bound<'_, PyType>,
        group: &Bound<'_, PyAny>,
        ham: &Bound<'_, PyAny>,
        seeds: Vec<String>,
    ) -> PyResult<Self> {
        let (n_sites, lhss) = group_n_sites_lhss(group)?;
        let byte_seeds = parse_seeds(&seeds, n_sites, lhss)?;
        let mut basis = SpinBasis::new(n_sites, lhss, SpaceKind::Symm).map_err(Error::from)?;
        replay_group_into_generic(group, &mut basis.inner)?;
        build_spin_basis(&mut basis, ham, n_sites, &byte_seeds)?;
        Ok(PySpinBasis { inner: basis })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    #[getter]
    fn n_sites(&self) -> usize {
        self.inner.inner.n_sites()
    }

    #[getter]
    fn lhss(&self) -> usize {
        self.inner.inner.lhss()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.inner.size()
    }

    #[getter]
    #[pyo3(name = "Ns")]
    fn size_alias(&self) -> usize {
        self.inner.inner.size()
    }

    #[getter]
    fn is_built(&self) -> bool {
        self.inner.inner.is_built()
    }

    /// Integer representation of every basis state, in array-index order.
    ///
    /// Matches the "integer repr." column of `print(basis)`.
    #[getter]
    fn states(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let decimals: Vec<String> = (0..self.inner.inner.size())
            .map(|i| self.inner.inner.state_at_decimal_str(i))
            .collect();
        states_to_pyarray(py, &decimals)
    }

    // ------------------------------------------------------------------
    // Methods
    // ------------------------------------------------------------------

    /// Return the `i`-th basis state as a string of site occupations.
    ///
    /// For `lhss == 2` returns a `'0'`/`'1'` string; for `lhss > 2` returns
    /// decimal digit characters (one per site, value in `0..lhss`).
    fn state_at(&self, i: usize) -> PyResult<String> {
        if i >= self.inner.inner.size() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {i} out of range for basis of size {}",
                self.inner.inner.size()
            )));
        }
        Ok(self.inner.inner.state_at_str(i))
    }

    /// Return the integer representation of `state_str`.
    ///
    /// Accepts ket notation (`"|01>"`), a plain per-site string (`"01"`), and
    /// whitespace- or comma-separated occupations (`"0 1"`, `"0,1"`).
    fn state_to_int(&self, py: Python<'_>, state_str: &str) -> PyResult<Py<PyAny>> {
        let bytes = parse_state_str(
            state_str,
            self.inner.inner.n_sites(),
            self.inner.inner.lhss(),
        )?;
        state_bytes_to_py_int(py, &bytes, self.inner.inner.lhss())
    }

    /// Return the Fock-state string for an integer-encoded state.
    #[pyo3(signature = (state_int, bracket_notation = true))]
    fn int_to_state(
        &self,
        state_int: &Bound<'_, PyAny>,
        bracket_notation: bool,
    ) -> PyResult<String> {
        let bytes = py_int_to_state_bytes(
            state_int,
            self.inner.inner.n_sites(),
            self.inner.inner.lhss(),
        )?;
        Ok(state_to_display_str(&bytes, bracket_notation))
    }

    /// Return the index of `state_str` in this basis, or `None` if absent.
    ///
    /// `state_str` must be a string of length `n_sites`. For `lhss == 2`: one
    /// `'0'`/`'1'` character per site. For `lhss > 2`: one decimal digit per
    /// site in the range `0..lhss`.
    #[pyo3(name = "index_str")]
    fn index_str(&self, state_str: &str) -> PyResult<Option<usize>> {
        let bytes = parse_state_str(
            state_str,
            self.inner.inner.n_sites(),
            self.inner.inner.lhss(),
        )?;
        Ok(self.inner.inner.index_of_bytes(&bytes))
    }

    /// Return the index of an integer-encoded basis state, or `None` if absent.
    ///
    /// `state_int` is the raw integer representation (same convention as the
    /// "integer repr." column in `print(basis)`).
    #[pyo3(name = "index")]
    fn index_int_raw(&self, state_int: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let bytes = py_int_to_state_bytes(
            state_int,
            self.inner.inner.n_sites(),
            self.inner.inner.lhss(),
        )?;
        Ok(self.inner.inner.index_of_bytes(&bytes))
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
        project_generic(py, &full_basis, &self.inner.inner, state)
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
        project_generic(py, &self.inner.inner, &full_basis, state)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "SpinBasis(n_sites={}, lhss={}, size={}, kind={})",
            self.inner.inner.n_sites(),
            self.inner.inner.lhss(),
            self.inner.inner.size(),
            self.inner.inner.kind(),
        )
    }
}
