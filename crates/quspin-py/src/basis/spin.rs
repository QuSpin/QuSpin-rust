use crate::basis::{apply_symmetries, parse_seeds, parse_state_str};
use crate::error::Error;
use crate::operator::bond::PyBondOperator;
use crate::operator::pauli::PyPauliOperator;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::{SpaceKind, SpinBasis};

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
    byte_seeds: &[Vec<u8>],
) -> PyResult<()> {
    if let Ok(op) = ham.downcast::<PyPauliOperator>() {
        basis
            .build_hardcore(&op.borrow().inner, byte_seeds)
            .map_err(Error::from)?;
    } else if let Ok(op) = ham.downcast::<PyBondOperator>() {
        basis
            .build_bond(&op.borrow().inner, byte_seeds)
            .map_err(Error::from)?;
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "ham must be a PauliOperator (lhss=2) or BondOperator",
        ));
    }
    Ok(())
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
    ///     seeds:   list of seed state strings (`'0'`/`'1'` chars per site).
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
        let byte_seeds = parse_seeds(&seeds, lhss)?;
        let mut basis = SpinBasis::new(n_sites, lhss, SpaceKind::Sub).map_err(Error::from)?;
        build_spin_basis(&mut basis, ham, &byte_seeds)?;
        Ok(PySpinBasis { inner: basis })
    }

    /// Symmetry-reduced subspace.
    ///
    /// Args:
    ///     n_sites:     number of lattice sites.
    ///     ham:         `PauliOperator` (LHSS=2) or `BondOperator` used for BFS.
    ///     seeds:       list of seed state strings.
    ///     symmetries:  list of `(perm, (re, im))` tuples, where `perm` is a
    ///                  site permutation (list of ints) and `(re, im)` is the
    ///                  group character (complex number as a float pair).
    ///     lhss:        local Hilbert-space size (default 2).
    #[classmethod]
    #[pyo3(signature = (n_sites, ham, seeds, symmetries, lhss = 2))]
    fn symmetric(
        _cls: &Bound<'_, PyType>,
        n_sites: usize,
        ham: &Bound<'_, PyAny>,
        seeds: Vec<String>,
        symmetries: Vec<(Vec<usize>, (f64, f64))>,
        lhss: usize,
    ) -> PyResult<Self> {
        let byte_seeds = parse_seeds(&seeds, lhss)?;
        let mut basis = SpinBasis::new(n_sites, lhss, SpaceKind::Symm).map_err(Error::from)?;
        apply_symmetries(&symmetries, |c, p| basis.add_lattice(c, p))?;
        build_spin_basis(&mut basis, ham, &byte_seeds)?;
        Ok(PySpinBasis { inner: basis })
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
        let bytes = parse_state_str(state_str, self.inner.lhss)?;
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
