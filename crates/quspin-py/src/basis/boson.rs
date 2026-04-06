use crate::basis::{apply_symmetries, parse_seeds, parse_state_str};
use crate::error::Error;
use crate::operator::bond::PyBondOperator;
use crate::operator::boson::PyBosonOperator;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::{BosonBasis, SpaceKind};

/// Python-facing bosonic basis.
///
/// `lhss` is the number of on-site Fock states (≥ 2).
#[pyclass(name = "BosonBasis", module = "quspin._rs")]
pub struct PyBosonBasis {
    pub inner: BosonBasis,
}

fn build_boson_basis(
    basis: &mut BosonBasis,
    ham: &Bound<'_, PyAny>,
    byte_seeds: &[Vec<u8>],
) -> PyResult<()> {
    if let Ok(op) = ham.downcast::<PyBosonOperator>() {
        basis
            .build_boson(&op.borrow().inner, byte_seeds)
            .map_err(Error::from)?;
    } else if let Ok(op) = ham.downcast::<PyBondOperator>() {
        basis
            .build_bond(&op.borrow().inner, byte_seeds)
            .map_err(Error::from)?;
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "ham must be a BosonOperator or BondOperator",
        ));
    }
    Ok(())
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

    /// Particle-number sector subspace.
    ///
    /// Args:
    ///     n_sites: number of lattice sites.
    ///     lhss:    on-site Fock-state count (≥ 2).
    ///     ham:     `BosonOperator` or `BondOperator` used for BFS.
    ///     seeds:   list of seed state strings (one digit per site for LHSS>2).
    #[classmethod]
    fn subspace(
        _cls: &Bound<'_, PyType>,
        n_sites: usize,
        lhss: usize,
        ham: &Bound<'_, PyAny>,
        seeds: Vec<String>,
    ) -> PyResult<Self> {
        let byte_seeds = parse_seeds(&seeds, lhss)?;
        let mut basis = BosonBasis::new(n_sites, lhss, SpaceKind::Sub).map_err(Error::from)?;
        build_boson_basis(&mut basis, ham, &byte_seeds)?;
        Ok(PyBosonBasis { inner: basis })
    }

    /// Symmetry-reduced subspace.
    ///
    /// Args:
    ///     n_sites:     number of lattice sites.
    ///     lhss:        on-site Fock-state count (≥ 2).
    ///     ham:         `BosonOperator` or `BondOperator` used for BFS.
    ///     seeds:       list of seed state strings.
    ///     symmetries:  list of `(perm, (re, im))` lattice symmetry tuples.
    #[classmethod]
    fn symmetric(
        _cls: &Bound<'_, PyType>,
        n_sites: usize,
        lhss: usize,
        ham: &Bound<'_, PyAny>,
        seeds: Vec<String>,
        symmetries: Vec<(Vec<usize>, (f64, f64))>,
    ) -> PyResult<Self> {
        let byte_seeds = parse_seeds(&seeds, lhss)?;
        let mut basis = BosonBasis::new(n_sites, lhss, SpaceKind::Symm).map_err(Error::from)?;
        apply_symmetries(&symmetries, |c, p| basis.add_lattice(c, p))?;
        build_boson_basis(&mut basis, ham, &byte_seeds)?;
        Ok(PyBosonBasis { inner: basis })
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
    fn index(&self, state_str: &str) -> PyResult<Option<usize>> {
        let bytes = parse_state_str(state_str, self.inner.lhss)?;
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
