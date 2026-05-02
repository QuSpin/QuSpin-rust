use crate::basis::{group_n_sites_lhss, parse_seeds, parse_state_str, replay_group_into_bit};
use crate::error::Error;
use crate::operator::bond::PyBondOperator;
use crate::operator::fermion::PyFermionOperator;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::{FermionBasis, SpaceKind};

/// Python-facing fermionic basis.
///
/// Fermions are always LHSS=2 (one bit per orbital).
#[pyclass(name = "FermionBasis", module = "quspin._rs")]
pub struct PyFermionBasis {
    pub inner: FermionBasis,
}

fn build_fermion_basis(
    basis: &mut FermionBasis,
    ham: &Bound<'_, PyAny>,
    byte_seeds: &[Vec<u8>],
) -> PyResult<()> {
    if let Ok(op) = ham.cast::<PyFermionOperator>() {
        basis
            .build(&op.borrow().inner, byte_seeds)
            .map_err(Error::from)?;
    } else if let Ok(op) = ham.cast::<PyBondOperator>() {
        basis
            .build(&op.borrow().inner, byte_seeds)
            .map_err(Error::from)?;
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "ham must be a FermionOperator or BondOperator",
        ));
    }
    Ok(())
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

    /// Particle-number sector subspace.
    ///
    /// Args:
    ///     n_sites: number of orbitals / lattice sites.
    ///     ham:     `FermionOperator` or `BondOperator` used for BFS.
    ///     seeds:   list of seed state strings (`'0'`/`'1'` chars per site).
    #[classmethod]
    fn subspace(
        _cls: &Bound<'_, PyType>,
        n_sites: usize,
        ham: &Bound<'_, PyAny>,
        seeds: Vec<String>,
    ) -> PyResult<Self> {
        let byte_seeds = parse_seeds(&seeds, 2)?;
        let mut basis = FermionBasis::new(n_sites, SpaceKind::Sub).map_err(Error::from)?;
        build_fermion_basis(&mut basis, ham, &byte_seeds)?;
        Ok(PyFermionBasis { inner: basis })
    }

    /// Symmetry-reduced subspace.
    ///
    /// Args:
    ///     group: a :class:`SymmetryGroup` describing the symmetry group.
    ///            Must have `lhss == 2` (fermions are always two-state).
    ///            `n_sites` is read from `group.n_sites`.
    ///     ham:   `FermionOperator` or `BondOperator` used for BFS.
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
        if lhss != 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "FermionBasis requires SymmetryGroup with lhss=2, got lhss={lhss}"
            )));
        }
        let byte_seeds = parse_seeds(&seeds, 2)?;
        let mut basis = FermionBasis::new(n_sites, SpaceKind::Symm).map_err(Error::from)?;
        replay_group_into_bit(group, &mut basis.inner)?;
        build_fermion_basis(&mut basis, ham, &byte_seeds)?;
        Ok(PyFermionBasis { inner: basis })
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
        let bytes = parse_state_str(state_str, 2)?;
        Ok(self.inner.inner.index_of_bytes(&bytes))
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "FermionBasis(n_sites={}, size={}, kind={})",
            self.inner.inner.n_sites(),
            self.inner.inner.size(),
            self.inner.inner.kind(),
        )
    }
}
