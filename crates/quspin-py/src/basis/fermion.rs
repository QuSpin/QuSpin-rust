use crate::error::Error;
use crate::operator::bond::PyBondOperator;
use crate::operator::fermion::PyFermionOperator;
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::PyType;
use quspin_core::basis::seed::seed_from_str;
use quspin_core::basis::{FermionBasis, SpaceKind};

/// Python-facing fermionic basis.
///
/// Fermions are always LHSS=2 (one bit per orbital).
#[pyclass(name = "FermionBasis", module = "quspin._rs")]
pub struct PyFermionBasis {
    pub inner: FermionBasis,
}

fn parse_seeds(seeds: &[String]) -> PyResult<Vec<Vec<u8>>> {
    seeds
        .iter()
        .map(|s| seed_from_str(s).map_err(Error::from).map_err(PyErr::from))
        .collect()
}

fn apply_symmetries(
    basis: &mut FermionBasis,
    symmetries: &[(Vec<usize>, (f64, f64))],
) -> PyResult<()> {
    for (perm, (re, im)) in symmetries {
        basis
            .add_lattice(Complex::new(*re, *im), perm.clone())
            .map_err(Error::from)?;
    }
    Ok(())
}

fn build_fermion_basis(
    basis: &mut FermionBasis,
    ham: &Bound<'_, PyAny>,
    byte_seeds: &[Vec<u8>],
) -> PyResult<()> {
    if let Ok(op) = ham.downcast::<PyFermionOperator>() {
        basis
            .build_fermion(&op.borrow().inner, byte_seeds)
            .map_err(Error::from)?;
    } else if let Ok(op) = ham.downcast::<PyBondOperator>() {
        basis
            .build_bond(&op.borrow().inner, byte_seeds)
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
        let byte_seeds = parse_seeds(&seeds)?;
        let mut basis = FermionBasis::new(n_sites, SpaceKind::Sub).map_err(Error::from)?;
        build_fermion_basis(&mut basis, ham, &byte_seeds)?;
        Ok(PyFermionBasis { inner: basis })
    }

    /// Symmetry-reduced subspace.
    ///
    /// Args:
    ///     n_sites:     number of orbitals / lattice sites.
    ///     ham:         `FermionOperator` or `BondOperator` used for BFS.
    ///     seeds:       list of seed state strings.
    ///     symmetries:  list of `(perm, (re, im))` lattice symmetry tuples.
    #[classmethod]
    fn symmetric(
        _cls: &Bound<'_, PyType>,
        n_sites: usize,
        ham: &Bound<'_, PyAny>,
        seeds: Vec<String>,
        symmetries: Vec<(Vec<usize>, (f64, f64))>,
    ) -> PyResult<Self> {
        let byte_seeds = parse_seeds(&seeds)?;
        let mut basis = FermionBasis::new(n_sites, SpaceKind::Symm).map_err(Error::from)?;
        apply_symmetries(&mut basis, &symmetries)?;
        build_fermion_basis(&mut basis, ham, &byte_seeds)?;
        Ok(PyFermionBasis { inner: basis })
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
