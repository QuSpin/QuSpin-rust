use crate::basis::{apply_symmetries, parse_seeds, parse_state_str};
use crate::error::Error;
use crate::operator::monomial::PyMonomialOperator;
use num_complex::Complex;
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

/// Parse a `local_symmetries` entry.  Each entry is either a 2-tuple
/// `(perm, (re, im))` (mask = all sites) or a 3-tuple
/// `(perm, (re, im), mask)`.
///
/// `perm` is a list/array of integer values in `0..lhss`.
fn apply_local_symmetries(
    py: Python<'_>,
    basis: &mut GenericBasis,
    local_symmetries: &[PyObject],
) -> PyResult<()> {
    let n_sites = basis.inner.n_sites();
    let lhss = basis.inner.lhss();

    for (i, sym_obj) in local_symmetries.iter().enumerate() {
        let sym = sym_obj.bind(py);
        let tuple = sym.downcast::<pyo3::types::PyTuple>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "local_symmetries[{i}]: expected a 2- or 3-tuple"
            ))
        })?;

        if tuple.len() < 2 || tuple.len() > 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "local_symmetries[{i}]: expected a 2- or 3-tuple, got {}-tuple",
                tuple.len()
            )));
        }

        // First element: perm as list of ints (0..lhss).
        let perm_vals: Vec<u8> = tuple
            .get_item(0)?
            .extract::<Vec<u64>>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "local_symmetries[{i}]: perm must be a list of non-negative integers"
                ))
            })?
            .into_iter()
            .map(|v| v as u8)
            .collect();

        if perm_vals.len() != lhss {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "local_symmetries[{i}]: perm length {} != lhss {}",
                perm_vals.len(),
                lhss
            )));
        }

        // Second element: complex character as (re, im) float pair.
        let (re, im): (f64, f64) = tuple.get_item(1)?.extract().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "local_symmetries[{i}]: expected (re, im) float pair for character"
            ))
        })?;
        let grp_char = Complex::new(re, im);

        // Optional third element: mask of site indices.  Default: all sites.
        let locs: Vec<usize> = if tuple.len() == 3 {
            tuple.get_item(2)?.extract::<Vec<usize>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "local_symmetries[{i}]: mask must be a list of site indices"
                ))
            })?
        } else {
            (0..n_sites).collect()
        };

        basis
            .add_local(grp_char, perm_vals, locs)
            .map_err(Error::from)?;
    }
    Ok(())
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
        let inner = GenericBasis::new(n_sites, lhss, SpaceKind::Full).map_err(Error::from)?;
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
        let mut basis = GenericBasis::new(n_sites, lhss, SpaceKind::Sub).map_err(Error::from)?;
        basis
            .build_monomial(&ham.inner, &byte_seeds)
            .map_err(Error::from)?;
        Ok(PyGenericBasis { inner: basis })
    }

    /// Symmetry-reduced subspace.
    ///
    /// Args:
    ///     n_sites:           number of lattice sites.
    ///     lhss:              on-site state count (≥ 2).
    ///     ham:               `MonomialOperator` used for BFS.
    ///     seeds:             list of seed state strings.
    ///     symmetries:        list of ``(perm, (re, im))`` lattice symmetry tuples.
    ///     local_symmetries:  list of 2- or 3-tuples:
    ///         - ``(perm, (re, im))``         — applies to all sites
    ///         - ``(perm, (re, im), mask)``   — applies to sites in ``mask``
    #[classmethod]
    #[pyo3(signature = (n_sites, lhss, ham, seeds, symmetries, local_symmetries = vec![]))]
    fn symmetric(
        _cls: &Bound<'_, PyType>,
        n_sites: usize,
        lhss: usize,
        ham: &PyMonomialOperator,
        seeds: Vec<String>,
        symmetries: Vec<(Vec<usize>, (f64, f64))>,
        local_symmetries: Vec<PyObject>,
    ) -> PyResult<Self> {
        let py = _cls.py();
        let byte_seeds = parse_seeds(&seeds, lhss)?;
        let mut basis = GenericBasis::new(n_sites, lhss, SpaceKind::Symm).map_err(Error::from)?;
        apply_symmetries(&symmetries, |c, p| basis.add_lattice(c, p))?;
        apply_local_symmetries(py, &mut basis, &local_symmetries)?;
        basis
            .build_monomial(&ham.inner, &byte_seeds)
            .map_err(Error::from)?;
        Ok(PyGenericBasis { inner: basis })
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
        let bytes = parse_state_str(state_str, self.inner.inner.lhss())?;
        Ok(self.inner.inner.index_of_bytes(&bytes))
    }

    fn __repr__(&self) -> String {
        format!(
            "GenericBasis(n_sites={}, lhss={}, size={}, kind={})",
            self.inner.inner.n_sites(),
            self.inner.inner.lhss(),
            self.inner.inner.size(),
            self.inner.inner.kind(),
        )
    }
}
