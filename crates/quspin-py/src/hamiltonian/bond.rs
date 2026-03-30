/// Python-facing `PyBondTerm` and `PyBondHamiltonian` pyclasses.
///
/// `PyBondTerm` is a thin wrapper around a parsed `(matrix, bonds)` pair.
/// Shallow validation (dtype, 2-D shape, int pairs) happens at construction;
/// semantic validation (perfect-square dim, lhss range, site bounds) is
/// delegated to `BondOperator::new` in quspin-core.
///
/// `PyBondHamiltonian` accepts a `list[PyBondTerm]`, assigns cindices by
/// position, and forwards to `BondOperator::new`.
use ndarray::Array2;
use num_complex::Complex;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyTuple};
use quspin_core::operator::bond::dispatch::BondOperatorInner;
use quspin_core::operator::bond::{BondOperator, BondTerm};

// ---------------------------------------------------------------------------
// PyBondTerm
// ---------------------------------------------------------------------------

#[pyclass(name = "PyBondTerm")]
pub struct PyBondTerm {
    pub matrix: Array2<Complex<f64>>,
    pub bonds: Vec<(u32, u32)>,
}

#[pymethods]
impl PyBondTerm {
    /// Construct a BondTerm from a dense matrix and a list of site-pair bonds.
    ///
    /// Args:
    ///     matrix (NDArray): 2-D array with ``dtype=complex128`` and shape
    ///         ``(lhss², lhss²)``.  Semantic validation (perfect-square
    ///         dimension, lhss in 2..=255) is deferred to
    ///         ``PyBondHamiltonian``.
    ///     bonds (list[tuple[int, int]]): Site pairs ``(si, sj)`` to apply
    ///         the matrix to.
    ///
    /// Raises:
    ///     ValueError: If ``matrix`` is not a 2-D ``complex128`` NumPy array
    ///         or ``bonds`` is not a list of ``(int, int)`` pairs.
    #[new]
    pub fn new(
        _py: Python<'_>,
        matrix: &Bound<'_, PyAny>,
        bonds: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        // --- matrix: must be a 2-D complex128 ndarray ---
        let matrix_arr: PyReadonlyArray2<Complex<f64>> = matrix
            .downcast::<PyArray2<Complex<f64>>>()
            .map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "matrix must be a 2-D numpy array with dtype=complex128",
                )
            })?
            .readonly();
        let owned = matrix_arr.as_array().to_owned();

        // --- bonds: list of (int, int) pairs ---
        let mut parsed_bonds: Vec<(u32, u32)> = Vec::with_capacity(bonds.len());
        for (bi, item) in bonds.iter().enumerate() {
            let tup = item.downcast::<PyTuple>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "bonds[{bi}] must be a (int, int) tuple"
                ))
            })?;
            if tup.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "bonds[{bi}] must be a 2-tuple (si, sj), got length {}",
                    tup.len()
                )));
            }
            let si: u32 = tup.get_item(0)?.extract()?;
            let sj: u32 = tup.get_item(1)?.extract()?;
            parsed_bonds.push((si, sj));
        }

        Ok(PyBondTerm {
            matrix: owned,
            bonds: parsed_bonds,
        })
    }

    /// Shape of the interaction matrix as ``(rows, cols)``.
    #[getter]
    pub fn matrix_shape(&self) -> (usize, usize) {
        let s = self.matrix.shape();
        (s[0], s[1])
    }

    /// Site-pair bonds applied by this term.
    #[getter]
    pub fn bonds(&self) -> Vec<(u32, u32)> {
        self.bonds.clone()
    }

    pub fn __repr__(&self) -> String {
        let s = self.matrix.shape();
        format!(
            "PyBondTerm(matrix_shape=({}, {}), n_bonds={})",
            s[0],
            s[1],
            self.bonds.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyBondHamiltonian
// ---------------------------------------------------------------------------

#[pyclass(name = "PyBondHamiltonian")]
pub struct PyBondHamiltonian {
    pub inner: BondOperatorInner,
}

#[pymethods]
impl PyBondHamiltonian {
    /// Construct a BondOperator from a list of `PyBondTerm` objects.
    ///
    /// Each term is assigned a ``cindex`` equal to its position in the list.
    /// ``n_sites`` is inferred from the maximum site index across all bonds,
    /// plus one.  All semantic validation (lhss, matrix shape, site bounds) is
    /// performed by `BondOperator::new` in quspin-core.
    ///
    /// Args:
    ///     terms (list[PyBondTerm]): One entry per ``cindex``.
    ///
    /// Raises:
    ///     ValueError: If ``terms`` is empty, matrices have inconsistent
    ///         shapes, ``lhss`` is out of range, or any site index is invalid.
    #[new]
    pub fn new(_py: Python<'_>, terms: &Bound<'_, PyList>) -> PyResult<Self> {
        if terms.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "terms must not be empty",
            ));
        }

        type RawTerm = (usize, Array2<Complex<f64>>, Vec<(u32, u32)>);
        let mut raw: Vec<RawTerm> = Vec::with_capacity(terms.len());
        let mut max_site = 0usize;

        for (cindex, item) in terms.iter().enumerate() {
            let term = item.downcast::<PyBondTerm>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "terms[{cindex}] must be a PyBondTerm"
                ))
            })?;
            let t = term.borrow();
            for &(si, sj) in &t.bonds {
                max_site = max_site.max(si as usize).max(sj as usize);
            }
            raw.push((cindex, t.matrix.clone(), t.bonds.clone()));
        }

        let max_cindex = raw.len() - 1;
        let needs_u16 = max_cindex > 255 || max_site > 255;

        let inner = if needs_u16 {
            let bond_terms: Vec<BondTerm<u16>> = raw
                .into_iter()
                .map(|(cindex, matrix, bonds)| BondTerm {
                    cindex: cindex as u16,
                    matrix,
                    bonds,
                })
                .collect();
            let ham = BondOperator::new(bond_terms)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            BondOperatorInner::Ham16(ham)
        } else {
            let bond_terms: Vec<BondTerm<u8>> = raw
                .into_iter()
                .map(|(cindex, matrix, bonds)| BondTerm {
                    cindex: cindex as u8,
                    matrix,
                    bonds,
                })
                .collect();
            let ham = BondOperator::new(bond_terms)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            BondOperatorInner::Ham8(ham)
        };

        Ok(PyBondHamiltonian { inner })
    }

    /// Maximum site index across all bonds.
    #[getter]
    pub fn max_site(&self) -> usize {
        self.inner.max_site()
    }

    /// Number of distinct coefficient indices (length of the terms list).
    #[getter]
    pub fn num_cindices(&self) -> usize {
        self.inner.num_cindices()
    }

    /// Local Hilbert-space size, inferred from ``sqrt(matrix.shape[0])``.
    #[getter]
    pub fn lhss(&self) -> usize {
        self.inner.lhss()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyBondHamiltonian(max_site={}, lhss={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.lhss(),
            self.inner.num_cindices(),
        )
    }
}
