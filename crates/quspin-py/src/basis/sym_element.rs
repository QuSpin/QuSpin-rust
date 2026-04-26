//! PyO3 element handle for SymmetryGroup. LHSS-agnostic — holds the
//! untyped triple (perm, perm_vals, locs); LHSS-specific construction
//! happens at *Basis.symmetric(...) time inside the dispatch enum.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum SymElementKind {
    Lattice,
    Local,
    Composite,
}

#[pyclass(name = "SymElement", module = "quspin_rs._rs", frozen)]
#[derive(Clone)]
pub struct PySymElement {
    pub(crate) kind: SymElementKind,
    pub(crate) perm: Option<Vec<usize>>,
    pub(crate) perm_vals: Option<Vec<u8>>,
    pub(crate) locs: Option<Vec<usize>>,
}

#[pymethods]
impl PySymElement {
    fn __repr__(&self) -> String {
        match self.kind {
            SymElementKind::Lattice => {
                format!("Lattice(perm={:?})", self.perm.as_ref().unwrap())
            }
            SymElementKind::Local => format!(
                "Local(perm_vals={:?}, locs={:?})",
                self.perm_vals.as_ref().unwrap(),
                self.locs,
            ),
            SymElementKind::Composite => format!(
                "Composite(perm={:?}, perm_vals={:?}, locs={:?})",
                self.perm.as_ref().unwrap(),
                self.perm_vals.as_ref().unwrap(),
                self.locs,
            ),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.perm == other.perm
            && self.perm_vals == other.perm_vals
            && self.locs == other.locs
    }

    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.kind.hash(&mut h);
        self.perm.hash(&mut h);
        self.perm_vals.hash(&mut h);
        self.locs.hash(&mut h);
        h.finish()
    }
}

#[pyfunction]
#[pyo3(name = "Lattice", signature = (perm))]
pub fn lattice(perm: Vec<i64>) -> PyResult<PySymElement> {
    if perm.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err(
            "Lattice perm contains negative integers — old-QuSpin used \
             negative ints to encode spin-flip+permutation; use \
             Composite(perm, perm_vals=[1, 0]) instead.",
        ));
    }
    let perm: Vec<usize> = perm.into_iter().map(|v| v as usize).collect();
    Ok(PySymElement {
        kind: SymElementKind::Lattice,
        perm: Some(perm),
        perm_vals: None,
        locs: None,
    })
}

#[pyfunction]
#[pyo3(name = "Local", signature = (perm_vals, locs = None))]
pub fn local(perm_vals: Vec<u64>, locs: Option<Vec<usize>>) -> PyResult<PySymElement> {
    let perm_vals: Vec<u8> = perm_vals
        .into_iter()
        .map(|v| {
            u8::try_from(v).map_err(|_| PyValueError::new_err("perm_vals values must fit in u8"))
        })
        .collect::<PyResult<_>>()?;
    Ok(PySymElement {
        kind: SymElementKind::Local,
        perm: None,
        perm_vals: Some(perm_vals),
        locs,
    })
}

#[pyfunction]
#[pyo3(name = "Composite", signature = (perm, perm_vals, locs = None))]
pub fn composite(
    perm: Vec<i64>,
    perm_vals: Vec<u64>,
    locs: Option<Vec<usize>>,
) -> PyResult<PySymElement> {
    if perm.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err(
            "Composite perm contains negative integers — perm indexes sites; \
             use perm_vals to encode local-op action.",
        ));
    }
    let perm: Vec<usize> = perm.into_iter().map(|v| v as usize).collect();
    let perm_vals: Vec<u8> = perm_vals
        .into_iter()
        .map(|v| {
            u8::try_from(v).map_err(|_| PyValueError::new_err("perm_vals values must fit in u8"))
        })
        .collect::<PyResult<_>>()?;
    Ok(PySymElement {
        kind: SymElementKind::Composite,
        perm: Some(perm),
        perm_vals: Some(perm_vals),
        locs,
    })
}
