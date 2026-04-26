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
        let locs_str = match &self.locs {
            Some(v) => format!("{:?}", v),
            None => "None".to_string(),
        };
        match self.kind {
            SymElementKind::Lattice => {
                format!("Lattice(perm={:?})", self.perm.as_ref().unwrap())
            }
            SymElementKind::Local => format!(
                "Local(perm_vals={:?}, locs={})",
                self.perm_vals.as_ref().unwrap(),
                locs_str,
            ),
            SymElementKind::Composite => format!(
                "Composite(perm={:?}, perm_vals={:?}, locs={})",
                self.perm.as_ref().unwrap(),
                self.perm_vals.as_ref().unwrap(),
                locs_str,
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

/// Pure-lattice symmetry element (site permutation only).
///
/// `perm[src] = dst` moves the dit at site `src` to site `dst`.
/// `perm` must have length `n_sites` and contain a permutation of
/// `0..n_sites` — validated downstream at *Basis.symmetric(...) time.
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

/// Pure-local symmetry element (per-site value-permutation only).
///
/// `perm_vals` is a permutation of `0..lhss` describing the per-site
/// action; `locs` selects the sites the local op applies to (None →
/// all sites). Validated downstream at *Basis.symmetric(...) time.
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

/// Compute the order of a permutation `p` over `0..N` as LCM of cycle lengths.
fn perm_order(p: &[usize]) -> usize {
    let n = p.len();
    let mut visited = vec![false; n];
    let mut lcm: usize = 1;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut len = 0usize;
        let mut i = start;
        while !visited[i] {
            visited[i] = true;
            i = p[i];
            len += 1;
        }
        if len > 1 {
            lcm = lcm_u(lcm, len);
        }
    }
    lcm
}

fn lcm_u(a: usize, b: usize) -> usize {
    a / gcd_u(a, b) * b
}

fn gcd_u(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Compute the order of a SymElement: the smallest positive integer N
/// such that g^N is the identity. Computed as the LCM of the
/// permutation component's cycle order and the perm_vals component's
/// cycle order. Returns 1 for the identity element.
///
/// `n_sites` and `lhss` are accepted for API uniformity / future use
/// but are currently unused — element shape encodes the cycle domains.
#[pyfunction]
#[pyo3(signature = (elem, n_sites, lhss))]
pub fn _order(elem: &PySymElement, n_sites: usize, lhss: usize) -> usize {
    let _ = (n_sites, lhss);
    let perm_o = elem.perm.as_deref().map(perm_order).unwrap_or(1);
    let pv_o = elem
        .perm_vals
        .as_deref()
        .map(|v| {
            let v_us: Vec<usize> = v.iter().map(|&x| x as usize).collect();
            perm_order(&v_us)
        })
        .unwrap_or(1);
    lcm_u(perm_o, pv_o)
}

/// Composite symmetry element (lattice + local applied atomically).
///
/// Both `perm` (site permutation) and `perm_vals` (per-site value
/// permutation) act as a single group element with one character.
/// Use this for symmetries like PZ where neither component alone is
/// a symmetry of the Hamiltonian.
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
