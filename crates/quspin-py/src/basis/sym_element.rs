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

/// Compute the cyclic order N of a SymElement: the smallest positive
/// integer such that g^N is the identity. Returned as the LCM of the
/// permutation component's cycle order and the perm_vals component's
/// cycle order; returns 1 for the identity element.
///
/// Trusts that `perm` and `perm_vals` are well-formed permutations
/// (each a bijection of its domain). Higher-level validation
/// (`perm.len() == n_sites`, `perm_vals.len() == lhss`, bijection
/// checks) lives in `SymBasis::add_symmetry` (perm) and the per-family
/// inner-enum `add_local` / `add_composite` methods (perm_vals / locs).
///
/// `_n_sites` and `_lhss` are accepted for API uniformity with other
/// `SymElement` helpers (e.g. `_compose`, `_validate_group`) but
/// currently unused — cycle structure is encoded entirely in the
/// element itself.
#[pyfunction]
#[pyo3(signature = (elem, n_sites, lhss))]
#[allow(unused_variables)]
pub fn _order(elem: &PySymElement, n_sites: usize, lhss: usize) -> usize {
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

/// Compose two permutations: `(a ∘ b)[src] = a[b[src]]` — meaning `b`
/// is applied first, then `a`. Both inputs must have the same length.
fn compose_perms(a: &[usize], b: &[usize]) -> Vec<usize> {
    debug_assert_eq!(a.len(), b.len(), "perm length mismatch");
    (0..b.len()).map(|s| a[b[s]]).collect()
}

/// Compose two `SymElement`s. Mirrors `SymElement::compose` for the
/// untyped triple — applied as `(a ∘ b)(state) = a(b(state))`.
///
/// - Pure-lattice ∘ pure-lattice → pure-lattice (perms must match length).
/// - Pure-local ∘ pure-local → pure-local (perm_vals must match length).
/// - Any mix promotes to composite.
/// - If either side is identity-shaped (the other side present), the
///   output adopts the present side's perm / perm_vals / locs.
/// - locs union: when both sides supply explicit locs, take their
///   set union (sorted, deduplicated); when one side is `None`, the
///   other's `locs` carries through unchanged.
///
/// Returns an error when the perms or perm_vals have mismatched
/// lengths, or when the result would be the identity (caller should
/// drop the element rather than store it).
#[pyfunction]
#[pyo3(name = "_compose")]
pub fn _compose(a: &PySymElement, b: &PySymElement) -> PyResult<PySymElement> {
    let perm = match (&a.perm, &b.perm) {
        (None, None) => None,
        (Some(p), None) | (None, Some(p)) => Some(p.clone()),
        (Some(x), Some(y)) => {
            if x.len() != y.len() {
                return Err(PyValueError::new_err(format!(
                    "_compose: perm lengths differ ({} vs {})",
                    x.len(),
                    y.len()
                )));
            }
            Some(compose_perms(x, y))
        }
    };
    let perm_vals = match (&a.perm_vals, &b.perm_vals) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v.clone()),
        (Some(x), Some(y)) => {
            if x.len() != y.len() {
                return Err(PyValueError::new_err(format!(
                    "_compose: perm_vals lengths differ ({} vs {})",
                    x.len(),
                    y.len()
                )));
            }
            let xu: Vec<usize> = x.iter().map(|&v| v as usize).collect();
            let yu: Vec<usize> = y.iter().map(|&v| v as usize).collect();
            Some(
                compose_perms(&xu, &yu)
                    .into_iter()
                    .map(|v| v as u8)
                    .collect(),
            )
        }
    };
    let locs = match (&a.locs, &b.locs) {
        (None, x) | (x, None) => x.clone(),
        (Some(x), Some(y)) => {
            let mut s: std::collections::BTreeSet<usize> = x.iter().copied().collect();
            s.extend(y.iter().copied());
            Some(s.into_iter().collect())
        }
    };
    let kind = match (perm.is_some(), perm_vals.is_some()) {
        (true, false) => SymElementKind::Lattice,
        (false, true) => SymElementKind::Local,
        (true, true) => SymElementKind::Composite,
        (false, false) => {
            return Err(PyValueError::new_err(
                "_compose produced identity (both components empty)",
            ));
        }
    };
    Ok(PySymElement {
        kind,
        perm,
        perm_vals,
        locs,
    })
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
