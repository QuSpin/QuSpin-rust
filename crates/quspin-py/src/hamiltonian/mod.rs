/// Python-facing `PyHardcoreHamiltonian` pyclass.
///
/// Wraps a `HardcoreHamiltonianInner` enum that selects between
/// `HardcoreHamiltonian<u8>` (≤ 255 cindices / site indices) and
/// `HardcoreHamiltonian<u16>` (larger).  The cindex type is chosen at
/// construction time based on the maximum cindex and site index seen in the
/// input.
///
/// ## Python constructor
///
/// ```python
/// H = PyHardcoreHamiltonian([
///     [("xx", [(J, 0, 1), (J, 1, 2)])],  # cindex = 0
///     [("z",  [(h, 0), (h, 1)])],         # cindex = 1
/// ])
/// ```
///
/// - Outer list index → `cindex` value stored in each `OpEntry`.
/// - Each inner list element is a `(op_str, coupling_list)` pair.
/// - `op_str` characters are parsed by `HardcoreOp::from_char`.
/// - Each `coupling_list` element is `(coeff, site_0, site_1, ...)`.
/// - `n_sites` is inferred from `max_site_index + 1`.
pub mod dispatch;

pub use dispatch::HardcoreHamiltonianInner;

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyTuple};
use quspin_core::operator::{HardcoreHamiltonian, HardcoreOp, OpEntry};
use smallvec::SmallVec;

use crate::error::Error;

// ---------------------------------------------------------------------------
// PyHardcoreHamiltonian
// ---------------------------------------------------------------------------

#[pyclass(name = "PyHardcoreHamiltonian")]
pub struct PyHardcoreHamiltonian {
    pub inner: HardcoreHamiltonianInner,
    /// Number of distinct cindex values (= length of the outer terms list).
    pub num_cindices: usize,
}

#[pymethods]
impl PyHardcoreHamiltonian {
    /// Construct from a Python list-of-lists.
    ///
    /// `terms[cindex_idx]` is a list of `(op_str, coupling_list)` pairs.
    /// Each `coupling_list` element is `(coeff, site_0, site_1, ...)`.
    #[new]
    pub fn new(py: Python<'_>, terms: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut raw: Vec<RawEntry> = Vec::new();
        let mut max_cindex: usize = 0;
        let mut max_site: usize = 0;
        let mut n_sites: usize = 0;

        for (cindex, cindex_terms) in terms.iter().enumerate() {
            max_cindex = max_cindex.max(cindex);
            let cindex_list = cindex_terms.downcast::<PyList>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "terms[{cindex}] must be a list of (op_str, coupling_list) pairs"
                ))
            })?;

            for item in cindex_list.iter() {
                let tup = item.downcast::<PyTuple>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "each term must be a (op_str, coupling_list) tuple",
                    )
                })?;
                if tup.len() != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "each term must be a 2-tuple (op_str, coupling_list)",
                    ));
                }

                let op_str: String = tup.get_item(0)?.extract()?;
                let coupling_item = tup.get_item(1)?;
                let coupling_list = coupling_item.downcast::<PyList>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "coupling_list must be a list of (coeff, site, ...) tuples",
                    )
                })?;

                for coupling in coupling_list.iter() {
                    let ctup = coupling.downcast::<PyTuple>().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(
                            "each coupling must be a (coeff, site_0, ...) tuple",
                        )
                    })?;
                    let nops = op_str.chars().count();
                    if ctup.len() != nops + 1 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "op_str '{op_str}' has {nops} operators but coupling has {} \
                             site indices (expected {nops})",
                            ctup.len() - 1,
                        )));
                    }

                    let coeff = extract_complex(py, &ctup.get_item(0)?)?;

                    let mut sites: SmallVec<[u32; 4]> = SmallVec::new();
                    for i in 1..ctup.len() {
                        let site: u32 = ctup.get_item(i)?.extract()?;
                        sites.push(site);
                        let s = site as usize;
                        max_site = max_site.max(s);
                        n_sites = n_sites.max(s + 1);
                    }

                    let mut ops: SmallVec<[(HardcoreOp, u32); 4]> = SmallVec::new();
                    for (ch, &site) in op_str.chars().zip(sites.iter()) {
                        let op = HardcoreOp::from_char(ch).ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "unknown operator character '{ch}'; \
                                 expected one of x, y, z, +, -, n"
                            ))
                        })?;
                        ops.push((op, site));
                    }

                    raw.push(RawEntry { cindex, coeff, ops });
                }
            }
        }

        let num_cindices = max_cindex + 1;
        let needs_u16 = max_cindex > 255 || max_site > 255;
        let inner = if needs_u16 {
            let entries: Vec<OpEntry<u16>> = raw
                .into_iter()
                .map(|r| OpEntry::new(r.cindex as u16, r.coeff, r.ops))
                .collect();
            HardcoreHamiltonianInner::Ham16(HardcoreHamiltonian::new(entries, n_sites))
        } else {
            let entries: Vec<OpEntry<u8>> = raw
                .into_iter()
                .map(|r| OpEntry::new(r.cindex as u8, r.coeff, r.ops))
                .collect();
            HardcoreHamiltonianInner::Ham8(HardcoreHamiltonian::new(entries, n_sites))
        };

        Ok(PyHardcoreHamiltonian {
            inner,
            num_cindices,
        })
    }

    /// Number of sites (inferred from the maximum site index + 1).
    #[getter]
    pub fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }

    /// Number of distinct cindex values (outer list length).
    #[getter]
    pub fn num_cindices(&self) -> usize {
        self.num_cindices
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Intermediate storage for a parsed operator entry before the cindex type
/// (`u8` vs `u16`) is decided.
struct RawEntry {
    cindex: usize,
    coeff: num_complex::Complex<f64>,
    ops: SmallVec<[(HardcoreOp, u32); 4]>,
}

/// Extract a Python scalar as `Complex<f64>`.
fn extract_complex(
    _py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> Result<num_complex::Complex<f64>, Error> {
    if let Ok(c) = obj.extract::<num_complex::Complex<f64>>() {
        return Ok(c);
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(num_complex::Complex::new(f, 0.0));
    }
    Err(Error(quspin_core::error::QuSpinError::ValueError(
        "coefficient must be a Python complex, float, or int".to_string(),
    )))
}
