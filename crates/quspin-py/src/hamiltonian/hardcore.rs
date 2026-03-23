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
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use quspin_core::hamiltonian::hardcore::dispatch::HardcoreHamiltonianInner;
use quspin_core::hamiltonian::hardcore::{HardcoreHamiltonian, HardcoreOp, OpEntry};

use super::parse::parse_term;

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
    /// Construct a Hamiltonian from a nested list of operator terms.
    ///
    /// Args:
    ///     terms (list[list[tuple[str, list[tuple]]]]): Outer list indexed by
    ///         ``cindex``. Each element is a list of ``(op_str, coupling_list)``
    ///         pairs where:
    ///
    ///         - ``op_str`` (str): Operator string, one character per site
    ///           (``'x'``, ``'y'``, ``'z'``, ``'+'``, ``'-'``, ``'n'``).
    ///         - ``coupling_list`` (list[tuple]): Each element is
    ///           ``(coeff, site_0, site_1, ...)`` with one site per character
    ///           in ``op_str``. ``coeff`` may be ``complex``, ``float``,
    ///           or ``int``.
    ///
    /// Raises:
    ///     ValueError: If ``op_str`` contains an unknown operator character,
    ///         if the number of site indices does not match ``len(op_str)``,
    ///         or if the input structure is malformed.
    #[new]
    pub fn new(py: Python<'_>, terms: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut raw = Vec::new();
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
                let (entries, term_max_site) = parse_term::<HardcoreOp>(py, &item, cindex)?;
                max_site = max_site.max(term_max_site);
                n_sites = n_sites.max(term_max_site + 1);
                raw.extend(entries);
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

    /// Number of sites, inferred from the maximum site index plus one.
    #[getter]
    pub fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }

    /// Number of distinct coefficient indices (outer list length).
    #[getter]
    pub fn num_cindices(&self) -> usize {
        self.num_cindices
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyHardcoreHamiltonian(n_sites={}, num_cindices={})",
            self.inner.n_sites(),
            self.num_cindices,
        )
    }
}
