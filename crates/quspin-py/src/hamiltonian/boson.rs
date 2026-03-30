/// Python-facing `PyBosonHamiltonian` pyclass.
///
/// Wraps a `BosonOperatorInner` enum that selects between
/// `BosonOperator<u8>` (≤ 255 cindices / site indices) and
/// `BosonOperator<u16>` (larger).  The cindex type is chosen at
/// construction time based on the maximum cindex and site index seen in the
/// input.
///
/// ## Python constructor
///
/// ```python
/// H = PyBosonHamiltonian(lhss=3, terms=[
///     [("+-", [(J, 0, 1), (J, 1, 2)])],  # cindex = 0
///     [("n",  [(mu, 0), (mu, 1)])],        # cindex = 1
/// ])
/// ```
///
/// - `lhss` (int): Local Hilbert space size (number of levels per site). Must be ≥ 2.
/// - Outer list index → `cindex` value stored in each entry.
/// - Each inner list element is a `(op_str, coupling_list)` pair.
/// - `op_str` characters: `'+'` (a†), `'-'` (a), `'n'` (n̂).
/// - Each `coupling_list` element is `(coeff, site_0, site_1, ...)`.
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use quspin_core::hamiltonian::boson::dispatch::BosonOperatorInner;
use quspin_core::hamiltonian::boson::{BosonOp, BosonOpEntry, BosonOperator};

use super::parse::parse_term;

// ---------------------------------------------------------------------------
// PyBosonHamiltonian
// ---------------------------------------------------------------------------

#[pyclass(name = "PyBosonHamiltonian")]
pub struct PyBosonHamiltonian {
    pub inner: BosonOperatorInner,
    /// Number of distinct cindex values (= length of the outer terms list).
    pub num_cindices: usize,
}

#[pymethods]
impl PyBosonHamiltonian {
    /// Construct a bosonic Hamiltonian from a local Hilbert space size and a
    /// nested list of operator terms.
    ///
    /// Args:
    ///     lhss (int): Local Hilbert space size (levels per site). Must be ≥ 2.
    ///     terms (list[list[tuple[str, list[tuple]]]]): Outer list indexed by
    ///         ``cindex``. Each element is a list of ``(op_str, coupling_list)``
    ///         pairs where:
    ///
    ///         - ``op_str`` (str): Operator string, one character per site
    ///           (``'+'``, ``'-'``, ``'n'``).
    ///         - ``coupling_list`` (list[tuple]): Each element is
    ///           ``(coeff, site_0, site_1, ...)`` with one site per character
    ///           in ``op_str``.
    ///
    /// Raises:
    ///     ValueError: If ``lhss < 2``, ``op_str`` contains an unknown
    ///         operator character, or the input structure is malformed.
    #[new]
    pub fn new(py: Python<'_>, lhss: usize, terms: &Bound<'_, PyList>) -> PyResult<Self> {
        if lhss < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lhss must be ≥ 2 for a bosonic Hamiltonian",
            ));
        }

        let mut raw = Vec::new();
        let mut max_cindex: usize = 0;
        let mut max_site: usize = 0;

        for (cindex, cindex_terms) in terms.iter().enumerate() {
            max_cindex = max_cindex.max(cindex);
            let cindex_list = cindex_terms.downcast::<PyList>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "terms[{cindex}] must be a list of (op_str, coupling_list) pairs"
                ))
            })?;

            for item in cindex_list.iter() {
                let (entries, term_max_site) = parse_term::<BosonOp>(py, &item, cindex)?;
                max_site = max_site.max(term_max_site);
                raw.extend(entries);
            }
        }

        let num_cindices = max_cindex + 1;
        let needs_u16 = max_cindex > 255 || max_site > 255;
        let inner = if needs_u16 {
            let entries: Vec<BosonOpEntry<u16>> = raw
                .into_iter()
                .map(|r| BosonOpEntry::new(r.cindex as u16, r.coeff, r.ops))
                .collect();
            BosonOperatorInner::Ham16(BosonOperator::new(entries, lhss))
        } else {
            let entries: Vec<BosonOpEntry<u8>> = raw
                .into_iter()
                .map(|r| BosonOpEntry::new(r.cindex as u8, r.coeff, r.ops))
                .collect();
            BosonOperatorInner::Ham8(BosonOperator::new(entries, lhss))
        };

        Ok(PyBosonHamiltonian {
            inner,
            num_cindices,
        })
    }

    /// Local Hilbert space size (number of levels per site).
    #[getter]
    pub fn lhss(&self) -> usize {
        self.inner.lhss()
    }

    /// Maximum site index across all operator strings.
    #[getter]
    pub fn max_site(&self) -> usize {
        self.inner.max_site()
    }

    /// Number of distinct coefficient indices (outer list length).
    #[getter]
    pub fn num_cindices(&self) -> usize {
        self.num_cindices
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyBosonHamiltonian(lhss={}, max_site={}, num_cindices={})",
            self.inner.lhss(),
            self.inner.max_site(),
            self.num_cindices,
        )
    }
}
