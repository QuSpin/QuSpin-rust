/// Python-facing `PyFermionHamiltonian` pyclass.
///
/// Wraps a `FermionOperatorInner` enum that selects the cindex type at
/// construction time based on the number of operator strings and site indices.
///
/// ## Python constructor
///
/// ```python
/// H = PyFermionHamiltonian([
///     [("+-", [(t, 0, 1), (t, 1, 0)])],  # cindex = 0  (hopping)
///     [("n",  [(U, 0), (U, 1)])],         # cindex = 1  (Hubbard U)
/// ])
/// ```
///
/// - Outer list index → `cindex` value stored in each `FermionOpEntry`.
/// - Each inner list element is a `(op_str, coupling_list)` pair.
/// - `op_str` characters: `'+'` (c†), `'-'` (c), `'n'` (n̂).
/// - Orbital labelling: site `2*i` = spin-down orbital `i`,
///   site `2*i+1` = spin-up orbital `i`.
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use quspin_core::operator::fermion::dispatch::FermionOperatorInner;
use quspin_core::operator::fermion::{FermionOp, FermionOpEntry, FermionOperator};

use super::parse::parse_term;

// ---------------------------------------------------------------------------
// PyFermionHamiltonian
// ---------------------------------------------------------------------------

#[pyclass(name = "PyFermionHamiltonian")]
pub struct PyFermionHamiltonian {
    pub inner: FermionOperatorInner,
    /// Number of distinct cindex values (= length of the outer terms list).
    pub num_cindices: usize,
}

#[pymethods]
impl PyFermionHamiltonian {
    /// Construct a fermionic Hamiltonian from a nested list of operator terms.
    ///
    /// Args:
    ///     terms (list[list[tuple[str, list[tuple]]]]): Outer list indexed by
    ///         ``cindex``. Each element is a list of ``(op_str, coupling_list)``
    ///         pairs where:
    ///
    ///         - ``op_str`` (str): Operator string, one character per site
    ///           (``'+'`` for c†, ``'-'`` for c, ``'n'`` for n̂).
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

        for (cindex, cindex_terms) in terms.iter().enumerate() {
            max_cindex = max_cindex.max(cindex);
            let cindex_list = cindex_terms.downcast::<PyList>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "terms[{cindex}] must be a list of (op_str, coupling_list) pairs"
                ))
            })?;

            for item in cindex_list.iter() {
                let (entries, term_max_site) = parse_term::<FermionOp>(py, &item, cindex)?;
                max_site = max_site.max(term_max_site);
                raw.extend(entries);
            }
        }

        let num_cindices = max_cindex + 1;
        let needs_u16 = max_cindex > 255 || max_site > 255;
        let inner = if needs_u16 {
            let entries: Vec<FermionOpEntry<u16>> = raw
                .into_iter()
                .map(|r| FermionOpEntry::new(r.cindex as u16, r.coeff, r.ops))
                .collect();
            FermionOperatorInner::Ham16(FermionOperator::new(entries))
        } else {
            let entries: Vec<FermionOpEntry<u8>> = raw
                .into_iter()
                .map(|r| FermionOpEntry::new(r.cindex as u8, r.coeff, r.ops))
                .collect();
            FermionOperatorInner::Ham8(FermionOperator::new(entries))
        };

        Ok(PyFermionHamiltonian {
            inner,
            num_cindices,
        })
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
            "PyFermionHamiltonian(max_site={}, num_cindices={})",
            self.inner.max_site(),
            self.num_cindices,
        )
    }
}
