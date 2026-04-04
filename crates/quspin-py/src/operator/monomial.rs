use crate::error::Error;
use num_complex::Complex;
use numpy::{Complex64, PyReadonlyArray1};
use pyo3::prelude::*;
use quspin_core::operator::monomial::{MonomialOperator, MonomialOperatorInner, MonomialTerm};
use smallvec::SmallVec;

/// Python-facing generic monomial operator.
///
/// A monomial operator is defined by a list of terms, each specifying:
/// - `perm`: 1-D integer array of length `lhss^k` — the output joint-state
///   index for each input joint-state index.
/// - `amp`: 1-D complex128 array of length `lhss^k` — the complex amplitude
///   for each input joint-state.
/// - `bonds`: list of k-tuples of site indices.  All bonds in one term must
///   have the same number of sites k.
///
/// Cindex (coupling-constant index) is implicit by position: the i-th term
/// gets cindex `i`.
///
/// Example (cyclic shift on nearest-neighbour bonds of a 3-site lhss=3 chain):
/// ```python
/// import numpy as np
/// lhss = 3
/// dim = lhss ** 2  # 2-site joint space
/// perm = np.array([
///     lhss*((a+1)%lhss) + (b+1)%lhss
///     for a in range(lhss) for b in range(lhss)
/// ], dtype=np.intp)
/// amp = np.ones(dim, dtype=complex)
/// bonds = [(0, 1), (1, 2)]
/// op = MonomialOperator(lhss, (perm, amp, bonds))
/// ```
#[pyclass(name = "MonomialOperator", module = "quspin._rs")]
pub struct PyMonomialOperator {
    pub inner: MonomialOperatorInner,
}

#[pymethods]
impl PyMonomialOperator {
    /// Construct from `lhss` and one or more terms.
    ///
    /// Args:
    ///     lhss: Local Hilbert-space size (number of states per site, ≥ 2).
    ///     *terms: Each term is a 3-tuple ``(perm, amp, bonds)`` where:
    ///         - ``perm`` — 1-D integer array of length ``lhss^k``
    ///         - ``amp`` — 1-D complex128 array of length ``lhss^k``
    ///         - ``bonds`` — list of k-tuples of site indices (``int``)
    #[new]
    #[pyo3(signature = (lhss, *terms))]
    fn new(py: Python<'_>, lhss: usize, terms: Vec<PyObject>) -> PyResult<Self> {
        if lhss < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("lhss must be >= 2"));
        }
        if terms.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "MonomialOperator requires at least one term",
            ));
        }

        // Parse all terms to determine max_site and max_cindex for type selection.
        let parsed = parse_terms(py, &terms, lhss)?;

        let max_cindex = parsed.len().saturating_sub(1);
        let max_site = parsed
            .iter()
            .flat_map(|(_, _, bonds)| bonds.iter())
            .flat_map(|b| b.iter())
            .copied()
            .max()
            .unwrap_or(0);

        let use_u8 = max_cindex <= 255 && max_site <= 255;

        if use_u8 {
            let mono_terms = build_mono_terms::<u8>(&parsed);
            let op = MonomialOperator::new(mono_terms, lhss).map_err(Error::from)?;
            Ok(PyMonomialOperator {
                inner: MonomialOperatorInner::Ham8(op),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let mono_terms = build_mono_terms::<u16>(&parsed);
            let op = MonomialOperator::new(mono_terms, lhss).map_err(Error::from)?;
            Ok(PyMonomialOperator {
                inner: MonomialOperatorInner::Ham16(op),
            })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "cindex and site indices must be <= 65535",
            ))
        }
    }

    #[getter]
    fn max_site(&self) -> usize {
        self.inner.max_site()
    }

    #[getter]
    fn num_coeffs(&self) -> usize {
        self.inner.num_cindices()
    }

    #[getter]
    fn lhss(&self) -> usize {
        self.inner.lhss()
    }

    fn __repr__(&self) -> String {
        format!(
            "MonomialOperator(lhss={}, max_site={}, num_coeffs={})",
            self.inner.lhss(),
            self.inner.max_site(),
            self.inner.num_cindices(),
        )
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Intermediate representation of one parsed term before type parameters.
type ParsedTerm = (Vec<usize>, Vec<Complex<f64>>, Vec<SmallVec<[u32; 4]>>);

/// Parse the Python `*terms` into an intermediate representation.
///
/// Each term is a `(perm, amp, bonds)` 3-tuple:
/// - `perm`: any array-like convertible to 1-D usize array
/// - `amp`:  any array-like convertible to 1-D complex128 array
/// - `bonds`: list of k-tuples of u32 site indices
fn parse_terms(py: Python<'_>, terms: &[PyObject], lhss: usize) -> PyResult<Vec<ParsedTerm>> {
    let mut out = Vec::with_capacity(terms.len());
    for (term_idx, term_obj) in terms.iter().enumerate() {
        let term = term_obj.bind(py);

        // Each term must be a 3-tuple (perm, amp, bonds).
        let tuple = term.downcast::<pyo3::types::PyTuple>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "term {term_idx}: expected a 3-tuple (perm, amp, bonds)"
            ))
        })?;
        if tuple.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "term {term_idx}: expected a 3-tuple (perm, amp, bonds), got {}-tuple",
                tuple.len()
            )));
        }

        // Extract perm as a 1-D integer array (accept np.intp / int64 / int32).
        // We try i64 first (most common on 64-bit), then i32 (32-bit platforms).
        let perm: Vec<usize> = {
            let perm_obj = tuple.get_item(0)?;
            if let Ok(arr) = perm_obj.extract::<PyReadonlyArray1<'_, i64>>() {
                arr.as_array().iter().map(|&v| v as usize).collect()
            } else if let Ok(arr) = perm_obj.extract::<PyReadonlyArray1<'_, i32>>() {
                arr.as_array().iter().map(|&v| v as usize).collect()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "term {term_idx}: perm must be a 1-D integer array (e.g. dtype=np.intp or np.int64)"
                )));
            }
        };

        // Extract amp as 1-D complex128 array.
        let amp: Vec<Complex<f64>> = {
            let amp_obj = tuple.get_item(1)?;
            let arr: PyReadonlyArray1<'_, Complex64> = amp_obj
                .extract::<PyReadonlyArray1<'_, Complex64>>()
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "term {term_idx}: amp must be a 1-D complex128 array"
                    ))
                })?;
            arr.as_array()
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect()
        };

        if perm.len() != amp.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "term {term_idx}: perm.len()={} != amp.len()={}",
                perm.len(),
                amp.len()
            )));
        }

        // Validate that length == lhss^k for some k.
        if !perm.is_empty() {
            let len = perm.len();
            let mut l = 1usize;
            while l < len {
                l *= lhss;
            }
            if l != len {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "term {term_idx}: perm/amp length {len} is not lhss^k for lhss={lhss}"
                )));
            }
        }

        // Extract bonds as list of k-tuples.
        let bonds_obj = tuple.get_item(2)?;
        let bonds_list = bonds_obj.extract::<Vec<Vec<u32>>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "term {term_idx}: bonds must be a list of tuples/lists of int site indices"
            ))
        })?;
        let bonds: Vec<SmallVec<[u32; 4]>> =
            bonds_list.into_iter().map(SmallVec::from_vec).collect();

        out.push((perm, amp, bonds));
    }
    Ok(out)
}

/// Convert parsed terms into `MonomialTerm<C>` with the correct cindex type.
fn build_mono_terms<C: Copy + TryFrom<usize>>(parsed: &[ParsedTerm]) -> Vec<MonomialTerm<C>>
where
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
{
    parsed
        .iter()
        .enumerate()
        .map(|(i, (perm, amp, bonds))| MonomialTerm {
            cindex: C::try_from(i).expect("cindex in range (checked above)"),
            perm: perm.clone(),
            amp: amp.clone(),
            bonds: bonds.clone(),
        })
        .collect()
}
