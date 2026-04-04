use crate::error::Error;
use num_complex::Complex;
use pyo3::prelude::*;
use quspin_core::operator::pauli::{HardcoreOp, HardcoreOperator, HardcoreOperatorInner, OpEntry};
use smallvec::SmallVec;

/// Python-facing Pauli / hardcore-boson / spin-½ operator.
///
/// Terms are grouped by coupling-constant index (cindex): the i-th element of
/// `terms` corresponds to cindex `i`.  Each element is a `(op_str, bonds)`
/// pair where each bond is `[coeff, site0, site1, ...]`.
///
/// Example (XX + ZZ nearest-neighbour chain on 4 sites):
/// ```python
/// op = PauliOperator([
///     ("XX", [[1.0, 0, 1], [1.0, 1, 2], [1.0, 2, 3]]),
///     ("ZZ", [[1.0, 0, 1], [1.0, 1, 2], [1.0, 2, 3]]),
/// ])
/// ```
#[pyclass(name = "PauliOperator", module = "quspin._rs")]
pub struct PyPauliOperator {
    pub inner: HardcoreOperatorInner,
}

#[pymethods]
impl PyPauliOperator {
    /// Construct from a list of `(op_str, bonds)` pairs, one per cindex.
    ///
    /// Each bond is `[coeff, site0, site1, ...]` where `coeff` is a complex
    /// scalar and the remaining elements are integer site indices.
    #[new]
    #[pyo3(signature = (terms))]
    fn new(py: Python<'_>, terms: Vec<(String, Vec<Vec<PyObject>>)>) -> PyResult<Self> {
        let max_cindex = terms.len().saturating_sub(1);
        let max_site = max_site_from_terms(py, &terms)?;

        let use_u8 = max_cindex <= 255 && max_site <= 255;

        if use_u8 {
            let entries = parse_terms::<u8>(py, &terms).map_err(Error::from)?;
            Ok(PyPauliOperator {
                inner: HardcoreOperatorInner::Ham8(HardcoreOperator::new(entries)),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let entries = parse_terms::<u16>(py, &terms).map_err(Error::from)?;
            Ok(PyPauliOperator {
                inner: HardcoreOperatorInner::Ham16(HardcoreOperator::new(entries)),
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
    fn num_cindices(&self) -> usize {
        self.inner.num_cindices()
    }

    #[getter]
    fn lhss(&self) -> usize {
        self.inner.lhss()
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliOperator(max_site={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.num_cindices(),
        )
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Extract the max site index from all bonds across all terms.
pub(crate) fn max_site_from_terms(
    py: Python<'_>,
    terms: &[(String, Vec<Vec<PyObject>>)],
) -> PyResult<usize> {
    let mut max = 0usize;
    for (_, bonds) in terms {
        for bond in bonds {
            // bond = [coeff, site0, site1, ...]  — skip index 0 (coeff)
            for obj in bond.iter().skip(1) {
                let site: u32 = obj.bind(py).extract()?;
                max = max.max(site as usize);
            }
        }
    }
    Ok(max)
}

/// Extract a complex coefficient from a Python scalar (int, float, or complex).
pub(crate) fn extract_coeff(py: Python<'_>, obj: &PyObject) -> PyResult<Complex<f64>> {
    let bound = obj.bind(py);
    if let Ok(z) = bound.extract::<Complex<f64>>() {
        return Ok(z);
    }
    let re: f64 = bound.extract()?;
    Ok(Complex::new(re, 0.0))
}

fn parse_terms<C: Copy + Ord + TryFrom<usize>>(
    py: Python<'_>,
    terms: &[(String, Vec<Vec<PyObject>>)],
) -> Result<Vec<OpEntry<C>>, quspin_core::error::QuSpinError>
where
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut entries = Vec::new();
    for (cindex_usize, (op_str, bonds)) in terms.iter().enumerate() {
        let cindex = C::try_from(cindex_usize).map_err(|_| {
            quspin_core::error::QuSpinError::ValueError(format!(
                "cindex {cindex_usize} out of range for chosen index type"
            ))
        })?;
        for bond in bonds {
            if bond.is_empty() {
                return Err(quspin_core::error::QuSpinError::ValueError(
                    "each bond must be [coeff, site0, site1, ...]".to_string(),
                ));
            }
            let coeff = extract_coeff(py, &bond[0]).map_err(|e| {
                quspin_core::error::QuSpinError::ValueError(format!("bond coefficient: {e}"))
            })?;
            let sites: Vec<u32> = bond[1..]
                .iter()
                .map(|s| {
                    s.bind(py).extract::<u32>().map_err(|e| {
                        quspin_core::error::QuSpinError::ValueError(format!("bond site index: {e}"))
                    })
                })
                .collect::<Result<_, _>>()?;
            if op_str.len() != sites.len() {
                return Err(quspin_core::error::QuSpinError::ValueError(format!(
                    "op_str length {} != number of sites {}",
                    op_str.len(),
                    sites.len()
                )));
            }
            let mut ops: SmallVec<[(HardcoreOp, u32); 4]> = SmallVec::new();
            for (ch, &site) in op_str.chars().zip(sites.iter()) {
                let op = HardcoreOp::from_char(ch).ok_or_else(|| {
                    quspin_core::error::QuSpinError::ValueError(format!(
                        "unknown operator character '{ch}'; expected one of x, y, z, +, -, n"
                    ))
                })?;
                ops.push((op, site));
            }
            entries.push(OpEntry::new(cindex, coeff, ops));
        }
    }
    Ok(entries)
}
