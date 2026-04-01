use crate::error::Error;
use num_complex::Complex;
use pyo3::prelude::*;
use quspin_core::operator::pauli::{HardcoreOp, HardcoreOperator, HardcoreOperatorInner, OpEntry};
use smallvec::SmallVec;

/// Python-facing Pauli / hardcore-boson / spin-½ operator.
///
/// Each term is `(coeff, op_str, sites, cindex)` where:
/// - `coeff`   – complex coefficient
/// - `op_str`  – ASCII string; each char is one of `x y z + - n` (case-insensitive)
/// - `sites`   – site index for each character in `op_str`
/// - `cindex`  – which coupling constant this term belongs to (default 0)
///
/// Example (Heisenberg XX+YY chain on 4 sites):
/// ```python
/// op = PauliOperator([
///     (1.0, "++", [0, 1], 0), (1.0, "--", [0, 1], 0),
///     (1.0, "++", [1, 2], 0), (1.0, "--", [1, 2], 0),
///     (1.0, "++", [2, 3], 0), (1.0, "--", [2, 3], 0),
/// ])
/// ```
#[pyclass(name = "PauliOperator", module = "quspin._rs")]
pub struct PyPauliOperator {
    pub inner: HardcoreOperatorInner,
}

#[pymethods]
impl PyPauliOperator {
    /// Construct from a list of `(coeff, op_str, sites, cindex)` tuples.
    #[new]
    #[pyo3(signature = (terms))]
    fn new(terms: Vec<(Complex<f64>, String, Vec<u32>, usize)>) -> PyResult<Self> {
        // Determine cindex type based on max_cindex and max_site.
        let max_cindex = terms.iter().map(|(_, _, _, c)| *c).max().unwrap_or(0);
        let max_site = terms
            .iter()
            .flat_map(|(_, _, sites, _)| sites.iter().copied())
            .max()
            .unwrap_or(0) as usize;

        let use_u8 = max_cindex <= 255 && max_site <= 255;

        if use_u8 {
            let entries = parse_terms::<u8>(&terms).map_err(Error::from)?;
            Ok(PyPauliOperator {
                inner: HardcoreOperatorInner::Ham8(HardcoreOperator::new(entries)),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let entries = parse_terms::<u16>(&terms).map_err(Error::from)?;
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

fn parse_terms<C: Copy + Ord + TryFrom<usize>>(
    terms: &[(Complex<f64>, String, Vec<u32>, usize)],
) -> Result<Vec<OpEntry<C>>, quspin_core::error::QuSpinError>
where
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut entries = Vec::with_capacity(terms.len());
    for (coeff, op_str, sites, cindex_usize) in terms {
        if op_str.len() != sites.len() {
            return Err(quspin_core::error::QuSpinError::ValueError(format!(
                "op_str length {} != sites length {}",
                op_str.len(),
                sites.len()
            )));
        }
        let cindex = C::try_from(*cindex_usize).map_err(|_| {
            quspin_core::error::QuSpinError::ValueError(format!(
                "cindex {cindex_usize} out of range for chosen index type"
            ))
        })?;
        let mut ops: SmallVec<[(HardcoreOp, u32); 4]> = SmallVec::new();
        for (ch, &site) in op_str.chars().zip(sites.iter()) {
            let op = HardcoreOp::from_char(ch).ok_or_else(|| {
                quspin_core::error::QuSpinError::ValueError(format!(
                    "unknown operator character '{ch}'; expected one of x, y, z, +, -, n"
                ))
            })?;
            ops.push((op, site));
        }
        entries.push(OpEntry::new(cindex, *coeff, ops));
    }
    Ok(entries)
}
