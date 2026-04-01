use crate::error::Error;
use num_complex::Complex;
use pyo3::prelude::*;
use quspin_core::operator::boson::{BosonOp, BosonOpEntry, BosonOperator, BosonOperatorInner};
use smallvec::SmallVec;

/// Python-facing bosonic operator (truncated harmonic oscillator).
///
/// Each term is `(coeff, op_str, sites, cindex)` where:
/// - `coeff`   – complex coefficient
/// - `op_str`  – ASCII string; each char is one of `+ - n N` (case-insensitive)
/// - `sites`   – site index for each character in `op_str`
/// - `cindex`  – which coupling constant this term belongs to (default 0)
/// - `lhss`    – number of on-site Fock states (≥ 2)
#[pyclass(name = "BosonOperator", module = "quspin._rs")]
pub struct PyBosonOperator {
    pub inner: BosonOperatorInner,
}

#[pymethods]
impl PyBosonOperator {
    #[new]
    #[pyo3(signature = (terms, lhss))]
    fn new(terms: Vec<(Complex<f64>, String, Vec<u32>, usize)>, lhss: usize) -> PyResult<Self> {
        if lhss < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("lhss must be >= 2"));
        }
        let max_cindex = terms.iter().map(|(_, _, _, c)| *c).max().unwrap_or(0);
        let max_site = terms
            .iter()
            .flat_map(|(_, _, sites, _)| sites.iter().copied())
            .max()
            .unwrap_or(0) as usize;

        let use_u8 = max_cindex <= 255 && max_site <= 255;

        if use_u8 {
            let entries = parse_terms::<u8>(&terms).map_err(Error::from)?;
            Ok(PyBosonOperator {
                inner: BosonOperatorInner::Ham8(BosonOperator::new(entries, lhss)),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let entries = parse_terms::<u16>(&terms).map_err(Error::from)?;
            Ok(PyBosonOperator {
                inner: BosonOperatorInner::Ham16(BosonOperator::new(entries, lhss)),
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
            "BosonOperator(max_site={}, lhss={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.lhss(),
            self.inner.num_cindices(),
        )
    }
}

fn parse_terms<C: Copy + Ord + TryFrom<usize>>(
    terms: &[(Complex<f64>, String, Vec<u32>, usize)],
) -> Result<Vec<BosonOpEntry<C>>, quspin_core::error::QuSpinError>
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
        let mut ops: SmallVec<[(BosonOp, u32); 4]> = SmallVec::new();
        for (ch, &site) in op_str.chars().zip(sites.iter()) {
            let op = BosonOp::from_char(ch).ok_or_else(|| {
                quspin_core::error::QuSpinError::ValueError(format!(
                    "unknown operator character '{ch}'; expected one of +, -, n"
                ))
            })?;
            ops.push((op, site));
        }
        entries.push(BosonOpEntry::new(cindex, *coeff, ops));
    }
    Ok(entries)
}
