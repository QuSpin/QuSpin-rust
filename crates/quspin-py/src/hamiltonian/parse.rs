/// Shared parsing helpers for Hamiltonian constructors.
///
/// These functions handle the Python-side parsing of the nested
/// `terms` structure used by all Hamiltonian types:
///
/// ```text
/// terms[cindex] = [(op_str, [(coeff, site_0, ...), ...]), ...]
/// ```
///
/// Each function corresponds to one level of that structure and is generic
/// over the operator type `Op: ParseOp`.  Implement `ParseOp` for a new
/// operator enum to reuse the full parsing pipeline.
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyTuple};
use quspin_core::ParseOp;
use smallvec::SmallVec;

use crate::error::Error;

// ---------------------------------------------------------------------------
// RawEntry
// ---------------------------------------------------------------------------

/// Intermediate storage for a parsed operator entry before the cindex type
/// (`u8` vs `u16`) is decided.
pub struct RawEntry<Op> {
    pub cindex: usize,
    pub coeff: num_complex::Complex<f64>,
    pub ops: SmallVec<[(Op, u32); 4]>,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse one `(op_str, coupling_list)` tuple.
///
/// Returns `(entries, max_site)` where `max_site` is the largest site index
/// seen across all couplings in this term.
pub fn parse_term<Op: ParseOp>(
    py: Python<'_>,
    item: &Bound<'_, PyAny>,
    cindex: usize,
) -> PyResult<(Vec<RawEntry<Op>>, usize)> {
    let tup = item.downcast::<PyTuple>().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("each term must be a (op_str, coupling_list) tuple")
    })?;
    if tup.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "each term must be a 2-tuple (op_str, coupling_list)",
        ));
    }

    let op_str: String = tup.get_item(0)?.extract()?;
    let coupling_list = tup.get_item(1)?.downcast_into::<PyList>().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "coupling_list must be a list of (coeff, site, ...) tuples",
        )
    })?;

    let mut entries: Vec<RawEntry<Op>> = Vec::with_capacity(coupling_list.len());
    let mut max_site: usize = 0;
    for coupling in coupling_list.iter() {
        let ctup = coupling.downcast::<PyTuple>().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "each coupling must be a (coeff, site_0, ...) tuple",
            )
        })?;
        let (entry, coupling_max_site) = parse_coupling(py, ctup, &op_str, cindex)?;
        max_site = max_site.max(coupling_max_site);
        entries.push(entry);
    }
    Ok((entries, max_site))
}

/// Parse one `(coeff, site_0, …)` coupling tuple for the given `op_str`.
///
/// Returns `(RawEntry, max_site)` where `max_site` is the largest site index
/// in this coupling.
pub fn parse_coupling<Op: ParseOp>(
    py: Python<'_>,
    ctup: &Bound<'_, PyTuple>,
    op_str: &str,
    cindex: usize,
) -> PyResult<(RawEntry<Op>, usize)> {
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
    let mut max_site: usize = 0;
    for i in 1..ctup.len() {
        let site: u32 = ctup.get_item(i)?.extract()?;
        sites.push(site);
        max_site = max_site.max(site as usize);
    }

    let ops = parse_ops::<Op>(op_str, &sites)?;
    Ok((RawEntry { cindex, coeff, ops }, max_site))
}

/// Build the `(Op, site)` pairs from an operator string and site list.
pub fn parse_ops<Op: ParseOp>(
    op_str: &str,
    sites: &SmallVec<[u32; 4]>,
) -> PyResult<SmallVec<[(Op, u32); 4]>> {
    op_str
        .chars()
        .zip(sites.iter().copied())
        .map(|(ch, site)| Ok((Op::from_char(ch).map_err(Error)?, site)))
        .collect()
}

/// Extract a Python scalar as `Complex<f64>`.
pub fn extract_complex(
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
