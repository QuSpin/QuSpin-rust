pub mod bond;
pub mod boson;
pub mod fermion;
pub mod monomial;
pub mod pauli;

pub use bond::PyBondOperator;
pub use boson::PyBosonOperator;
pub use fermion::PyFermionOperator;
pub use monomial::PyMonomialOperator;
pub use pauli::PyPauliOperator;

use crate::basis::{PyBosonBasis, PyFermionBasis, PyGenericBasis, PySpinBasis};
use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use quspin_core::ParseOp;
use quspin_core::basis::dispatch::SpaceInner;
use smallvec::SmallVec;

/// Extract a reference to `SpaceInner` from any Python basis object,
/// passing it to a closure while the PyRef borrow is live.
pub(crate) fn with_space_inner<F, R>(basis: &Bound<'_, PyAny>, f: F) -> PyResult<R>
where
    F: FnOnce(&SpaceInner) -> R,
{
    if let Ok(b) = basis.downcast::<PySpinBasis>() {
        Ok(f(&b.borrow().inner.inner))
    } else if let Ok(b) = basis.downcast::<PyFermionBasis>() {
        Ok(f(&b.borrow().inner.inner))
    } else if let Ok(b) = basis.downcast::<PyBosonBasis>() {
        Ok(f(&b.borrow().inner.inner))
    } else if let Ok(b) = basis.downcast::<PyGenericBasis>() {
        Ok(f(&b.borrow().inner.inner))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "basis must be SpinBasis, FermionBasis, BosonBasis, or GenericBasis",
        ))
    }
}

/// Same as `with_space_inner` but for two bases at once.
pub(crate) fn with_two_space_inners<F, R>(
    basis_a: &Bound<'_, PyAny>,
    basis_b: &Bound<'_, PyAny>,
    f: F,
) -> PyResult<R>
where
    F: FnOnce(&SpaceInner, &SpaceInner) -> R,
{
    with_space_inner(basis_a, |a| with_space_inner(basis_b, |b| f(a, b)))?
}

/// Read a Complex64 numpy array into a Vec<Complex<f64>>.
///
/// # Safety
/// Caller must ensure the array is valid.
pub(crate) unsafe fn as_c64_vec(arr: &Bound<'_, PyArray1<Complex64>>) -> Vec<Complex<f64>> {
    unsafe {
        arr.as_array()
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect()
    }
}

/// Write Vec<Complex<f64>> back into a Complex64 numpy array.
///
/// # Safety
/// Caller must ensure lengths match and array is writable.
pub(crate) unsafe fn write_c64_back(arr: &Bound<'_, PyArray1<Complex64>>, data: &[Complex<f64>]) {
    let mut out = unsafe { arr.as_array_mut() };
    for (dst, src) in out.iter_mut().zip(data.iter()) {
        *dst = Complex64::new(src.re, src.im);
    }
}

// ---------------------------------------------------------------------------
// Shared operator-parsing helpers (used by pauli, boson, fermion)
// ---------------------------------------------------------------------------

/// A single term: a list of `(op_str, bonds)` pairs sharing one cindex.
pub(crate) type Term = Vec<(String, Vec<Vec<PyObject>>)>;
/// All terms passed to a constructor (one per cindex).
pub(crate) type Terms = Vec<Term>;

/// Extract the max site index from all bonds across all terms.
pub(crate) fn max_site_from_terms(py: Python<'_>, terms: &[Term]) -> PyResult<usize> {
    let mut max = 0usize;
    for term in terms {
        for (_, bonds) in term {
            for bond in bonds {
                // bond = [coeff, site0, site1, ...]  — skip index 0 (coeff)
                for obj in bond.iter().skip(1) {
                    let site: u32 = obj.bind(py).extract()?;
                    max = max.max(site as usize);
                }
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

/// Generic parsing of `*terms` for any operator type that implements `ParseOp`.
///
/// The `make_entry` closure constructs the concrete entry type (e.g. `OpEntry`,
/// `BosonOpEntry`, `FermionOpEntry`) from the parsed components, avoiding the
/// need for a shared trait on the entry types in `quspin-core`.
pub(crate) fn parse_terms_generic<C, Op, E, F>(
    py: Python<'_>,
    terms: &[Term],
    make_entry: F,
) -> Result<Vec<E>, quspin_core::error::QuSpinError>
where
    C: Copy + Ord + TryFrom<usize>,
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
    Op: ParseOp,
    F: Fn(C, Complex<f64>, SmallVec<[(Op, u32); 4]>) -> E,
{
    let mut entries = Vec::new();
    for (cindex_usize, term) in terms.iter().enumerate() {
        let cindex = C::try_from(cindex_usize).map_err(|_| {
            quspin_core::error::QuSpinError::ValueError(format!(
                "cindex {cindex_usize} out of range for chosen index type"
            ))
        })?;
        for (op_str, bonds) in term {
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
                            quspin_core::error::QuSpinError::ValueError(format!(
                                "bond site index: {e}"
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let op_len = op_str.chars().count();
                if op_len != sites.len() {
                    return Err(quspin_core::error::QuSpinError::ValueError(format!(
                        "op_str length {} != number of sites {}",
                        op_len,
                        sites.len()
                    )));
                }
                let mut ops: SmallVec<[(Op, u32); 4]> = SmallVec::new();
                for (ch, &site) in op_str.chars().zip(sites.iter()) {
                    let op = Op::from_char(ch)?;
                    ops.push((op, site));
                }
                entries.push(make_entry(cindex, coeff, ops));
            }
        }
    }
    Ok(entries)
}
