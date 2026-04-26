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
use crate::error::Error;
use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use quspin_core::OperatorDispatch;
use quspin_core::ParseOp;
use quspin_core::basis::dispatch::GenericBasis;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// Basis-aware operator dispatch
// ---------------------------------------------------------------------------

/// Dispatch a single-basis `apply` call to the right typed entry point on
/// `op` based on the concrete Python basis type.
///
/// `PyFermionBasis` routes through [`OperatorDispatch::apply_bit`] (with
/// its `&BitBasis`); the other three wrappers route through
/// [`OperatorDispatch::apply`] (with their `&GenericBasis`).
pub(crate) fn dispatch_apply<OP>(
    op: &OP,
    basis: &Bound<'_, PyAny>,
    coeffs: &[Complex<f64>],
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
    overwrite: bool,
) -> PyResult<()>
where
    OP: OperatorDispatch,
{
    if let Ok(b) = basis.downcast::<PyFermionBasis>() {
        op.apply_bit(&b.borrow().inner.inner, coeffs, input, output, overwrite)
            .map_err(Error::from)?;
    } else if let Ok(b) = basis.downcast::<PySpinBasis>() {
        op.apply(&b.borrow().inner.inner, coeffs, input, output, overwrite)
            .map_err(Error::from)?;
    } else if let Ok(b) = basis.downcast::<PyBosonBasis>() {
        op.apply(&b.borrow().inner.inner, coeffs, input, output, overwrite)
            .map_err(Error::from)?;
    } else if let Ok(b) = basis.downcast::<PyGenericBasis>() {
        op.apply(&b.borrow().inner, coeffs, input, output, overwrite)
            .map_err(Error::from)?;
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "basis must be SpinBasis, FermionBasis, BosonBasis, or GenericBasis",
        ));
    }
    Ok(())
}

/// Dispatch a two-basis `apply_and_project_to` call. Both bases must
/// belong to the same family (both `PyFermionBasis` for the bit path,
/// or both spin/boson/generic for the GenericBasis path) — mixing is
/// rejected at the boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_apply_and_project_to<OP>(
    op: &OP,
    input_basis: &Bound<'_, PyAny>,
    output_basis: &Bound<'_, PyAny>,
    coeffs: &[Complex<f64>],
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
    overwrite: bool,
) -> PyResult<()>
where
    OP: OperatorDispatch,
{
    let in_is_fermion = input_basis.is_instance_of::<PyFermionBasis>();
    let out_is_fermion = output_basis.is_instance_of::<PyFermionBasis>();

    if in_is_fermion != out_is_fermion {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "input and output basis must both be FermionBasis or both not be FermionBasis",
        ));
    }

    if in_is_fermion {
        let in_b = input_basis.downcast::<PyFermionBasis>()?.borrow();
        let out_b = output_basis.downcast::<PyFermionBasis>()?.borrow();
        op.apply_and_project_to_bit(
            &in_b.inner.inner,
            &out_b.inner.inner,
            coeffs,
            input,
            output,
            overwrite,
        )
        .map_err(Error::from)?;
    } else {
        // Both bases must resolve to GenericBasis. Hold both PyRefs so the
        // borrows live as long as the call.
        let in_spin = input_basis
            .downcast::<PySpinBasis>()
            .ok()
            .map(|b| b.borrow());
        let in_boson = input_basis
            .downcast::<PyBosonBasis>()
            .ok()
            .map(|b| b.borrow());
        let in_generic = input_basis
            .downcast::<PyGenericBasis>()
            .ok()
            .map(|b| b.borrow());
        let in_space = pick_generic_basis(
            input_basis,
            in_spin.as_deref(),
            in_boson.as_deref(),
            in_generic.as_deref(),
        )?;
        let out_spin = output_basis
            .downcast::<PySpinBasis>()
            .ok()
            .map(|b| b.borrow());
        let out_boson = output_basis
            .downcast::<PyBosonBasis>()
            .ok()
            .map(|b| b.borrow());
        let out_generic = output_basis
            .downcast::<PyGenericBasis>()
            .ok()
            .map(|b| b.borrow());
        let out_space = pick_generic_basis(
            output_basis,
            out_spin.as_deref(),
            out_boson.as_deref(),
            out_generic.as_deref(),
        )?;
        op.apply_and_project_to(in_space, out_space, coeffs, input, output, overwrite)
            .map_err(Error::from)?;
    }
    Ok(())
}

fn pick_generic_basis<'a>(
    basis: &Bound<'_, PyAny>,
    spin: Option<&'a PySpinBasis>,
    boson: Option<&'a PyBosonBasis>,
    generic: Option<&'a PyGenericBasis>,
) -> PyResult<&'a GenericBasis> {
    if let Some(b) = spin {
        Ok(&b.inner.inner)
    } else if let Some(b) = boson {
        Ok(&b.inner.inner)
    } else if let Some(b) = generic {
        Ok(&b.inner)
    } else {
        let _ = basis;
        Err(pyo3::exceptions::PyTypeError::new_err(
            "basis must be SpinBasis, BosonBasis, or GenericBasis",
        ))
    }
}

// ---------------------------------------------------------------------------
// numpy helpers
// ---------------------------------------------------------------------------

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
