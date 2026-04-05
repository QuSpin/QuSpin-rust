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
use quspin_core::basis::dispatch::SpaceInner;

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
