//! Python bindings for the matrix-free `(operator, basis)` linear operator.
//!
//! [`PyOperatorLinearOperator`] is the matrix-free twin of
//! [`PyQMatrixLinearOperator`](crate::linear_operator::PyQMatrixLinearOperator):
//! same SciPy duck-typed surface, but nothing is assembled. It is built by
//! `operator.as_linearoperator(basis, coeffs)` and is the right choice when
//! the `QMatrix` would not fit in memory.
//!
//! The concrete `OperatorOnBasis<OP, B>` stays statically dispatched inside
//! `quspin-matrix`; the type erasure to `dyn LinearOperator` happens here, at
//! the FFI boundary, where `DynLinearOperator` already lives.

use std::sync::Arc;

use ndarray::Array2;
use num_complex::Complex;
use numpy::{
    Complex64, PyArray1, PyArray2, PyArrayDescr, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use quspin_core::{BasisSource, LinearOperator, OperatorOnBasis};

use crate::basis::{PyBosonBasis, PyFermionBasis, PyGenericBasis, PySpinBasis};
use crate::error::Error;
use crate::operator::{
    PyBondOperator, PyBosonOperator, PyFermionOperator, PyMonomialOperator, PyPauliOperator,
    PySpinOperator,
};

type C64 = Complex<f64>;

/// Type-erased matrix-free operator handed to Python and to `ExpmOp`.
pub type SharedLinearOperator = Arc<dyn LinearOperator<C64> + Send + Sync>;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Resolve a Python basis object to a shared [`BasisSource`].
fn basis_source(basis: &Bound<'_, PyAny>) -> PyResult<Arc<dyn BasisSource>> {
    if let Ok(b) = basis.cast::<PySpinBasis>() {
        return Ok(Arc::clone(&b.borrow().inner) as Arc<dyn BasisSource>);
    }
    if let Ok(b) = basis.cast::<PyBosonBasis>() {
        return Ok(Arc::clone(&b.borrow().inner) as Arc<dyn BasisSource>);
    }
    if let Ok(b) = basis.cast::<PyFermionBasis>() {
        return Ok(Arc::clone(&b.borrow().inner) as Arc<dyn BasisSource>);
    }
    if let Ok(b) = basis.cast::<PyGenericBasis>() {
        return Ok(Arc::clone(&b.borrow().inner) as Arc<dyn BasisSource>);
    }
    Err(PyTypeError::new_err(
        "basis must be SpinBasis, FermionBasis, BosonBasis, or GenericBasis",
    ))
}

/// Bundle a Python operator + basis + coefficient snapshot into a shared,
/// type-erased matrix-free [`LinearOperator`].
pub(crate) fn build_matrix_free(
    op: &Bound<'_, PyAny>,
    basis: &Bound<'_, PyAny>,
    coeffs: Vec<C64>,
) -> PyResult<SharedLinearOperator> {
    let source = basis_source(basis)?;

    // One arm per operator type keeps operator dispatch static; the basis is
    // already erased behind `dyn BasisSource`.
    macro_rules! try_op {
        ($($py_ty:ty),+ $(,)?) => {
            $(
                if let Ok(o) = op.cast::<$py_ty>() {
                    let inner = o.borrow().inner.clone();
                    let wrapped = OperatorOnBasis::new(inner, source, coeffs)
                        .map_err(Error::from)?;
                    return Ok(Arc::new(wrapped) as SharedLinearOperator);
                }
            )+
        };
    }

    try_op!(
        PyPauliOperator,
        PySpinOperator,
        PyBosonOperator,
        PyFermionOperator,
        PyBondOperator,
        PyMonomialOperator,
    );

    Err(PyTypeError::new_err(
        "op must be a PauliOperator, SpinOperator, BosonOperator, FermionOperator, \
         BondOperator, or MonomialOperator",
    ))
}

// ---------------------------------------------------------------------------
// PyOperatorLinearOperator
// ---------------------------------------------------------------------------

/// Matrix-free `LinearOperator` over an `(operator, basis)` pair.
///
/// Exposes the SciPy `LinearOperator` duck-typed interface (`shape`, `dtype`,
/// `matvec`, `matmat`, `rmatvec`, `rmatmat`, `@`) so instances can be passed
/// straight to `scipy.sparse.linalg`, plus `trace` and `onenorm`.
///
/// Construct with `operator.as_linearoperator(basis, coeffs)`; not
/// constructible from Python directly.
#[pyclass(name = "OperatorLinearOperator", module = "quspin_rs._rs", frozen)]
pub struct PyOperatorLinearOperator {
    pub inner: SharedLinearOperator,
}

impl PyOperatorLinearOperator {
    fn check_len(&self, len: usize) -> PyResult<usize> {
        let n = self.inner.dim();
        if len != n {
            return Err(PyValueError::new_err(format!(
                "x must have length {n}, got {len}",
            )));
        }
        Ok(n)
    }
}

#[pymethods]
impl PyOperatorLinearOperator {
    /// Opt out of numpy ufuncs so `np.ndarray @ qop` defers to our
    /// `__rmatmul__` instead of numpy attempting an elementwise op.
    #[classattr]
    fn __array_ufunc__(py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// SciPy-compatible ``(dim, dim)`` shape tuple.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        let n = self.inner.dim();
        (n, n)
    }

    /// SciPy-compatible ``numpy.dtype``.  Always ``complex128``.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        numpy::dtype::<Complex64>(py)
    }

    /// ``trace(A)``, computed in one sweep over the basis.
    fn trace(&self) -> Complex64 {
        self.inner.trace()
    }

    /// ``‖A − shift·I‖₁`` (column 1-norm), computed in one sweep.
    #[pyo3(signature = (shift = Complex64::new(0.0, 0.0)))]
    fn onenorm(&self, shift: Complex64) -> f64 {
        self.inner.onenorm(shift)
    }

    /// ``A @ x`` for a 1-D complex128 input.
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let n = self.check_len(x.len())?;
        let in_slice = x
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = PyArray1::<Complex64>::zeros(py, [n], false);
        {
            let mut rw = out.try_readwrite()?;
            let mut view = rw.as_array_mut();
            let out_slice = view
                .as_slice_mut()
                .expect("freshly-allocated PyArray1 is contiguous");
            py.detach(|| self.inner.dot(true, in_slice, out_slice))
                .map_err(Error::from)?;
        }
        Ok(out)
    }

    /// ``A @ X`` for a 2-D complex128 input of shape ``(dim, k)``.
    fn matmat<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let n = self.inner.dim();
        let shape = x.shape();
        if shape[0] != n {
            return Err(PyValueError::new_err(format!(
                "x.shape[0] must be {n}, got {}",
                shape[0],
            )));
        }
        let k = shape[1];
        let in_view = x.as_array();
        let out = PyArray2::<Complex64>::zeros(py, [n, k], false);
        {
            let mut rw = out.try_readwrite()?;
            let out_view = rw.as_array_mut();
            py.detach(|| self.inner.dot_many(true, in_view, out_view))
                .map_err(Error::from)?;
        }
        Ok(out)
    }

    /// ``A^H @ x`` for a 1-D input — SciPy `rmatvec`.  Computed as
    /// ``conj(A^T @ conj(x))`` so it is the Hermitian adjoint even when the
    /// operator has complex matrix elements.
    fn rmatvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let n = self.check_len(x.len())?;
        let in_conj: Vec<C64> = x.as_array().iter().map(|c| c.conj()).collect();
        let out = PyArray1::<Complex64>::zeros(py, [n], false);
        {
            let mut rw = out.try_readwrite()?;
            let mut view = rw.as_array_mut();
            let out_slice = view
                .as_slice_mut()
                .expect("freshly-allocated PyArray1 is contiguous");
            py.detach(|| self.inner.dot_transpose(true, &in_conj, out_slice))
                .map_err(Error::from)?;
            for v in out_slice.iter_mut() {
                *v = v.conj();
            }
        }
        Ok(out)
    }

    /// ``A^H @ X`` for a 2-D input of shape ``(dim, k)`` — SciPy `rmatmat`.
    fn rmatmat<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let n = self.inner.dim();
        let shape = x.shape();
        if shape[0] != n {
            return Err(PyValueError::new_err(format!(
                "x.shape[0] must be {n}, got {}",
                shape[0],
            )));
        }
        let k = shape[1];
        let x_arr = x.as_array();
        let out = PyArray2::<Complex64>::zeros(py, [n, k], false);
        {
            let mut rw = out.try_readwrite()?;
            let mut view = rw.as_array_mut();
            // No batched transpose on the matrix-free path — go column by
            // column through `dot_transpose`.
            let mut in_col = vec![C64::default(); n];
            let mut out_col = vec![C64::default(); n];
            for c in 0..k {
                for (dst, src) in in_col.iter_mut().zip(x_arr.column(c).iter()) {
                    *dst = src.conj();
                }
                py.detach(|| self.inner.dot_transpose(true, &in_col, &mut out_col))
                    .map_err(Error::from)?;
                for (dst, src) in view.column_mut(c).iter_mut().zip(out_col.iter()) {
                    *dst = src.conj();
                }
            }
        }
        Ok(out)
    }

    /// ``A @ x`` — dispatches to `matvec` (1-D) or `matmat` (2-D).
    fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(x) = other.extract::<PyReadonlyArray1<'py, Complex64>>() {
            return Ok(self.matvec(py, x)?.into_any());
        }
        if let Ok(x) = other.extract::<PyReadonlyArray2<'py, Complex64>>() {
            return Ok(self.matmat(py, x)?.into_any());
        }
        Err(PyTypeError::new_err(
            "OperatorLinearOperator @ x: x must be a 1-D or 2-D complex128 ndarray",
        ))
    }

    /// ``x @ A`` — for a 1-D input returns ``A^T @ x``; for a 2-D ``(m, dim)``
    /// input returns ``x @ A``.  Plain transpose, not the Hermitian adjoint,
    /// matching numpy's ``@`` semantics.
    fn __rmatmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.dim();
        if let Ok(x) = other.extract::<PyReadonlyArray1<'py, Complex64>>() {
            self.check_len(x.len())?;
            let in_slice = x
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let out = PyArray1::<Complex64>::zeros(py, [n], false);
            {
                let mut rw = out.try_readwrite()?;
                let mut view = rw.as_array_mut();
                let out_slice = view
                    .as_slice_mut()
                    .expect("freshly-allocated PyArray1 is contiguous");
                py.detach(|| self.inner.dot_transpose(true, in_slice, out_slice))
                    .map_err(Error::from)?;
            }
            return Ok(out.into_any());
        }
        if let Ok(x) = other.extract::<PyReadonlyArray2<'py, Complex64>>() {
            let shape = x.shape();
            if shape[1] != n {
                return Err(PyValueError::new_err(format!(
                    "x.shape[1] must be {n}, got {}",
                    shape[1],
                )));
            }
            let m = shape[0];
            let x_arr = x.as_array();
            let mut tmp = Array2::<C64>::zeros((m, n));
            let mut in_row = vec![C64::default(); n];
            let mut out_row = vec![C64::default(); n];
            for r in 0..m {
                for (dst, src) in in_row.iter_mut().zip(x_arr.row(r).iter()) {
                    *dst = *src;
                }
                py.detach(|| self.inner.dot_transpose(true, &in_row, &mut out_row))
                    .map_err(Error::from)?;
                for (dst, src) in tmp.row_mut(r).iter_mut().zip(out_row.iter()) {
                    *dst = *src;
                }
            }
            let out = PyArray2::<Complex64>::zeros(py, [m, n], false);
            {
                let mut rw = out.try_readwrite()?;
                let mut view = rw.as_array_mut();
                view.assign(&tmp);
            }
            return Ok(out.into_any());
        }
        Err(PyTypeError::new_err(
            "x @ OperatorLinearOperator: x must be a 1-D or 2-D complex128 ndarray",
        ))
    }

    fn __repr__(&self) -> String {
        let n = self.inner.dim();
        format!("OperatorLinearOperator(shape=({n}, {n}), dtype=complex128, matrix_free=True)")
    }
}

// ---------------------------------------------------------------------------
// Shared `as_linearoperator` entry point
// ---------------------------------------------------------------------------

/// Backing implementation of `<Operator>.as_linearoperator(basis, coeffs)`.
///
/// Each operator pyclass forwards here so the six wrappers stay one line
/// apiece.
pub(crate) fn as_linearoperator(
    op: &Bound<'_, PyAny>,
    basis: &Bound<'_, PyAny>,
    coeffs: PyReadonlyArray1<'_, Complex64>,
) -> PyResult<PyOperatorLinearOperator> {
    let coeffs_vec: Vec<C64> = coeffs
        .as_array()
        .iter()
        .map(|c| C64::new(c.re, c.im))
        .collect();
    Ok(PyOperatorLinearOperator {
        inner: build_matrix_free(op, basis, coeffs_vec)?,
    })
}
