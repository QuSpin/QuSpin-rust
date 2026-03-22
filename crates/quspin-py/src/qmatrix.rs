/// Python-facing `PyQMatrix` pyclass.
///
/// Wraps `QMatrixInner` (type-erased `QMatrix<V, i64, C>`) and exposes
/// `build`, `dot`, `dot_transpose`, `__add__`, `__sub__`, `dim`, and `nnz`.
///
/// ## Python API
///
/// ```python
/// mat = PyQMatrix.build(ham=H, basis=basis, dtype=np.float64)
///
/// out = mat.dot(coeff, input, output, overwrite=True)
/// out = mat.dot_transpose(coeff, input, output, overwrite=True)
///
/// mat3 = mat1 + mat2
/// mat3 = mat1 - mat2
///
/// mat.dim   # int
/// mat.nnz   # int
/// ```
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use quspin_core::qmatrix::build::{build_from_basis, build_from_symmetric};

use crate::basis::PyHardcoreBasis;
use crate::dispatch::{IntoQMatrixInner, QMatrixInner};
use crate::dtype::MatrixDType;
use crate::error::Error;
use crate::hamiltonian::{PauliHamiltonianInner, PyPauliHamiltonian};
use crate::{with_plain_basis, with_sym_basis, with_value_dtype};

// ---------------------------------------------------------------------------
// PyQMatrix
// ---------------------------------------------------------------------------

#[pyclass(name = "PyQMatrix")]
pub struct PyQMatrix {
    pub inner: QMatrixInner,
}

#[pymethods]
impl PyQMatrix {
    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    /// Build a sparse quantum matrix from a Hamiltonian and a basis.
    ///
    /// Args:
    ///   ham:   The `PyPauliHamiltonian` defining operator strings.
    ///   basis: The `PyHardcoreBasis` defining the Hilbert space.
    ///   dtype: NumPy dtype object for matrix element storage.
    #[staticmethod]
    pub fn build(
        py: Python<'_>,
        ham: &PyPauliHamiltonian,
        basis: &PyHardcoreBasis,
        dtype: &Bound<'_, numpy::PyArrayDescr>,
    ) -> PyResult<Self> {
        // Determine value dtype from the numpy dtype descriptor.
        let v_dtype = MatrixDType::from_descr(py, dtype).map_err(Error::from)?;

        // Dispatch over value type; cindex type is inferred from the Ham variant.
        let mat_inner = if basis.inner.is_symmetric() {
            // Symmetric basis: build_from_symmetric applies symmetry scaling.
            with_value_dtype!(v_dtype, V, {
                with_sym_basis!(&basis.inner, B, sym_basis, {
                    match &ham.inner {
                        PauliHamiltonianInner::Ham8(h) => {
                            build_from_symmetric::<B, V, i64, u8>(h, sym_basis).into_qmatrix_inner()
                        }
                        PauliHamiltonianInner::Ham16(h) => {
                            build_from_symmetric::<B, V, i64, u16>(h, sym_basis)
                                .into_qmatrix_inner()
                        }
                    }
                })
            })
        } else {
            // Full or subspace: build_from_basis (no symmetry scaling).
            with_value_dtype!(v_dtype, V, {
                with_plain_basis!(&basis.inner, B, plain_basis, {
                    match &ham.inner {
                        PauliHamiltonianInner::Ham8(h) => {
                            build_from_basis::<B, V, i64, u8, _>(h, plain_basis)
                                .into_qmatrix_inner()
                        }
                        PauliHamiltonianInner::Ham16(h) => {
                            build_from_basis::<B, V, i64, u16, _>(h, plain_basis)
                                .into_qmatrix_inner()
                        }
                    }
                })
            })
        };

        Ok(PyQMatrix { inner: mat_inner })
    }

    // ------------------------------------------------------------------
    // dot / dot_transpose
    // ------------------------------------------------------------------

    /// Compute `output[r] = Σ coeff[c] * value * input[col]` for each row.
    ///
    /// Args:
    ///   coeff:     1-D numpy array of length `num_coeff`.
    ///   input:     1-D numpy array of length `dim`.
    ///   output:    1-D numpy array of length `dim` (modified in-place).
    ///   overwrite: If `True`, zero `output` before accumulating.
    #[pyo3(signature = (coeff, input, output, overwrite=true))]
    pub fn dot(
        &self,
        coeff: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        overwrite: bool,
    ) -> PyResult<()> {
        use crate::with_qmatrix;
        with_qmatrix!(&self.inner, V, _C, mat, {
            let c = coeff.downcast::<PyArray1<V>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "coeff dtype does not match the matrix element type",
                )
            })?;
            let inp = input.downcast::<PyArray1<V>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "input dtype does not match the matrix element type",
                )
            })?;
            let out = output.downcast::<PyArray1<V>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "output dtype does not match the matrix element type",
                )
            })?;

            let c_ro = c.readonly();
            let inp_ro = inp.readonly();
            let c_slice = c_ro
                .as_slice()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let inp_slice = inp_ro
                .as_slice()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            // SAFETY: We hold the GIL and the only live borrows on `out` are
            // the readonly views above (which are on different arrays).
            let out_slice = unsafe { out.as_slice_mut() }
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            mat.dot(overwrite, c_slice, inp_slice, out_slice)
                .map_err(Error::from)?;
            Ok(())
        })
    }

    /// Compute `output[col] += Σ coeff[c] * value * input[r]` (transpose matvec).
    ///
    /// Args:
    ///   coeff:     1-D numpy array of length `num_coeff`.
    ///   input:     1-D numpy array of length `dim`.
    ///   output:    1-D numpy array of length `dim` (modified in-place).
    ///   overwrite: If `True`, zero `output` before accumulating.
    #[pyo3(signature = (coeff, input, output, overwrite=true))]
    pub fn dot_transpose(
        &self,
        coeff: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        overwrite: bool,
    ) -> PyResult<()> {
        use crate::with_qmatrix;
        with_qmatrix!(&self.inner, V, _C, mat, {
            let c = coeff.downcast::<PyArray1<V>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "coeff dtype does not match the matrix element type",
                )
            })?;
            let inp = input.downcast::<PyArray1<V>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "input dtype does not match the matrix element type",
                )
            })?;
            let out = output.downcast::<PyArray1<V>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "output dtype does not match the matrix element type",
                )
            })?;

            let c_ro = c.readonly();
            let inp_ro = inp.readonly();
            let c_slice = c_ro
                .as_slice()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let inp_slice = inp_ro
                .as_slice()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            // SAFETY: same as dot above.
            let out_slice = unsafe { out.as_slice_mut() }
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            mat.dot_transpose(overwrite, c_slice, inp_slice, out_slice)
                .map_err(Error::from)?;
            Ok(())
        })
    }

    // ------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------

    /// Element-wise addition (clones both operands).
    pub fn __add__(&self, other: &PyQMatrix) -> PyResult<PyQMatrix> {
        // We need to clone both sides to satisfy the owned Add impl.
        // TODO: add AddAssign / in-place ops to avoid double allocation.
        let lhs = clone_qmatrix_inner(&self.inner);
        let rhs = clone_qmatrix_inner(&other.inner);
        let result = lhs.try_add(rhs)?;
        Ok(PyQMatrix { inner: result })
    }

    /// Element-wise subtraction (clones both operands).
    pub fn __sub__(&self, other: &PyQMatrix) -> PyResult<PyQMatrix> {
        let lhs = clone_qmatrix_inner(&self.inner);
        let rhs = clone_qmatrix_inner(&other.inner);
        let result = lhs.try_sub(rhs)?;
        Ok(PyQMatrix { inner: result })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    /// Matrix dimension (number of rows / columns).
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Number of stored non-zero entries.
    #[getter]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Clone a `QMatrixInner` by delegating to `QMatrixInner: Clone`.
fn clone_qmatrix_inner(inner: &QMatrixInner) -> QMatrixInner {
    inner.clone()
}
