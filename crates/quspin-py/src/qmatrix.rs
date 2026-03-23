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
use quspin_core::qmatrix::dispatch::{IntoQMatrixInner, QMatrixInner};
use quspin_core::{with_plain_basis, with_sym_basis};

use crate::basis::hardcore::PyHardcoreBasis;
use crate::dtype::{FromPyDescr, ValueDType};
use crate::error::Error;
use crate::hamiltonian::PyHardcoreHamiltonian;
use quspin_core::hamiltonian::hardcore::dispatch::HardcoreHamiltonianInner;
use quspin_core::with_value_dtype;

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

    /// Build a sparse matrix from a Hamiltonian and a basis.
    ///
    /// Args:
    ///     ham (PyHardcoreHamiltonian): The Hamiltonian defining operator
    ///         strings and coupling coefficients.
    ///     basis (PyHardcoreBasis): The Hilbert space basis (full, subspace,
    ///         or symmetric).
    ///     dtype (numpy.dtype): NumPy dtype for matrix element storage.
    ///         Supported: ``int8``, ``int16``, ``float32``, ``float64``,
    ///         ``complex64``, ``complex128``.
    ///
    /// Returns:
    ///     PyQMatrix: Sparse matrix representation of the Hamiltonian.
    ///
    /// Raises:
    ///     ValueError: If ``ham.n_sites != basis.n_sites``, or if ``dtype``
    ///         is not supported.
    #[staticmethod]
    pub fn build_hardcore_hamiltonian(
        py: Python<'_>,
        ham: &PyHardcoreHamiltonian,
        basis: &PyHardcoreBasis,
        dtype: &Bound<'_, numpy::PyArrayDescr>,
    ) -> PyResult<Self> {
        if basis.inner.n_sites() != ham.inner.n_sites() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites mismatch: basis has {} sites but Hamiltonian has {}",
                basis.inner.n_sites(),
                ham.inner.n_sites()
            )));
        }
        let v_dtype = <ValueDType as FromPyDescr>::from_descr(py, dtype).map_err(Error::from)?;

        let mat_inner = if basis.inner.is_symmetric() {
            with_value_dtype!(v_dtype, V, {
                with_sym_basis!(&basis.inner, B, sym_basis, {
                    match &ham.inner {
                        HardcoreHamiltonianInner::Ham8(h) => {
                            build_from_symmetric::<B, V, i64, u8>(h, sym_basis).into_qmatrix_inner()
                        }
                        HardcoreHamiltonianInner::Ham16(h) => {
                            build_from_symmetric::<B, V, i64, u16>(h, sym_basis)
                                .into_qmatrix_inner()
                        }
                    }
                })
            })
        } else {
            with_value_dtype!(v_dtype, V, {
                with_plain_basis!(&basis.inner, B, plain_basis, {
                    match &ham.inner {
                        HardcoreHamiltonianInner::Ham8(h) => {
                            build_from_basis::<B, V, i64, u8, _>(h, plain_basis)
                                .into_qmatrix_inner()
                        }
                        HardcoreHamiltonianInner::Ham16(h) => {
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

    /// Compute a matrix-vector product, accumulating into ``output``.
    ///
    /// Computes ``output[row] = Σ_c coeff[c] * Σ_col M[c, row, col] * input[col]``.
    ///
    /// Args:
    ///     coeff (NDArray): 1-D array of length ``num_cindices``. dtype must
    ///         match the matrix element type.
    ///     input (NDArray): 1-D array of length ``dim``. dtype must match the
    ///         matrix element type.
    ///     output (NDArray): 1-D array of length ``dim``, modified in-place.
    ///         dtype must match the matrix element type.
    ///     overwrite (bool): If ``True`` (default), zero ``output`` before
    ///         accumulating. If ``False``, add to existing values.
    ///
    /// Raises:
    ///     TypeError: If any array dtype does not match the matrix element type.
    ///     ValueError: If any array is not C-contiguous or has the wrong shape.
    #[pyo3(signature = (coeff, input, output, overwrite=true))]
    pub fn dot(
        &self,
        coeff: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        overwrite: bool,
    ) -> PyResult<()> {
        use quspin_core::with_qmatrix;
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

    /// Compute a transpose matrix-vector product, accumulating into ``output``.
    ///
    /// Computes ``output[col] = Σ_c coeff[c] * Σ_row M[c, row, col] * input[row]``.
    ///
    /// Args:
    ///     coeff (NDArray): 1-D array of length ``num_cindices``. dtype must
    ///         match the matrix element type.
    ///     input (NDArray): 1-D array of length ``dim``. dtype must match the
    ///         matrix element type.
    ///     output (NDArray): 1-D array of length ``dim``, modified in-place.
    ///         dtype must match the matrix element type.
    ///     overwrite (bool): If ``True`` (default), zero ``output`` before
    ///         accumulating. If ``False``, add to existing values.
    ///
    /// Raises:
    ///     TypeError: If any array dtype does not match the matrix element type.
    ///     ValueError: If any array is not C-contiguous or has the wrong shape.
    #[pyo3(signature = (coeff, input, output, overwrite=true))]
    pub fn dot_transpose(
        &self,
        coeff: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        overwrite: bool,
    ) -> PyResult<()> {
        use quspin_core::with_qmatrix;
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
    // to_csr
    // ------------------------------------------------------------------

    /// Materialise the matrix as a plain CSR sparse matrix.
    ///
    /// Multiplies stored values by ``coeff`` and sums entries that share the
    /// same ``(row, col)`` position across different operator strings.
    ///
    /// Args:
    ///     coeff (NDArray): 1-D array of length ``num_cindices``. dtype must
    ///         match the matrix element type.
    ///     drop_zeros (bool): If ``True`` (default), omit entries whose
    ///         accumulated value is exactly zero from the output.
    ///
    /// Returns:
    ///     tuple[NDArray, NDArray, NDArray]: ``(indptr, indices, data)`` arrays
    ///     suitable for constructing a ``scipy.sparse.csr_array``:
    ///
    ///     .. code-block:: python
    ///
    ///         ip, idx, d = mat.to_csr(coeff)
    ///         A = scipy.sparse.csr_array((d, idx, ip), shape=(mat.dim, mat.dim))
    ///
    ///     - ``indptr``: int64, length ``dim + 1``
    ///     - ``indices``: int64, length ``nnz``
    ///     - ``data``: same dtype as the matrix, length ``nnz``
    ///
    /// Raises:
    ///     TypeError: If ``coeff`` dtype does not match the matrix element type.
    ///     ValueError: If ``coeff`` has the wrong length or is not C-contiguous.
    #[pyo3(signature = (coeff, drop_zeros = true))]
    pub fn to_csr<'py>(
        &self,
        py: Python<'py>,
        coeff: &Bound<'_, PyAny>,
        drop_zeros: bool,
    ) -> PyResult<PyObject> {
        use numpy::PyArray1;
        use quspin_core::with_qmatrix;
        with_qmatrix!(&self.inner, V, _C, mat, {
            let c = coeff.downcast::<PyArray1<V>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "coeff dtype does not match the matrix element type",
                )
            })?;
            let c_ro = c.readonly();
            let c_slice = c_ro
                .as_slice()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            let (indptr, indices, data) = mat.to_csr(c_slice, drop_zeros).map_err(Error::from)?;

            let indptr_arr = PyArray1::from_vec(py, indptr);
            let indices_arr = PyArray1::from_vec(py, indices);
            let data_arr = PyArray1::from_vec(py, data);

            Ok(pyo3::types::PyTuple::new(
                py,
                [
                    indptr_arr.into_any(),
                    indices_arr.into_any(),
                    data_arr.into_any(),
                ],
            )?
            .into())
        })
    }

    /// Materialise the matrix as a dense 2-D NumPy array.
    ///
    /// Returns a C-contiguous ``(dim, dim)`` array where element
    /// ``[r, col] = Σ_c coeff[c] * M[c, r, col]``.
    ///
    /// Args:
    ///     coeff (NDArray): 1-D array of length ``num_cindices``. dtype must
    ///         match the matrix element type.
    ///
    /// Returns:
    ///     NDArray: 2-D array of shape ``(dim, dim)`` with the same dtype as
    ///     the matrix.
    ///
    /// Raises:
    ///     TypeError: If ``coeff`` dtype does not match the matrix element type.
    ///     ValueError: If ``coeff`` has the wrong length or is not C-contiguous.
    pub fn to_dense<'py>(&self, py: Python<'py>, coeff: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        use numpy::{PyArray2, PyArrayMethods};
        use quspin_core::with_qmatrix;
        with_qmatrix!(&self.inner, V, _C, mat, {
            let c = coeff.downcast::<PyArray1<V>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "coeff dtype does not match the matrix element type",
                )
            })?;
            let c_ro = c.readonly();
            let c_slice = c_ro
                .as_slice()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            let flat = mat.to_dense(c_slice).map_err(Error::from)?;
            let dim = mat.dim();

            let arr2d = numpy::ndarray::Array2::from_shape_vec((dim, dim), flat).map_err(
                |e: numpy::ndarray::ShapeError| {
                    pyo3::exceptions::PyValueError::new_err(e.to_string())
                },
            )?;
            let arr = PyArray2::from_owned_array(py, arr2d);
            Ok(arr.into_any().into())
        })
    }

    // ------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------

    /// Return the element-wise sum of two matrices.
    ///
    /// Args:
    ///     other (PyQMatrix): Matrix to add. Must have the same dtype and
    ///         dimension.
    ///
    /// Returns:
    ///     PyQMatrix: New matrix equal to ``self + other``.
    ///
    /// Raises:
    ///     ValueError: If the matrices have incompatible dtypes or dimensions.
    pub fn __add__(&self, other: &PyQMatrix) -> PyResult<PyQMatrix> {
        let result = self
            .inner
            .clone()
            .try_add(other.inner.clone())
            .map_err(Error)?;
        Ok(PyQMatrix { inner: result })
    }

    /// Return the element-wise difference of two matrices.
    ///
    /// Args:
    ///     other (PyQMatrix): Matrix to subtract. Must have the same dtype and
    ///         dimension.
    ///
    /// Returns:
    ///     PyQMatrix: New matrix equal to ``self - other``.
    ///
    /// Raises:
    ///     ValueError: If the matrices have incompatible dtypes or dimensions.
    pub fn __sub__(&self, other: &PyQMatrix) -> PyResult<PyQMatrix> {
        let result = self
            .inner
            .clone()
            .try_sub(other.inner.clone())
            .map_err(Error)?;
        Ok(PyQMatrix { inner: result })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    /// Matrix dimension (number of rows and columns).
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Number of stored non-zero entries.
    #[getter]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyQMatrix(dim={}, nnz={}, dtype={})",
            self.inner.dim(),
            self.inner.nnz(),
            self.inner.dtype_name(),
        )
    }
}
