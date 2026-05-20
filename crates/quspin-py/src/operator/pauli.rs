use crate::error::Error;
use crate::operator::{
    Terms, as_c64_vec, dispatch_apply, dispatch_apply_and_project_to, max_site_from_terms,
    parse_terms_generic, write_c64_back,
};
use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyArrayDescr, PyArrayDescrMethods, ToPyArray};
use pyo3::prelude::*;
use quspin_core::operator::pauli::{HardcoreOp, HardcoreOperator, HardcoreOperatorInner, OpEntry};

/// Python-facing Pauli / hardcore-boson / spin-½ operator.
///
/// Each positional argument is a *term* (sharing one coupling-constant index).
/// A term is a list of `(op_str, bonds)` pairs.  Each bond is
/// `[coeff, site0, site1, ...]`.
///
/// Example (XX + ZZ nearest-neighbour chain on 4 sites, two separate terms):
/// ```python
/// bonds = [[1.0, 0, 1], [1.0, 1, 2], [1.0, 2, 3]]
/// # Two cindices — XX and ZZ can have independent coefficients:
/// op = PauliOperator([("XX", bonds)], [("ZZ", bonds)])
///
/// # One cindex — XX and ZZ always share the same coefficient:
/// op = PauliOperator([("XX", bonds), ("ZZ", bonds)])
/// ```
#[pyclass(name = "PauliOperator", module = "quspin._rs")]
pub struct PyPauliOperator {
    pub inner: HardcoreOperatorInner,
}

#[pymethods]
impl PyPauliOperator {
    /// Construct from one or more terms (variadic).
    ///
    /// Each positional argument is a list of `(op_str, bonds)` pairs that
    /// share the same cindex.  Each bond is `[coeff, site0, site1, ...]`.
    #[new]
    #[pyo3(signature = (*terms))]
    fn new(py: Python<'_>, terms: Terms) -> PyResult<Self> {
        let max_cindex = terms.len().saturating_sub(1);
        let max_site = max_site_from_terms(py, &terms)?;

        let use_u8 = max_cindex <= 255 && max_site <= 255;

        if use_u8 {
            let entries = parse_terms_generic::<u8, HardcoreOp, _, _>(py, &terms, OpEntry::new)
                .map_err(Error::from)?;
            Ok(PyPauliOperator {
                inner: HardcoreOperatorInner::Ham8(HardcoreOperator::new(entries)),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let entries = parse_terms_generic::<u16, HardcoreOp, _, _>(py, &terms, OpEntry::new)
                .map_err(Error::from)?;
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

    /// Apply operator to a vector in ``input_basis`` and project into ``output_basis``.
    #[pyo3(signature = (input_basis, output_basis, coeffs, input, output, overwrite = true))]
    #[allow(clippy::too_many_arguments)]
    fn apply_and_project_to(
        &self,
        input_basis: &Bound<'_, PyAny>,
        output_basis: &Bound<'_, PyAny>,
        coeffs: &Bound<'_, PyArray1<Complex64>>,
        input: &Bound<'_, PyArray1<Complex64>>,
        output: &Bound<'_, PyArray1<Complex64>>,
        overwrite: bool,
    ) -> PyResult<()> {
        let coeffs_vec = unsafe { as_c64_vec(coeffs) };
        let input_vec = unsafe { as_c64_vec(input) };
        let mut output_vec = unsafe { as_c64_vec(output) };

        dispatch_apply_and_project_to(
            &self.inner,
            input_basis,
            output_basis,
            &coeffs_vec,
            &input_vec,
            &mut output_vec,
            overwrite,
        )?;

        unsafe { write_c64_back(output, &output_vec) };
        Ok(())
    }

    /// Apply operator to a vector, projecting back into the same basis.
    #[pyo3(signature = (basis, coeffs, input, output, overwrite = true))]
    fn apply(
        &self,
        basis: &Bound<'_, PyAny>,
        coeffs: &Bound<'_, PyArray1<Complex64>>,
        input: &Bound<'_, PyArray1<Complex64>>,
        output: &Bound<'_, PyArray1<Complex64>>,
        overwrite: bool,
    ) -> PyResult<()> {
        let coeffs_vec = unsafe { as_c64_vec(coeffs) };
        let input_vec = unsafe { as_c64_vec(input) };
        let mut output_vec = unsafe { as_c64_vec(output) };

        dispatch_apply(
            &self.inner,
            basis,
            &coeffs_vec,
            &input_vec,
            &mut output_vec,
            overwrite,
        )?;

        unsafe { write_c64_back(output, &output_vec) };
        Ok(())
    }

    /// Materialise rows ``[row_start, row_end)`` as CSR without building a
    /// global QMatrix.  Designed for petsc4py distributed assembly: each
    /// MPI rank calls this with its locally-owned row range.
    ///
    /// Args:
    ///     basis:      ``SpinBasis`` or ``FermionBasis``.
    ///     coeffs:     1-D complex128 array of length ``num_cindices``.
    ///     row_start:  inclusive, 0-based.
    ///     row_end:    exclusive.
    ///     dtype:      output value dtype (default complex128).
    ///     drop_zeros: omit entries with ``|acc| <= scale * 4 * f64::EPSILON``.
    ///
    /// Returns:
    ///     ``(indptr, indices, data)`` as numpy arrays.  ``indptr`` length
    ///     ``row_end - row_start + 1``, indices are **global** column
    ///     indices (zero-based).  Both indptr and indices are int64; the
    ///     petsc4py user does ``arr.astype(PETSc.IntType, copy=False)``.
    #[pyo3(signature = (basis, coeffs, row_start, row_end, dtype, drop_zeros = true))]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn csr_slab<'py>(
        &self,
        py: Python<'py>,
        basis: &Bound<'py, PyAny>,
        coeffs: numpy::PyReadonlyArray1<'py, Complex64>,
        row_start: usize,
        row_end: usize,
        dtype: &Bound<'py, PyArrayDescr>,
        drop_zeros: bool,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<i64>>,
        Bound<'py, numpy::PyArray1<i64>>,
        Bound<'py, pyo3::types::PyAny>,
    )> {
        use crate::basis::fermion::PyFermionBasis;
        use crate::basis::spin::PySpinBasis;
        use quspin_core::{csr_slab_pauli_bit, csr_slab_pauli_generic};

        // Convert numpy::Complex64 -> num_complex::Complex<f64> (same memory layout).
        let coeffs_vec: Vec<Complex<f64>> = coeffs
            .as_array()
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();

        // Dispatch on basis type.
        let (indptr, indices, data) = if let Ok(b) = basis.cast::<PySpinBasis>() {
            let basis_ref = b.borrow();
            if !basis_ref.inner.inner.is_built() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "basis must be built (call .full / .subspace / .symmetric first)",
                ));
            }
            csr_slab_pauli_generic(
                &self.inner,
                &basis_ref.inner.inner,
                &coeffs_vec,
                row_start,
                row_end,
                drop_zeros,
            )
            .map_err(Error::from)?
        } else if let Ok(b) = basis.cast::<PyFermionBasis>() {
            let basis_ref = b.borrow();
            if !basis_ref.inner.inner.is_built() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "basis must be built (call .full / .subspace / .symmetric first)",
                ));
            }
            csr_slab_pauli_bit(
                &self.inner,
                &basis_ref.inner.inner,
                &coeffs_vec,
                row_start,
                row_end,
                drop_zeros,
            )
            .map_err(Error::from)?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "basis must be SpinBasis or FermionBasis for PauliOperator.csr_slab",
            ));
        };

        // Convert to numpy arrays.
        let indptr_py = indptr.to_pyarray(py);
        let indices_py = indices.to_pyarray(py);

        // Output dtype: build a complex128 numpy array and `astype` if user
        // asked for something else.  For complex128 (the common case) this
        // is a no-op cast.
        let data_c128: Bound<'py, numpy::PyArray1<Complex64>> = data
            .iter()
            .map(|c| numpy::Complex64::new(c.re, c.im))
            .collect::<Vec<_>>()
            .to_pyarray(py);
        let data_out: Bound<'py, pyo3::types::PyAny> =
            if dtype.is_equiv_to(&numpy::dtype::<Complex64>(py)) {
                data_c128.into_any()
            } else {
                data_c128.call_method1("astype", (dtype.clone(),))?
            };

        Ok((indptr_py, indices_py, data_out))
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliOperator(max_site={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.num_cindices(),
        )
    }
}
