use crate::error::Error;
use crate::operator::{
    Terms, as_c64_vec, dispatch_apply, dispatch_apply_and_project_to, max_site_from_terms,
    parse_terms_generic, write_c64_back,
};
use numpy::{Complex64, PyArray1};
use pyo3::prelude::*;
use quspin_core::operator::fermion::{
    FermionOp, FermionOpEntry, FermionOperator, FermionOperatorInner,
};

/// Python-facing fermionic operator.
///
/// Same variadic `*terms` format as `PauliOperator`, using fermion op strings
/// (`+`, `-`, `n`).  Jordan-Wigner signs are handled automatically.
#[pyclass(name = "FermionOperator", module = "quspin._rs")]
pub struct PyFermionOperator {
    pub inner: FermionOperatorInner,
}

#[pymethods]
impl PyFermionOperator {
    #[new]
    #[pyo3(signature = (*terms))]
    fn new(py: Python<'_>, terms: Terms) -> PyResult<Self> {
        let max_cindex = terms.len().saturating_sub(1);
        let max_site = max_site_from_terms(py, &terms)?;

        let use_u8 = max_cindex <= 255 && max_site <= 255;

        if use_u8 {
            let entries =
                parse_terms_generic::<u8, FermionOp, _, _>(py, &terms, FermionOpEntry::new)
                    .map_err(Error::from)?;
            Ok(PyFermionOperator {
                inner: FermionOperatorInner::Ham8(FermionOperator::new(entries)),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let entries =
                parse_terms_generic::<u16, FermionOp, _, _>(py, &terms, FermionOpEntry::new)
                    .map_err(Error::from)?;
            Ok(PyFermionOperator {
                inner: FermionOperatorInner::Ham16(FermionOperator::new(entries)),
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

    fn __repr__(&self) -> String {
        format!(
            "FermionOperator(max_site={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.num_cindices(),
        )
    }
}
