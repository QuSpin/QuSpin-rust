use crate::error::Error;
use crate::operator::{
    Terms, as_c64_vec, dispatch_apply, dispatch_apply_and_project_to, max_site_from_terms,
    parse_terms_generic, write_c64_back,
};
use numpy::{Complex64, PyArray1};
use pyo3::prelude::*;
use quspin_core::operator::boson::{BosonOp, BosonOpEntry, BosonOperator, BosonOperatorInner};

/// Python-facing bosonic operator (truncated harmonic oscillator).
///
/// Same variadic `*terms` format as `PauliOperator`, using boson op strings
/// (`+`, `-`, `n`).  `lhss` (local Hilbert-space size) is keyword-only.
#[pyclass(name = "BosonOperator", module = "quspin._rs")]
pub struct PyBosonOperator {
    pub inner: BosonOperatorInner,
}

#[pymethods]
impl PyBosonOperator {
    #[new]
    #[pyo3(signature = (*terms, lhss))]
    fn new(py: Python<'_>, terms: Terms, lhss: usize) -> PyResult<Self> {
        if lhss < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("lhss must be >= 2"));
        }
        let max_cindex = terms.len().saturating_sub(1);
        let max_site = max_site_from_terms(py, &terms)?;

        let use_u8 = max_cindex <= 255 && max_site <= 255;

        if use_u8 {
            let entries = parse_terms_generic::<u8, BosonOp, _, _>(py, &terms, BosonOpEntry::new)
                .map_err(Error::from)?;
            Ok(PyBosonOperator {
                inner: BosonOperatorInner::Ham8(BosonOperator::new(entries, lhss)),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let entries = parse_terms_generic::<u16, BosonOp, _, _>(py, &terms, BosonOpEntry::new)
                .map_err(Error::from)?;
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
            "BosonOperator(max_site={}, lhss={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.lhss(),
            self.inner.num_cindices(),
        )
    }
}
