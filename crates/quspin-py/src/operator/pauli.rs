use crate::error::Error;
use crate::operator::{
    Terms, as_c64_vec, max_site_from_terms, parse_terms_generic, with_space_inner,
    with_two_space_inners, write_c64_back,
};
use numpy::{Complex64, PyArray1};
use pyo3::prelude::*;
use quspin_core::OperatorDispatch;
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

        with_two_space_inners(input_basis, output_basis, |in_space, out_space| {
            self.inner
                .apply_and_project_to(
                    in_space,
                    out_space,
                    &coeffs_vec,
                    &input_vec,
                    &mut output_vec,
                    overwrite,
                )
                .map_err(Error::from)
        })??;

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

        with_space_inner(basis, |space| {
            self.inner
                .apply(space, &coeffs_vec, &input_vec, &mut output_vec, overwrite)
                .map_err(Error::from)
        })??;

        unsafe { write_c64_back(output, &output_vec) };
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliOperator(max_site={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.num_cindices(),
        )
    }
}
