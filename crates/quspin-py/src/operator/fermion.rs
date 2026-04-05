use crate::error::Error;
use crate::operator::pauli::{Terms, extract_coeff, max_site_from_terms};
use crate::operator::{as_c64_vec, with_space_inner, with_two_space_inners, write_c64_back};
use numpy::{Complex64, PyArray1};
use pyo3::prelude::*;
use quspin_core::operator::fermion::{
    FermionOp, FermionOpEntry, FermionOperator, FermionOperatorInner,
};
use smallvec::SmallVec;

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
            let entries = parse_terms::<u8>(py, &terms).map_err(Error::from)?;
            Ok(PyFermionOperator {
                inner: FermionOperatorInner::Ham8(FermionOperator::new(entries)),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let entries = parse_terms::<u16>(py, &terms).map_err(Error::from)?;
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
            "FermionOperator(max_site={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.num_cindices(),
        )
    }
}

fn parse_terms<C: Copy + Ord + TryFrom<usize>>(
    py: Python<'_>,
    terms: &[crate::operator::pauli::Term],
) -> Result<Vec<FermionOpEntry<C>>, quspin_core::error::QuSpinError>
where
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
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
                if op_str.len() != sites.len() {
                    return Err(quspin_core::error::QuSpinError::ValueError(format!(
                        "op_str length {} != number of sites {}",
                        op_str.len(),
                        sites.len()
                    )));
                }
                let mut ops: SmallVec<[(FermionOp, u32); 4]> = SmallVec::new();
                for (ch, &site) in op_str.chars().zip(sites.iter()) {
                    let op = FermionOp::from_char(ch).ok_or_else(|| {
                        quspin_core::error::QuSpinError::ValueError(format!(
                            "unknown operator character '{ch}'; expected one of +, -, n"
                        ))
                    })?;
                    ops.push((op, site));
                }
                entries.push(FermionOpEntry::new(cindex, coeff, ops));
            }
        }
    }
    Ok(entries)
}
