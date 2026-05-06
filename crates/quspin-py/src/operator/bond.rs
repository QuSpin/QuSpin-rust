use crate::error::Error;
use crate::operator::{as_c64_vec, dispatch_apply, dispatch_apply_and_project_to, write_c64_back};
use ndarray::Array2;
use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use quspin_core::operator::bond::{BondOperator, BondOperatorInner, BondTerm};

/// One `(matrix, bonds)` pair within a term. Multiple pairs in the same
/// term share one cindex (= position of the term in `*terms`).
type BondPair<'py> = (PyReadonlyArray2<'py, Complex64>, Vec<(u32, u32)>);

/// Python-facing bond (dense two-site matrix) operator.
///
/// Variadic `*terms` — each positional argument is one term and the term
/// position is its cindex (matching `PauliOperator`, `BosonOperator`, etc.).
/// Each term is a list of `(matrix, bonds)` pairs that all share that
/// cindex:
/// - `matrix`  – (lhss² × lhss²) complex128 ndarray; row/col index = `a * lhss + b`
/// - `bonds`   – list of `(site_i, site_j)` pairs to apply the matrix to
///
/// Example:
/// ```python
/// # Two cindices; second cindex carries two matrices on different bond sets:
/// op = BondOperator(
///     [(M1, [(0, 1), (1, 2)])],            # cindex 0
///     [(M2, [(0, 1)]), (M3, [(1, 2)])],    # cindex 1
/// )
/// ```
#[pyclass(name = "BondOperator", module = "quspin._rs")]
pub struct PyBondOperator {
    pub inner: BondOperatorInner,
}

#[pymethods]
impl PyBondOperator {
    #[new]
    #[pyo3(signature = (*terms))]
    fn new(terms: Vec<Vec<BondPair<'_>>>) -> PyResult<Self> {
        let max_cindex = terms.len().saturating_sub(1);
        let max_site = terms
            .iter()
            .flat_map(|term| term.iter())
            .flat_map(|(_, bonds)| bonds.iter())
            .flat_map(|&(si, sj)| [si as usize, sj as usize])
            .max()
            .unwrap_or(0);

        let use_u8 = max_cindex <= 255 && max_site <= 255;

        if use_u8 {
            let bond_terms = extract_terms::<u8>(&terms)?;
            Ok(PyBondOperator {
                inner: BondOperatorInner::Ham8(BondOperator::new(bond_terms).map_err(Error::from)?),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let bond_terms = extract_terms::<u16>(&terms)?;
            Ok(PyBondOperator {
                inner: BondOperatorInner::Ham16(
                    BondOperator::new(bond_terms).map_err(Error::from)?,
                ),
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
            "BondOperator(max_site={}, lhss={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.lhss(),
            self.inner.num_cindices(),
        )
    }
}

fn extract_terms<C: Copy + Ord + TryFrom<usize>>(
    terms: &[Vec<BondPair<'_>>],
) -> PyResult<Vec<BondTerm<C>>>
where
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut out = Vec::new();
    for (cindex_usize, term) in terms.iter().enumerate() {
        let cindex = C::try_from(cindex_usize).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "cindex {cindex_usize} out of range for chosen index type"
            ))
        })?;
        for (mat, bonds) in term {
            // Copy matrix into an owned ndarray with Complex<f64> elements.
            let arr = mat.as_array();
            let nrows = arr.nrows();
            let ncols = arr.ncols();
            let owned: Array2<Complex<f64>> = Array2::from_shape_fn((nrows, ncols), |(r, c)| {
                let v = arr[[r, c]];
                Complex::new(v.re, v.im)
            });
            out.push(BondTerm {
                cindex,
                matrix: owned,
                bonds: bonds.clone(),
            });
        }
    }
    Ok(out)
}
