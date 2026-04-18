use crate::error::Error;
use crate::operator::{as_c64_vec, with_space_inner, with_two_space_inners, write_c64_back};
use ndarray::Array2;
use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use quspin_core::OperatorDispatch;
use quspin_core::operator::bond::{BondOperator, BondOperatorInner, BondTerm};

type BondTermInput<'py> = (PyReadonlyArray2<'py, Complex64>, Vec<(u32, u32)>, usize);

/// Python-facing bond (dense two-site matrix) operator.
///
/// Each term is `(matrix, bonds, cindex)` where:
/// - `matrix`  – (lhss² × lhss²) complex128 ndarray; row/col index = `a * lhss + b`
/// - `bonds`   – list of `(site_i, site_j)` pairs to apply the matrix to
/// - `cindex`  – which coupling constant this term belongs to (default 0)
#[pyclass(name = "BondOperator", module = "quspin._rs")]
pub struct PyBondOperator {
    pub inner: BondOperatorInner,
}

#[pymethods]
impl PyBondOperator {
    #[new]
    #[pyo3(signature = (terms))]
    fn new(terms: Vec<BondTermInput<'_>>) -> PyResult<Self> {
        let max_cindex = terms.iter().map(|(_, _, c)| *c).max().unwrap_or(0);
        let max_site = terms
            .iter()
            .flat_map(|(_, bonds, _)| bonds.iter())
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
            "BondOperator(max_site={}, lhss={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.lhss(),
            self.inner.num_cindices(),
        )
    }
}

fn extract_terms<C: Copy + Ord + TryFrom<usize>>(
    terms: &[BondTermInput<'_>],
) -> PyResult<Vec<BondTerm<C>>>
where
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
{
    terms
        .iter()
        .map(|(mat, bonds, cindex_usize)| {
            let cindex = C::try_from(*cindex_usize).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "cindex {cindex_usize} out of range for chosen index type"
                ))
            })?;
            // Copy matrix into an owned ndarray with Complex<f64> elements.
            let arr = mat.as_array();
            let nrows = arr.nrows();
            let ncols = arr.ncols();
            let owned: Array2<Complex<f64>> = Array2::from_shape_fn((nrows, ncols), |(r, c)| {
                let v = arr[[r, c]];
                Complex::new(v.re, v.im)
            });
            Ok(BondTerm {
                cindex,
                matrix: owned,
                bonds: bonds.clone(),
            })
        })
        .collect()
}
