use crate::dtype::FromPyDescr;
use crate::error::Error;
use ndarray::Array2;
use num_complex::Complex;
use numpy::{
    Complex32, Complex64, PyArray2, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use quspin_core::dtype::ValueDType;
use quspin_core::operator::bond::{BondOperator, BondOperatorInner, BondTerm};

/// Python-facing bond (dense two-site matrix) operator.
///
/// Each term is `(matrix, i, j)` or `(matrix, i, j, cindex)` where:
/// - `matrix`  – (lhss² × lhss²) ndarray of any supported dtype
///               (int8, int16, float32, float64, complex64, complex128);
///               converted to complex128 internally.
///               Row/col index = `a * lhss + b`.
/// - `i`, `j`  – site indices for this bond
/// - `cindex`  – which coupling constant (default 0)
#[pyclass(name = "BondOperator", module = "quspin._rs")]
pub struct PyBondOperator {
    pub inner: BondOperatorInner,
}

#[pymethods]
impl PyBondOperator {
    #[new]
    #[pyo3(signature = (terms))]
    fn new(py: Python<'_>, terms: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        // Parse every entry into (matrix, i, j, cindex).
        let mut entries: Vec<(Array2<Complex<f64>>, u32, u32, usize)> =
            Vec::with_capacity(terms.len());
        for (idx, term) in terms.iter().enumerate() {
            entries.push(extract_bond_entry(py, term, idx)?);
        }

        let max_cindex = entries.iter().map(|&(_, _, _, c)| c).max().unwrap_or(0);
        let max_site = entries
            .iter()
            .flat_map(|&(_, i, j, _)| [i as usize, j as usize])
            .max()
            .unwrap_or(0);

        let use_u8 = max_cindex <= 255 && max_site <= 255;

        if use_u8 {
            let bond_terms = to_bond_terms::<u8>(&entries)?;
            Ok(PyBondOperator {
                inner: BondOperatorInner::Ham8(BondOperator::new(bond_terms).map_err(Error::from)?),
            })
        } else if max_cindex <= 65535 && max_site <= 65535 {
            let bond_terms = to_bond_terms::<u16>(&entries)?;
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

    fn __repr__(&self) -> String {
        format!(
            "BondOperator(max_site={}, lhss={}, num_cindices={})",
            self.inner.max_site(),
            self.inner.lhss(),
            self.inner.num_cindices(),
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract `(matrix, i, j, cindex)` from one Python term entry.
///
/// Accepts a sequence of length 3 `(matrix, i, j)` or 4 `(matrix, i, j, cindex)`.
fn extract_bond_entry(
    py: Python<'_>,
    term: &Bound<'_, PyAny>,
    idx: usize,
) -> PyResult<(Array2<Complex<f64>>, u32, u32, usize)> {
    let len = term.len()?;
    if !(3..=4).contains(&len) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "term {idx}: expected (matrix, i, j) or (matrix, i, j, cindex), got {len} elements"
        )));
    }
    let mat_obj = term.get_item(0)?;
    let i: u32 = term.get_item(1)?.extract()?;
    let j: u32 = term.get_item(2)?.extract()?;
    let cindex: usize = if len == 4 {
        term.get_item(3)?.extract()?
    } else {
        0
    };
    let matrix = pyany_to_complex_array(py, &mat_obj, idx)?;
    Ok((matrix, i, j, cindex))
}

/// Downcast a numpy array of any supported dtype to `Array2<Complex<f64>>`.
///
/// Supported dtypes: int8, int16, float32, float64, complex64, complex128.
fn pyany_to_complex_array(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    idx: usize,
) -> PyResult<Array2<Complex<f64>>> {
    let untyped = obj.downcast::<PyUntypedArray>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!("term {idx}: matrix must be a numpy array"))
    })?;

    if untyped.ndim() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "term {idx}: matrix must be 2-D, got {} dimensions",
            untyped.ndim()
        )));
    }

    let vdtype = ValueDType::from_descr(py, &untyped.dtype()).map_err(Error::from)?;

    match vdtype {
        ValueDType::Int8 => {
            let a = obj.downcast::<PyArray2<i8>>()?.readonly();
            let a = a.as_array();
            Ok(Array2::from_shape_fn(a.dim(), |(r, c)| {
                Complex::new(a[[r, c]] as f64, 0.0)
            }))
        }
        ValueDType::Int16 => {
            let a = obj.downcast::<PyArray2<i16>>()?.readonly();
            let a = a.as_array();
            Ok(Array2::from_shape_fn(a.dim(), |(r, c)| {
                Complex::new(a[[r, c]] as f64, 0.0)
            }))
        }
        ValueDType::Float32 => {
            let a = obj.downcast::<PyArray2<f32>>()?.readonly();
            let a = a.as_array();
            Ok(Array2::from_shape_fn(a.dim(), |(r, c)| {
                Complex::new(a[[r, c]] as f64, 0.0)
            }))
        }
        ValueDType::Float64 => {
            let a = obj.downcast::<PyArray2<f64>>()?.readonly();
            let a = a.as_array();
            Ok(Array2::from_shape_fn(a.dim(), |(r, c)| {
                Complex::new(a[[r, c]], 0.0)
            }))
        }
        ValueDType::Complex64 => {
            let a = obj.downcast::<PyArray2<Complex32>>()?.readonly();
            let a = a.as_array();
            Ok(Array2::from_shape_fn(a.dim(), |(r, c)| {
                Complex::new(a[[r, c]].re as f64, a[[r, c]].im as f64)
            }))
        }
        ValueDType::Complex128 => {
            let a = obj.downcast::<PyArray2<Complex64>>()?.readonly();
            let a = a.as_array();
            Ok(Array2::from_shape_fn(a.dim(), |(r, c)| {
                Complex::new(a[[r, c]].re, a[[r, c]].im)
            }))
        }
    }
}

/// Convert flat `(matrix, i, j, cindex)` entries to `BondTerm<C>` list.
fn to_bond_terms<C: Copy + Ord + TryFrom<usize>>(
    entries: &[(Array2<Complex<f64>>, u32, u32, usize)],
) -> PyResult<Vec<BondTerm<C>>>
where
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
{
    entries
        .iter()
        .enumerate()
        .map(|(idx, (matrix, i, j, cindex_usize))| {
            let cindex = C::try_from(*cindex_usize).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "term {idx}: cindex {cindex_usize} out of range for chosen index type"
                ))
            })?;
            Ok(BondTerm {
                cindex,
                matrix: matrix.clone(),
                bonds: vec![(*i, *j)],
            })
        })
        .collect()
}
