//! `numpy` ⇄ Rust state-vector interop and the shared `project_to` /
//! `project_from` implementation.
//!
//! The four Python basis wrappers differ only in which pair of bases they
//! project between, so the whole body lives here and they call
//! [`project_generic`] / [`project_bit`]. Keeping it in one place also keeps
//! the intermediate [`PyStateVector`] representation private to this module —
//! nothing outside needs to know how a Python object becomes a column-major
//! `Vec<Complex<f64>>`.

use crate::error::Error;
use ndarray::Array2;
use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use quspin_core::QuSpinError;
use quspin_core::basis::GenericBasis;
use quspin_core::basis::dispatch::BitBasis;

type C64 = Complex<f64>;

/// A Python state vector (or a matrix of column vectors) unpacked into
/// column-major complex storage.
struct PyStateVector {
    data: Vec<C64>,
    /// Whether the Python object was complex-valued. Drives whether the
    /// result is handed back as a real or complex array.
    is_complex: bool,
    nrows: usize,
    ncols: usize,
    is_matrix: bool,
}

impl PyStateVector {
    fn vector(data: Vec<C64>, is_complex: bool) -> Self {
        let nrows = data.len();
        Self {
            data,
            is_complex,
            nrows,
            ncols: 1,
            is_matrix: false,
        }
    }

    fn matrix(data: Vec<C64>, is_complex: bool, nrows: usize, ncols: usize) -> Self {
        Self {
            data,
            is_complex,
            nrows,
            ncols,
            is_matrix: true,
        }
    }
}

/// Collect a 2-D `numpy` view into column-major complex storage.
fn column_major<T, F>(view: ndarray::ArrayView2<'_, T>, to_c64: F) -> (Vec<C64>, usize, usize)
where
    T: Copy,
    F: Fn(T) -> C64,
{
    let (nrows, ncols) = view.dim();
    let mut data = Vec::with_capacity(nrows * ncols);
    for col in 0..ncols {
        for row in 0..nrows {
            data.push(to_c64(view[(row, col)]));
        }
    }
    (data, nrows, ncols)
}

/// Flatten nested Python sequences into column-major complex storage.
fn rows_to_column_major(rows: &[Vec<C64>]) -> PyResult<(Vec<C64>, usize, usize)> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != ncols) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "2-D state array must be rectangular",
        ));
    }
    let mut data = Vec::with_capacity(nrows * ncols);
    for col in 0..ncols {
        for row in rows {
            data.push(row[col]);
        }
    }
    Ok((data, nrows, ncols))
}

/// Convert a Python object into [`PyStateVector`].
///
/// Accepts 1-D and 2-D `numpy` arrays of `float64` / `complex128` and the
/// equivalent nested Python sequences.
fn py_any_to_c64_vec(vec_obj: &Bound<'_, PyAny>) -> PyResult<PyStateVector> {
    if let Ok(arr) = vec_obj.cast::<PyArray1<Complex64>>() {
        let data = unsafe {
            arr.as_array()
                .iter()
                .map(|c| C64::new(c.re, c.im))
                .collect()
        };
        return Ok(PyStateVector::vector(data, true));
    }

    if let Ok(arr) = vec_obj.cast::<PyArray1<f64>>() {
        let data = unsafe { arr.as_array().iter().map(|&x| C64::new(x, 0.0)).collect() };
        return Ok(PyStateVector::vector(data, false));
    }

    if let Ok(arr) = vec_obj.cast::<PyArray2<Complex64>>() {
        let (data, nrows, ncols) =
            column_major(unsafe { arr.as_array() }, |c| C64::new(c.re, c.im));
        return Ok(PyStateVector::matrix(data, true, nrows, ncols));
    }

    if let Ok(arr) = vec_obj.cast::<PyArray2<f64>>() {
        let (data, nrows, ncols) = column_major(unsafe { arr.as_array() }, |x| C64::new(x, 0.0));
        return Ok(PyStateVector::matrix(data, false, nrows, ncols));
    }

    if let Ok(v) = vec_obj.extract::<Vec<C64>>() {
        let is_complex = v.iter().any(|z| z.im != 0.0);
        return Ok(PyStateVector::vector(v, is_complex));
    }

    if let Ok(v) = vec_obj.extract::<Vec<f64>>() {
        let data = v.into_iter().map(|x| C64::new(x, 0.0)).collect();
        return Ok(PyStateVector::vector(data, false));
    }

    if let Ok(rows) = vec_obj.extract::<Vec<Vec<C64>>>() {
        let is_complex = rows.iter().flatten().any(|z| z.im != 0.0);
        let (data, nrows, ncols) = rows_to_column_major(&rows)?;
        return Ok(PyStateVector::matrix(data, is_complex, nrows, ncols));
    }

    if let Ok(rows) = vec_obj.extract::<Vec<Vec<f64>>>() {
        let rows: Vec<Vec<C64>> = rows
            .into_iter()
            .map(|row| row.into_iter().map(|x| C64::new(x, 0.0)).collect())
            .collect();
        let (data, nrows, ncols) = rows_to_column_major(&rows)?;
        return Ok(PyStateVector::matrix(data, false, nrows, ncols));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "state vector must be a 1-D or 2-D sequence or numpy array of real/complex numbers",
    ))
}

/// Hand column-major complex data back to Python.
///
/// Downcasts to a real array when the input was real *and* the result has no
/// imaginary part — projecting into a sector with complex characters turns a
/// real input into a genuinely complex output, which must stay complex.
fn state_vec_to_pyarray(
    py: Python<'_>,
    data: &[C64],
    prefer_complex: bool,
    nrows: usize,
    ncols: usize,
    is_matrix: bool,
) -> Py<PyAny> {
    let max_abs = data.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
    let imag_tol = (max_abs * 64.0 * f64::EPSILON).max(1e-14);
    let is_numerically_real = data.iter().all(|z| z.im.abs() <= imag_tol);

    match (prefer_complex || !is_numerically_real, is_matrix) {
        (false, false) => data
            .iter()
            .map(|z| z.re)
            .collect::<Vec<f64>>()
            .to_pyarray(py)
            .unbind()
            .into_any(),
        (false, true) => Array2::from_shape_fn((nrows, ncols), |(r, c)| data[c * nrows + r].re)
            .to_pyarray(py)
            .unbind()
            .into_any(),
        (true, false) => data
            .iter()
            .map(|z| Complex64::new(z.re, z.im))
            .collect::<Vec<Complex64>>()
            .to_pyarray(py)
            .unbind()
            .into_any(),
        (true, true) => Array2::from_shape_fn((nrows, ncols), |(r, c)| {
            let z = data[c * nrows + r];
            Complex64::new(z.re, z.im)
        })
        .to_pyarray(py)
        .unbind()
        .into_any(),
    }
}

/// Reject `sparse=True` rather than silently returning a dense array.
pub(crate) fn reject_sparse(sparse: bool) -> PyResult<()> {
    if sparse {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "sparse projection is not implemented yet; pass sparse=False for a dense \
             numpy array",
        ));
    }
    Ok(())
}

/// Shared body of `project_to` / `project_from`: unpack the Python state,
/// run `project_col` once per column, and repack the result.
fn project_columns<F>(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
    out_rows: usize,
    project_col: F,
) -> PyResult<Py<PyAny>>
where
    F: Fn(&[C64], &mut [C64]) -> Result<(), QuSpinError>,
{
    let in_vec = py_any_to_c64_vec(state)?;

    let mut out = vec![C64::new(0.0, 0.0); out_rows * in_vec.ncols];
    for col in 0..in_vec.ncols {
        let in_col = &in_vec.data[col * in_vec.nrows..(col + 1) * in_vec.nrows];
        let out_col = &mut out[col * out_rows..(col + 1) * out_rows];
        project_col(in_col, out_col).map_err(Error::from)?;
    }

    Ok(state_vec_to_pyarray(
        py,
        &out,
        in_vec.is_complex,
        out_rows,
        in_vec.ncols,
        in_vec.is_matrix,
    ))
}

/// Project a Python state vector from `input` to `output` (generic family).
pub(crate) fn project_generic(
    py: Python<'_>,
    input: &GenericBasis,
    output: &GenericBasis,
    state: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    project_columns(py, state, output.size(), |i, o| {
        quspin_core::project_to(input, output, i, o, true)
    })
}

/// Project a Python state vector from `input` to `output` (bit family).
pub(crate) fn project_bit(
    py: Python<'_>,
    input: &BitBasis,
    output: &BitBasis,
    state: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    project_columns(py, state, output.size(), |i, o| {
        quspin_core::project_to_bit(input, output, i, o, true)
    })
}
