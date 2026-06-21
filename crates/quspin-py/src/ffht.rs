//! Python bindings for the FFHT (Fast Fast Hadamard Transform), backed by
//! `quspin-ffht` (via the `quspin-core` facade).
//!
//! Exposes a single [`ffht_py`] function that dispatches on the input
//! array's dtype (``float32`` / ``float64``) and on the ``inplace``
//! flag. Inputs are validated (C-contiguity, power-of-two length) and
//! violations are translated into ``ValueError``/``TypeError`` before
//! calling into `quspin_core::ffht`, which uses runtime SIMD dispatch
//! (scalar / SSE2 / AVX2+FMA) under the hood.

use numpy::{
    PyArray1, PyArrayDescrMethods, PyReadonlyArray1, PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use quspin_core::ffht as ffht_impl;

const CONTIGUITY_ERR: &str =
    "input array must be C-contiguous; use np.ascontiguousarray(arr) first";

fn check_power_of_two(len: usize) -> PyResult<()> {
    if len == 0 || (len & (len - 1)) != 0 {
        return Err(PyValueError::new_err(format!(
            "array length must be a power of two, got {len}"
        )));
    }
    Ok(())
}

fn ffht_f32_inplace(mut arr: PyReadwriteArray1<f32>) -> PyResult<()> {
    let mut array = arr.as_array_mut();
    let slice = array
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err(CONTIGUITY_ERR))?;
    check_power_of_two(slice.len())?;
    ffht_impl::fht_f32(slice);
    Ok(())
}

fn ffht_f64_inplace(mut arr: PyReadwriteArray1<f64>) -> PyResult<()> {
    let mut array = arr.as_array_mut();
    let slice = array
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err(CONTIGUITY_ERR))?;
    check_power_of_two(slice.len())?;
    ffht_impl::fht_f64(slice);
    Ok(())
}

fn ffht_f32_oop<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let in_array = input.as_array();
    // as_slice() needs an owned copy since the C signature takes `*mut`
    // for the input pointer too (even though it only reads from it).
    let mut in_buf: Vec<f32> = in_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err(CONTIGUITY_ERR))?
        .to_vec();
    check_power_of_two(in_buf.len())?;

    let mut out_buf = vec![0f32; in_buf.len()];
    ffht_impl::fht_f32_oop(&mut in_buf, &mut out_buf);
    Ok(PyArray1::from_vec(py, out_buf))
}

fn ffht_f64_oop<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let in_array = input.as_array();
    let mut in_buf: Vec<f64> = in_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err(CONTIGUITY_ERR))?
        .to_vec();
    check_power_of_two(in_buf.len())?;

    let mut out_buf = vec![0f64; in_buf.len()];
    ffht_impl::fht_f64_oop(&mut in_buf, &mut out_buf);
    Ok(PyArray1::from_vec(py, out_buf))
}

/// Fast Hadamard Transform of a 1-D ``float32`` or ``float64`` array.
///
/// Dispatches on ``arr.dtype`` and on ``inplace`` to the appropriate
/// scalar/SSE2/AVX2 implementation under the hood.
///
/// Args:
///     arr: 1-D ``float32`` or ``float64`` array. ``arr.shape[0]`` must
///         be a power of two and ``arr`` must be C-contiguous.
///     inplace: If ``True``, transform ``arr`` in place and return
///         ``None``. If ``False`` (default), leave ``arr`` unchanged
///         and return a new array containing the transform.
///
/// Returns:
///     A new array containing the transform if ``inplace`` is
///     ``False``; otherwise ``None``.
///
/// Raises:
///     TypeError: If ``arr.dtype`` is not ``float32`` or ``float64``.
///     ValueError: If ``arr`` is not C-contiguous or its length is not
///         a power of two.
#[pyfunction]
#[pyo3(name = "ffht", signature = (arr, inplace=false))]
pub fn ffht_py<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    inplace: bool,
) -> PyResult<Option<Py<PyAny>>> {
    let array = arr
        .cast::<numpy::PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err("ffht: expected a numpy ndarray"))?;

    let dtype = array.dtype();

    if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        if inplace {
            let arr: PyReadwriteArray1<f32> = arr.extract()?;
            ffht_f32_inplace(arr)?;
            Ok(None)
        } else {
            let arr: PyReadonlyArray1<f32> = arr.extract()?;
            Ok(Some(ffht_f32_oop(py, arr)?.unbind().into_any()))
        }
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        if inplace {
            let arr: PyReadwriteArray1<f64> = arr.extract()?;
            ffht_f64_inplace(arr)?;
            Ok(None)
        } else {
            let arr: PyReadonlyArray1<f64> = arr.extract()?;
            Ok(Some(ffht_f64_oop(py, arr)?.unbind().into_any()))
        }
    } else {
        Err(PyTypeError::new_err(format!(
            "ffht: unsupported dtype {dtype}; expected float32 or float64"
        )))
    }
}
