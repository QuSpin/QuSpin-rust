//! Python bindings for the FHT (Fast Hadamard Transform), backed by
//! `quspin-ffht`.
//!
//! All four functions validate their inputs (C-contiguity, power-of-two
//! length) and translate violations into `ValueError` before calling into
//! `quspin_ffht::fht`, which uses runtime SIMD dispatch
//! (scalar / SSE2 / AVX2+FMA) under the hood.

use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quspin_core::ffht;

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

/// In-place Fast Hadamard Transform for a 1-D ``float32`` array.
///
/// ``arr.shape[0]`` must be a power of two and ``arr`` must be
/// C-contiguous.
#[pyfunction]
pub fn fht_f32(mut arr: PyReadwriteArray1<f32>) -> PyResult<()> {
    let mut array = arr.as_array_mut();

    let slice = array
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err(CONTIGUITY_ERR))?;

    check_power_of_two(slice.len())?;

    ffht::fht_f32(slice);
    Ok(())
}

/// In-place Fast Hadamard Transform for a 1-D ``float64`` array.
///
/// ``arr.shape[0]`` must be a power of two and ``arr`` must be
/// C-contiguous.
#[pyfunction]
pub fn fht_f64(mut arr: PyReadwriteArray1<f64>) -> PyResult<()> {
    let mut array = arr.as_array_mut();

    let slice = array
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err(CONTIGUITY_ERR))?;

    check_power_of_two(slice.len())?;

    ffht::fht_f64(slice);
    Ok(())
}

/// Out-of-place Fast Hadamard Transform for a 1-D ``float32`` array.
///
/// Returns a new array; ``input`` is left unchanged. ``input.shape[0]``
/// must be a power of two and ``input`` must be C-contiguous.
#[pyfunction]
pub fn fht_f32_oop<'py>(
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
    ffht::fht_f32_oop(&mut in_buf, &mut out_buf);

    Ok(PyArray1::from_vec(py, out_buf))
}

/// Out-of-place Fast Hadamard Transform for a 1-D ``float64`` array.
///
/// Returns a new array; ``input`` is left unchanged. ``input.shape[0]``
/// must be a power of two and ``input`` must be C-contiguous.
#[pyfunction]
pub fn fht_f64_oop<'py>(
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
    ffht::fht_f64_oop(&mut in_buf, &mut out_buf);

    Ok(PyArray1::from_vec(py, out_buf))
}
