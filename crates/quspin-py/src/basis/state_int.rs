//! Conversions between Python integers and per-site occupation bytes.
//!
//! The packing convention is the one `quspin-bitbasis` uses internally: site
//! `i` occupies bits `[i * bits, (i + 1) * bits)` of the state integer, least
//! significant first, where `bits = DynamicDitManip::new(lhss).bits`. The
//! bit-width is read from `quspin-bitbasis` rather than recomputed here so the
//! two can never drift apart.
//!
//! Python integers are arbitrary precision, so conversion goes through
//! `int.to_bytes` / `int.from_bytes` rather than a fixed-width Rust integer.

use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};
use quspin_core::bitbasis_crate::manip::DynamicDitManip;

/// Bits used to encode one site's occupation for a local Hilbert space of
/// size `lhss`.
#[inline]
fn bits_per_site(lhss: usize) -> usize {
    DynamicDitManip::new(lhss).bits
}

/// Read `bits` bits starting at bit offset `shift` from a little-endian buffer.
///
/// `bits <= 8` (lhss is capped at 255), so the value spans at most two bytes.
#[inline]
fn read_bits(buf: &[u8], shift: usize, bits: usize) -> u8 {
    let byte = shift / 8;
    let off = shift % 8;
    let lo = buf.get(byte).copied().unwrap_or(0) as u16;
    let hi = buf.get(byte + 1).copied().unwrap_or(0) as u16;
    (((lo | (hi << 8)) >> off) & ((1u16 << bits) - 1)) as u8
}

/// Write `bits` bits of `value` at bit offset `shift` into a little-endian
/// buffer. Inverse of [`read_bits`].
#[inline]
fn write_bits(buf: &mut [u8], shift: usize, bits: usize, value: u8) {
    let byte = shift / 8;
    let off = shift % 8;
    let word = (value as u16 & ((1u16 << bits) - 1)) << off;
    buf[byte] |= word as u8;
    if off + bits > 8 {
        buf[byte + 1] |= (word >> 8) as u8;
    }
}

/// Number of bytes needed to hold `n_sites` sites of `lhss` occupations.
#[inline]
fn buffer_len(n_sites: usize, lhss: usize) -> usize {
    (n_sites * bits_per_site(lhss)).div_ceil(8)
}

/// Encode per-site occupation bytes as a Python integer.
pub(crate) fn state_bytes_to_py_int(
    py: Python<'_>,
    bytes: &[u8],
    lhss: usize,
) -> PyResult<Py<PyAny>> {
    let bits = bits_per_site(lhss);
    let mut buf = vec![0u8; buffer_len(bytes.len(), lhss)];
    for (site, &value) in bytes.iter().enumerate() {
        write_bits(&mut buf, site * bits, bits, value);
    }

    Ok(py
        .get_type::<PyInt>()
        .call_method1("from_bytes", (PyBytes::new(py, &buf), "little"))?
        .unbind())
}

/// Decode a Python integer into per-site occupation bytes.
///
/// Accepts anything implementing `__index__` (so `numpy` integer scalars work
/// as well as `int`). Rejects negative values, values wider than `n_sites`
/// sites, and values encoding a per-site occupation `>= lhss`.
pub(crate) fn py_int_to_state_bytes(
    state_int: &Bound<'_, PyAny>,
    n_sites: usize,
    lhss: usize,
) -> PyResult<Vec<u8>> {
    let index = state_int.call_method0("__index__").map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "state_int must be an integer, got {}",
            state_int.get_type().name().map_or_else(
                |_| "<unknown>".to_string(),
                |name| name.to_string_lossy().into_owned()
            ),
        ))
    })?;

    if index.lt(0i32)? {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "state_int must be non-negative",
        ));
    }

    let bits = bits_per_site(lhss);
    let bit_length: usize = index.call_method0("bit_length")?.extract()?;
    if bit_length > n_sites * bits {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "state_int needs {bit_length} bits but the basis encodes only \
             {n_sites} sites of lhss={lhss} ({} bits)",
            n_sites * bits,
        )));
    }

    let buf: Vec<u8> = index
        .call_method1("to_bytes", (buffer_len(n_sites, lhss), "little"))?
        .extract()?;

    (0..n_sites)
        .map(|site| {
            let value = read_bits(&buf, site * bits, bits);
            if value as usize >= lhss {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "state_int encodes invalid local value {value} at site {site} \
                     for lhss={lhss}"
                )));
            }
            Ok(value)
        })
        .collect()
}

/// Build the `basis.states` array from the basis' raw decimal state strings.
///
/// Returns a `uint64` `numpy` array when every state fits in 64 bits (always
/// true without the `large-int` feature), and an object-dtype array of Python
/// integers otherwise, so the values are exact either way.
pub(crate) fn states_to_pyarray(py: Python<'_>, decimal_strs: &[String]) -> PyResult<Py<PyAny>> {
    if let Some(narrow) = decimal_strs
        .iter()
        .map(|s| s.parse::<u64>().ok())
        .collect::<Option<Vec<u64>>>()
    {
        return Ok(narrow.to_pyarray(py).unbind().into_any());
    }

    let int_type = py.get_type::<PyInt>();
    let wide = decimal_strs
        .iter()
        .map(|s| int_type.call1((s.as_str(),)))
        .collect::<PyResult<Vec<_>>>()?;

    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", "object")?;
    Ok(np.call_method("array", (wide,), Some(&kwargs))?.unbind())
}
