use numpy::{Complex32, Complex64, PyArrayDescr, PyArrayDescrMethods, dtype};
use pyo3::{Bound, Python};
use quspin_core::error::QuSpinError;

pub use quspin_core::dtype::{CIndexDType, ValueDType};

/// Extension trait to construct dtype enums from a NumPy `PyArrayDescr`.
///
/// This is the only PyO3-specific piece of dtype handling; all other dtype
/// logic lives in `quspin-core`.
pub trait FromPyDescr: Sized {
    fn from_descr(py: Python<'_>, descr: &Bound<'_, PyArrayDescr>) -> Result<Self, QuSpinError>;
}

impl FromPyDescr for ValueDType {
    /// Inspect a NumPy dtype descriptor and return the matching variant.
    ///
    /// Returns `ValueError` for any dtype that is not one of the six supported
    /// element types.
    fn from_descr(py: Python<'_>, descr: &Bound<'_, PyArrayDescr>) -> Result<Self, QuSpinError> {
        if descr.is_equiv_to(&dtype::<i8>(py)) {
            Ok(ValueDType::Int8)
        } else if descr.is_equiv_to(&dtype::<i16>(py)) {
            Ok(ValueDType::Int16)
        } else if descr.is_equiv_to(&dtype::<f32>(py)) {
            Ok(ValueDType::Float32)
        } else if descr.is_equiv_to(&dtype::<f64>(py)) {
            Ok(ValueDType::Float64)
        } else if descr.is_equiv_to(&dtype::<Complex32>(py)) {
            Ok(ValueDType::Complex64)
        } else if descr.is_equiv_to(&dtype::<Complex64>(py)) {
            Ok(ValueDType::Complex128)
        } else {
            Err(QuSpinError::ValueError(
                "unsupported dtype; expected one of int8, int16, float32, float64, \
                 complex64, complex128"
                    .to_string(),
            ))
        }
    }
}

impl FromPyDescr for CIndexDType {
    /// Inspect a NumPy dtype descriptor and return the matching cindex variant.
    ///
    /// Returns `ValueError` for anything other than `uint8` or `uint16`.
    fn from_descr(py: Python<'_>, descr: &Bound<'_, PyArrayDescr>) -> Result<Self, QuSpinError> {
        if descr.is_equiv_to(&dtype::<u8>(py)) {
            Ok(CIndexDType::U8)
        } else if descr.is_equiv_to(&dtype::<u16>(py)) {
            Ok(CIndexDType::U16)
        } else {
            Err(QuSpinError::ValueError(
                "unsupported cindex dtype; expected uint8 or uint16".to_string(),
            ))
        }
    }
}
