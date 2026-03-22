use numpy::{Complex32, Complex64, PyArrayDescr, PyArrayDescrMethods, dtype};
use pyo3::{Bound, Python};
use quspin_core::error::QuSpinError;

/// The six element types supported by `QMatrix` / `PyQMatrix`.
///
/// Mirrors the six `Primitive` impls in `quspin-core` and the six value
/// columns in the dispatch table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixDType {
    Int8,
    Int16,
    Float32,
    Float64,
    /// `complex64` in NumPy (two `f32` components).
    Complex64,
    /// `complex128` in NumPy (two `f64` components).
    Complex128,
}

/// The two operator-string index types (`cindex`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CIndexDType {
    U8,
    U16,
}

impl MatrixDType {
    /// Inspect a NumPy dtype descriptor and return the matching variant.
    ///
    /// Returns `ValueError` for any dtype that is not one of the six supported
    /// element types.
    pub fn from_descr(
        py: Python<'_>,
        descr: &Bound<'_, PyArrayDescr>,
    ) -> Result<Self, QuSpinError> {
        if descr.is_equiv_to(&dtype::<i8>(py)) {
            Ok(MatrixDType::Int8)
        } else if descr.is_equiv_to(&dtype::<i16>(py)) {
            Ok(MatrixDType::Int16)
        } else if descr.is_equiv_to(&dtype::<f32>(py)) {
            Ok(MatrixDType::Float32)
        } else if descr.is_equiv_to(&dtype::<f64>(py)) {
            Ok(MatrixDType::Float64)
        } else if descr.is_equiv_to(&dtype::<Complex32>(py)) {
            Ok(MatrixDType::Complex64)
        } else if descr.is_equiv_to(&dtype::<Complex64>(py)) {
            Ok(MatrixDType::Complex128)
        } else {
            Err(QuSpinError::ValueError(
                "unsupported dtype; expected one of int8, int16, float32, float64, \
                 complex64, complex128"
                    .to_string(),
            ))
        }
    }
}

impl CIndexDType {
    /// Inspect a NumPy dtype descriptor and return the matching cindex variant.
    ///
    /// Returns `ValueError` for anything other than `uint8` or `uint16`.
    pub fn from_descr(
        py: Python<'_>,
        descr: &Bound<'_, PyArrayDescr>,
    ) -> Result<Self, QuSpinError> {
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
