//! Runtime dtype tags and dispatch macros shared by all FFI consumers.
//!
//! `ValueDType` and `CIndexDType` are the canonical runtime representations of
//! the supported element types.  All FFI crates (`quspin-py`, `quspin-c`, …)
//! convert their own dtype representations into these enums at their boundary,
//! then use the macros here to dispatch into generic `quspin-core` code.

/// The six element types supported by `QMatrix`.
///
/// Mirrors the six `Primitive` impls in `quspin-core`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueDType {
    Int8,
    Int16,
    Float32,
    Float64,
    /// `complex64` in NumPy / C (two `f32` components).
    Complex64,
    /// `complex128` in NumPy / C (two `f64` components).
    Complex128,
}

/// The two operator-string index types (`cindex`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CIndexDType {
    U8,
    U16,
}

/// Inject a type alias `$V` for the concrete element type matching `$dtype`,
/// then evaluate `$body`.
///
/// `$dtype` must be a `ValueDType` expression.  `$body` may reference `$V`
/// as a concrete type.
#[macro_export]
macro_rules! with_value_dtype {
    ($dtype:expr, $V:ident, $body:block) => {
        match $dtype {
            $crate::dtype::ValueDType::Int8 => {
                type $V = i8;
                $body
            }
            $crate::dtype::ValueDType::Int16 => {
                type $V = i16;
                $body
            }
            $crate::dtype::ValueDType::Float32 => {
                type $V = f32;
                $body
            }
            $crate::dtype::ValueDType::Float64 => {
                type $V = f64;
                $body
            }
            $crate::dtype::ValueDType::Complex64 => {
                type $V = ::num_complex::Complex<f32>;
                $body
            }
            $crate::dtype::ValueDType::Complex128 => {
                type $V = ::num_complex::Complex<f64>;
                $body
            }
        }
    };
}

/// Inject a type alias `$C` for the concrete cindex type matching `$dtype`,
/// then evaluate `$body`.
///
/// `$dtype` must be a `CIndexDType` expression.
#[macro_export]
macro_rules! with_cindex_dtype {
    ($dtype:expr, $C:ident, $body:block) => {
        match $dtype {
            $crate::dtype::CIndexDType::U8 => {
                type $C = u8;
                $body
            }
            $crate::dtype::CIndexDType::U16 => {
                type $C = u16;
                $body
            }
        }
    };
}
