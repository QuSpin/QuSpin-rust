//! Shared dispatch macros used across multiple modules.
//!
//! `with_value_dtype!` and `with_cindex_dtype!` map the runtime dtype enums
//! to concrete Rust types via type-alias-in-block, enabling zero-cost static
//! dispatch at the PyO3 boundary.

/// Inject a type alias `$V` for the concrete element type matching `$dtype`.
///
/// The `$body` block may reference `$V` as a concrete type.
#[macro_export]
macro_rules! with_value_dtype {
    ($dtype:expr, $V:ident, $body:block) => {
        match $dtype {
            $crate::dtype::MatrixDType::Int8 => {
                type $V = i8;
                $body
            }
            $crate::dtype::MatrixDType::Int16 => {
                type $V = i16;
                $body
            }
            $crate::dtype::MatrixDType::Float32 => {
                type $V = f32;
                $body
            }
            $crate::dtype::MatrixDType::Float64 => {
                type $V = f64;
                $body
            }
            $crate::dtype::MatrixDType::Complex64 => {
                type $V = ::num_complex::Complex<f32>;
                $body
            }
            $crate::dtype::MatrixDType::Complex128 => {
                type $V = ::num_complex::Complex<f64>;
                $body
            }
        }
    };
}

/// Inject a type alias `$C` for the concrete cindex type matching `$dtype`.
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
