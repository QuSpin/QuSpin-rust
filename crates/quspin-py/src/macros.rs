//! Shared dispatch macros used across multiple modules.
//!
//! `with_value_dtype!` and `with_cindex_dtype!` map the runtime dtype enums
//! to concrete Rust types via type-alias-in-block, enabling zero-cost static
//! dispatch at the PyO3 boundary.
//!
//! `select_b_for_n_sites!` maps a runtime `n_sites: usize` to the smallest
//! `B: BitInt` that fits all site indices, injecting a local type alias.

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

/// Select the smallest `B: BitInt` that fits `n_sites` site indices, inject it
/// as a local type alias `$B`, and evaluate `$body`.
///
/// The ladder is: ≤32 → `u32`, ≤64 → `u64`, ≤128 → `Uint<128,2>`, …, ≤8192 →
/// `Uint<8192,128>`.  Returns a `PyValueError` for `n_sites > 8192`.
///
/// Used by `PySymmetryGrp::new` and `PyHardcoreBasis::subspace` so the type
/// selection logic lives in exactly one place.
#[macro_export]
macro_rules! select_b_for_n_sites {
    ($n_sites:expr, $B:ident, $body:block) => {
        if $n_sites <= 32 {
            type $B = u32;
            $body
        } else if $n_sites <= 64 {
            type $B = u64;
            $body
        } else if $n_sites <= 128 {
            type $B = ::ruint::Uint<128, 2>;
            $body
        } else if $n_sites <= 256 {
            type $B = ::ruint::Uint<256, 4>;
            $body
        } else if $n_sites <= 512 {
            type $B = ::ruint::Uint<512, 8>;
            $body
        } else if $n_sites <= 1024 {
            type $B = ::ruint::Uint<1024, 16>;
            $body
        } else if $n_sites <= 2048 {
            type $B = ::ruint::Uint<2048, 32>;
            $body
        } else if $n_sites <= 4096 {
            type $B = ::ruint::Uint<4096, 64>;
            $body
        } else if $n_sites <= 8192 {
            type $B = ::ruint::Uint<8192, 128>;
            $body
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites={} exceeds the maximum supported value of 8192",
                $n_sites
            )));
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
