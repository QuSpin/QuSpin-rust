//! [`ExpmComputation`] trait and its four implementations.

use std::fmt::Debug;
use std::ops::{Add, Mul};

use num_complex::Complex;

use crate::primitive::Primitive;

/// Extension of [`Primitive`] with the operations needed by the Taylor-series
/// matrix-exponential algorithm.
///
/// Implemented for `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
pub trait ExpmComputation: Primitive {
    /// The real scalar type: `f32` for `f32`/`Complex<f32>`,
    /// `f64` for `f64`/`Complex<f64>`.
    type Real: Copy
        + PartialOrd
        + Add<Output = Self::Real>
        + Mul<Output = Self::Real>
        + Default
        + Debug;

    /// `|self|` as `Self::Real` (modulus for complex types).
    fn abs_val(self) -> Self::Real;

    /// `e^self` (real or complex exponential).
    fn exp_val(self) -> Self;

    /// Wrap a real value `r` as `Self` (e.g. `3.0 → Complex::new(3.0, 0.0)`).
    fn from_real(r: Self::Real) -> Self;

    /// Cast an `f64` to `Self::Real` (lossy for `f32`).
    fn real_from_f64(v: f64) -> Self::Real;

    /// Machine epsilon / 2: default convergence tolerance.
    fn machine_eps() -> Self::Real;
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

impl ExpmComputation for f32 {
    type Real = f32;

    #[inline]
    fn abs_val(self) -> f32 {
        self.abs()
    }
    #[inline]
    fn exp_val(self) -> f32 {
        self.exp()
    }
    #[inline]
    fn from_real(r: f32) -> f32 {
        r
    }
    #[inline]
    fn real_from_f64(v: f64) -> f32 {
        v as f32
    }
    #[inline]
    fn machine_eps() -> f32 {
        f32::EPSILON / 2.0
    }
}

impl ExpmComputation for f64 {
    type Real = f64;

    #[inline]
    fn abs_val(self) -> f64 {
        self.abs()
    }
    #[inline]
    fn exp_val(self) -> f64 {
        self.exp()
    }
    #[inline]
    fn from_real(r: f64) -> f64 {
        r
    }
    #[inline]
    fn real_from_f64(v: f64) -> f64 {
        v
    }
    #[inline]
    fn machine_eps() -> f64 {
        f64::EPSILON / 2.0
    }
}

impl ExpmComputation for Complex<f32> {
    type Real = f32;

    #[inline]
    fn abs_val(self) -> f32 {
        self.norm()
    }
    #[inline]
    fn exp_val(self) -> Complex<f32> {
        self.exp()
    }
    #[inline]
    fn from_real(r: f32) -> Complex<f32> {
        Complex::new(r, 0.0)
    }
    #[inline]
    fn real_from_f64(v: f64) -> f32 {
        v as f32
    }
    #[inline]
    fn machine_eps() -> f32 {
        f32::EPSILON / 2.0
    }
}

impl ExpmComputation for Complex<f64> {
    type Real = f64;

    #[inline]
    fn abs_val(self) -> f64 {
        self.norm()
    }
    #[inline]
    fn exp_val(self) -> Complex<f64> {
        self.exp()
    }
    #[inline]
    fn from_real(r: f64) -> Complex<f64> {
        Complex::new(r, 0.0)
    }
    #[inline]
    fn real_from_f64(v: f64) -> f64 {
        v
    }
    #[inline]
    fn machine_eps() -> f64 {
        f64::EPSILON / 2.0
    }
}
