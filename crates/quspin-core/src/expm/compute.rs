//! [`ExpmComputation`] trait and its four implementations.

use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use num_complex::Complex;

use crate::primitive::Primitive;

// ---------------------------------------------------------------------------
// AtomicAccum — parallel scatter-write accumulator
// ---------------------------------------------------------------------------

/// A single atomically-accumatable output slot for parallel scatter-writes.
///
/// Used by [`crate::expm::linear_operator::LinearOperator::dot_transpose_chunk`]
/// so that multiple threads can safely scatter-add into a shared output array
/// without per-thread intermediate buffers.
///
/// All operations use `Relaxed` ordering; the caller is responsible for
/// inserting the appropriate synchronisation barrier (e.g. the rayon thread-
/// pool join) before reading the accumulated results.
pub trait AtomicAccum: Send + Sync {
    type Value;
    /// Atomically add `val` to this slot.
    fn fetch_add(&self, val: Self::Value);
    /// Load the current value.
    fn load(&self) -> Self::Value;
    /// Construct a zero-valued slot.
    fn zero() -> Self;
}

// ---------------------------------------------------------------------------
// Private helpers: CAS-loop atomic add for f32 / f64
// ---------------------------------------------------------------------------

#[inline]
fn cas_add_f32(atom: &AtomicU32, val: f32) {
    atom.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
        Some((f32::from_bits(bits) + val).to_bits())
    })
    .unwrap();
}

#[inline]
fn cas_add_f64(atom: &AtomicU64, val: f64) {
    atom.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
        Some((f64::from_bits(bits) + val).to_bits())
    })
    .unwrap();
}

// ---------------------------------------------------------------------------
// Concrete atomic accumulators
// ---------------------------------------------------------------------------

/// Atomic accumulator for `f32`.
pub struct AtomicF32(AtomicU32);

impl AtomicAccum for AtomicF32 {
    type Value = f32;
    #[inline]
    fn fetch_add(&self, val: f32) {
        cas_add_f32(&self.0, val);
    }
    #[inline]
    fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
    #[inline]
    fn zero() -> Self {
        AtomicF32(AtomicU32::new(0))
    }
}

/// Atomic accumulator for `f64`.
pub struct AtomicF64(AtomicU64);

impl AtomicAccum for AtomicF64 {
    type Value = f64;
    #[inline]
    fn fetch_add(&self, val: f64) {
        cas_add_f64(&self.0, val);
    }
    #[inline]
    fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
    #[inline]
    fn zero() -> Self {
        AtomicF64(AtomicU64::new(0))
    }
}

/// Atomic accumulator for `Complex<f32>`.
///
/// The real and imaginary parts are updated independently (each with a
/// separate CAS loop).  No cross-field atomicity is provided; observers
/// must not read until all threads have finished.
pub struct AtomicComplex32 {
    re: AtomicU32,
    im: AtomicU32,
}

impl AtomicAccum for AtomicComplex32 {
    type Value = Complex<f32>;
    #[inline]
    fn fetch_add(&self, val: Complex<f32>) {
        cas_add_f32(&self.re, val.re);
        cas_add_f32(&self.im, val.im);
    }
    #[inline]
    fn load(&self) -> Complex<f32> {
        Complex::new(
            f32::from_bits(self.re.load(Ordering::Relaxed)),
            f32::from_bits(self.im.load(Ordering::Relaxed)),
        )
    }
    #[inline]
    fn zero() -> Self {
        AtomicComplex32 {
            re: AtomicU32::new(0),
            im: AtomicU32::new(0),
        }
    }
}

/// Atomic accumulator for `Complex<f64>`.
///
/// The real and imaginary parts are updated independently (each with a
/// separate CAS loop).  No cross-field atomicity is provided; observers
/// must not read until all threads have finished.
pub struct AtomicComplex64 {
    re: AtomicU64,
    im: AtomicU64,
}

impl AtomicAccum for AtomicComplex64 {
    type Value = Complex<f64>;
    #[inline]
    fn fetch_add(&self, val: Complex<f64>) {
        cas_add_f64(&self.re, val.re);
        cas_add_f64(&self.im, val.im);
    }
    #[inline]
    fn load(&self) -> Complex<f64> {
        Complex::new(
            f64::from_bits(self.re.load(Ordering::Relaxed)),
            f64::from_bits(self.im.load(Ordering::Relaxed)),
        )
    }
    #[inline]
    fn zero() -> Self {
        AtomicComplex64 {
            re: AtomicU64::new(0),
            im: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// ExpmComputation trait
// ---------------------------------------------------------------------------

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

    /// Atomic accumulator for parallel scatter-writes into a shared output
    /// slice.  Used by
    /// [`LinearOperator::dot_transpose_chunk`](crate::expm::linear_operator::LinearOperator::dot_transpose_chunk).
    type Atomic: AtomicAccum<Value = Self>;

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
    type Atomic = AtomicF32;

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
    type Atomic = AtomicF64;

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
    type Atomic = AtomicComplex32;

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
    type Atomic = AtomicComplex64;

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
