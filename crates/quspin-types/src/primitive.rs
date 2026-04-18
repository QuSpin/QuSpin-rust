use num_complex::Complex;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

mod private {
    pub trait Sealed {}
}

/// Sealed trait over the six element types supported by `QMatrix`:
/// `i8`, `i16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
///
/// Replaces the `PrimativeTypes` / `QMatrixValueTypes` C++ concepts.
/// All core functions are generic over `V: Primitive`.
pub trait Primitive:
    private::Sealed
    + Copy
    + Default
    + Debug
    + Send
    + Sync
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// Widen to `Complex<f64>` for intermediate computation.
    fn to_complex(self) -> Complex<f64>;

    /// Narrow from `Complex<f64>`.  Real types take only the real part;
    /// `Complex<f32>` casts both components; `Complex<f64>` is a no-op.
    ///
    /// Mirrors `detail::cast<T>` from `cast.hpp`.
    fn from_complex(c: Complex<f64>) -> Self;

    /// Absolute value (real types) or modulus (complex types) as `f64`.
    ///
    /// Preferred over `self.to_complex().norm()` because it avoids
    /// constructing an intermediate `Complex<f64>` and the unnecessary
    /// `sqrt` call for real types where `|x|` is just `x.abs()`.
    fn magnitude(self) -> f64;
}

macro_rules! impl_primitive_real {
    ($($T:ty),*) => {
        $(
            impl private::Sealed for $T {}
            impl Primitive for $T {
                #[inline]
                fn to_complex(self) -> Complex<f64> {
                    Complex::new(self as f64, 0.0)
                }
                #[inline]
                fn from_complex(c: Complex<f64>) -> Self {
                    c.re as $T
                }
                #[inline]
                fn magnitude(self) -> f64 {
                    (self as f64).abs()
                }
            }
        )*
    };
}

impl_primitive_real!(i8, i16, f32, f64);

impl private::Sealed for Complex<f32> {}
impl Primitive for Complex<f32> {
    #[inline]
    fn to_complex(self) -> Complex<f64> {
        Complex::new(self.re as f64, self.im as f64)
    }
    #[inline]
    fn from_complex(c: Complex<f64>) -> Self {
        Complex::new(c.re as f32, c.im as f32)
    }
    #[inline]
    fn magnitude(self) -> f64 {
        // Upcast components before squaring to avoid f32 overflow.
        ((self.re as f64).powi(2) + (self.im as f64).powi(2)).sqrt()
    }
}

impl private::Sealed for Complex<f64> {}
impl Primitive for Complex<f64> {
    #[inline]
    fn to_complex(self) -> Complex<f64> {
        self
    }
    #[inline]
    fn from_complex(c: Complex<f64>) -> Self {
        c
    }
    #[inline]
    fn magnitude(self) -> f64 {
        self.norm()
    }
}
