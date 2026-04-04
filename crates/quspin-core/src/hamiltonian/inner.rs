/// Type-erased `HamiltonianInner` and the `with_hamiltonian!` macro.
///
/// 12 variants: 6 value types × 2 cindex types.  The index type `I` is fixed
/// to `i64` at the FFI boundary.
///
/// Naming convention: `HM` prefix, value type abbreviation, cindex type
/// abbreviation.  For example `HMf64U8` is `Hamiltonian<f64, i64, u8>`.
use super::ham::{CoeffFn, Hamiltonian};
use crate::error::QuSpinError;
use crate::qmatrix::QMatrixInner;
use ndarray::{ArrayView2, ArrayViewMut2};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// with_hamiltonian! macro  (must be defined before HamiltonianInner impl uses it)
// ---------------------------------------------------------------------------

/// Match on a `&HamiltonianInner`, injecting type aliases `$M` (stored element
/// type) and `$C` (cindex type), and binding `$ham` to the inner `Hamiltonian`
/// reference.
#[macro_export]
macro_rules! with_hamiltonian {
    ($inner:expr, $M:ident, $C:ident, $ham:ident, $body:block) => {
        match $inner {
            $crate::hamiltonian::inner::HamiltonianInner::HMi8U8($ham) => {
                type $M = i8;
                type $C = u8;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMi8U16($ham) => {
                type $M = i8;
                type $C = u16;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMi16U8($ham) => {
                type $M = i16;
                type $C = u8;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMi16U16($ham) => {
                type $M = i16;
                type $C = u16;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMf32U8($ham) => {
                type $M = f32;
                type $C = u8;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMf32U16($ham) => {
                type $M = f32;
                type $C = u16;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMf64U8($ham) => {
                type $M = f64;
                type $C = u8;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMf64U16($ham) => {
                type $M = f64;
                type $C = u16;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMc32U8($ham) => {
                type $M = ::num_complex::Complex<f32>;
                type $C = u8;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMc32U16($ham) => {
                type $M = ::num_complex::Complex<f32>;
                type $C = u16;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMc64U8($ham) => {
                type $M = ::num_complex::Complex<f64>;
                type $C = u8;
                $body
            }
            $crate::hamiltonian::inner::HamiltonianInner::HMc64U16($ham) => {
                type $M = ::num_complex::Complex<f64>;
                type $C = u16;
                $body
            }
        }
    };
}

// ---------------------------------------------------------------------------
// HamiltonianInner
// ---------------------------------------------------------------------------

pub enum HamiltonianInner {
    HMi8U8(Hamiltonian<i8, i64, u8>),
    HMi8U16(Hamiltonian<i8, i64, u16>),
    HMi16U8(Hamiltonian<i16, i64, u8>),
    HMi16U16(Hamiltonian<i16, i64, u16>),
    HMf32U8(Hamiltonian<f32, i64, u8>),
    HMf32U16(Hamiltonian<f32, i64, u16>),
    HMf64U8(Hamiltonian<f64, i64, u8>),
    HMf64U16(Hamiltonian<f64, i64, u16>),
    /// `complex64` (2 × f32).
    HMc32U8(Hamiltonian<num_complex::Complex<f32>, i64, u8>),
    HMc32U16(Hamiltonian<num_complex::Complex<f32>, i64, u16>),
    /// `complex128` (2 × f64).
    HMc64U8(Hamiltonian<num_complex::Complex<f64>, i64, u8>),
    HMc64U16(Hamiltonian<num_complex::Complex<f64>, i64, u16>),
}

impl HamiltonianInner {
    /// Build a `HamiltonianInner` from a type-erased `QMatrixInner` and a list
    /// of coefficient descriptors (one per cindex).
    ///
    /// Each entry is `None` for a static term (coefficient 1.0) or
    /// `Some(f)` for a time-dependent term.  The matrix element type `M` and
    /// cindex type `C` are inferred from the `QMatrixInner` variant.
    pub fn from_qmatrix_inner(
        qmatrix: QMatrixInner,
        coeff_fns: Vec<Option<CoeffFn>>,
    ) -> Result<Self, QuSpinError> {
        crate::with_qmatrix!(qmatrix, _M, _C, mat, {
            Ok(Hamiltonian::new(mat, coeff_fns)?.into_hamiltonian_inner())
        })
    }

    pub fn dim(&self) -> usize {
        with_hamiltonian!(self, _M, _C, h, { h.dim() })
    }

    pub fn num_coeff(&self) -> usize {
        with_hamiltonian!(self, _M, _C, h, { h.num_coeff() })
    }

    /// NumPy dtype name for the matrix element type (e.g. `"float64"`).
    pub fn dtype_name(&self) -> &'static str {
        match self {
            HamiltonianInner::HMi8U8(_) | HamiltonianInner::HMi8U16(_) => "int8",
            HamiltonianInner::HMi16U8(_) | HamiltonianInner::HMi16U16(_) => "int16",
            HamiltonianInner::HMf32U8(_) | HamiltonianInner::HMf32U16(_) => "float32",
            HamiltonianInner::HMf64U8(_) | HamiltonianInner::HMf64U16(_) => "float64",
            HamiltonianInner::HMc32U8(_) | HamiltonianInner::HMc32U16(_) => "complex64",
            HamiltonianInner::HMc64U8(_) | HamiltonianInner::HMc64U16(_) => "complex128",
        }
    }

    /// Element-wise addition.  Both operands must have the same dtype and cindex type.
    pub fn try_add(self, rhs: Self) -> Result<Self, QuSpinError> {
        macro_rules! add_variant {
            ($a:expr, $b:expr, $variant:ident) => {
                match $b {
                    HamiltonianInner::$variant(r) => Ok(HamiltonianInner::$variant($a + r)),
                    _ => Err(QuSpinError::ValueError(
                        "Hamiltonian dtype mismatch in addition".to_string(),
                    )),
                }
            };
        }
        match self {
            HamiltonianInner::HMi8U8(l) => add_variant!(l, rhs, HMi8U8),
            HamiltonianInner::HMi8U16(l) => add_variant!(l, rhs, HMi8U16),
            HamiltonianInner::HMi16U8(l) => add_variant!(l, rhs, HMi16U8),
            HamiltonianInner::HMi16U16(l) => add_variant!(l, rhs, HMi16U16),
            HamiltonianInner::HMf32U8(l) => add_variant!(l, rhs, HMf32U8),
            HamiltonianInner::HMf32U16(l) => add_variant!(l, rhs, HMf32U16),
            HamiltonianInner::HMf64U8(l) => add_variant!(l, rhs, HMf64U8),
            HamiltonianInner::HMf64U16(l) => add_variant!(l, rhs, HMf64U16),
            HamiltonianInner::HMc32U8(l) => add_variant!(l, rhs, HMc32U8),
            HamiltonianInner::HMc32U16(l) => add_variant!(l, rhs, HMc32U16),
            HamiltonianInner::HMc64U8(l) => add_variant!(l, rhs, HMc64U8),
            HamiltonianInner::HMc64U16(l) => add_variant!(l, rhs, HMc64U16),
        }
    }

    /// Element-wise subtraction.  Both operands must have the same dtype and cindex type.
    pub fn try_sub(self, rhs: Self) -> Result<Self, QuSpinError> {
        macro_rules! sub_variant {
            ($a:expr, $b:expr, $variant:ident) => {
                match $b {
                    HamiltonianInner::$variant(r) => Ok(HamiltonianInner::$variant($a - r)),
                    _ => Err(QuSpinError::ValueError(
                        "Hamiltonian dtype mismatch in subtraction".to_string(),
                    )),
                }
            };
        }
        match self {
            HamiltonianInner::HMi8U8(l) => sub_variant!(l, rhs, HMi8U8),
            HamiltonianInner::HMi8U16(l) => sub_variant!(l, rhs, HMi8U16),
            HamiltonianInner::HMi16U8(l) => sub_variant!(l, rhs, HMi16U8),
            HamiltonianInner::HMi16U16(l) => sub_variant!(l, rhs, HMi16U16),
            HamiltonianInner::HMf32U8(l) => sub_variant!(l, rhs, HMf32U8),
            HamiltonianInner::HMf32U16(l) => sub_variant!(l, rhs, HMf32U16),
            HamiltonianInner::HMf64U8(l) => sub_variant!(l, rhs, HMf64U8),
            HamiltonianInner::HMf64U16(l) => sub_variant!(l, rhs, HMf64U16),
            HamiltonianInner::HMc32U8(l) => sub_variant!(l, rhs, HMc32U8),
            HamiltonianInner::HMc32U16(l) => sub_variant!(l, rhs, HMc32U16),
            HamiltonianInner::HMc64U8(l) => sub_variant!(l, rhs, HMc64U8),
            HamiltonianInner::HMc64U16(l) => sub_variant!(l, rhs, HMc64U16),
        }
    }

    // ------------------------------------------------------------------
    // Time-parameterised output methods (output type fixed to Complex<f64>)
    // ------------------------------------------------------------------

    pub fn to_csr_nnz(&self, time: f64, drop_zeros: bool) -> Result<usize, QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, { h.to_csr_nnz(time, drop_zeros) })
    }

    #[allow(clippy::type_complexity)]
    pub fn to_csr(
        &self,
        time: f64,
        drop_zeros: bool,
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<Complex<f64>>), QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, { h.to_csr(time, drop_zeros) })
    }

    pub fn to_csr_into(
        &self,
        time: f64,
        drop_zeros: bool,
        indptr: &mut [i64],
        indices: &mut [i64],
        data: &mut [Complex<f64>],
    ) -> Result<(), QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, {
            h.to_csr_into(time, drop_zeros, indptr, indices, data)
        })
    }

    pub fn to_dense_into(&self, time: f64, output: &mut [Complex<f64>]) -> Result<(), QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, { h.to_dense_into(time, output) })
    }

    pub fn to_dense(&self, time: f64) -> Result<Vec<Complex<f64>>, QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, { h.to_dense(time) })
    }

    pub fn dot(
        &self,
        overwrite: bool,
        time: f64,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
    ) -> Result<(), QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, { h.dot(overwrite, time, input, output) })
    }

    pub fn dot_transpose(
        &self,
        overwrite: bool,
        time: f64,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
    ) -> Result<(), QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, {
            h.dot_transpose(overwrite, time, input, output)
        })
    }

    pub fn dot_many(
        &self,
        overwrite: bool,
        time: f64,
        input: ArrayView2<'_, Complex<f64>>,
        output: ArrayViewMut2<'_, Complex<f64>>,
    ) -> Result<(), QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, {
            h.dot_many(overwrite, time, input, output)
        })
    }

    pub fn dot_transpose_many(
        &self,
        overwrite: bool,
        time: f64,
        input: ArrayView2<'_, Complex<f64>>,
        output: ArrayViewMut2<'_, Complex<f64>>,
    ) -> Result<(), QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, {
            h.dot_transpose_many(overwrite, time, input, output)
        })
    }

    pub fn expm_dot(
        &self,
        time: f64,
        a: Complex<f64>,
        f: &mut [Complex<f64>],
    ) -> Result<(), QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, { h.expm_dot(time, a, f) })
    }

    pub fn expm_dot_many(
        &self,
        time: f64,
        a: Complex<f64>,
        f: ArrayViewMut2<'_, Complex<f64>>,
    ) -> Result<(), QuSpinError> {
        with_hamiltonian!(self, _M, _C, h, { h.expm_dot_many(time, a, f) })
    }
}

// ---------------------------------------------------------------------------
// IntoHamiltonianInner
// ---------------------------------------------------------------------------

/// Convert a concrete `Hamiltonian<M, i64, C>` to the type-erased `HamiltonianInner`.
pub trait IntoHamiltonianInner {
    fn into_hamiltonian_inner(self) -> HamiltonianInner;
}

macro_rules! impl_into_hamiltonian_inner {
    ($M:ty, $C:ty, $variant:ident) => {
        impl IntoHamiltonianInner for Hamiltonian<$M, i64, $C> {
            #[inline]
            fn into_hamiltonian_inner(self) -> HamiltonianInner {
                HamiltonianInner::$variant(self)
            }
        }
    };
}

impl_into_hamiltonian_inner!(i8, u8, HMi8U8);
impl_into_hamiltonian_inner!(i8, u16, HMi8U16);
impl_into_hamiltonian_inner!(i16, u8, HMi16U8);
impl_into_hamiltonian_inner!(i16, u16, HMi16U16);
impl_into_hamiltonian_inner!(f32, u8, HMf32U8);
impl_into_hamiltonian_inner!(f32, u16, HMf32U16);
impl_into_hamiltonian_inner!(f64, u8, HMf64U8);
impl_into_hamiltonian_inner!(f64, u16, HMf64U16);
impl_into_hamiltonian_inner!(num_complex::Complex<f32>, u8, HMc32U8);
impl_into_hamiltonian_inner!(num_complex::Complex<f32>, u16, HMc32U16);
impl_into_hamiltonian_inner!(num_complex::Complex<f64>, u8, HMc64U8);
impl_into_hamiltonian_inner!(num_complex::Complex<f64>, u16, HMc64U16);
