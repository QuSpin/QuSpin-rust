/// Dispatch layer between Python and the generic `quspin-core` types.
///
/// Defines the inner enums that erase the generic type parameters of
/// `FullSpace<B>`, `Subspace<B>`, `SymmetricSubspace<B>`, and
/// `QMatrix<V, I, C>`, plus a set of `macro_rules!` dispatch macros that
/// restore those parameters in a match arm so the body can call generic core
/// functions with static dispatch.
///
/// # Design
///
/// All variant types are concrete — no trait objects, no virtual calls.
/// Dispatch happens once at the PyO3 boundary; everything downstream is
/// statically monomorphised.
///
/// ## Supported basis sizes
///
/// | Variant suffix | Rust type                  | Bit width |
/// |----------------|----------------------------|-----------|
/// | `32`           | `u32`                      | 32        |
/// | `64`           | `u64`                      | 64        |
/// | `128`          | `ruint::Uint<128,  2>`     | 128       |
/// | `256`          | `ruint::Uint<256,  4>`     | 256       |
/// | `512`          | `ruint::Uint<512,  8>`     | 512       |
/// | `1024`         | `ruint::Uint<1024, 16>`    | 1024      |
/// | `2048`         | `ruint::Uint<2048, 32>`    | 2048      |
/// | `4096`         | `ruint::Uint<4096, 64>`    | 4096      |
/// | `8192`         | `ruint::Uint<8192, 128>`   | 8192      |
///
/// `FullSpace` is only instantiated for `u32` and `u64`; larger full spaces
/// are not physically meaningful for hardcore bosons.
use quspin_core::basis::{
    space::{FullSpace, Subspace},
    sym::SymmetricSubspace,
};
use quspin_core::qmatrix::QMatrix;

// Type alias for the ruint sizes used in Sub/Sym variants.
type B128 = ruint::Uint<128, 2>;
type B256 = ruint::Uint<256, 4>;
type B512 = ruint::Uint<512, 8>;
type B1024 = ruint::Uint<1024, 16>;
type B2048 = ruint::Uint<2048, 32>;
type B4096 = ruint::Uint<4096, 64>;
type B8192 = ruint::Uint<8192, 128>;

// ---------------------------------------------------------------------------
// HardcoreBasisInner — erases the BitInt type parameter of basis spaces
// ---------------------------------------------------------------------------

/// Type-erased wrapper for the three basis-space variants over all supported
/// integer widths.
///
/// 16 variants total:
/// - 2 `Full` variants (u32, u64)
/// - 7 `Sub` variants (128–8192 bit ruint integers)
/// - 7 `Sym` variants (128–8192 bit ruint integers)
pub enum HardcoreBasisInner {
    // Full Hilbert spaces (small n_sites only).
    Full32(FullSpace<u32>),
    Full64(FullSpace<u64>),

    // Subspaces (particle-number or energy sector).
    Sub128(Subspace<B128>),
    Sub256(Subspace<B256>),
    Sub512(Subspace<B512>),
    Sub1024(Subspace<B1024>),
    Sub2048(Subspace<B2048>),
    Sub4096(Subspace<B4096>),
    Sub8192(Subspace<B8192>),

    // Symmetry-reduced subspaces.
    Sym128(SymmetricSubspace<B128>),
    Sym256(SymmetricSubspace<B256>),
    Sym512(SymmetricSubspace<B512>),
    Sym1024(SymmetricSubspace<B1024>),
    Sym2048(SymmetricSubspace<B2048>),
    Sym4096(SymmetricSubspace<B4096>),
    Sym8192(SymmetricSubspace<B8192>),
}

impl HardcoreBasisInner {
    /// Number of basis states.
    pub fn size(&self) -> usize {
        use quspin_core::basis::BasisSpace;
        match self {
            HardcoreBasisInner::Full32(b) => b.size(),
            HardcoreBasisInner::Full64(b) => b.size(),
            HardcoreBasisInner::Sub128(b) => b.size(),
            HardcoreBasisInner::Sub256(b) => b.size(),
            HardcoreBasisInner::Sub512(b) => b.size(),
            HardcoreBasisInner::Sub1024(b) => b.size(),
            HardcoreBasisInner::Sub2048(b) => b.size(),
            HardcoreBasisInner::Sub4096(b) => b.size(),
            HardcoreBasisInner::Sub8192(b) => b.size(),
            HardcoreBasisInner::Sym128(b) => b.size(),
            HardcoreBasisInner::Sym256(b) => b.size(),
            HardcoreBasisInner::Sym512(b) => b.size(),
            HardcoreBasisInner::Sym1024(b) => b.size(),
            HardcoreBasisInner::Sym2048(b) => b.size(),
            HardcoreBasisInner::Sym4096(b) => b.size(),
            HardcoreBasisInner::Sym8192(b) => b.size(),
        }
    }

    /// Returns `true` for `Sym*` variants (symmetry-reduced subspaces).
    pub fn is_symmetric(&self) -> bool {
        matches!(
            self,
            HardcoreBasisInner::Sym128(_)
                | HardcoreBasisInner::Sym256(_)
                | HardcoreBasisInner::Sym512(_)
                | HardcoreBasisInner::Sym1024(_)
                | HardcoreBasisInner::Sym2048(_)
                | HardcoreBasisInner::Sym4096(_)
                | HardcoreBasisInner::Sym8192(_)
        )
    }
}

// ---------------------------------------------------------------------------
// QMatrixInner — erases the V, I, C type parameters of QMatrix
// ---------------------------------------------------------------------------

/// Type-erased wrapper for `QMatrix<V, i64, C>`.
///
/// 12 variants: 6 value types × 2 cindex types.  The index type `I` is
/// fixed to `i64` at the PyO3 boundary for all basis sizes.
///
/// Naming convention: `QM` prefix, then value type abbreviation, then cindex
/// type abbreviation.  For example `QMf64U8` is `QMatrix<f64, i64, u8>`.
pub enum QMatrixInner {
    QMi8U8(QMatrix<i8, i64, u8>),
    QMi8U16(QMatrix<i8, i64, u16>),
    QMi16U8(QMatrix<i16, i64, u8>),
    QMi16U16(QMatrix<i16, i64, u16>),
    QMf32U8(QMatrix<f32, i64, u8>),
    QMf32U16(QMatrix<f32, i64, u16>),
    QMf64U8(QMatrix<f64, i64, u8>),
    QMf64U16(QMatrix<f64, i64, u16>),
    /// `complex64` (2 × f32).
    QMc32U8(QMatrix<num_complex::Complex<f32>, i64, u8>),
    QMc32U16(QMatrix<num_complex::Complex<f32>, i64, u16>),
    /// `complex128` (2 × f64).
    QMc64U8(QMatrix<num_complex::Complex<f64>, i64, u8>),
    QMc64U16(QMatrix<num_complex::Complex<f64>, i64, u16>),
}

impl QMatrixInner {
    pub fn dim(&self) -> usize {
        match self {
            QMatrixInner::QMi8U8(m) => m.dim(),
            QMatrixInner::QMi8U16(m) => m.dim(),
            QMatrixInner::QMi16U8(m) => m.dim(),
            QMatrixInner::QMi16U16(m) => m.dim(),
            QMatrixInner::QMf32U8(m) => m.dim(),
            QMatrixInner::QMf32U16(m) => m.dim(),
            QMatrixInner::QMf64U8(m) => m.dim(),
            QMatrixInner::QMf64U16(m) => m.dim(),
            QMatrixInner::QMc32U8(m) => m.dim(),
            QMatrixInner::QMc32U16(m) => m.dim(),
            QMatrixInner::QMc64U8(m) => m.dim(),
            QMatrixInner::QMc64U16(m) => m.dim(),
        }
    }

    pub fn nnz(&self) -> usize {
        match self {
            QMatrixInner::QMi8U8(m) => m.nnz(),
            QMatrixInner::QMi8U16(m) => m.nnz(),
            QMatrixInner::QMi16U8(m) => m.nnz(),
            QMatrixInner::QMi16U16(m) => m.nnz(),
            QMatrixInner::QMf32U8(m) => m.nnz(),
            QMatrixInner::QMf32U16(m) => m.nnz(),
            QMatrixInner::QMf64U8(m) => m.nnz(),
            QMatrixInner::QMf64U16(m) => m.nnz(),
            QMatrixInner::QMc32U8(m) => m.nnz(),
            QMatrixInner::QMc32U16(m) => m.nnz(),
            QMatrixInner::QMc64U8(m) => m.nnz(),
            QMatrixInner::QMc64U16(m) => m.nnz(),
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------
//
// Each macro matches on a discriminant (enum value or reference) and executes
// a body block with local type aliases injected for the erased type parameters.
//
// Usage pattern:
//
//   with_value_dtype!(v_dtype, V, {
//       with_cindex_dtype!(c_dtype, C, {
//           with_basis!(basis_inner, B, basis, {
//               let mat: QMatrix<V, i64, C> = build_from_basis::<B, V, i64, C>(&ham, basis);
//               QMatrixInner::...  // wrap
//           })
//       })
//   })

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

/// Match on a `HardcoreBasisInner` reference, injecting a type alias `$B` for
/// the concrete `BitInt` type and binding `$basis` to the inner basis reference.
///
/// `$inner` must be an expression of type `&HardcoreBasisInner`.
#[macro_export]
macro_rules! with_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::dispatch::HardcoreBasisInner::Full32($basis) => {
                type $B = u32;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Full64($basis) => {
                type $B = u64;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::dispatch::HardcoreBasisInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
        }
    };
}

/// Match on a `&QMatrixInner`, injecting type aliases `$V` and `$C` and
/// binding `$mat` to the inner `QMatrix` reference.
#[macro_export]
macro_rules! with_qmatrix {
    ($inner:expr, $V:ident, $C:ident, $mat:ident, $body:block) => {
        match $inner {
            $crate::dispatch::QMatrixInner::QMi8U8($mat) => {
                type $V = i8;
                type $C = u8;
                $body
            }
            $crate::dispatch::QMatrixInner::QMi8U16($mat) => {
                type $V = i8;
                type $C = u16;
                $body
            }
            $crate::dispatch::QMatrixInner::QMi16U8($mat) => {
                type $V = i16;
                type $C = u8;
                $body
            }
            $crate::dispatch::QMatrixInner::QMi16U16($mat) => {
                type $V = i16;
                type $C = u16;
                $body
            }
            $crate::dispatch::QMatrixInner::QMf32U8($mat) => {
                type $V = f32;
                type $C = u8;
                $body
            }
            $crate::dispatch::QMatrixInner::QMf32U16($mat) => {
                type $V = f32;
                type $C = u16;
                $body
            }
            $crate::dispatch::QMatrixInner::QMf64U8($mat) => {
                type $V = f64;
                type $C = u8;
                $body
            }
            $crate::dispatch::QMatrixInner::QMf64U16($mat) => {
                type $V = f64;
                type $C = u16;
                $body
            }
            $crate::dispatch::QMatrixInner::QMc32U8($mat) => {
                type $V = ::num_complex::Complex<f32>;
                type $C = u8;
                $body
            }
            $crate::dispatch::QMatrixInner::QMc32U16($mat) => {
                type $V = ::num_complex::Complex<f32>;
                type $C = u16;
                $body
            }
            $crate::dispatch::QMatrixInner::QMc64U8($mat) => {
                type $V = ::num_complex::Complex<f64>;
                type $C = u8;
                $body
            }
            $crate::dispatch::QMatrixInner::QMc64U16($mat) => {
                type $V = ::num_complex::Complex<f64>;
                type $C = u16;
                $body
            }
        }
    };
}
