/// Type-erased `HardcoreBasisInner` and its dispatch macros.
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

type B128 = ruint::Uint<128, 2>;
type B256 = ruint::Uint<256, 4>;
type B512 = ruint::Uint<512, 8>;
type B1024 = ruint::Uint<1024, 16>;
type B2048 = ruint::Uint<2048, 32>;
type B4096 = ruint::Uint<4096, 64>;
type B8192 = ruint::Uint<8192, 128>;

// ---------------------------------------------------------------------------
// HardcoreBasisInner
// ---------------------------------------------------------------------------

/// Type-erased wrapper for the three basis-space variants over all supported
/// integer widths.
///
/// 20 variants total:
/// - 2 `Full` variants (u32, u64)
/// - 9 `Sub` variants (u32, u64, and 128–8192 bit ruint integers)
/// - 9 `Sym` variants (u32, u64, and 128–8192 bit ruint integers)
pub enum HardcoreBasisInner {
    // Full Hilbert spaces (small n_sites only).
    Full32(FullSpace<u32>),
    Full64(FullSpace<u64>),

    // Subspaces (particle-number or energy sector).
    Sub32(Subspace<u32>),
    Sub64(Subspace<u64>),
    Sub128(Subspace<B128>),
    Sub256(Subspace<B256>),
    Sub512(Subspace<B512>),
    Sub1024(Subspace<B1024>),
    Sub2048(Subspace<B2048>),
    Sub4096(Subspace<B4096>),
    Sub8192(Subspace<B8192>),

    // Symmetry-reduced subspaces.
    Sym32(SymmetricSubspace<u32>),
    Sym64(SymmetricSubspace<u64>),
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
            HardcoreBasisInner::Sub32(b) => b.size(),
            HardcoreBasisInner::Sub64(b) => b.size(),
            HardcoreBasisInner::Sub128(b) => b.size(),
            HardcoreBasisInner::Sub256(b) => b.size(),
            HardcoreBasisInner::Sub512(b) => b.size(),
            HardcoreBasisInner::Sub1024(b) => b.size(),
            HardcoreBasisInner::Sub2048(b) => b.size(),
            HardcoreBasisInner::Sub4096(b) => b.size(),
            HardcoreBasisInner::Sub8192(b) => b.size(),
            HardcoreBasisInner::Sym32(b) => b.size(),
            HardcoreBasisInner::Sym64(b) => b.size(),
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
            HardcoreBasisInner::Sym32(_)
                | HardcoreBasisInner::Sym64(_)
                | HardcoreBasisInner::Sym128(_)
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
// Dispatch macros
// ---------------------------------------------------------------------------

/// Match on a `HardcoreBasisInner` reference, injecting a type alias `$B` for
/// the concrete `BitInt` type and binding `$basis` to the inner basis reference.
///
/// Covers all 20 variants (Full*, Sub*, Sym*).
#[macro_export]
macro_rules! with_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Full32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Full64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
        }
    };
}

/// Like `with_basis!` but restricted to Full* and Sub* (non-symmetric) variants.
#[macro_export]
macro_rules! with_plain_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Full32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Full64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_plain_basis! called on a symmetric variant"),
        }
    };
}

/// Like `with_basis!` but restricted to Sym* (symmetric) variants.
#[macro_export]
macro_rules! with_sym_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::hardcore::dispatch::HardcoreBasisInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_sym_basis! called on a non-symmetric variant"),
        }
    };
}
