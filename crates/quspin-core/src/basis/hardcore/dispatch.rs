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
use crate::basis::{
    BasisSpace,
    seed::{seed_from_bytes, state_to_str},
    space::{FullSpace, Subspace},
    sym::SymmetricSubspace,
    symmetry::group::SymGrpInner,
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
    Sym32(SymmetricSubspace<SymGrpInner<u32>>),
    Sym64(SymmetricSubspace<SymGrpInner<u64>>),
    Sym128(SymmetricSubspace<SymGrpInner<B128>>),
    Sym256(SymmetricSubspace<SymGrpInner<B256>>),
    Sym512(SymmetricSubspace<SymGrpInner<B512>>),
    Sym1024(SymmetricSubspace<SymGrpInner<B1024>>),
    Sym2048(SymmetricSubspace<SymGrpInner<B2048>>),
    Sym4096(SymmetricSubspace<SymGrpInner<B4096>>),
    Sym8192(SymmetricSubspace<SymGrpInner<B8192>>),
}

impl HardcoreBasisInner {
    /// Number of lattice sites.
    pub fn n_sites(&self) -> usize {
        match self {
            HardcoreBasisInner::Full32(b) => b.n_sites(),
            HardcoreBasisInner::Full64(b) => b.n_sites(),
            HardcoreBasisInner::Sub32(b) => b.n_sites(),
            HardcoreBasisInner::Sub64(b) => b.n_sites(),
            HardcoreBasisInner::Sub128(b) => b.n_sites(),
            HardcoreBasisInner::Sub256(b) => b.n_sites(),
            HardcoreBasisInner::Sub512(b) => b.n_sites(),
            HardcoreBasisInner::Sub1024(b) => b.n_sites(),
            HardcoreBasisInner::Sub2048(b) => b.n_sites(),
            HardcoreBasisInner::Sub4096(b) => b.n_sites(),
            HardcoreBasisInner::Sub8192(b) => b.n_sites(),
            HardcoreBasisInner::Sym32(b) => b.n_sites(),
            HardcoreBasisInner::Sym64(b) => b.n_sites(),
            HardcoreBasisInner::Sym128(b) => b.n_sites(),
            HardcoreBasisInner::Sym256(b) => b.n_sites(),
            HardcoreBasisInner::Sym512(b) => b.n_sites(),
            HardcoreBasisInner::Sym1024(b) => b.n_sites(),
            HardcoreBasisInner::Sym2048(b) => b.n_sites(),
            HardcoreBasisInner::Sym4096(b) => b.n_sites(),
            HardcoreBasisInner::Sym8192(b) => b.n_sites(),
        }
    }

    /// Number of basis states.
    pub fn size(&self) -> usize {
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

    /// Return the `i`-th basis state as a `'0'`/`'1'` string (site 0 = index 0).
    pub fn state_at_str(&self, i: usize) -> String {
        match self {
            HardcoreBasisInner::Full32(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Full64(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sub32(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sub64(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sub128(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sub256(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sub512(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sub1024(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sub2048(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sub4096(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sub8192(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sym32(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sym64(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sym128(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sym256(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sym512(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sym1024(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sym2048(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sym4096(b) => state_to_str(b.state_at(i), b.n_sites()),
            HardcoreBasisInner::Sym8192(b) => state_to_str(b.state_at(i), b.n_sites()),
        }
    }

    /// Look up the index of the state encoded as a site-occupation byte slice.
    ///
    /// Returns `None` if the state is not in the basis.
    pub fn index_of_bytes(&self, bytes: &[u8]) -> Option<usize> {
        match self {
            HardcoreBasisInner::Full32(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Full64(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sub32(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sub64(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sub128(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sub256(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sub512(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sub1024(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sub2048(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sub4096(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sub8192(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sym32(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sym64(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sym128(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sym256(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sym512(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sym1024(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sym2048(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sym4096(b) => b.index(seed_from_bytes(bytes)),
            HardcoreBasisInner::Sym8192(b) => b.index(seed_from_bytes(bytes)),
        }
    }

    /// One of `"full"`, `"subspace"`, or `"symmetric"`.
    pub fn kind(&self) -> &'static str {
        match self {
            HardcoreBasisInner::Full32(_) | HardcoreBasisInner::Full64(_) => "full",
            HardcoreBasisInner::Sub32(_)
            | HardcoreBasisInner::Sub64(_)
            | HardcoreBasisInner::Sub128(_)
            | HardcoreBasisInner::Sub256(_)
            | HardcoreBasisInner::Sub512(_)
            | HardcoreBasisInner::Sub1024(_)
            | HardcoreBasisInner::Sub2048(_)
            | HardcoreBasisInner::Sub4096(_)
            | HardcoreBasisInner::Sub8192(_) => "subspace",
            HardcoreBasisInner::Sym32(_)
            | HardcoreBasisInner::Sym64(_)
            | HardcoreBasisInner::Sym128(_)
            | HardcoreBasisInner::Sym256(_)
            | HardcoreBasisInner::Sym512(_)
            | HardcoreBasisInner::Sym1024(_)
            | HardcoreBasisInner::Sym2048(_)
            | HardcoreBasisInner::Sym4096(_)
            | HardcoreBasisInner::Sym8192(_) => "symmetric",
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
// Display
// ---------------------------------------------------------------------------

const DISPLAY_HEAD: usize = 25;
const DISPLAY_TAIL: usize = 25;

impl std::fmt::Display for HardcoreBasisInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = self.size();
        let sym_display = if self.is_symmetric() {
            "[symmetric]"
        } else {
            "[]"
        };
        let index_width = size.saturating_sub(1).to_string().len();

        write!(
            f,
            "{}(n_sites={}, size={}, symmetries={}):",
            self.kind(),
            self.n_sites(),
            size,
            sym_display,
        )?;

        let truncate = size > DISPLAY_HEAD + DISPLAY_TAIL;
        let indices: Box<dyn Iterator<Item = usize>> = if truncate {
            Box::new((0..DISPLAY_HEAD).chain(size - DISPLAY_TAIL..size))
        } else {
            Box::new(0..size)
        };

        let mut prev: Option<usize> = None;
        for i in indices {
            if truncate
                && let Some(p) = prev
                && i > p + 1
            {
                write!(f, "\n  {:>width$}", "...", width = index_width + 1)?;
            }
            write!(
                f,
                "\n  {:>width$}. |{}>",
                i,
                self.state_at_str(i),
                width = index_width,
            )?;
            prev = Some(i);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// From impls — wrap a concrete basis space without naming the variant
// ---------------------------------------------------------------------------

macro_rules! impl_from_basis_spaces {
    ($B:ty, $sub_variant:ident, $sym_variant:ident) => {
        impl From<Subspace<$B>> for HardcoreBasisInner {
            #[inline]
            fn from(b: Subspace<$B>) -> Self {
                HardcoreBasisInner::$sub_variant(b)
            }
        }
        impl From<SymmetricSubspace<SymGrpInner<$B>>> for HardcoreBasisInner {
            #[inline]
            fn from(b: SymmetricSubspace<SymGrpInner<$B>>) -> Self {
                HardcoreBasisInner::$sym_variant(b)
            }
        }
    };
}

impl_from_basis_spaces!(u32, Sub32, Sym32);
impl_from_basis_spaces!(u64, Sub64, Sym64);
impl_from_basis_spaces!(B128, Sub128, Sym128);
impl_from_basis_spaces!(B256, Sub256, Sym256);
impl_from_basis_spaces!(B512, Sub512, Sym512);
impl_from_basis_spaces!(B1024, Sub1024, Sym1024);
impl_from_basis_spaces!(B2048, Sub2048, Sym2048);
impl_from_basis_spaces!(B4096, Sub4096, Sym4096);
impl_from_basis_spaces!(B8192, Sub8192, Sym8192);

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

/// Select the smallest `B: BitInt` that fits `$n_sites` site indices, inject
/// it as a local type alias `$B`, and evaluate `$body`.
///
/// The ladder is: ≤32 → `u32`, ≤64 → `u64`, ≤128 → `Uint<128,2>`, …,
/// ≤8192 → `Uint<8192,128>`.
///
/// `$on_overflow` is evaluated (and must diverge or return) when
/// `n_sites > 8192`.  Each FFI consumer supplies its own expression:
///
/// ```rust,ignore
/// // quspin-py
/// select_b_for_n_sites!(n, B,
///     return Err(pyo3::exceptions::PyValueError::new_err("n_sites > 8192")),
///     { ... }
/// );
///
/// // quspin-c
/// select_b_for_n_sites!(n, B,
///     return write_error(err, QuSpinError::ValueError("n_sites > 8192".into())),
///     { ... }
/// );
/// ```
#[macro_export]
macro_rules! select_b_for_n_sites {
    ($n_sites:expr, $B:ident, $on_overflow:expr, $body:block) => {
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
            $on_overflow
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::space::{FullSpace, Subspace};

    #[test]
    fn display_full_space() {
        let inner = HardcoreBasisInner::Full32(FullSpace::new(2, 4));
        let s = inner.to_string();
        assert!(s.starts_with("full(n_sites=2, size=4, symmetries=[]):"));
        assert!(s.contains("|11>"));
        assert!(s.contains("|00>"));
    }

    #[test]
    fn display_subspace() {
        let mut sub = Subspace::<u32>::new(2);
        sub.build(0b01u32, |s| {
            vec![(num_complex::Complex::new(1.0, 0.0), s ^ 0b11, 0u8)]
        });
        let inner = HardcoreBasisInner::Sub32(sub);
        let s = inner.to_string();
        assert!(s.starts_with("subspace(n_sites=2, size="));
        assert!(s.contains("symmetries=[]"));
    }

    #[test]
    fn display_index_alignment() {
        // 16 states → indices 0-15, width 2; row 9 and 10 should be right-aligned
        let inner = HardcoreBasisInner::Full32(FullSpace::new(4, 16));
        let s = inner.to_string();
        assert!(s.contains("  9."));
        assert!(s.contains(" 10."));
    }

    #[test]
    fn display_truncation() {
        // 64 states > 50 → should truncate with "..."
        let inner = HardcoreBasisInner::Full32(FullSpace::new(6, 64));
        let s = inner.to_string();
        assert!(s.contains("..."), "expected truncation marker");
        // First 25 rows present (index 0 and 24)
        assert!(s.contains("\n   0."), "expected row 0");
        assert!(s.contains("\n  24."), "expected row 24");
        // Row 25 should be absent (truncated)
        assert!(!s.contains("\n  25."), "row 25 should be truncated");
        // Last 25 rows present (index 39 and 63)
        assert!(s.contains("\n  39."), "expected row 39");
        assert!(s.contains("\n  63."), "expected row 63");
    }
}
