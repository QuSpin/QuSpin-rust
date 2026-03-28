/// Type-erased `BasisInner` and its dispatch macros.
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
/// are not physically meaningful.
use crate::basis::{
    BasisSpace,
    seed::{seed_from_bytes, state_to_str},
    space::{FullSpace, Subspace},
    sym::SymmetricSubspace,
    sym_grp::{DitGrpInner, HardcoreGrpInner},
};

type B128 = ruint::Uint<128, 2>;
type B256 = ruint::Uint<256, 4>;
type B512 = ruint::Uint<512, 8>;
type B1024 = ruint::Uint<1024, 16>;
type B2048 = ruint::Uint<2048, 32>;
type B4096 = ruint::Uint<4096, 64>;
type B8192 = ruint::Uint<8192, 128>;

// ---------------------------------------------------------------------------
// BasisInner
// ---------------------------------------------------------------------------

/// Type-erased wrapper for all basis-space variants over all supported
/// integer widths.
///
/// 29 variants total:
/// - 2 `Full` variants (u32, u64)
/// - 9 `Sub` variants (u32, u64, and 128–8192 bit ruint integers)
/// - 9 `Sym` variants — LHSS=2 symmetric (hardcore bosons / spin-½ / fermions)
/// - 9 `DitSym` variants — LHSS≥3 symmetric (bosons / higher spin)
pub enum BasisInner {
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

    // LHSS=2 symmetry-reduced subspaces (hardcore bosons / spin-½ / fermions).
    Sym32(SymmetricSubspace<HardcoreGrpInner<u32>>),
    Sym64(SymmetricSubspace<HardcoreGrpInner<u64>>),
    Sym128(SymmetricSubspace<HardcoreGrpInner<B128>>),
    Sym256(SymmetricSubspace<HardcoreGrpInner<B256>>),
    Sym512(SymmetricSubspace<HardcoreGrpInner<B512>>),
    Sym1024(SymmetricSubspace<HardcoreGrpInner<B1024>>),
    Sym2048(SymmetricSubspace<HardcoreGrpInner<B2048>>),
    Sym4096(SymmetricSubspace<HardcoreGrpInner<B4096>>),
    Sym8192(SymmetricSubspace<HardcoreGrpInner<B8192>>),

    // LHSS≥3 symmetry-reduced subspaces (bosons / higher spin).
    DitSym32(SymmetricSubspace<DitGrpInner<u32>>),
    DitSym64(SymmetricSubspace<DitGrpInner<u64>>),
    DitSym128(SymmetricSubspace<DitGrpInner<B128>>),
    DitSym256(SymmetricSubspace<DitGrpInner<B256>>),
    DitSym512(SymmetricSubspace<DitGrpInner<B512>>),
    DitSym1024(SymmetricSubspace<DitGrpInner<B1024>>),
    DitSym2048(SymmetricSubspace<DitGrpInner<B2048>>),
    DitSym4096(SymmetricSubspace<DitGrpInner<B4096>>),
    DitSym8192(SymmetricSubspace<DitGrpInner<B8192>>),
}

impl BasisInner {
    /// Number of lattice sites.
    pub fn n_sites(&self) -> usize {
        match self {
            BasisInner::Full32(b) => b.n_sites(),
            BasisInner::Full64(b) => b.n_sites(),
            BasisInner::Sub32(b) => b.n_sites(),
            BasisInner::Sub64(b) => b.n_sites(),
            BasisInner::Sub128(b) => b.n_sites(),
            BasisInner::Sub256(b) => b.n_sites(),
            BasisInner::Sub512(b) => b.n_sites(),
            BasisInner::Sub1024(b) => b.n_sites(),
            BasisInner::Sub2048(b) => b.n_sites(),
            BasisInner::Sub4096(b) => b.n_sites(),
            BasisInner::Sub8192(b) => b.n_sites(),
            BasisInner::Sym32(b) => b.n_sites(),
            BasisInner::Sym64(b) => b.n_sites(),
            BasisInner::Sym128(b) => b.n_sites(),
            BasisInner::Sym256(b) => b.n_sites(),
            BasisInner::Sym512(b) => b.n_sites(),
            BasisInner::Sym1024(b) => b.n_sites(),
            BasisInner::Sym2048(b) => b.n_sites(),
            BasisInner::Sym4096(b) => b.n_sites(),
            BasisInner::Sym8192(b) => b.n_sites(),
            BasisInner::DitSym32(b) => b.n_sites(),
            BasisInner::DitSym64(b) => b.n_sites(),
            BasisInner::DitSym128(b) => b.n_sites(),
            BasisInner::DitSym256(b) => b.n_sites(),
            BasisInner::DitSym512(b) => b.n_sites(),
            BasisInner::DitSym1024(b) => b.n_sites(),
            BasisInner::DitSym2048(b) => b.n_sites(),
            BasisInner::DitSym4096(b) => b.n_sites(),
            BasisInner::DitSym8192(b) => b.n_sites(),
        }
    }

    /// Number of basis states.
    pub fn size(&self) -> usize {
        match self {
            BasisInner::Full32(b) => b.size(),
            BasisInner::Full64(b) => b.size(),
            BasisInner::Sub32(b) => b.size(),
            BasisInner::Sub64(b) => b.size(),
            BasisInner::Sub128(b) => b.size(),
            BasisInner::Sub256(b) => b.size(),
            BasisInner::Sub512(b) => b.size(),
            BasisInner::Sub1024(b) => b.size(),
            BasisInner::Sub2048(b) => b.size(),
            BasisInner::Sub4096(b) => b.size(),
            BasisInner::Sub8192(b) => b.size(),
            BasisInner::Sym32(b) => b.size(),
            BasisInner::Sym64(b) => b.size(),
            BasisInner::Sym128(b) => b.size(),
            BasisInner::Sym256(b) => b.size(),
            BasisInner::Sym512(b) => b.size(),
            BasisInner::Sym1024(b) => b.size(),
            BasisInner::Sym2048(b) => b.size(),
            BasisInner::Sym4096(b) => b.size(),
            BasisInner::Sym8192(b) => b.size(),
            BasisInner::DitSym32(b) => b.size(),
            BasisInner::DitSym64(b) => b.size(),
            BasisInner::DitSym128(b) => b.size(),
            BasisInner::DitSym256(b) => b.size(),
            BasisInner::DitSym512(b) => b.size(),
            BasisInner::DitSym1024(b) => b.size(),
            BasisInner::DitSym2048(b) => b.size(),
            BasisInner::DitSym4096(b) => b.size(),
            BasisInner::DitSym8192(b) => b.size(),
        }
    }

    /// Return the `i`-th basis state as a bit-string (site 0 = index 0).
    pub fn state_at_str(&self, i: usize) -> String {
        match self {
            BasisInner::Full32(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Full64(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sub32(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sub64(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sub128(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sub256(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sub512(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sub1024(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sub2048(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sub4096(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sub8192(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sym32(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sym64(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sym128(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sym256(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sym512(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sym1024(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sym2048(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sym4096(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::Sym8192(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::DitSym32(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::DitSym64(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::DitSym128(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::DitSym256(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::DitSym512(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::DitSym1024(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::DitSym2048(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::DitSym4096(b) => state_to_str(b.state_at(i), b.n_sites()),
            BasisInner::DitSym8192(b) => state_to_str(b.state_at(i), b.n_sites()),
        }
    }

    /// Look up the index of the state encoded as a site-occupation byte slice.
    ///
    /// Returns `None` if the state is not in the basis.
    pub fn index_of_bytes(&self, bytes: &[u8]) -> Option<usize> {
        match self {
            BasisInner::Full32(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Full64(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sub32(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sub64(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sub128(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sub256(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sub512(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sub1024(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sub2048(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sub4096(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sub8192(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sym32(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sym64(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sym128(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sym256(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sym512(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sym1024(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sym2048(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sym4096(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::Sym8192(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::DitSym32(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::DitSym64(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::DitSym128(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::DitSym256(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::DitSym512(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::DitSym1024(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::DitSym2048(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::DitSym4096(b) => b.index(seed_from_bytes(bytes)),
            BasisInner::DitSym8192(b) => b.index(seed_from_bytes(bytes)),
        }
    }

    /// One of `"full"`, `"subspace"`, or `"symmetric"`.
    pub fn kind(&self) -> &'static str {
        match self {
            BasisInner::Full32(_) | BasisInner::Full64(_) => "full",
            BasisInner::Sub32(_)
            | BasisInner::Sub64(_)
            | BasisInner::Sub128(_)
            | BasisInner::Sub256(_)
            | BasisInner::Sub512(_)
            | BasisInner::Sub1024(_)
            | BasisInner::Sub2048(_)
            | BasisInner::Sub4096(_)
            | BasisInner::Sub8192(_) => "subspace",
            BasisInner::Sym32(_)
            | BasisInner::Sym64(_)
            | BasisInner::Sym128(_)
            | BasisInner::Sym256(_)
            | BasisInner::Sym512(_)
            | BasisInner::Sym1024(_)
            | BasisInner::Sym2048(_)
            | BasisInner::Sym4096(_)
            | BasisInner::Sym8192(_)
            | BasisInner::DitSym32(_)
            | BasisInner::DitSym64(_)
            | BasisInner::DitSym128(_)
            | BasisInner::DitSym256(_)
            | BasisInner::DitSym512(_)
            | BasisInner::DitSym1024(_)
            | BasisInner::DitSym2048(_)
            | BasisInner::DitSym4096(_)
            | BasisInner::DitSym8192(_) => "symmetric",
        }
    }

    /// Returns `true` for `Sym*` and `DitSym*` variants (symmetry-reduced subspaces).
    pub fn is_symmetric(&self) -> bool {
        matches!(
            self,
            BasisInner::Sym32(_)
                | BasisInner::Sym64(_)
                | BasisInner::Sym128(_)
                | BasisInner::Sym256(_)
                | BasisInner::Sym512(_)
                | BasisInner::Sym1024(_)
                | BasisInner::Sym2048(_)
                | BasisInner::Sym4096(_)
                | BasisInner::Sym8192(_)
                | BasisInner::DitSym32(_)
                | BasisInner::DitSym64(_)
                | BasisInner::DitSym128(_)
                | BasisInner::DitSym256(_)
                | BasisInner::DitSym512(_)
                | BasisInner::DitSym1024(_)
                | BasisInner::DitSym2048(_)
                | BasisInner::DitSym4096(_)
                | BasisInner::DitSym8192(_)
        )
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

const DISPLAY_HEAD: usize = 25;
const DISPLAY_TAIL: usize = 25;

impl std::fmt::Display for BasisInner {
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
    ($B:ty, $sub_variant:ident, $sym_variant:ident, $dit_sym_variant:ident) => {
        impl From<Subspace<$B>> for BasisInner {
            #[inline]
            fn from(b: Subspace<$B>) -> Self {
                BasisInner::$sub_variant(b)
            }
        }
        impl From<SymmetricSubspace<HardcoreGrpInner<$B>>> for BasisInner {
            #[inline]
            fn from(b: SymmetricSubspace<HardcoreGrpInner<$B>>) -> Self {
                BasisInner::$sym_variant(b)
            }
        }
        impl From<SymmetricSubspace<DitGrpInner<$B>>> for BasisInner {
            #[inline]
            fn from(b: SymmetricSubspace<DitGrpInner<$B>>) -> Self {
                BasisInner::$dit_sym_variant(b)
            }
        }
    };
}

impl_from_basis_spaces!(u32, Sub32, Sym32, DitSym32);
impl_from_basis_spaces!(u64, Sub64, Sym64, DitSym64);
impl_from_basis_spaces!(B128, Sub128, Sym128, DitSym128);
impl_from_basis_spaces!(B256, Sub256, Sym256, DitSym256);
impl_from_basis_spaces!(B512, Sub512, Sym512, DitSym512);
impl_from_basis_spaces!(B1024, Sub1024, Sym1024, DitSym1024);
impl_from_basis_spaces!(B2048, Sub2048, Sym2048, DitSym2048);
impl_from_basis_spaces!(B4096, Sub4096, Sym4096, DitSym4096);
impl_from_basis_spaces!(B8192, Sub8192, Sym8192, DitSym8192);

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------

/// Match on a [`BasisInner`] reference, injecting a type alias `$B` for
/// the concrete `BitInt` type and binding `$basis` to the inner basis reference.
///
/// Covers all 29 variants (Full*, Sub*, Sym*, DitSym*).
#[macro_export]
macro_rules! with_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::BasisInner::Full32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::BasisInner::Full64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym8192($basis) => {
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
            $crate::basis::dispatch::BasisInner::Full32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::BasisInner::Full64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_plain_basis! called on a symmetric variant"),
        }
    };
}

/// Like `with_basis!` but restricted to `Sym*` (LHSS=2 symmetric) variants.
///
/// Panics if called on a `DitSym*` variant.
#[macro_export]
macro_rules! with_sym_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::BasisInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::BasisInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_sym_basis! called on a non-Sym variant"),
        }
    };
}

/// Like `with_basis!` but restricted to `DitSym*` (LHSS≥3 symmetric) variants.
///
/// Panics if called on a `Sym*` or non-symmetric variant.
#[macro_export]
macro_rules! with_dit_sym_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::BasisInner::DitSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::BasisInner::DitSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_dit_sym_basis! called on a non-DitSym variant"),
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
        let inner = BasisInner::Full32(FullSpace::new(2, 4));
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
        let inner = BasisInner::Sub32(sub);
        let s = inner.to_string();
        assert!(s.starts_with("subspace(n_sites=2, size="));
        assert!(s.contains("symmetries=[]"));
    }

    #[test]
    fn display_index_alignment() {
        // 16 states → indices 0-15, width 2; row 9 and 10 should be right-aligned
        let inner = BasisInner::Full32(FullSpace::new(4, 16));
        let s = inner.to_string();
        assert!(s.contains("  9."));
        assert!(s.contains(" 10."));
    }

    #[test]
    fn display_truncation() {
        // 64 states > 50 → should truncate with "..."
        let inner = BasisInner::Full32(FullSpace::new(6, 64));
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
