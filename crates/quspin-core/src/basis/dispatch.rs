/// Type-erased `SpaceInner` and its dispatch macros.
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
    seed::{dit_seed_from_bytes, dit_state_to_str, seed_from_bytes, state_to_str},
    space::{FullSpace, Subspace},
    sym::SymBasis,
};
use crate::bitbasis::manip::DynamicDitManip;
use crate::bitbasis::{BitInt, DynamicPermDitValues, PermDitMask, PermDitValues};
use crate::error::QuSpinError;
use num_complex::Complex;

type B128 = ruint::Uint<128, 2>;
type B256 = ruint::Uint<256, 4>;
#[cfg(feature = "large-int")]
type B512 = ruint::Uint<512, 8>;
#[cfg(feature = "large-int")]
type B1024 = ruint::Uint<1024, 16>;
#[cfg(feature = "large-int")]
type B2048 = ruint::Uint<2048, 32>;
#[cfg(feature = "large-int")]
type B4096 = ruint::Uint<4096, 64>;
#[cfg(feature = "large-int")]
type B8192 = ruint::Uint<8192, 128>;

// ---------------------------------------------------------------------------
// SpaceInner
// ---------------------------------------------------------------------------

/// Type-erased wrapper for all basis-space variants over all supported
/// integer widths.
///
/// 47 variants total:
/// - 2 `Full` variants (u32, u64)
/// - 9 `Sub` variants (u32, u64, and 128–8192 bit ruint integers)
/// - 9 `Sym` variants — LHSS=2 symmetric (hardcore bosons / spin-½ / fermions)
/// - 9 `TritSym` variants — LHSS=3 symmetric (`PermDitValues<3>`)
/// - 9 `QuatSym` variants — LHSS=4 symmetric (`PermDitValues<4>`)
/// - 9 `DitSym` variants — LHSS≥5 symmetric (bosons / higher spin)
pub enum SpaceInner {
    // Full Hilbert spaces (small n_sites only).
    Full32(FullSpace<u32>),
    Full64(FullSpace<u64>),

    // Subspaces (particle-number or energy sector).
    Sub32(Subspace<u32>),
    Sub64(Subspace<u64>),
    Sub128(Subspace<B128>),
    Sub256(Subspace<B256>),
    #[cfg(feature = "large-int")]
    Sub512(Subspace<B512>),
    #[cfg(feature = "large-int")]
    Sub1024(Subspace<B1024>),
    #[cfg(feature = "large-int")]
    Sub2048(Subspace<B2048>),
    #[cfg(feature = "large-int")]
    Sub4096(Subspace<B4096>),
    #[cfg(feature = "large-int")]
    Sub8192(Subspace<B8192>),

    // LHSS=2 symmetry-reduced subspaces (hardcore bosons / spin-½ / fermions).
    Sym32(SymBasis<u32, PermDitMask<u32>, u8>),
    Sym64(SymBasis<u64, PermDitMask<u64>, u16>),
    Sym128(SymBasis<B128, PermDitMask<B128>, u32>),
    Sym256(SymBasis<B256, PermDitMask<B256>, u32>),
    #[cfg(feature = "large-int")]
    Sym512(SymBasis<B512, PermDitMask<B512>, u32>),
    #[cfg(feature = "large-int")]
    Sym1024(SymBasis<B1024, PermDitMask<B1024>, u32>),
    #[cfg(feature = "large-int")]
    Sym2048(SymBasis<B2048, PermDitMask<B2048>, u32>),
    #[cfg(feature = "large-int")]
    Sym4096(SymBasis<B4096, PermDitMask<B4096>, u32>),
    #[cfg(feature = "large-int")]
    Sym8192(SymBasis<B8192, PermDitMask<B8192>, u32>),

    // LHSS=3 symmetry-reduced subspaces (PermDitValues<3>).
    TritSym32(SymBasis<u32, PermDitValues<3>, u8>),
    TritSym64(SymBasis<u64, PermDitValues<3>, u16>),
    TritSym128(SymBasis<B128, PermDitValues<3>, u32>),
    TritSym256(SymBasis<B256, PermDitValues<3>, u32>),
    #[cfg(feature = "large-int")]
    TritSym512(SymBasis<B512, PermDitValues<3>, u32>),
    #[cfg(feature = "large-int")]
    TritSym1024(SymBasis<B1024, PermDitValues<3>, u32>),
    #[cfg(feature = "large-int")]
    TritSym2048(SymBasis<B2048, PermDitValues<3>, u32>),
    #[cfg(feature = "large-int")]
    TritSym4096(SymBasis<B4096, PermDitValues<3>, u32>),
    #[cfg(feature = "large-int")]
    TritSym8192(SymBasis<B8192, PermDitValues<3>, u32>),

    // LHSS=4 symmetry-reduced subspaces (PermDitValues<4>).
    QuatSym32(SymBasis<u32, PermDitValues<4>, u8>),
    QuatSym64(SymBasis<u64, PermDitValues<4>, u16>),
    QuatSym128(SymBasis<B128, PermDitValues<4>, u32>),
    QuatSym256(SymBasis<B256, PermDitValues<4>, u32>),
    #[cfg(feature = "large-int")]
    QuatSym512(SymBasis<B512, PermDitValues<4>, u32>),
    #[cfg(feature = "large-int")]
    QuatSym1024(SymBasis<B1024, PermDitValues<4>, u32>),
    #[cfg(feature = "large-int")]
    QuatSym2048(SymBasis<B2048, PermDitValues<4>, u32>),
    #[cfg(feature = "large-int")]
    QuatSym4096(SymBasis<B4096, PermDitValues<4>, u32>),
    #[cfg(feature = "large-int")]
    QuatSym8192(SymBasis<B8192, PermDitValues<4>, u32>),

    // LHSS≥5 symmetry-reduced subspaces (bosons / higher spin).
    DitSym32(SymBasis<u32, DynamicPermDitValues, u8>),
    DitSym64(SymBasis<u64, DynamicPermDitValues, u16>),
    DitSym128(SymBasis<B128, DynamicPermDitValues, u32>),
    DitSym256(SymBasis<B256, DynamicPermDitValues, u32>),
    #[cfg(feature = "large-int")]
    DitSym512(SymBasis<B512, DynamicPermDitValues, u32>),
    #[cfg(feature = "large-int")]
    DitSym1024(SymBasis<B1024, DynamicPermDitValues, u32>),
    #[cfg(feature = "large-int")]
    DitSym2048(SymBasis<B2048, DynamicPermDitValues, u32>),
    #[cfg(feature = "large-int")]
    DitSym4096(SymBasis<B4096, DynamicPermDitValues, u32>),
    #[cfg(feature = "large-int")]
    DitSym8192(SymBasis<B8192, DynamicPermDitValues, u32>),
}

/// Format a basis state as a string, using bit-encoding for LHSS=2 and
/// decimal dit-encoding for LHSS≥3.
#[inline]
fn fmt_state<B: crate::bitbasis::BitInt>(state: B, n_sites: usize, lhss: usize) -> String {
    if lhss == 2 {
        state_to_str(state, n_sites)
    } else {
        let manip = DynamicDitManip::new(lhss);
        dit_state_to_str(state, n_sites, &manip)
    }
}

impl SpaceInner {
    /// Number of lattice sites.
    pub fn n_sites(&self) -> usize {
        match self {
            SpaceInner::Full32(b) => b.n_sites(),
            SpaceInner::Full64(b) => b.n_sites(),
            SpaceInner::Sub32(b) => b.n_sites(),
            SpaceInner::Sub64(b) => b.n_sites(),
            SpaceInner::Sub128(b) => b.n_sites(),
            SpaceInner::Sub256(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub512(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub1024(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub2048(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub4096(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub8192(b) => b.n_sites(),
            SpaceInner::Sym32(b) => b.n_sites(),
            SpaceInner::Sym64(b) => b.n_sites(),
            SpaceInner::Sym128(b) => b.n_sites(),
            SpaceInner::Sym256(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym1024(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym2048(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym4096(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym8192(b) => b.n_sites(),
            SpaceInner::DitSym32(b) => b.n_sites(),
            SpaceInner::DitSym64(b) => b.n_sites(),
            SpaceInner::DitSym128(b) => b.n_sites(),
            SpaceInner::DitSym256(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym512(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym1024(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym2048(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym4096(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym8192(b) => b.n_sites(),
            SpaceInner::TritSym32(b) => b.n_sites(),
            SpaceInner::TritSym64(b) => b.n_sites(),
            SpaceInner::TritSym128(b) => b.n_sites(),
            SpaceInner::TritSym256(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym512(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym1024(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym2048(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym4096(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym8192(b) => b.n_sites(),
            SpaceInner::QuatSym32(b) => b.n_sites(),
            SpaceInner::QuatSym64(b) => b.n_sites(),
            SpaceInner::QuatSym128(b) => b.n_sites(),
            SpaceInner::QuatSym256(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym512(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym1024(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym2048(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym4096(b) => b.n_sites(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym8192(b) => b.n_sites(),
        }
    }

    /// Local Hilbert-space size (number of states per site).
    pub fn lhss(&self) -> usize {
        match self {
            SpaceInner::Full32(b) => b.lhss(),
            SpaceInner::Full64(b) => b.lhss(),
            SpaceInner::Sub32(b) => b.lhss(),
            SpaceInner::Sub64(b) => b.lhss(),
            SpaceInner::Sub128(b) => b.lhss(),
            SpaceInner::Sub256(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub512(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub1024(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub2048(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub4096(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub8192(b) => b.lhss(),
            SpaceInner::Sym32(b) => b.lhss(),
            SpaceInner::Sym64(b) => b.lhss(),
            SpaceInner::Sym128(b) => b.lhss(),
            SpaceInner::Sym256(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym1024(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym2048(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym4096(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym8192(b) => b.lhss(),
            SpaceInner::DitSym32(b) => b.lhss(),
            SpaceInner::DitSym64(b) => b.lhss(),
            SpaceInner::DitSym128(b) => b.lhss(),
            SpaceInner::DitSym256(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym512(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym1024(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym2048(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym4096(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym8192(b) => b.lhss(),
            SpaceInner::TritSym32(b) => b.lhss(),
            SpaceInner::TritSym64(b) => b.lhss(),
            SpaceInner::TritSym128(b) => b.lhss(),
            SpaceInner::TritSym256(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym512(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym1024(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym2048(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym4096(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym8192(b) => b.lhss(),
            SpaceInner::QuatSym32(b) => b.lhss(),
            SpaceInner::QuatSym64(b) => b.lhss(),
            SpaceInner::QuatSym128(b) => b.lhss(),
            SpaceInner::QuatSym256(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym512(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym1024(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym2048(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym4096(b) => b.lhss(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym8192(b) => b.lhss(),
        }
    }

    /// Which kind of space this basis represents.
    pub fn space_kind(&self) -> crate::basis::spin::SpaceKind {
        use crate::basis::spin::SpaceKind;
        match self {
            SpaceInner::Full32(_) | SpaceInner::Full64(_) => SpaceKind::Full,
            SpaceInner::Sub32(_)
            | SpaceInner::Sub64(_)
            | SpaceInner::Sub128(_)
            | SpaceInner::Sub256(_) => SpaceKind::Sub,
            #[cfg(feature = "large-int")]
            SpaceInner::Sub512(_)
            | SpaceInner::Sub1024(_)
            | SpaceInner::Sub2048(_)
            | SpaceInner::Sub4096(_)
            | SpaceInner::Sub8192(_) => SpaceKind::Sub,
            _ => SpaceKind::Symm,
        }
    }

    /// Number of basis states.
    pub fn size(&self) -> usize {
        match self {
            SpaceInner::Full32(b) => b.size(),
            SpaceInner::Full64(b) => b.size(),
            SpaceInner::Sub32(b) => b.size(),
            SpaceInner::Sub64(b) => b.size(),
            SpaceInner::Sub128(b) => b.size(),
            SpaceInner::Sub256(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub512(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub1024(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub2048(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub4096(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub8192(b) => b.size(),
            SpaceInner::Sym32(b) => b.size(),
            SpaceInner::Sym64(b) => b.size(),
            SpaceInner::Sym128(b) => b.size(),
            SpaceInner::Sym256(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym1024(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym2048(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym4096(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym8192(b) => b.size(),
            SpaceInner::DitSym32(b) => b.size(),
            SpaceInner::DitSym64(b) => b.size(),
            SpaceInner::DitSym128(b) => b.size(),
            SpaceInner::DitSym256(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym512(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym1024(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym2048(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym4096(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym8192(b) => b.size(),
            SpaceInner::TritSym32(b) => b.size(),
            SpaceInner::TritSym64(b) => b.size(),
            SpaceInner::TritSym128(b) => b.size(),
            SpaceInner::TritSym256(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym512(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym1024(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym2048(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym4096(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym8192(b) => b.size(),
            SpaceInner::QuatSym32(b) => b.size(),
            SpaceInner::QuatSym64(b) => b.size(),
            SpaceInner::QuatSym128(b) => b.size(),
            SpaceInner::QuatSym256(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym512(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym1024(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym2048(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym4096(b) => b.size(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym8192(b) => b.size(),
        }
    }

    /// Return the `i`-th basis state as a bit-string (site 0 = index 0).
    pub fn state_at_str(&self, i: usize) -> String {
        match self {
            SpaceInner::Full32(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::Full64(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::Sub32(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::Sub64(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::Sub128(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::Sub256(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub512(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub1024(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub2048(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub4096(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub8192(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::Sym32(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::Sym64(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::Sym128(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::Sym256(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym1024(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym2048(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym4096(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym8192(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::DitSym32(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::DitSym64(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::DitSym128(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::DitSym256(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym512(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym1024(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym2048(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym4096(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym8192(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::TritSym32(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::TritSym64(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::TritSym128(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::TritSym256(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym512(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym1024(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym2048(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym4096(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym8192(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::QuatSym32(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::QuatSym64(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::QuatSym128(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            SpaceInner::QuatSym256(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym512(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym1024(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym2048(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym4096(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym8192(b) => fmt_state(b.state_at(i), b.n_sites(), b.lhss()),
        }
    }

    /// Return the `i`-th basis state as a decimal integer string.
    ///
    /// The integer is the raw internal state value (LSB = site 0).  Useful for
    /// the third column of the QuSpin-style display.
    pub fn state_at_decimal_str(&self, i: usize) -> String {
        match self {
            SpaceInner::Full32(b) => format!("{}", b.state_at(i)),
            SpaceInner::Full64(b) => format!("{}", b.state_at(i)),
            SpaceInner::Sub32(b) => format!("{}", b.state_at(i)),
            SpaceInner::Sub64(b) => format!("{}", b.state_at(i)),
            SpaceInner::Sub128(b) => format!("{}", b.state_at(i)),
            SpaceInner::Sub256(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub512(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub1024(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub2048(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub4096(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub8192(b) => format!("{}", b.state_at(i)),
            SpaceInner::Sym32(b) => format!("{}", b.state_at(i)),
            SpaceInner::Sym64(b) => format!("{}", b.state_at(i)),
            SpaceInner::Sym128(b) => format!("{}", b.state_at(i)),
            SpaceInner::Sym256(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym1024(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym2048(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym4096(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym8192(b) => format!("{}", b.state_at(i)),
            SpaceInner::DitSym32(b) => format!("{}", b.state_at(i)),
            SpaceInner::DitSym64(b) => format!("{}", b.state_at(i)),
            SpaceInner::DitSym128(b) => format!("{}", b.state_at(i)),
            SpaceInner::DitSym256(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym512(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym1024(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym2048(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym4096(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym8192(b) => format!("{}", b.state_at(i)),
            SpaceInner::TritSym32(b) => format!("{}", b.state_at(i)),
            SpaceInner::TritSym64(b) => format!("{}", b.state_at(i)),
            SpaceInner::TritSym128(b) => format!("{}", b.state_at(i)),
            SpaceInner::TritSym256(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym512(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym1024(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym2048(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym4096(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym8192(b) => format!("{}", b.state_at(i)),
            SpaceInner::QuatSym32(b) => format!("{}", b.state_at(i)),
            SpaceInner::QuatSym64(b) => format!("{}", b.state_at(i)),
            SpaceInner::QuatSym128(b) => format!("{}", b.state_at(i)),
            SpaceInner::QuatSym256(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym512(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym1024(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym2048(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym4096(b) => format!("{}", b.state_at(i)),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym8192(b) => format!("{}", b.state_at(i)),
        }
    }

    /// Look up the index of the state encoded as a site-occupation byte slice.
    ///
    /// For `lhss == 2` each byte is a 0/1 occupation; for `lhss > 2` each byte
    /// is a dit value in `0..lhss`.  The encoding must match what was used when
    /// building the basis (i.e. `seed_from_bytes` for binary, `dit_seed_from_bytes`
    /// for multi-valued).
    ///
    /// Returns `None` if the state is not in the basis.
    pub fn index_of_bytes(&self, bytes: &[u8]) -> Option<usize> {
        // Converts `bytes` to the correct integer seed type, branching on lhss.
        macro_rules! index_bytes {
            ($b:expr) => {{
                let lhss = $b.lhss();
                let seed = if lhss == 2 {
                    seed_from_bytes(bytes)
                } else {
                    dit_seed_from_bytes(bytes, &DynamicDitManip::new(lhss))
                };
                $b.index(seed)
            }};
        }
        match self {
            SpaceInner::Full32(b) => index_bytes!(b),
            SpaceInner::Full64(b) => index_bytes!(b),
            SpaceInner::Sub32(b) => index_bytes!(b),
            SpaceInner::Sub64(b) => index_bytes!(b),
            SpaceInner::Sub128(b) => index_bytes!(b),
            SpaceInner::Sub256(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub512(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub1024(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub2048(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub4096(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub8192(b) => index_bytes!(b),
            SpaceInner::Sym32(b) => index_bytes!(b),
            SpaceInner::Sym64(b) => index_bytes!(b),
            SpaceInner::Sym128(b) => index_bytes!(b),
            SpaceInner::Sym256(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym1024(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym2048(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym4096(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym8192(b) => index_bytes!(b),
            SpaceInner::DitSym32(b) => index_bytes!(b),
            SpaceInner::DitSym64(b) => index_bytes!(b),
            SpaceInner::DitSym128(b) => index_bytes!(b),
            SpaceInner::DitSym256(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym512(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym1024(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym2048(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym4096(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym8192(b) => index_bytes!(b),
            SpaceInner::TritSym32(b) => index_bytes!(b),
            SpaceInner::TritSym64(b) => index_bytes!(b),
            SpaceInner::TritSym128(b) => index_bytes!(b),
            SpaceInner::TritSym256(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym512(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym1024(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym2048(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym4096(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym8192(b) => index_bytes!(b),
            SpaceInner::QuatSym32(b) => index_bytes!(b),
            SpaceInner::QuatSym64(b) => index_bytes!(b),
            SpaceInner::QuatSym128(b) => index_bytes!(b),
            SpaceInner::QuatSym256(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym512(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym1024(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym2048(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym4096(b) => index_bytes!(b),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym8192(b) => index_bytes!(b),
        }
    }

    /// One of `"full"`, `"subspace"`, or `"symmetric"`.
    pub fn kind(&self) -> &'static str {
        match self {
            SpaceInner::Full32(_) | SpaceInner::Full64(_) => "full",
            SpaceInner::Sub32(_)
            | SpaceInner::Sub64(_)
            | SpaceInner::Sub128(_)
            | SpaceInner::Sub256(_) => "subspace",
            #[cfg(feature = "large-int")]
            SpaceInner::Sub512(_)
            | SpaceInner::Sub1024(_)
            | SpaceInner::Sub2048(_)
            | SpaceInner::Sub4096(_)
            | SpaceInner::Sub8192(_) => "subspace",
            SpaceInner::Sym32(_)
            | SpaceInner::Sym64(_)
            | SpaceInner::Sym128(_)
            | SpaceInner::Sym256(_)
            | SpaceInner::TritSym32(_)
            | SpaceInner::TritSym64(_)
            | SpaceInner::TritSym128(_)
            | SpaceInner::TritSym256(_)
            | SpaceInner::QuatSym32(_)
            | SpaceInner::QuatSym64(_)
            | SpaceInner::QuatSym128(_)
            | SpaceInner::QuatSym256(_)
            | SpaceInner::DitSym32(_)
            | SpaceInner::DitSym64(_)
            | SpaceInner::DitSym128(_)
            | SpaceInner::DitSym256(_) => "symmetric",
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(_)
            | SpaceInner::Sym1024(_)
            | SpaceInner::Sym2048(_)
            | SpaceInner::Sym4096(_)
            | SpaceInner::Sym8192(_)
            | SpaceInner::TritSym512(_)
            | SpaceInner::TritSym1024(_)
            | SpaceInner::TritSym2048(_)
            | SpaceInner::TritSym4096(_)
            | SpaceInner::TritSym8192(_)
            | SpaceInner::QuatSym512(_)
            | SpaceInner::QuatSym1024(_)
            | SpaceInner::QuatSym2048(_)
            | SpaceInner::QuatSym4096(_)
            | SpaceInner::QuatSym8192(_)
            | SpaceInner::DitSym512(_)
            | SpaceInner::DitSym1024(_)
            | SpaceInner::DitSym2048(_)
            | SpaceInner::DitSym4096(_)
            | SpaceInner::DitSym8192(_) => "symmetric",
        }
    }

    /// Returns `true` once `build` has been called on the inner basis.
    ///
    /// - `Full*` → always `true` (no build step required)
    /// - `Sub*`  → `subspace.is_built()`
    /// - `Sym*` / `DitSym*` → `sym_basis.is_built()`
    pub fn is_built(&self) -> bool {
        match self {
            SpaceInner::Full32(_) | SpaceInner::Full64(_) => true,
            SpaceInner::Sub32(b) => b.is_built(),
            SpaceInner::Sub64(b) => b.is_built(),
            SpaceInner::Sub128(b) => b.is_built(),
            SpaceInner::Sub256(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub512(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub1024(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub2048(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub4096(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sub8192(b) => b.is_built(),
            SpaceInner::Sym32(b) => b.is_built(),
            SpaceInner::Sym64(b) => b.is_built(),
            SpaceInner::Sym128(b) => b.is_built(),
            SpaceInner::Sym256(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym1024(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym2048(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym4096(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym8192(b) => b.is_built(),
            SpaceInner::DitSym32(b) => b.is_built(),
            SpaceInner::DitSym64(b) => b.is_built(),
            SpaceInner::DitSym128(b) => b.is_built(),
            SpaceInner::DitSym256(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym512(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym1024(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym2048(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym4096(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym8192(b) => b.is_built(),
            SpaceInner::TritSym32(b) => b.is_built(),
            SpaceInner::TritSym64(b) => b.is_built(),
            SpaceInner::TritSym128(b) => b.is_built(),
            SpaceInner::TritSym256(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym512(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym1024(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym2048(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym4096(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym8192(b) => b.is_built(),
            SpaceInner::QuatSym32(b) => b.is_built(),
            SpaceInner::QuatSym64(b) => b.is_built(),
            SpaceInner::QuatSym128(b) => b.is_built(),
            SpaceInner::QuatSym256(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym512(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym1024(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym2048(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym4096(b) => b.is_built(),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym8192(b) => b.is_built(),
        }
    }

    /// Add a lattice (site-permutation) symmetry element.
    ///
    /// # Errors
    /// - Basis is not symmetric (`Sym*`, `TritSym*`, `QuatSym*`, or `DitSym*`)
    /// - Basis is already built
    /// - `perm.len() != n_sites`
    pub fn add_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
    ) -> Result<(), QuSpinError> {
        if !self.is_symmetric() {
            return Err(QuSpinError::ValueError(
                "add_lattice requires a symmetric (Sym*, TritSym*, QuatSym*, or DitSym*) basis"
                    .into(),
            ));
        }
        if self.is_built() {
            return Err(QuSpinError::ValueError(
                "cannot add symmetry elements after basis is built".into(),
            ));
        }
        let n_sites = self.n_sites();
        if perm.len() != n_sites {
            return Err(QuSpinError::ValueError(format!(
                "perm.len()={} but n_sites={}",
                perm.len(),
                n_sites
            )));
        }
        // Validate perm is a valid permutation of 0..n_sites.
        let mut seen = vec![false; n_sites];
        for (i, &p) in perm.iter().enumerate() {
            if p >= n_sites {
                return Err(QuSpinError::ValueError(format!(
                    "perm[{i}]={p} is out of range 0..{n_sites}"
                )));
            }
            if seen[p] {
                return Err(QuSpinError::ValueError(format!(
                    "perm has duplicate target site {p}"
                )));
            }
            seen[p] = true;
        }
        match self {
            SpaceInner::Sym32(b) => b.add_lattice(grp_char, perm),
            SpaceInner::Sym64(b) => b.add_lattice(grp_char, perm),
            SpaceInner::Sym128(b) => b.add_lattice(grp_char, perm),
            SpaceInner::Sym256(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym1024(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym2048(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym4096(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym8192(b) => b.add_lattice(grp_char, perm),
            SpaceInner::DitSym32(b) => b.add_lattice(grp_char, perm),
            SpaceInner::DitSym64(b) => b.add_lattice(grp_char, perm),
            SpaceInner::DitSym128(b) => b.add_lattice(grp_char, perm),
            SpaceInner::DitSym256(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym512(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym1024(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym2048(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym4096(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::DitSym8192(b) => b.add_lattice(grp_char, perm),
            SpaceInner::TritSym32(b) => b.add_lattice(grp_char, perm),
            SpaceInner::TritSym64(b) => b.add_lattice(grp_char, perm),
            SpaceInner::TritSym128(b) => b.add_lattice(grp_char, perm),
            SpaceInner::TritSym256(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym512(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym1024(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym2048(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym4096(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::TritSym8192(b) => b.add_lattice(grp_char, perm),
            SpaceInner::QuatSym32(b) => b.add_lattice(grp_char, perm),
            SpaceInner::QuatSym64(b) => b.add_lattice(grp_char, perm),
            SpaceInner::QuatSym128(b) => b.add_lattice(grp_char, perm),
            SpaceInner::QuatSym256(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym512(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym1024(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym2048(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym4096(b) => b.add_lattice(grp_char, perm),
            #[cfg(feature = "large-int")]
            SpaceInner::QuatSym8192(b) => b.add_lattice(grp_char, perm),
            _ => unreachable!("non-symmetric variants ruled out above"),
        }
        Ok(())
    }

    /// Add an inversion (XOR bit-flip) symmetry element for LHSS=2 (`Sym*`) bases.
    ///
    /// Flips the local state at each site in `locs` (`v → 1 − v`).
    ///
    /// # Errors
    /// - Basis is not symmetric
    /// - Basis is already built
    /// - `lhss != 2`
    pub fn add_inv(&mut self, grp_char: Complex<f64>, locs: &[usize]) -> Result<(), QuSpinError> {
        if !self.is_symmetric() {
            return Err(QuSpinError::ValueError(
                "add_inv requires a symmetric basis".into(),
            ));
        }
        if self.is_built() {
            return Err(QuSpinError::ValueError(
                "cannot add symmetry elements after basis is built".into(),
            ));
        }
        if self.lhss() != 2 {
            return Err(QuSpinError::ValueError(format!(
                "add_inv requires lhss=2, got lhss={}",
                self.lhss()
            )));
        }
        let n_sites = self.n_sites();
        for (i, &loc) in locs.iter().enumerate() {
            if loc >= n_sites {
                return Err(QuSpinError::ValueError(format!(
                    "locs[{i}]={loc} is out of range 0..{n_sites}"
                )));
            }
        }
        macro_rules! build_mask_and_push {
            ($basis:expr, $B:ty) => {{
                let mask = locs.iter().fold(<$B>::from_u64(0), |acc, &site| {
                    if site < <$B>::BITS as usize {
                        acc | (<$B>::from_u64(1) << site)
                    } else {
                        acc
                    }
                });
                $basis.add_local(grp_char, PermDitMask::new(mask));
            }};
        }
        match self {
            SpaceInner::Sym32(b) => build_mask_and_push!(b, u32),
            SpaceInner::Sym64(b) => build_mask_and_push!(b, u64),
            SpaceInner::Sym128(b) => build_mask_and_push!(b, B128),
            SpaceInner::Sym256(b) => build_mask_and_push!(b, B256),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym512(b) => build_mask_and_push!(b, B512),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym1024(b) => build_mask_and_push!(b, B1024),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym2048(b) => build_mask_and_push!(b, B2048),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym4096(b) => build_mask_and_push!(b, B4096),
            #[cfg(feature = "large-int")]
            SpaceInner::Sym8192(b) => build_mask_and_push!(b, B8192),
            _ => {
                return Err(QuSpinError::ValueError(
                    "add_inv requires a Sym* variant (lhss=2)".into(),
                ));
            }
        }
        Ok(())
    }

    /// Add a local dit-permutation symmetry element.
    ///
    /// `perm_vals[v] = w` maps local-state `v` to `w` at each site in `locs`.
    /// Dispatches to the appropriate compiled path based on LHSS:
    /// - 2 → delegates to [`add_inv`](Self::add_inv) (perm_vals must be `[1, 0]`)
    /// - 3 → `PermDitValues<3>`
    /// - 4 → `PermDitValues<4>`
    /// - ≥5 → `DynamicPermDitValues`
    ///
    /// # Errors
    /// - Basis is not symmetric
    /// - Basis is already built
    /// - `perm_vals.len() != lhss`
    /// - LHSS=2 with `perm_vals` other than `[1, 0]`
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        if !self.is_symmetric() {
            return Err(QuSpinError::ValueError(
                "add_local requires a symmetric basis".into(),
            ));
        }
        if self.is_built() {
            return Err(QuSpinError::ValueError(
                "cannot add symmetry elements after basis is built".into(),
            ));
        }
        let lhss = self.lhss();
        if perm_vals.len() != lhss {
            return Err(QuSpinError::ValueError(format!(
                "perm_vals.len()={} but lhss={lhss}",
                perm_vals.len()
            )));
        }
        // Validate perm_vals is a permutation of 0..lhss.
        let mut seen = vec![false; lhss];
        for (i, &v) in perm_vals.iter().enumerate() {
            let v = v as usize;
            if v >= lhss {
                return Err(QuSpinError::ValueError(format!(
                    "perm_vals[{i}]={v} is out of range 0..{lhss}"
                )));
            }
            if seen[v] {
                return Err(QuSpinError::ValueError(format!(
                    "perm_vals has duplicate value {v}"
                )));
            }
            seen[v] = true;
        }
        // Validate locs are within n_sites.
        let n_sites = self.n_sites();
        for (i, &loc) in locs.iter().enumerate() {
            if loc >= n_sites {
                return Err(QuSpinError::ValueError(format!(
                    "locs[{i}]={loc} is out of range 0..{n_sites}"
                )));
            }
        }
        match lhss {
            2 => {
                // For lhss=2, the only non-trivial local symmetry is bit-flip [1, 0].
                if perm_vals != [1, 0] {
                    return Err(QuSpinError::ValueError(format!(
                        "add_local with lhss=2 only supports inversion perm_vals=[1,0], got {perm_vals:?}"
                    )));
                }
                return self.add_inv(grp_char, &locs);
            }
            3 => {
                let arr: [u8; 3] = perm_vals.try_into().expect("length validated above");
                match self {
                    SpaceInner::TritSym32(b) => {
                        b.add_local(grp_char, PermDitValues::<3>::new(arr, locs))
                    }
                    SpaceInner::TritSym64(b) => {
                        b.add_local(grp_char, PermDitValues::<3>::new(arr, locs))
                    }
                    SpaceInner::TritSym128(b) => {
                        b.add_local(grp_char, PermDitValues::<3>::new(arr, locs))
                    }
                    SpaceInner::TritSym256(b) => {
                        b.add_local(grp_char, PermDitValues::<3>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::TritSym512(b) => {
                        b.add_local(grp_char, PermDitValues::<3>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::TritSym1024(b) => {
                        b.add_local(grp_char, PermDitValues::<3>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::TritSym2048(b) => {
                        b.add_local(grp_char, PermDitValues::<3>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::TritSym4096(b) => {
                        b.add_local(grp_char, PermDitValues::<3>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::TritSym8192(b) => {
                        b.add_local(grp_char, PermDitValues::<3>::new(arr, locs))
                    }
                    _ => {
                        return Err(QuSpinError::ValueError(
                            "add_local with lhss=3 requires a TritSym* variant".into(),
                        ));
                    }
                }
            }
            4 => {
                let arr: [u8; 4] = perm_vals.try_into().expect("length validated above");
                match self {
                    SpaceInner::QuatSym32(b) => {
                        b.add_local(grp_char, PermDitValues::<4>::new(arr, locs))
                    }
                    SpaceInner::QuatSym64(b) => {
                        b.add_local(grp_char, PermDitValues::<4>::new(arr, locs))
                    }
                    SpaceInner::QuatSym128(b) => {
                        b.add_local(grp_char, PermDitValues::<4>::new(arr, locs))
                    }
                    SpaceInner::QuatSym256(b) => {
                        b.add_local(grp_char, PermDitValues::<4>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::QuatSym512(b) => {
                        b.add_local(grp_char, PermDitValues::<4>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::QuatSym1024(b) => {
                        b.add_local(grp_char, PermDitValues::<4>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::QuatSym2048(b) => {
                        b.add_local(grp_char, PermDitValues::<4>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::QuatSym4096(b) => {
                        b.add_local(grp_char, PermDitValues::<4>::new(arr, locs))
                    }
                    #[cfg(feature = "large-int")]
                    SpaceInner::QuatSym8192(b) => {
                        b.add_local(grp_char, PermDitValues::<4>::new(arr, locs))
                    }
                    _ => {
                        return Err(QuSpinError::ValueError(
                            "add_local with lhss=4 requires a QuatSym* variant".into(),
                        ));
                    }
                }
            }
            _ => match self {
                SpaceInner::DitSym32(b) => {
                    b.add_local(grp_char, DynamicPermDitValues::new(lhss, perm_vals, locs))
                }
                SpaceInner::DitSym64(b) => {
                    b.add_local(grp_char, DynamicPermDitValues::new(lhss, perm_vals, locs))
                }
                SpaceInner::DitSym128(b) => {
                    b.add_local(grp_char, DynamicPermDitValues::new(lhss, perm_vals, locs))
                }
                SpaceInner::DitSym256(b) => {
                    b.add_local(grp_char, DynamicPermDitValues::new(lhss, perm_vals, locs))
                }
                #[cfg(feature = "large-int")]
                SpaceInner::DitSym512(b) => {
                    b.add_local(grp_char, DynamicPermDitValues::new(lhss, perm_vals, locs))
                }
                #[cfg(feature = "large-int")]
                SpaceInner::DitSym1024(b) => {
                    b.add_local(grp_char, DynamicPermDitValues::new(lhss, perm_vals, locs))
                }
                #[cfg(feature = "large-int")]
                SpaceInner::DitSym2048(b) => {
                    b.add_local(grp_char, DynamicPermDitValues::new(lhss, perm_vals, locs))
                }
                #[cfg(feature = "large-int")]
                SpaceInner::DitSym4096(b) => {
                    b.add_local(grp_char, DynamicPermDitValues::new(lhss, perm_vals, locs))
                }
                #[cfg(feature = "large-int")]
                SpaceInner::DitSym8192(b) => {
                    b.add_local(grp_char, DynamicPermDitValues::new(lhss, perm_vals, locs))
                }
                _ => {
                    return Err(QuSpinError::ValueError(
                        "add_local with lhss>=5 requires a DitSym* variant".into(),
                    ));
                }
            },
        }
        Ok(())
    }

    /// Returns `true` for `Sym*` and `DitSym*` variants (symmetry-reduced subspaces).
    pub fn is_symmetric(&self) -> bool {
        self.kind() == "symmetric"
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

const DISPLAY_HEAD: usize = 25;
const DISPLAY_TAIL: usize = 25;

impl std::fmt::Display for SpaceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = self.size();
        let lhss = self.lhss();
        let n_sites = self.n_sites();

        writeln!(f, "reference states:")?;
        writeln!(f, "array index   /   Fock state   /   integer repr.")?;

        if size == 0 {
            if self.is_symmetric() {
                write!(
                    f,
                    "\nThe states printed do NOT correspond to the physical states: \
                     see review arXiv:1101.3281 for more details about reference \
                     states for symmetry-reduced blocks."
                )?;
            }
            return Ok(());
        }

        // Width of the index column (right-aligned).
        let w_idx = (size - 1).to_string().len();
        // Digits per site in the Fock-state column.
        let n_space = (lhss - 1).to_string().len().max(1);
        // Width of the Fock string "|s0 s1 ... sN>".
        let fock_w = 1 + n_sites * n_space + n_sites.saturating_sub(1) + 1;
        // Width of the integer-repr column (last state has the largest value).
        let w_int = self.state_at_decimal_str(size - 1).len();

        // Build and write a single row given its basis index `i`.
        macro_rules! write_row {
            ($i:expr) => {{
                let compact = self.state_at_str($i);
                let spaced: String = compact
                    .chars()
                    .map(|c| format!("{:>n_space$}", c))
                    .collect::<Vec<_>>()
                    .join(" ");
                let fock = format!("|{}>", spaced);
                let int_str = self.state_at_decimal_str($i);
                writeln!(
                    f,
                    " {:>w_idx$}.  {:<fock_w$}  {:>w_int$}",
                    $i, fock, int_str,
                )?;
            }};
        }

        let truncate = size > DISPLAY_HEAD + DISPLAY_TAIL;
        if !truncate {
            for i in 0..size {
                write_row!(i);
            }
        } else {
            for i in 0..DISPLAY_HEAD {
                write_row!(i);
            }
            // Separator: ":" aligned under the centre of the Fock column.
            // Position of "|" in the row: 1 + w_idx + 3  (leading space + index + ".  ")
            let pipe_pos = 1 + w_idx + 3;
            let colon_col = pipe_pos + fock_w / 2;
            writeln!(f, "{:>colon_col$}", ":")?;
            for i in size - DISPLAY_TAIL..size {
                write_row!(i);
            }
        }

        if self.is_symmetric() {
            write!(
                f,
                "\nThe states printed do NOT correspond to the physical states: \
                 see review arXiv:1101.3281 for more details about reference \
                 states for symmetry-reduced blocks."
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// From impls — wrap a concrete basis space without naming the variant
// ---------------------------------------------------------------------------

macro_rules! impl_from_basis_spaces {
    ($B:ty, $N:ty, $sub_variant:ident, $sym_variant:ident, $dit_sym_variant:ident) => {
        impl From<Subspace<$B>> for SpaceInner {
            #[inline]
            fn from(b: Subspace<$B>) -> Self {
                SpaceInner::$sub_variant(b)
            }
        }
        impl From<SymBasis<$B, PermDitMask<$B>, $N>> for SpaceInner {
            #[inline]
            fn from(b: SymBasis<$B, PermDitMask<$B>, $N>) -> Self {
                SpaceInner::$sym_variant(b)
            }
        }
        impl From<SymBasis<$B, DynamicPermDitValues, $N>> for SpaceInner {
            #[inline]
            fn from(b: SymBasis<$B, DynamicPermDitValues, $N>) -> Self {
                SpaceInner::$dit_sym_variant(b)
            }
        }
    };
}

impl_from_basis_spaces!(u32, u8, Sub32, Sym32, DitSym32);
impl_from_basis_spaces!(u64, u16, Sub64, Sym64, DitSym64);
impl_from_basis_spaces!(B128, u32, Sub128, Sym128, DitSym128);
impl_from_basis_spaces!(B256, u32, Sub256, Sym256, DitSym256);
#[cfg(feature = "large-int")]
impl_from_basis_spaces!(B512, u32, Sub512, Sym512, DitSym512);
#[cfg(feature = "large-int")]
impl_from_basis_spaces!(B1024, u32, Sub1024, Sym1024, DitSym1024);
#[cfg(feature = "large-int")]
impl_from_basis_spaces!(B2048, u32, Sub2048, Sym2048, DitSym2048);
#[cfg(feature = "large-int")]
impl_from_basis_spaces!(B4096, u32, Sub4096, Sym4096, DitSym4096);
#[cfg(feature = "large-int")]
impl_from_basis_spaces!(B8192, u32, Sub8192, Sym8192, DitSym8192);

macro_rules! impl_from_trit_sym_spaces {
    ($B:ty, $N:ty, $trit_sym_variant:ident) => {
        impl From<SymBasis<$B, PermDitValues<3>, $N>> for SpaceInner {
            #[inline]
            fn from(b: SymBasis<$B, PermDitValues<3>, $N>) -> Self {
                SpaceInner::$trit_sym_variant(b)
            }
        }
    };
}

impl_from_trit_sym_spaces!(u32, u8, TritSym32);
impl_from_trit_sym_spaces!(u64, u16, TritSym64);
impl_from_trit_sym_spaces!(B128, u32, TritSym128);
impl_from_trit_sym_spaces!(B256, u32, TritSym256);
#[cfg(feature = "large-int")]
impl_from_trit_sym_spaces!(B512, u32, TritSym512);
#[cfg(feature = "large-int")]
impl_from_trit_sym_spaces!(B1024, u32, TritSym1024);
#[cfg(feature = "large-int")]
impl_from_trit_sym_spaces!(B2048, u32, TritSym2048);
#[cfg(feature = "large-int")]
impl_from_trit_sym_spaces!(B4096, u32, TritSym4096);
#[cfg(feature = "large-int")]
impl_from_trit_sym_spaces!(B8192, u32, TritSym8192);

macro_rules! impl_from_quat_sym_spaces {
    ($B:ty, $N:ty, $quat_sym_variant:ident) => {
        impl From<SymBasis<$B, PermDitValues<4>, $N>> for SpaceInner {
            #[inline]
            fn from(b: SymBasis<$B, PermDitValues<4>, $N>) -> Self {
                SpaceInner::$quat_sym_variant(b)
            }
        }
    };
}

impl_from_quat_sym_spaces!(u32, u8, QuatSym32);
impl_from_quat_sym_spaces!(u64, u16, QuatSym64);
impl_from_quat_sym_spaces!(B128, u32, QuatSym128);
impl_from_quat_sym_spaces!(B256, u32, QuatSym256);
#[cfg(feature = "large-int")]
impl_from_quat_sym_spaces!(B512, u32, QuatSym512);
#[cfg(feature = "large-int")]
impl_from_quat_sym_spaces!(B1024, u32, QuatSym1024);
#[cfg(feature = "large-int")]
impl_from_quat_sym_spaces!(B2048, u32, QuatSym2048);
#[cfg(feature = "large-int")]
impl_from_quat_sym_spaces!(B4096, u32, QuatSym4096);
#[cfg(feature = "large-int")]
impl_from_quat_sym_spaces!(B8192, u32, QuatSym8192);

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------

/// Match on a [`SpaceInner`] reference, injecting a type alias `$B` for
/// the concrete `BitInt` type and binding `$basis` to the inner basis reference.
///
/// Covers all 47 variants (Full*, Sub*, Sym*, TritSym*, QuatSym*, DitSym*).
#[macro_export]
macro_rules! with_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::Full32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Full64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym8192($basis) => {
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
            $crate::basis::dispatch::SpaceInner::Full32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Full64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub8192($basis) => {
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
            $crate::basis::dispatch::SpaceInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym8192($basis) => {
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
            $crate::basis::dispatch::SpaceInner::DitSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_dit_sym_basis! called on a non-DitSym variant"),
        }
    };
}

/// Like `with_sym_basis!` but binds `$basis` as `&mut`.
#[macro_export]
macro_rules! with_sym_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_sym_basis_mut! called on a non-Sym variant"),
        }
    };
}

/// Like `with_dit_sym_basis!` but binds `$basis` as `&mut`.
#[macro_export]
macro_rules! with_dit_sym_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::DitSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::DitSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_dit_sym_basis_mut! called on a non-DitSym variant"),
        }
    };
}

/// Like `with_basis!` but restricted to `TritSym*` (LHSS=3) variants.
#[macro_export]
macro_rules! with_trit_sym_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::TritSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_trit_sym_basis! called on a non-TritSym variant"),
        }
    };
}

/// Like `with_trit_sym_basis!` but binds `$basis` as `&mut`.
#[macro_export]
macro_rules! with_trit_sym_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::TritSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::TritSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::TritSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_trit_sym_basis_mut! called on a non-TritSym variant"),
        }
    };
}

/// Like `with_basis!` but restricted to `QuatSym*` (LHSS=4) variants.
#[macro_export]
macro_rules! with_quat_sym_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::QuatSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_quat_sym_basis! called on a non-QuatSym variant"),
        }
    };
}

/// Like `with_quat_sym_basis!` but binds `$basis` as `&mut`.
#[macro_export]
macro_rules! with_quat_sym_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::QuatSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::QuatSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::QuatSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_quat_sym_basis_mut! called on a non-QuatSym variant"),
        }
    };
}

/// Like `with_plain_basis!` but restricted to `Sub*` variants and binds `$basis` as `&mut`.
///
/// Does not match `Full*` — full spaces are always built and cannot be mutated.
#[macro_export]
macro_rules! with_sub_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            #[cfg(feature = "large-int")]
            $crate::basis::dispatch::SpaceInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_sub_basis_mut! called on a non-Sub variant"),
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
        } else {
            $crate::select_b_for_large_n_sites!($n_sites, $B, $on_overflow, $body)
        }
    };
}

/// Extension of [`select_b_for_n_sites!`] for >256-bit integers.
///
/// When the `large-int` feature is disabled this immediately evaluates
/// `$on_overflow`; when enabled it continues the ladder up to 8192 bits.
#[cfg(not(feature = "large-int"))]
#[macro_export]
macro_rules! select_b_for_large_n_sites {
    ($n_sites:expr, $B:ident, $on_overflow:expr, $body:block) => {
        $on_overflow
    };
}

/// Extension of [`select_b_for_n_sites!`] for >256-bit integers (large-int enabled).
#[cfg(feature = "large-int")]
#[macro_export]
macro_rules! select_b_for_large_n_sites {
    ($n_sites:expr, $B:ident, $on_overflow:expr, $body:block) => {
        if $n_sites <= 512 {
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
    use crate::basis::space::FullSpace;

    #[test]
    fn display_header() {
        let inner = SpaceInner::Full32(FullSpace::new(2, 2, false));
        let s = inner.to_string();
        assert!(
            s.starts_with("reference states:\narray index   /   Fock state   /   integer repr.")
        );
    }

    #[test]
    fn display_fock_format() {
        // lhss=2, n_sites=2: states |0 0>, |0 1>, |1 0>, |1 1>
        let inner = SpaceInner::Full32(FullSpace::new(2, 2, false));
        let s = inner.to_string();
        assert!(s.contains("|0 0>"), "expected spaced fock string");
        assert!(s.contains("|1 1>"), "expected spaced fock string");
    }

    #[test]
    fn display_integer_column() {
        // Full basis lhss=2, n_sites=2: state 0b11 = 3, state 0b01 = 1
        let inner = SpaceInner::Full32(FullSpace::new(2, 2, false));
        let s = inner.to_string();
        // Each row should end with a decimal integer
        assert!(s.contains("  3"), "expected integer repr 3");
        assert!(s.contains("  0"), "expected integer repr 0");
    }

    #[test]
    fn display_index_alignment() {
        // 16 states → indices 0-15, width 2; rows 9 and 10 should be right-aligned
        let inner = SpaceInner::Full32(FullSpace::new(2, 4, false));
        let s = inner.to_string();
        assert!(s.contains("  9."), "expected right-aligned index 9");
        assert!(s.contains(" 10."), "expected right-aligned index 10");
    }

    #[test]
    fn display_truncation() {
        // 64 states > 50 → should truncate with ":"
        let inner = SpaceInner::Full32(FullSpace::new(2, 6, false));
        let s = inner.to_string();
        assert!(s.contains(':'), "expected truncation marker");
        // First 25 rows present (index 0 and 24)
        assert!(
            s.contains("\n  0.") || s.contains("\n   0."),
            "expected row 0"
        );
        assert!(s.contains(" 24."), "expected row 24");
        // Row 25 should be absent (truncated)
        assert!(
            !s.contains("\n  25.") && !s.contains("\n 25."),
            "row 25 should be truncated"
        );
        // Last 25 rows: for 64 states, tail starts at 39 (64 - 25 = 39)
        assert!(s.contains(" 39."), "expected row 39");
        assert!(s.contains(" 63."), "expected row 63");
    }

    #[test]
    fn display_symmetric_note() {
        use crate::basis::sym::SymBasis;
        use crate::bitbasis::PermDitMask;
        let inner = SpaceInner::Sym32(SymBasis::<u32, PermDitMask<u32>, u8>::new_empty(
            2, 2, false,
        ));
        let s = inner.to_string();
        assert!(
            s.contains("do NOT correspond to the physical states"),
            "expected symmetry note"
        );
    }
}
