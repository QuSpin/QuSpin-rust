//! `Bit` family — LHSS = 2 (hard-core bosons / spin-½ / fermions).
//!
//! Three-level dispatch:
//!
//! - [`BitBasis`] — switches `Default` (always) vs `LargeInt`
//!   (feature-gated whole-type).
//! - [`BitBasisDefault`] — concrete variants over `u32`, `u64`,
//!   `Uint<128>`, `Uint<256>`.
//! - [`BitBasisLargeInt`] — concrete variants over the wide
//!   integer widths gated behind the `large-int` feature; the type
//!   itself only exists when the feature is on.
//!
//! Local-op type for the symmetric variants is `PermDitMask<B>`.
//! Fermion-sign tracking applies to this family only — the
//! `fermionic` flag on the underlying [`SymBasis`] is meaningful here.

use super::macros::{impl_family_dispatch_enum, impl_inner_dispatch_enum};
use super::types::{B128, B256};
#[cfg(feature = "large-int")]
use super::types::{B512, B1024, B2048, B4096, B8192};
use super::validate::{validate_locs, validate_perm_vals};
use crate::space::{FullSpace, Subspace};
use crate::sym::SymBasis;
use crate::traits::BasisSpace;
use crate::{SymElement, seed::seed_from_bytes};
use num_complex::Complex;
use quspin_bitbasis::{BitInt, PermDitMask, StateTransitions};
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `B`-typed bit-mask with the bits at every site index in
/// `locs` set. Sites whose index exceeds `B::BITS` are silently
/// dropped — `BitBasisDefault::add_inv` /
/// `BitBasisLargeInt::add_inv` validate `loc < n_sites` before calling
/// this helper, and `n_sites <= B::BITS` by construction.
#[inline]
fn build_mask<B: BitInt>(locs: &[usize]) -> B {
    locs.iter().fold(B::from_u64(0), |acc, &site| {
        if site < B::BITS as usize {
            acc | (B::from_u64(1) << site)
        } else {
            acc
        }
    })
}

// ---------------------------------------------------------------------------
// Default-width inner enum (u32, u64, Uint<128>, Uint<256>)
// ---------------------------------------------------------------------------

/// Concrete variants over the always-available integer widths.
pub enum BitBasisDefault {
    Full32(FullSpace<u32>),
    Full64(FullSpace<u64>),

    Sub32(Subspace<u32>),
    Sub64(Subspace<u64>),
    Sub128(Subspace<B128>),
    Sub256(Subspace<B256>),

    Sym32(SymBasis<u32, PermDitMask<u32>, u8>),
    Sym64(SymBasis<u64, PermDitMask<u64>, u16>),
    Sym128(SymBasis<B128, PermDitMask<B128>, u32>),
    Sym256(SymBasis<B256, PermDitMask<B256>, u32>),
}

impl_inner_dispatch_enum!(
    BitBasisDefault,
    full = [Full32, Full64],
    sub = [Sub32, Sub64, Sub128, Sub256],
    sym = [Sym32, Sym64, Sym128, Sym256],
);

impl BitBasisDefault {
    /// Add a local bit-flip (LHSS = 2 inversion) at every site in `locs`.
    ///
    /// `locs` is validated against `n_sites` here — this is the level
    /// where the typed `PermDitMask<B>` is constructed from the user's
    /// `Vec<usize>`, so it owns the input-shape check.
    pub fn add_inv(&mut self, grp_char: Complex<f64>, locs: &[usize]) -> Result<(), QuSpinError> {
        validate_locs(locs, self.n_sites())?;
        match self {
            Self::Sym32(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitMask::new(build_mask::<u32>(locs))),
            ),
            Self::Sym64(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitMask::new(build_mask::<u64>(locs))),
            ),
            Self::Sym128(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitMask::new(build_mask::<B128>(locs))),
            ),
            Self::Sym256(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitMask::new(build_mask::<B256>(locs))),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_inv requires a Sym* variant on BitBasisDefault".into(),
            )),
        }
    }

    /// Add a local dit-permutation symmetry element. For the Bit family
    /// `perm_vals` must equal `[1, 0]` (the only non-trivial LHSS = 2
    /// permutation); otherwise an error is returned. Internally
    /// dispatches to [`add_inv`](Self::add_inv) which validates `locs`.
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        validate_perm_vals(&perm_vals, 2)?;
        if perm_vals != [1, 0] {
            return Err(QuSpinError::ValueError(format!(
                "add_local on Bit family (LHSS=2) requires perm_vals=[1,0], got {perm_vals:?}"
            )));
        }
        self.add_inv(grp_char, &locs)
    }

    /// Build the basis from `seeds`. Bit-family seeds are bit-encoded.
    pub fn build_seeds<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        macro_rules! sub_build {
            ($b:ident, $B:ty) => {{
                for s in seeds {
                    $b.build(seed_from_bytes::<$B>(s), graph)?;
                }
            }};
        }
        macro_rules! sym_build {
            ($b:ident, $B:ty) => {{
                for s in seeds {
                    $b.build(seed_from_bytes::<$B>(s), graph)?;
                }
            }};
        }
        match self {
            Self::Full32(_) | Self::Full64(_) => {
                return Err(QuSpinError::ValueError(
                    "Full basis requires no build step".into(),
                ));
            }
            Self::Sub32(b) => sub_build!(b, u32),
            Self::Sub64(b) => sub_build!(b, u64),
            Self::Sub128(b) => sub_build!(b, B128),
            Self::Sub256(b) => sub_build!(b, B256),
            Self::Sym32(b) => sym_build!(b, u32),
            Self::Sym64(b) => sym_build!(b, u64),
            Self::Sym128(b) => sym_build!(b, B128),
            Self::Sym256(b) => sym_build!(b, B256),
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Large-int inner enum (gated; widths 512..8192)
// ---------------------------------------------------------------------------

/// Concrete variants over the `large-int`-gated integer widths.
#[cfg(feature = "large-int")]
pub enum BitBasisLargeInt {
    Sub512(Subspace<B512>),
    Sub1024(Subspace<B1024>),
    Sub2048(Subspace<B2048>),
    Sub4096(Subspace<B4096>),
    Sub8192(Subspace<B8192>),

    Sym512(SymBasis<B512, PermDitMask<B512>, u32>),
    Sym1024(SymBasis<B1024, PermDitMask<B1024>, u32>),
    Sym2048(SymBasis<B2048, PermDitMask<B2048>, u32>),
    Sym4096(SymBasis<B4096, PermDitMask<B4096>, u32>),
    Sym8192(SymBasis<B8192, PermDitMask<B8192>, u32>),
}

#[cfg(feature = "large-int")]
impl_inner_dispatch_enum!(
    BitBasisLargeInt,
    full = [],
    sub = [Sub512, Sub1024, Sub2048, Sub4096, Sub8192],
    sym = [Sym512, Sym1024, Sym2048, Sym4096, Sym8192],
);

#[cfg(feature = "large-int")]
impl BitBasisLargeInt {
    pub fn add_inv(&mut self, grp_char: Complex<f64>, locs: &[usize]) -> Result<(), QuSpinError> {
        validate_locs(locs, self.n_sites())?;
        match self {
            Self::Sym512(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitMask::new(build_mask::<B512>(locs))),
            ),
            Self::Sym1024(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitMask::new(build_mask::<B1024>(locs))),
            ),
            Self::Sym2048(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitMask::new(build_mask::<B2048>(locs))),
            ),
            Self::Sym4096(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitMask::new(build_mask::<B4096>(locs))),
            ),
            Self::Sym8192(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitMask::new(build_mask::<B8192>(locs))),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_inv requires a Sym* variant on BitBasisLargeInt".into(),
            )),
        }
    }

    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        validate_perm_vals(&perm_vals, 2)?;
        if perm_vals != [1, 0] {
            return Err(QuSpinError::ValueError(format!(
                "add_local on Bit family (LHSS=2) requires perm_vals=[1,0], got {perm_vals:?}"
            )));
        }
        self.add_inv(grp_char, &locs)
    }

    pub fn build_seeds<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        macro_rules! sub_build {
            ($b:ident, $B:ty) => {{
                for s in seeds {
                    $b.build(seed_from_bytes::<$B>(s), graph)?;
                }
            }};
        }
        macro_rules! sym_build {
            ($b:ident, $B:ty) => {{
                for s in seeds {
                    $b.build(seed_from_bytes::<$B>(s), graph)?;
                }
            }};
        }
        match self {
            Self::Sub512(b) => sub_build!(b, B512),
            Self::Sub1024(b) => sub_build!(b, B1024),
            Self::Sub2048(b) => sub_build!(b, B2048),
            Self::Sub4096(b) => sub_build!(b, B4096),
            Self::Sub8192(b) => sub_build!(b, B8192),
            Self::Sym512(b) => sym_build!(b, B512),
            Self::Sym1024(b) => sym_build!(b, B1024),
            Self::Sym2048(b) => sym_build!(b, B2048),
            Self::Sym4096(b) => sym_build!(b, B4096),
            Self::Sym8192(b) => sym_build!(b, B8192),
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Family enum (Default vs LargeInt)
// ---------------------------------------------------------------------------

/// `Bit` family dispatcher. `LargeInt` arm vanishes when the
/// `large-int` feature is off — methods on this type never use a
/// `#[cfg]` match arm.
pub enum BitBasis {
    Default(BitBasisDefault),
    #[cfg(feature = "large-int")]
    LargeInt(BitBasisLargeInt),
}

impl_family_dispatch_enum!(BitBasis);

impl BitBasis {
    #[inline]
    pub fn add_inv(&mut self, grp_char: Complex<f64>, locs: &[usize]) -> Result<(), QuSpinError> {
        match self {
            Self::Default(inner) => inner.add_inv(grp_char, locs),
            #[cfg(feature = "large-int")]
            Self::LargeInt(inner) => inner.add_inv(grp_char, locs),
        }
    }

    #[inline]
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        match self {
            Self::Default(inner) => inner.add_local(grp_char, perm_vals, locs),
            #[cfg(feature = "large-int")]
            Self::LargeInt(inner) => inner.add_local(grp_char, perm_vals, locs),
        }
    }

    #[inline]
    pub fn build_seeds<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        match self {
            Self::Default(inner) => inner.build_seeds(graph, seeds),
            #[cfg(feature = "large-int")]
            Self::LargeInt(inner) => inner.build_seeds(graph, seeds),
        }
    }
}
