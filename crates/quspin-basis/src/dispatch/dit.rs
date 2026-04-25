//! `Dit` family — LHSS ≥ 5 (bosons / higher spin).
//!
//! Local-op type for the symmetric variants is
//! [`DynamicPermDitValues`](quspin_bitbasis::DynamicPermDitValues),
//! which carries an LHSS value at runtime.

use super::macros::{impl_family_dispatch_enum, impl_inner_dispatch_enum};
use super::types::{B128, B256};
#[cfg(feature = "large-int")]
use super::types::{B512, B1024, B2048, B4096, B8192};
use crate::space::{FullSpace, Subspace};
use crate::sym::SymBasis;
use crate::traits::BasisSpace;
use crate::{SymElement, seed::dit_seed_from_bytes};
use num_complex::Complex;
use quspin_bitbasis::manip::DynamicDitManip;
use quspin_bitbasis::{DynamicPermDitValues, StateTransitions};
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// Default-width inner enum
// ---------------------------------------------------------------------------

pub enum SpaceInnerDitDefault {
    Full32(FullSpace<u32>),
    Full64(FullSpace<u64>),

    Sub32(Subspace<u32>),
    Sub64(Subspace<u64>),
    Sub128(Subspace<B128>),
    Sub256(Subspace<B256>),

    Sym32(SymBasis<u32, DynamicPermDitValues, u8>),
    Sym64(SymBasis<u64, DynamicPermDitValues, u16>),
    Sym128(SymBasis<B128, DynamicPermDitValues, u32>),
    Sym256(SymBasis<B256, DynamicPermDitValues, u32>),
}

impl_inner_dispatch_enum!(
    SpaceInnerDitDefault,
    full = [Full32, Full64],
    sub = [Sub32, Sub64, Sub128, Sub256],
    sym = [Sym32, Sym64, Sym128, Sym256],
);

impl SpaceInnerDitDefault {
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        let lhss = self.lhss();
        if perm_vals.len() != lhss {
            return Err(QuSpinError::ValueError(format!(
                "add_local on Dit family requires perm_vals.len()=lhss={lhss}, got {}",
                perm_vals.len()
            )));
        }
        match self {
            Self::Sym32(b) => b.add_symmetry(
                grp_char,
                SymElement::local(DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym64(b) => b.add_symmetry(
                grp_char,
                SymElement::local(DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym128(b) => b.add_symmetry(
                grp_char,
                SymElement::local(DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym256(b) => b.add_symmetry(
                grp_char,
                SymElement::local(DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_local requires a Sym* variant on SpaceInnerDitDefault".into(),
            )),
        }
    }

    pub fn build_seeds<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        let manip = DynamicDitManip::new(self.lhss());
        macro_rules! sub_build {
            ($b:ident, $B:ty) => {{
                for s in seeds {
                    $b.build(dit_seed_from_bytes::<$B>(s, &manip), graph);
                }
            }};
        }
        macro_rules! sym_build {
            ($b:ident, $B:ty) => {{
                for s in seeds {
                    $b.build(dit_seed_from_bytes::<$B>(s, &manip), graph)?;
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
// Large-int inner enum (gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "large-int")]
pub enum SpaceInnerDitLargeInt {
    Sub512(Subspace<B512>),
    Sub1024(Subspace<B1024>),
    Sub2048(Subspace<B2048>),
    Sub4096(Subspace<B4096>),
    Sub8192(Subspace<B8192>),

    Sym512(SymBasis<B512, DynamicPermDitValues, u32>),
    Sym1024(SymBasis<B1024, DynamicPermDitValues, u32>),
    Sym2048(SymBasis<B2048, DynamicPermDitValues, u32>),
    Sym4096(SymBasis<B4096, DynamicPermDitValues, u32>),
    Sym8192(SymBasis<B8192, DynamicPermDitValues, u32>),
}

#[cfg(feature = "large-int")]
impl_inner_dispatch_enum!(
    SpaceInnerDitLargeInt,
    full = [],
    sub = [Sub512, Sub1024, Sub2048, Sub4096, Sub8192],
    sym = [Sym512, Sym1024, Sym2048, Sym4096, Sym8192],
);

#[cfg(feature = "large-int")]
impl SpaceInnerDitLargeInt {
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        let lhss = self.lhss();
        if perm_vals.len() != lhss {
            return Err(QuSpinError::ValueError(format!(
                "add_local on Dit family requires perm_vals.len()=lhss={lhss}, got {}",
                perm_vals.len()
            )));
        }
        match self {
            Self::Sym512(b) => b.add_symmetry(
                grp_char,
                SymElement::local(DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym1024(b) => b.add_symmetry(
                grp_char,
                SymElement::local(DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym2048(b) => b.add_symmetry(
                grp_char,
                SymElement::local(DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym4096(b) => b.add_symmetry(
                grp_char,
                SymElement::local(DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym8192(b) => b.add_symmetry(
                grp_char,
                SymElement::local(DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_local requires a Sym* variant on SpaceInnerDitLargeInt".into(),
            )),
        }
    }

    pub fn build_seeds<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        let manip = DynamicDitManip::new(self.lhss());
        macro_rules! sub_build {
            ($b:ident, $B:ty) => {{
                for s in seeds {
                    $b.build(dit_seed_from_bytes::<$B>(s, &manip), graph);
                }
            }};
        }
        macro_rules! sym_build {
            ($b:ident, $B:ty) => {{
                for s in seeds {
                    $b.build(dit_seed_from_bytes::<$B>(s, &manip), graph)?;
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
// Family enum
// ---------------------------------------------------------------------------

pub enum SpaceInnerDit {
    Default(SpaceInnerDitDefault),
    #[cfg(feature = "large-int")]
    LargeInt(SpaceInnerDitLargeInt),
}

impl_family_dispatch_enum!(SpaceInnerDit);

impl SpaceInnerDit {
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
