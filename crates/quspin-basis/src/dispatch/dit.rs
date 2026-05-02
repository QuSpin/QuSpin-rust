//! `Dit` family — LHSS ≥ 5 (bosons / higher spin).
//!
//! Local-op type for the symmetric variants is
//! [`DynamicPermDitValues`](quspin_bitbasis::DynamicPermDitValues),
//! which carries an LHSS value at runtime.

use super::macros::{impl_family_dispatch_enum, impl_inner_dispatch_enum};
use super::types::{B128, B256};
#[cfg(feature = "large-int")]
use super::types::{B512, B1024, B2048, B4096, B8192};
use super::validate::{validate_locs, validate_perm_vals};
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

pub enum DynDitBasisDefault {
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
    DynDitBasisDefault,
    full = [Full32, Full64],
    sub = [Sub32, Sub64, Sub128, Sub256],
    sym = [Sym32, Sym64, Sym128, Sym256],
);

impl DynDitBasisDefault {
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        let lhss = self.lhss();
        validate_perm_vals(&perm_vals, lhss)?;
        validate_locs(&locs, self.n_sites())?;
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
                "add_local requires a Sym* variant on DynDitBasisDefault".into(),
            )),
        }
    }

    /// Add a composite (lattice + local) element. `perm_vals` must be a
    /// length-`lhss` bijection on `0..lhss`; `locs` is validated against
    /// `n_sites`.
    pub fn add_composite(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        let lhss = self.lhss();
        validate_perm_vals(&perm_vals, lhss)?;
        validate_locs(&locs, self.n_sites())?;
        match self {
            Self::Sym32(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym64(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym128(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym256(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_composite requires a Sym* variant on DynDitBasisDefault".into(),
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
                    $b.build(dit_seed_from_bytes::<$B>(s, &manip), graph)?;
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
pub enum DynDitBasisLargeInt {
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
    DynDitBasisLargeInt,
    full = [],
    sub = [Sub512, Sub1024, Sub2048, Sub4096, Sub8192],
    sym = [Sym512, Sym1024, Sym2048, Sym4096, Sym8192],
);

#[cfg(feature = "large-int")]
impl DynDitBasisLargeInt {
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        let lhss = self.lhss();
        validate_perm_vals(&perm_vals, lhss)?;
        validate_locs(&locs, self.n_sites())?;
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
                "add_local requires a Sym* variant on DynDitBasisLargeInt".into(),
            )),
        }
    }

    /// Add a composite (lattice + local) element. `perm_vals` must be a
    /// length-`lhss` bijection on `0..lhss`; `locs` is validated against
    /// `n_sites`.
    pub fn add_composite(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        let lhss = self.lhss();
        validate_perm_vals(&perm_vals, lhss)?;
        validate_locs(&locs, self.n_sites())?;
        match self {
            Self::Sym512(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym1024(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym2048(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym4096(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            Self::Sym8192(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, DynamicPermDitValues::new(lhss, perm_vals, locs)),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_composite requires a Sym* variant on DynDitBasisLargeInt".into(),
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
                    $b.build(dit_seed_from_bytes::<$B>(s, &manip), graph)?;
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

pub enum DynDitBasis {
    Default(DynDitBasisDefault),
    #[cfg(feature = "large-int")]
    LargeInt(DynDitBasisLargeInt),
}

impl_family_dispatch_enum!(DynDitBasis);

impl DynDitBasis {
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
    pub fn add_composite(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        match self {
            Self::Default(inner) => inner.add_composite(grp_char, perm, perm_vals, locs),
            #[cfg(feature = "large-int")]
            Self::LargeInt(inner) => inner.add_composite(grp_char, perm, perm_vals, locs),
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

#[cfg(test)]
mod tests {
    #[test]
    fn add_composite_3site_lhss5() {
        use crate::SpaceKind;
        let mut basis = crate::dispatch::DitBasis::new(3, 5, SpaceKind::Symm).unwrap();
        if let crate::dispatch::DitBasis::Dyn(ref mut d) = basis {
            // perm cycles 3 sites; perm_vals swaps states 0<->1, leaves rest.
            d.add_composite(
                num_complex::Complex::new(1.0, 0.0),
                &[1, 2, 0],
                vec![1, 0, 2, 3, 4],
                vec![0, 1, 2],
            )
            .unwrap();
        } else {
            panic!("expected Dyn variant");
        }
    }

    #[test]
    fn add_composite_rejects_non_sym_variant_dyn() {
        use crate::SpaceKind;
        let mut basis = crate::dispatch::DitBasis::new(3, 5, SpaceKind::Sub).unwrap();
        if let crate::dispatch::DitBasis::Dyn(ref mut d) = basis {
            let r = d.add_composite(
                num_complex::Complex::new(1.0, 0.0),
                &[1, 2, 0],
                vec![1, 0, 2, 3, 4],
                vec![0, 1, 2],
            );
            assert!(r.is_err());
        } else {
            panic!("expected Dyn variant");
        }
    }
}
