//! `Quat` family — LHSS = 4.
//!
//! Local-op type for the symmetric variants is
//! [`PermDitValues<4>`](quspin_bitbasis::PermDitValues).

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
use quspin_bitbasis::{PermDitValues, StateTransitions};
use quspin_types::QuSpinError;

const LHSS: usize = 4;

// ---------------------------------------------------------------------------
// Default-width inner enum
// ---------------------------------------------------------------------------

pub enum QuatBasisDefault {
    Full32(FullSpace<u32>),
    Full64(FullSpace<u64>),

    Sub32(Subspace<u32>),
    Sub64(Subspace<u64>),
    Sub128(Subspace<B128>),
    Sub256(Subspace<B256>),

    Sym32(SymBasis<u32, PermDitValues<4>, u8>),
    Sym64(SymBasis<u64, PermDitValues<4>, u16>),
    Sym128(SymBasis<B128, PermDitValues<4>, u32>),
    Sym256(SymBasis<B256, PermDitValues<4>, u32>),
}

impl_inner_dispatch_enum!(
    QuatBasisDefault,
    full = [Full32, Full64],
    sub = [Sub32, Sub64, Sub128, Sub256],
    sym = [Sym32, Sym64, Sym128, Sym256],
);

impl QuatBasisDefault {
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        validate_perm_vals(&perm_vals, LHSS)?;
        validate_locs(&locs, self.n_sites())?;
        let arr: [u8; LHSS] = perm_vals.try_into().map_err(|v: Vec<u8>| {
            QuSpinError::ValueError(format!(
                "add_local on Quat family (LHSS=4) requires perm_vals.len()=4, got len={}",
                v.len()
            ))
        })?;
        match self {
            Self::Sym32(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym64(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym128(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym256(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitValues::<4>::new(arr, locs)),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_local requires a Sym* variant on QuatBasisDefault".into(),
            )),
        }
    }

    /// Add a composite (lattice + local) element. `perm_vals` must be a
    /// length-4 bijection on `0..4`; `locs` is validated against `n_sites`.
    pub fn add_composite(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        validate_perm_vals(&perm_vals, LHSS)?;
        validate_locs(&locs, self.n_sites())?;
        let arr: [u8; LHSS] = perm_vals.try_into().map_err(|v: Vec<u8>| {
            QuSpinError::ValueError(format!(
                "add_composite on Quat family (LHSS=4) requires perm_vals.len()=4, got len={}",
                v.len()
            ))
        })?;
        match self {
            Self::Sym32(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym64(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym128(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym256(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, PermDitValues::<4>::new(arr, locs)),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_composite requires a Sym* variant on QuatBasisDefault".into(),
            )),
        }
    }

    pub fn build_seeds<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        let manip = DynamicDitManip::new(LHSS);
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
pub enum QuatBasisLargeInt {
    Sub512(Subspace<B512>),
    Sub1024(Subspace<B1024>),
    Sub2048(Subspace<B2048>),
    Sub4096(Subspace<B4096>),
    Sub8192(Subspace<B8192>),

    Sym512(SymBasis<B512, PermDitValues<4>, u32>),
    Sym1024(SymBasis<B1024, PermDitValues<4>, u32>),
    Sym2048(SymBasis<B2048, PermDitValues<4>, u32>),
    Sym4096(SymBasis<B4096, PermDitValues<4>, u32>),
    Sym8192(SymBasis<B8192, PermDitValues<4>, u32>),
}

#[cfg(feature = "large-int")]
impl_inner_dispatch_enum!(
    QuatBasisLargeInt,
    full = [],
    sub = [Sub512, Sub1024, Sub2048, Sub4096, Sub8192],
    sym = [Sym512, Sym1024, Sym2048, Sym4096, Sym8192],
);

#[cfg(feature = "large-int")]
impl QuatBasisLargeInt {
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        validate_perm_vals(&perm_vals, LHSS)?;
        validate_locs(&locs, self.n_sites())?;
        let arr: [u8; LHSS] = perm_vals.try_into().map_err(|v: Vec<u8>| {
            QuSpinError::ValueError(format!(
                "add_local on Quat family (LHSS=4) requires perm_vals.len()=4, got len={}",
                v.len()
            ))
        })?;
        match self {
            Self::Sym512(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym1024(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym2048(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym4096(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym8192(b) => b.add_symmetry(
                grp_char,
                SymElement::local(PermDitValues::<4>::new(arr, locs)),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_local requires a Sym* variant on QuatBasisLargeInt".into(),
            )),
        }
    }

    /// Add a composite (lattice + local) element. `perm_vals` must be a
    /// length-4 bijection on `0..4`; `locs` is validated against `n_sites`.
    pub fn add_composite(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        validate_perm_vals(&perm_vals, LHSS)?;
        validate_locs(&locs, self.n_sites())?;
        let arr: [u8; LHSS] = perm_vals.try_into().map_err(|v: Vec<u8>| {
            QuSpinError::ValueError(format!(
                "add_composite on Quat family (LHSS=4) requires perm_vals.len()=4, got len={}",
                v.len()
            ))
        })?;
        match self {
            Self::Sym512(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym1024(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym2048(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym4096(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, PermDitValues::<4>::new(arr, locs)),
            ),
            Self::Sym8192(b) => b.add_symmetry(
                grp_char,
                SymElement::composite(perm, PermDitValues::<4>::new(arr, locs)),
            ),
            _ => Err(QuSpinError::ValueError(
                "add_composite requires a Sym* variant on QuatBasisLargeInt".into(),
            )),
        }
    }

    pub fn build_seeds<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        let manip = DynamicDitManip::new(LHSS);
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

pub enum QuatBasis {
    Default(QuatBasisDefault),
    #[cfg(feature = "large-int")]
    LargeInt(QuatBasisLargeInt),
}

impl_family_dispatch_enum!(QuatBasis);

impl QuatBasis {
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
    fn add_composite_4site_lhss4() {
        use crate::SpaceKind;
        let mut basis = crate::dispatch::DitBasis::new(4, 4, SpaceKind::Symm).unwrap();
        if let crate::dispatch::DitBasis::Quat(ref mut q) = basis {
            // perm cycles 4 sites; perm_vals swaps 0<->1 and 2<->3.
            q.add_composite(
                num_complex::Complex::new(1.0, 0.0),
                &[1, 2, 3, 0],
                vec![1, 0, 3, 2],
                vec![0, 1, 2, 3],
            )
            .unwrap();
        } else {
            panic!("expected Quat variant");
        }
    }

    #[test]
    fn add_composite_rejects_non_sym_variant_quat() {
        use crate::SpaceKind;
        let mut basis = crate::dispatch::DitBasis::new(4, 4, SpaceKind::Sub).unwrap();
        if let crate::dispatch::DitBasis::Quat(ref mut q) = basis {
            let r = q.add_composite(
                num_complex::Complex::new(1.0, 0.0),
                &[1, 2, 3, 0],
                vec![1, 0, 3, 2],
                vec![0, 1, 2, 3],
            );
            assert!(r.is_err());
        } else {
            panic!("expected Quat variant");
        }
    }
}
