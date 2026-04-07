pub(crate) mod bfs;
pub mod boson;
pub mod dispatch;
pub mod expand;
pub mod fermion;
pub mod generic;
pub(crate) mod lattice;
pub(crate) mod orbit;
pub mod seed;
pub mod space;
pub mod spin;
pub mod sym;
pub mod traits;

pub use boson::BosonBasis;
pub use fermion::FermionBasis;
pub use generic::GenericBasis;
pub use seed::{
    dit_seed_from_bytes, dit_seed_from_str, dit_state_to_str, seed_from_bytes, seed_from_str,
    state_to_str,
};
pub use space::{FullSpace, Subspace};
pub use spin::{SpaceKind, SpinBasis};
pub use sym::{NormInt, SymBasis};
pub use traits::{BasisSpace, SymGrp};

// ---------------------------------------------------------------------------
// Shared basis-construction helpers
// ---------------------------------------------------------------------------

use self::dispatch::SpaceInner;
use crate::bitbasis::{DynamicPermDitValues, PermDitMask, PermDitValues};
use crate::error::QuSpinError;
use crate::operator::BondOperatorInner;
use crate::{
    with_dit_sym_basis_mut, with_quat_sym_basis_mut, with_sub_basis_mut, with_sym_basis_mut,
    with_trit_sym_basis_mut,
};
use num_complex::Complex;
use smallvec::SmallVec;

/// Construct a [`SpaceInner`] for the given lattice parameters.
///
/// Shared by [`SpinBasis::new`], [`BosonBasis::new`], [`GenericBasis::new`],
/// and [`FermionBasis::new`].
pub(crate) fn make_space_inner(
    n_sites: usize,
    lhss: usize,
    space_kind: SpaceKind,
    fermionic: bool,
) -> Result<SpaceInner, QuSpinError> {
    if lhss < 2 {
        return Err(QuSpinError::ValueError(format!(
            "lhss must be >= 2, got {lhss}"
        )));
    }
    let bits_per_dit = if lhss <= 2 {
        1
    } else {
        (usize::BITS - (lhss - 1).leading_zeros()) as usize
    };
    let n_bits = n_sites * bits_per_dit;

    let inner = match space_kind {
        SpaceKind::Full => {
            if n_bits > 64 {
                return Err(QuSpinError::ValueError(format!(
                    "Full basis requires n_bits <= 64, but n_sites={n_sites} with \
                     lhss={lhss} needs {n_bits} bits"
                )));
            }
            if n_bits <= 32 {
                SpaceInner::Full32(FullSpace::<u32>::new(lhss, n_sites, fermionic))
            } else {
                SpaceInner::Full64(FullSpace::<u64>::new(lhss, n_sites, fermionic))
            }
        }
        SpaceKind::Sub => crate::select_b_for_n_sites!(
            n_bits,
            B,
            return Err(QuSpinError::ValueError(format!(
                "n_sites={n_sites} with lhss={lhss} requires {n_bits} bits, \
                 exceeding the 8192-bit maximum"
            ))),
            { SpaceInner::from(Subspace::<B>::new_empty(lhss, n_sites, fermionic)) }
        ),
        SpaceKind::Symm => {
            macro_rules! overflow_err {
                () => {
                    return Err(QuSpinError::ValueError(format!(
                        "n_sites={n_sites} with lhss={lhss} requires {n_bits} bits, \
                         exceeding the 8192-bit maximum"
                    )))
                };
            }
            match lhss {
                2 => crate::select_b_for_n_sites!(n_bits, B, overflow_err!(), {
                    SpaceInner::from(SymBasis::<B, PermDitMask<B>, _>::new_empty(
                        lhss, n_sites, fermionic,
                    ))
                }),
                3 => crate::select_b_for_n_sites!(n_bits, B, overflow_err!(), {
                    SpaceInner::from(SymBasis::<B, PermDitValues<3>, _>::new_empty(
                        lhss, n_sites, fermionic,
                    ))
                }),
                4 => crate::select_b_for_n_sites!(n_bits, B, overflow_err!(), {
                    SpaceInner::from(SymBasis::<B, PermDitValues<4>, _>::new_empty(
                        lhss, n_sites, fermionic,
                    ))
                }),
                _ => crate::select_b_for_n_sites!(n_bits, B, overflow_err!(), {
                    SpaceInner::from(SymBasis::<B, DynamicPermDitValues, _>::new_empty(
                        lhss, n_sites, fermionic,
                    ))
                }),
            }
        }
    };
    Ok(inner)
}

/// Build a basis subspace from `seeds` using a [`BondOperatorInner`].
///
/// Shared by [`SpinBasis::build_bond`], [`BosonBasis::build_bond`], and
/// [`FermionBasis::build_bond`].
pub(crate) fn build_bond_inner(
    inner: &mut SpaceInner,
    space_kind: SpaceKind,
    lhss: usize,
    ham: &BondOperatorInner,
    seeds: &[Vec<u8>],
) -> Result<(), QuSpinError> {
    use crate::operator::Operator;

    if space_kind == SpaceKind::Full {
        return Err(QuSpinError::ValueError(
            "Full basis requires no build step".into(),
        ));
    }
    if inner.is_built() {
        return Err(QuSpinError::ValueError("basis is already built".into()));
    }
    if ham.lhss() != lhss {
        return Err(QuSpinError::ValueError(format!(
            "ham.lhss()={} does not match basis lhss={}",
            ham.lhss(),
            lhss
        )));
    }

    macro_rules! bond_apply_fn {
        ($ham_inner:expr, $state:expr) => {{
            let mut out: SmallVec<[(Complex<f64>, _, u8); 8]> = SmallVec::new();
            match $ham_inner {
                BondOperatorInner::Ham8(h) => {
                    h.apply($state, |c, amp, ns| out.push((amp, ns, c)));
                }
                BondOperatorInner::Ham16(h) => {
                    h.apply($state, |c, amp, ns| out.push((amp, ns, c as u8)));
                }
            }
            out
        }};
    }

    macro_rules! decode_seed {
        ($B:ty, $seed:expr) => {
            if lhss == 2 {
                seed_from_bytes::<$B>($seed)
            } else {
                use crate::bitbasis::manip::DynamicDitManip;
                dit_seed_from_bytes::<$B>($seed, &DynamicDitManip::new(lhss))
            }
        };
    }

    match space_kind {
        SpaceKind::Sub => {
            with_sub_basis_mut!(inner, B, subspace, {
                for seed in seeds {
                    let s = decode_seed!(B, seed);
                    subspace.build(s, |state| bond_apply_fn!(ham, state).into_iter());
                }
            });
        }
        SpaceKind::Symm if lhss == 2 => {
            with_sym_basis_mut!(inner, B, sym_basis, {
                for seed in seeds {
                    let s = decode_seed!(B, seed);
                    sym_basis.build(s, |state| bond_apply_fn!(ham, state).into_iter());
                }
            });
        }
        SpaceKind::Symm if lhss == 3 => {
            with_trit_sym_basis_mut!(inner, B, sym_basis, {
                for seed in seeds {
                    let s = decode_seed!(B, seed);
                    sym_basis.build(s, |state| bond_apply_fn!(ham, state).into_iter());
                }
            });
        }
        SpaceKind::Symm if lhss == 4 => {
            with_quat_sym_basis_mut!(inner, B, sym_basis, {
                for seed in seeds {
                    let s = decode_seed!(B, seed);
                    sym_basis.build(s, |state| bond_apply_fn!(ham, state).into_iter());
                }
            });
        }
        SpaceKind::Symm => {
            with_dit_sym_basis_mut!(inner, B, sym_basis, {
                for seed in seeds {
                    let s = decode_seed!(B, seed);
                    sym_basis.build(s, |state| bond_apply_fn!(ham, state).into_iter());
                }
            });
        }
        SpaceKind::Full => unreachable!(),
    }

    Ok(())
}
