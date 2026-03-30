/// Spin basis type [`SpinBasis`].
use super::dispatch::SpaceInner;
use super::seed::{dit_seed_from_bytes, seed_from_bytes};
use super::space::{FullSpace, Subspace};
use super::sym::SymBasis;
use crate::bitbasis::{DynamicPermDitValues, PermDitMask};
use crate::error::QuSpinError;
use crate::operator::{BondOperatorInner, SpinOperatorInner};
use crate::{with_dit_sym_basis_mut, with_sub_basis_mut, with_sym_basis_mut};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// SpaceKind
// ---------------------------------------------------------------------------

/// Selects which kind of Hilbert space a basis represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpaceKind {
    /// Full Hilbert space — no projection, no build step required.
    Full,
    /// Particle-number (or energy) sector subspace.
    Sub,
    /// Symmetry-reduced subspace.
    Symm,
}

// ---------------------------------------------------------------------------
// SpinBasis
// ---------------------------------------------------------------------------

/// A unified spin basis combining space-kind selection, symmetry group
/// building, and basis construction into one type.
///
/// - [`SpaceKind::Full`]  — full Hilbert space; no build step required.
/// - [`SpaceKind::Sub`]   — particle-number / energy subspace built by BFS.
/// - [`SpaceKind::Symm`]  — symmetry-reduced subspace; add symmetry elements
///   with [`add_lattice`](SpinBasis::add_lattice) /
///   [`add_inv`](SpinBasis::add_inv) before calling a `build_*` method.
pub struct SpinBasis {
    pub n_sites: usize,
    pub lhss: usize,
    space_kind: SpaceKind,
    pub inner: SpaceInner,
}

impl SpinBasis {
    /// Construct a new spin basis.
    ///
    /// # Errors
    /// - `lhss < 2`
    /// - [`SpaceKind::Full`] with more than 64 bits required
    /// - [`SpaceKind::Sub`] / [`SpaceKind::Symm`] with more than 8192 bits
    pub fn new(n_sites: usize, lhss: usize, space_kind: SpaceKind) -> Result<Self, QuSpinError> {
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
                    SpaceInner::Full32(FullSpace::<u32>::new(lhss, n_sites, false))
                } else {
                    SpaceInner::Full64(FullSpace::<u64>::new(lhss, n_sites, false))
                }
            }
            SpaceKind::Sub => crate::select_b_for_n_sites!(
                n_bits,
                B,
                return Err(QuSpinError::ValueError(format!(
                    "n_sites={n_sites} with lhss={lhss} requires {n_bits} bits, \
                     exceeding the 8192-bit maximum"
                ))),
                { SpaceInner::from(Subspace::<B>::new_empty(lhss, n_sites, false)) }
            ),
            SpaceKind::Symm => {
                if lhss == 2 {
                    crate::select_b_for_n_sites!(
                        n_bits,
                        B,
                        return Err(QuSpinError::ValueError(format!(
                            "n_sites={n_sites} with lhss={lhss} requires {n_bits} bits, \
                             exceeding the 8192-bit maximum"
                        ))),
                        {
                            SpaceInner::from(SymBasis::<B, PermDitMask<B>, _>::new_empty(
                                lhss, n_sites, false,
                            ))
                        }
                    )
                } else {
                    crate::select_b_for_n_sites!(
                        n_bits,
                        B,
                        return Err(QuSpinError::ValueError(format!(
                            "n_sites={n_sites} with lhss={lhss} requires {n_bits} bits, \
                             exceeding the 8192-bit maximum"
                        ))),
                        {
                            SpaceInner::from(SymBasis::<B, DynamicPermDitValues, _>::new_empty(
                                lhss, n_sites, false,
                            ))
                        }
                    )
                }
            }
        };

        Ok(SpinBasis {
            n_sites,
            lhss,
            space_kind,
            inner,
        })
    }

    /// The [`SpaceKind`] this basis was constructed with.
    pub fn space_kind(&self) -> SpaceKind {
        self.space_kind
    }

    /// Add a lattice (site-permutation) symmetry element.
    ///
    /// Valid only for [`SpaceKind::Symm`] bases before [`build_spin`](Self::build_spin) /
    /// [`build_bond`](Self::build_bond) is called.
    pub fn add_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        if self.space_kind != SpaceKind::Symm {
            return Err(QuSpinError::ValueError(
                "add_lattice requires SpaceKind::Symm".into(),
            ));
        }
        if self.inner.is_built() {
            return Err(QuSpinError::ValueError(
                "cannot add symmetry elements after basis is built".into(),
            ));
        }
        if perm.len() != self.n_sites {
            return Err(QuSpinError::ValueError(format!(
                "perm.len()={} but n_sites={}",
                perm.len(),
                self.n_sites
            )));
        }
        self.inner.push_lattice(grp_char, &perm)
    }

    /// Add a spin-inversion symmetry element.
    ///
    /// - LHSS = 2: XOR bit-flip (`v → 1 − v`) at each site in `locs`.
    /// - LHSS > 2: value inversion (`v → lhss − v − 1`) at each site in `locs`.
    ///
    /// `locs = None` applies the operation to all sites.
    pub fn add_inv(&mut self, locs: Option<Vec<u32>>) -> Result<(), QuSpinError> {
        if self.space_kind != SpaceKind::Symm {
            return Err(QuSpinError::ValueError(
                "add_inv requires SpaceKind::Symm".into(),
            ));
        }
        if self.inner.is_built() {
            return Err(QuSpinError::ValueError(
                "cannot add symmetry elements after basis is built".into(),
            ));
        }
        let locs_u32 = locs.unwrap_or_else(|| (0..self.n_sites as u32).collect());
        let locs_usize: Vec<usize> = locs_u32.iter().map(|&v| v as usize).collect();

        if self.lhss == 2 {
            self.inner
                .push_local_mask(Complex::new(-1.0, 0.0), &locs_usize)
        } else {
            let perm: Vec<u8> = (0..self.lhss).rev().map(|v| v as u8).collect();
            self.inner
                .push_local_perm(Complex::new(-1.0, 0.0), perm, locs_usize)
        }
    }

    /// Build the subspace reachable from `seeds` using a [`SpinOperatorInner`].
    ///
    /// Not valid for [`SpaceKind::Full`] (full spaces require no build step).
    ///
    /// # Errors
    /// - Called on a [`SpaceKind::Full`] basis
    /// - Basis is already built
    /// - `ham.lhss() != self.lhss`
    pub fn build_spin(
        &mut self,
        ham: &SpinOperatorInner,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        if self.space_kind == SpaceKind::Full {
            return Err(QuSpinError::ValueError(
                "Full basis requires no build step".into(),
            ));
        }
        if self.inner.is_built() {
            return Err(QuSpinError::ValueError("basis is already built".into()));
        }
        if ham.lhss() != self.lhss {
            return Err(QuSpinError::ValueError(format!(
                "ham.lhss()={} does not match basis lhss={}",
                ham.lhss(),
                self.lhss
            )));
        }

        // For lhss=2, seeds are bit-encoded; for lhss>2 they are dit-encoded.
        let lhss = self.lhss;
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

        match self.space_kind {
            SpaceKind::Sub => {
                with_sub_basis_mut!(&mut self.inner, B, subspace, {
                    for seed in seeds {
                        let s = decode_seed!(B, seed);
                        match ham {
                            SpinOperatorInner::Ham8(h) => {
                                subspace.build(s, |state| h.apply_smallvec(state).into_iter());
                            }
                            SpinOperatorInner::Ham16(h) => {
                                subspace.build(s, |state| h.apply_smallvec(state).into_iter());
                            }
                        }
                    }
                });
            }
            SpaceKind::Symm if self.lhss == 2 => {
                with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
                    for seed in seeds {
                        let s = decode_seed!(B, seed);
                        match ham {
                            SpinOperatorInner::Ham8(h) => {
                                sym_basis.build(s, |state| h.apply_smallvec(state).into_iter());
                            }
                            SpinOperatorInner::Ham16(h) => {
                                sym_basis.build(s, |state| h.apply_smallvec(state).into_iter());
                            }
                        }
                    }
                });
            }
            SpaceKind::Symm => {
                with_dit_sym_basis_mut!(&mut self.inner, B, sym_basis, {
                    for seed in seeds {
                        let s = decode_seed!(B, seed);
                        match ham {
                            SpinOperatorInner::Ham8(h) => {
                                sym_basis.build(s, |state| h.apply_smallvec(state).into_iter());
                            }
                            SpinOperatorInner::Ham16(h) => {
                                sym_basis.build(s, |state| h.apply_smallvec(state).into_iter());
                            }
                        }
                    }
                });
            }
            SpaceKind::Full => unreachable!(),
        }

        Ok(())
    }

    /// Build the subspace reachable from `seeds` using a [`BondOperatorInner`].
    ///
    /// Not valid for [`SpaceKind::Full`] (full spaces require no build step).
    ///
    /// # Errors
    /// - Called on a [`SpaceKind::Full`] basis
    /// - Basis is already built
    /// - `ham.lhss() != self.lhss`
    pub fn build_bond(
        &mut self,
        ham: &BondOperatorInner,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        use crate::operator::Operator;
        use smallvec::SmallVec;

        if self.space_kind == SpaceKind::Full {
            return Err(QuSpinError::ValueError(
                "Full basis requires no build step".into(),
            ));
        }
        if self.inner.is_built() {
            return Err(QuSpinError::ValueError("basis is already built".into()));
        }
        if ham.lhss() != self.lhss {
            return Err(QuSpinError::ValueError(format!(
                "ham.lhss()={} does not match basis lhss={}",
                ham.lhss(),
                self.lhss
            )));
        }

        // Helper closure factory: wraps BondOperatorInner::apply into
        // the (amplitude, state, cindex) iterator expected by build().
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

        // For lhss=2, seeds are bit-encoded; for lhss>2 they are dit-encoded.
        let lhss = self.lhss;
        macro_rules! decode_seed_bond {
            ($B:ty, $seed:expr) => {
                if lhss == 2 {
                    seed_from_bytes::<$B>($seed)
                } else {
                    use crate::bitbasis::manip::DynamicDitManip;
                    dit_seed_from_bytes::<$B>($seed, &DynamicDitManip::new(lhss))
                }
            };
        }

        match self.space_kind {
            SpaceKind::Sub => {
                with_sub_basis_mut!(&mut self.inner, B, subspace, {
                    for seed in seeds {
                        let s = decode_seed_bond!(B, seed);
                        subspace.build(s, |state| bond_apply_fn!(ham, state).into_iter());
                    }
                });
            }
            SpaceKind::Symm if self.lhss == 2 => {
                with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
                    for seed in seeds {
                        let s = decode_seed_bond!(B, seed);
                        sym_basis.build(s, |state| bond_apply_fn!(ham, state).into_iter());
                    }
                });
            }
            SpaceKind::Symm => {
                with_dit_sym_basis_mut!(&mut self.inner, B, sym_basis, {
                    for seed in seeds {
                        let s = decode_seed_bond!(B, seed);
                        sym_basis.build(s, |state| bond_apply_fn!(ham, state).into_iter());
                    }
                });
            }
            SpaceKind::Full => unreachable!(),
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_basis_new_full_ok() {
        let basis = SpinBasis::new(4, 2, SpaceKind::Full).unwrap();
        assert_eq!(basis.inner.size(), 16);
        assert!(basis.inner.is_built());
    }

    #[test]
    fn spin_basis_new_sub_ok() {
        let basis = SpinBasis::new(4, 2, SpaceKind::Sub).unwrap();
        assert!(!basis.inner.is_built());
    }

    #[test]
    fn spin_basis_new_symm_lhss2_ok() {
        let basis = SpinBasis::new(4, 2, SpaceKind::Symm).unwrap();
        assert!(!basis.inner.is_built());
        assert_eq!(basis.lhss, 2);
        assert_eq!(basis.n_sites, 4);
    }

    #[test]
    fn spin_basis_new_symm_lhss3_ok() {
        let basis = SpinBasis::new(4, 3, SpaceKind::Symm).unwrap();
        assert!(!basis.inner.is_built());
        assert_eq!(basis.lhss, 3);
    }

    #[test]
    fn spin_basis_lhss1_errors() {
        assert!(SpinBasis::new(4, 1, SpaceKind::Sub).is_err());
    }

    #[test]
    fn spin_basis_add_lattice_on_non_symm_errors() {
        let mut basis = SpinBasis::new(4, 2, SpaceKind::Sub).unwrap();
        let result = basis.add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 3, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn spin_basis_add_inv_all_sites() {
        let mut basis = SpinBasis::new(4, 2, SpaceKind::Symm).unwrap();
        // add identity lattice element first so symmetry group is non-trivial
        basis
            .add_lattice(Complex::new(1.0, 0.0), vec![0, 1, 2, 3])
            .unwrap();
        let result = basis.add_inv(None);
        assert!(result.is_ok(), "add_inv(None) should succeed: {result:?}");
    }

    #[test]
    fn spin_basis_build_spin_half() {
        use crate::operator::{SpinOp, SpinOpEntry, SpinOperator, SpinOperatorInner};
        use smallvec::smallvec;

        // H = S+_0 S-_1 + S-_0 S+_1  (hopping / XX+YY-type), lhss=2
        let n_sites = 4usize;
        let mut terms = vec![];
        for i in 0..n_sites as u32 - 1 {
            terms.push(SpinOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(SpinOp::Plus, i), (SpinOp::Minus, i + 1)],
            ));
            terms.push(SpinOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(SpinOp::Minus, i), (SpinOp::Plus, i + 1)],
            ));
        }
        let ham = SpinOperatorInner::Ham8(SpinOperator::new(terms, 2));

        let mut basis = SpinBasis::new(n_sites, 2, SpaceKind::Sub).unwrap();
        // seed: lowest 2 bits set = 2-particle sector, state 0b0011
        let seed = vec![1u8, 1, 0, 0];
        basis.build_spin(&ham, &[seed]).unwrap();

        // 2-particle sector of 4 sites: C(4,2) = 6
        assert_eq!(basis.inner.size(), 6);
    }

    #[test]
    fn spin_basis_build_spin_one_lhss3() {
        use crate::operator::{SpinOp, SpinOpEntry, SpinOperator, SpinOperatorInner};
        use smallvec::smallvec;

        // H = S+_0 S-_1 + S-_0 S+_1  (spin-1 hopping), lhss=3
        let n_sites = 3usize;
        let mut terms = vec![];
        for i in 0..n_sites as u32 - 1 {
            terms.push(SpinOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(SpinOp::Plus, i), (SpinOp::Minus, i + 1)],
            ));
            terms.push(SpinOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(SpinOp::Minus, i), (SpinOp::Plus, i + 1)],
            ));
        }
        let ham = SpinOperatorInner::Ham8(SpinOperator::new(terms, 3));

        let mut basis = SpinBasis::new(n_sites, 3, SpaceKind::Sub).unwrap();
        // seed: all sites in m=0 state (dit value 1)
        let seed = vec![1u8, 1, 1];
        basis.build_spin(&ham, &[seed]).unwrap();

        // The sector with total Sz=0 for 3 spin-1 sites.
        // States where sum of (1 - dit_value) = 0 in the spin-projection convention.
        // dit=1 means m=0. Hopping connects states with same total Sz.
        // Distinct states reachable = number of ways to distribute 3 sites with
        // total Sz=0 (sum of m_i = 0 where m_i in {+1,0,-1}).
        // This is the multinomial count for (n+,n0,n-) with n+ = n-, n+ + n0 + n- = 3:
        // (0,3,0), (1,1,1): 1 + 3!/(1!1!1!) = 1 + 6 = 7
        assert_eq!(basis.inner.size(), 7);
    }
}
