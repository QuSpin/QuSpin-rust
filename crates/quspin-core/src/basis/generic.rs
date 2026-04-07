/// Generic basis type [`GenericBasis`].
use super::dispatch::SpaceInner;
use super::seed::{dit_seed_from_bytes, seed_from_bytes};
use crate::basis::spin::SpaceKind;
use crate::error::QuSpinError;
use crate::operator::MonomialOperatorInner;
use crate::{
    with_dit_sym_basis_mut, with_quat_sym_basis_mut, with_sub_basis_mut, with_sym_basis_mut,
    with_trit_sym_basis_mut,
};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// GenericBasis
// ---------------------------------------------------------------------------

/// A basis for models with any on-site Hilbert-space size, supporting both
/// lattice (site-permutation) and local (dit-permutation) symmetries.
///
/// Unlike [`BosonBasis`](super::boson::BosonBasis), the symmetric variant
/// uses the appropriate `Sym*`/`TritSym*`/`QuatSym*`/`DitSym*` path based
/// on LHSS, so that local symmetry generators can be added via
/// [`add_local`](GenericBasis::add_local).
///
/// - [`SpaceKind::Full`]  — full Hilbert space; no build step required.
/// - [`SpaceKind::Sub`]   — particle-number / energy subspace built by BFS.
/// - [`SpaceKind::Symm`]  — symmetry-reduced subspace; add symmetry elements
///   with [`add_lattice`](GenericBasis::add_lattice) and/or
///   [`add_local`](GenericBasis::add_local) before building.
pub struct GenericBasis {
    pub inner: SpaceInner,
}

impl GenericBasis {
    /// Construct a new generic basis.
    ///
    /// # Errors
    /// - `lhss < 2`
    /// - [`SpaceKind::Full`] with more than 64 bits required
    /// - [`SpaceKind::Sub`] / [`SpaceKind::Symm`] with more than 8192 bits
    pub fn new(n_sites: usize, lhss: usize, space_kind: SpaceKind) -> Result<Self, QuSpinError> {
        let inner = super::make_space_inner(n_sites, lhss, space_kind, false)?;
        Ok(GenericBasis { inner })
    }

    /// The [`SpaceKind`] this basis was constructed with.
    pub fn space_kind(&self) -> SpaceKind {
        self.inner.space_kind()
    }

    /// Add a lattice (site-permutation) symmetry element.
    ///
    /// Valid only for [`SpaceKind::Symm`] bases before building.
    pub fn add_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        self.inner.add_lattice(grp_char, &perm)
    }

    /// Add a local dit-permutation symmetry element.
    ///
    /// `perm_vals[v] = w` maps local-state `v` to `w` at each site in `locs`.
    /// `perm_vals.len()` must equal `self.lhss`.
    ///
    /// Valid only for [`SpaceKind::Symm`] bases before building.
    ///
    /// # Errors
    /// - Basis is not [`SpaceKind::Symm`]
    /// - Basis is already built
    /// - `perm_vals.len() != lhss`
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        self.inner.add_local(grp_char, perm_vals, locs)
    }

    /// Build the subspace reachable from `seeds` using a [`MonomialOperatorInner`].
    ///
    /// Not valid for [`SpaceKind::Full`] (full spaces require no build step).
    ///
    /// Seeds are per-site occupation byte slices.
    ///
    /// # Errors
    /// - Called on a [`SpaceKind::Full`] basis
    /// - Basis is already built
    /// - `ham.lhss() != self.lhss`
    pub fn build_monomial(
        &mut self,
        ham: &MonomialOperatorInner,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        use crate::operator::Operator;

        if self.inner.space_kind() == SpaceKind::Full {
            return Err(QuSpinError::ValueError(
                "Full basis requires no build step".into(),
            ));
        }
        if self.inner.is_built() {
            return Err(QuSpinError::ValueError("basis is already built".into()));
        }
        let lhss = self.inner.lhss();
        if ham.lhss() != lhss {
            return Err(QuSpinError::ValueError(format!(
                "ham.lhss()={} does not match basis lhss={}",
                ham.lhss(),
                lhss
            )));
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

        macro_rules! monomial_apply_fn {
            ($ham_inner:expr, $state:expr) => {{
                let mut out: smallvec::SmallVec<[(Complex<f64>, _, u8); 8]> =
                    smallvec::SmallVec::new();
                match $ham_inner {
                    MonomialOperatorInner::Ham8(h) => {
                        h.apply($state, |c, amp, ns| out.push((amp, ns, c)));
                    }
                    MonomialOperatorInner::Ham16(h) => {
                        h.apply($state, |c, amp, ns| out.push((amp, ns, c as u8)));
                    }
                }
                out
            }};
        }

        match self.inner.space_kind() {
            SpaceKind::Sub => {
                with_sub_basis_mut!(&mut self.inner, B, subspace, {
                    for seed in seeds {
                        let s = decode_seed!(B, seed);
                        subspace.build(s, |state| monomial_apply_fn!(ham, state).into_iter());
                    }
                });
            }
            SpaceKind::Symm if lhss == 2 => {
                with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
                    for seed in seeds {
                        let s = decode_seed!(B, seed);
                        sym_basis.build(s, |state| monomial_apply_fn!(ham, state).into_iter());
                    }
                });
            }
            SpaceKind::Symm if lhss == 3 => {
                with_trit_sym_basis_mut!(&mut self.inner, B, sym_basis, {
                    for seed in seeds {
                        let s = decode_seed!(B, seed);
                        sym_basis.build(s, |state| monomial_apply_fn!(ham, state).into_iter());
                    }
                });
            }
            SpaceKind::Symm if lhss == 4 => {
                with_quat_sym_basis_mut!(&mut self.inner, B, sym_basis, {
                    for seed in seeds {
                        let s = decode_seed!(B, seed);
                        sym_basis.build(s, |state| monomial_apply_fn!(ham, state).into_iter());
                    }
                });
            }
            SpaceKind::Symm => {
                with_dit_sym_basis_mut!(&mut self.inner, B, sym_basis, {
                    for seed in seeds {
                        let s = decode_seed!(B, seed);
                        sym_basis.build(s, |state| monomial_apply_fn!(ham, state).into_iter());
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
    use crate::operator::monomial::{MonomialOperator, MonomialTerm};
    use num_complex::Complex;
    use smallvec::smallvec;

    fn cyclic_term(lhss: usize, n_sites: usize) -> MonomialTerm<u8> {
        // Cyclic permutation of local states: v -> (v+1) % lhss, at all NN bonds.
        let dim = lhss * lhss;
        let mut perm = vec![0usize; dim];
        let amp = vec![Complex::new(1.0, 0.0); dim];
        for a in 0..lhss {
            for b in 0..lhss {
                let in_idx = a * lhss + b;
                // output: (a+1)%lhss, (b+1)%lhss
                let out_idx = ((a + 1) % lhss) * lhss + ((b + 1) % lhss);
                perm[in_idx] = out_idx;
            }
        }
        let bonds: Vec<smallvec::SmallVec<[u32; 4]>> = (0..n_sites as u32 - 1)
            .map(|i| smallvec![i, i + 1])
            .collect();
        MonomialTerm {
            cindex: 0u8,
            perm,
            amp,
            bonds,
        }
    }

    #[test]
    fn generic_basis_full_lhss3() {
        let basis = GenericBasis::new(2, 3, SpaceKind::Full).unwrap();
        // Full 2-site lhss=3: 3^2 = 9 states.
        assert_eq!(basis.inner.size(), 9);
        assert!(basis.inner.is_built());
    }

    #[test]
    fn generic_basis_sub_lhss3_build() {
        let n_sites = 3;
        let lhss = 3;
        let term = cyclic_term(lhss, n_sites);
        let ham = MonomialOperatorInner::Ham8(MonomialOperator::new(vec![term], lhss).unwrap());

        let mut basis = GenericBasis::new(n_sites, lhss, SpaceKind::Sub).unwrap();
        // Seed: |000>.  NN cyclic-shift bonds on 3 sites with lhss=3 reach
        // exactly the 9 states {(a, a+b mod 3, b) | a,b ∈ {0,1,2}}.
        let seed = vec![0u8, 0, 0];
        basis.build_monomial(&ham, &[seed]).unwrap();
        assert_eq!(basis.inner.size(), 9);
    }

    #[test]
    fn generic_basis_symm_lhss2_build() {
        let n_sites = 4;
        let lhss = 2;
        let term = cyclic_term(lhss, n_sites);
        let ham = MonomialOperatorInner::Ham8(MonomialOperator::new(vec![term], lhss).unwrap());

        let mut basis = GenericBasis::new(n_sites, lhss, SpaceKind::Symm).unwrap();
        // Translation symmetry k=0: perm = [1, 2, 3, 0], char = 1.
        basis
            .add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 3, 0])
            .unwrap();
        // Seed |0000> is its own canonical representative in any translation sector.
        let seed = vec![0u8, 0, 0, 0];
        basis.build_monomial(&ham, &[seed]).unwrap();
        assert!(basis.inner.size() > 0);
    }

    #[test]
    fn generic_basis_add_local_wrong_kind_errors() {
        let mut basis = GenericBasis::new(4, 3, SpaceKind::Sub).unwrap();
        let result = basis.add_local(Complex::new(1.0, 0.0), vec![1, 2, 0], vec![0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn generic_basis_add_local_wrong_perm_len_errors() {
        let mut basis = GenericBasis::new(4, 3, SpaceKind::Symm).unwrap();
        let result = basis.add_local(Complex::new(1.0, 0.0), vec![1, 0], vec![0]); // perm len=2, lhss=3
        assert!(result.is_err());
    }

    #[test]
    fn generic_basis_lhss1_errors() {
        assert!(GenericBasis::new(4, 1, SpaceKind::Sub).is_err());
    }
}
