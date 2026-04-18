/// Fermionic basis type [`FermionBasis`].
use super::dispatch::SpaceInner;
use super::seed::seed_from_bytes;
use crate::spin::SpaceKind;
use crate::{with_sub_basis_mut, with_sym_basis_mut};
use num_complex::Complex;
use quspin_operator::{BondOperatorInner, FermionOperatorInner};
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// FermionBasis
// ---------------------------------------------------------------------------

/// A unified fermionic basis combining space-kind selection, symmetry group
/// building, and basis construction into one type.
///
/// Fermions are always LHSS=2 (one bit per orbital) with Jordan-Wigner sign
/// tracking.
///
/// - [`SpaceKind::Full`]  — full Hilbert space; no build step required.
/// - [`SpaceKind::Sub`]   — particle-number sector subspace built by BFS.
/// - [`SpaceKind::Symm`]  — symmetry-reduced subspace; add lattice symmetry
///   elements with [`add_lattice`](FermionBasis::add_lattice) before calling
///   a `build_*` method.
pub struct FermionBasis {
    pub inner: SpaceInner,
}

impl FermionBasis {
    /// Construct a new fermionic basis.
    ///
    /// Fermions are always LHSS=2 (one bit per site).
    ///
    /// # Errors
    /// - [`SpaceKind::Full`] with `n_sites > 64`
    /// - [`SpaceKind::Sub`] / [`SpaceKind::Symm`] with `n_sites > 8192`
    pub fn new(n_sites: usize, space_kind: SpaceKind) -> Result<Self, QuSpinError> {
        // Fermions: lhss=2, 1 bit per site, fermionic=true.
        let inner = super::make_space_inner(n_sites, 2, space_kind, true)?;
        Ok(FermionBasis { inner })
    }

    /// The [`SpaceKind`] this basis was constructed with.
    pub fn space_kind(&self) -> SpaceKind {
        self.inner.space_kind()
    }

    /// Add a lattice (site-permutation) symmetry element with fermionic sign tracking.
    ///
    /// Valid only for [`SpaceKind::Symm`] bases before a `build_*` method is called.
    pub fn add_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        self.inner.add_lattice(grp_char, &perm)
    }

    /// Build the subspace reachable from `seeds` using a [`FermionOperatorInner`].
    ///
    /// Not valid for [`SpaceKind::Full`] (full spaces require no build step).
    ///
    /// # Errors
    /// - Called on a [`SpaceKind::Full`] basis
    /// - Basis is already built
    pub fn build_fermion(
        &mut self,
        ham: &FermionOperatorInner,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        if self.inner.space_kind() == SpaceKind::Full {
            return Err(QuSpinError::ValueError(
                "Full basis requires no build step".into(),
            ));
        }
        if self.inner.is_built() {
            return Err(QuSpinError::ValueError("basis is already built".into()));
        }

        match self.inner.space_kind() {
            SpaceKind::Sub => {
                with_sub_basis_mut!(&mut self.inner, B, subspace, {
                    for seed in seeds {
                        let s = seed_from_bytes::<B>(seed);
                        match ham {
                            FermionOperatorInner::Ham8(h) => {
                                subspace.build(s, |state| h.apply_smallvec(state).into_iter());
                            }
                            FermionOperatorInner::Ham16(h) => {
                                subspace.build(s, |state| h.apply_smallvec(state).into_iter());
                            }
                        }
                    }
                });
            }
            SpaceKind::Symm => {
                with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
                    for seed in seeds {
                        let s = seed_from_bytes::<B>(seed);
                        match ham {
                            FermionOperatorInner::Ham8(h) => {
                                sym_basis.build(s, |state| h.apply_smallvec(state).into_iter());
                            }
                            FermionOperatorInner::Ham16(h) => {
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
    /// - `ham.lhss() != 2`
    pub fn build_bond(
        &mut self,
        ham: &BondOperatorInner,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        let space_kind = self.inner.space_kind();
        super::build_bond_inner(&mut self.inner, space_kind, 2, ham, seeds)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fermion_basis_new_full_ok() {
        let basis = FermionBasis::new(4, SpaceKind::Full).unwrap();
        // Full 4-site fermion basis: 2^4 = 16 states.
        assert_eq!(basis.inner.size(), 16);
        assert!(basis.inner.is_built());
    }

    #[test]
    fn fermion_basis_new_sub_ok() {
        let basis = FermionBasis::new(4, SpaceKind::Sub).unwrap();
        assert!(!basis.inner.is_built());
        assert_eq!(basis.inner.n_sites(), 4);
    }

    #[test]
    fn fermion_basis_new_symm_ok() {
        let basis = FermionBasis::new(4, SpaceKind::Symm).unwrap();
        assert!(!basis.inner.is_built());
        assert_eq!(basis.inner.n_sites(), 4);
    }

    #[test]
    fn fermion_basis_full_too_large_errors() {
        // n_sites=65 > 64 → error for Full
        assert!(FermionBasis::new(65, SpaceKind::Full).is_err());
    }

    #[test]
    fn fermion_basis_add_lattice_on_non_symm_errors() {
        let mut basis = FermionBasis::new(4, SpaceKind::Sub).unwrap();
        let result = basis.add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 3, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn fermion_basis_build_fermion() {
        use quspin_operator::fermion::{FermionOp, FermionOpEntry, FermionOperator};
        use smallvec::smallvec;

        // Hopping Hamiltonian: H = sum_i (c†_i c_{i+1} + c†_{i+1} c_i), 4 sites.
        let n_sites = 4usize;
        let mut terms = vec![];
        for i in 0..n_sites as u32 - 1 {
            // c†_i c_{i+1}
            terms.push(FermionOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(FermionOp::Plus, i), (FermionOp::Minus, i + 1)],
            ));
            // c†_{i+1} c_i
            terms.push(FermionOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(FermionOp::Plus, i + 1), (FermionOp::Minus, i)],
            ));
        }
        let ham = FermionOperatorInner::Ham8(FermionOperator::new(terms));

        let mut basis = FermionBasis::new(n_sites, SpaceKind::Sub).unwrap();
        // Seed: state 0b0011 = sites 0 and 1 occupied (2-particle sector).
        let seed = vec![1u8, 1, 0, 0];
        basis.build_fermion(&ham, &[seed]).unwrap();

        // 2-particle sector of 4 sites: C(4,2) = 6 states.
        assert_eq!(basis.inner.size(), 6);
    }

    #[test]
    fn fermion_basis_build_bond() {
        use ndarray::array;
        use quspin_operator::bond::{BondOperator, BondTerm};

        // Hopping matrix for LHSS=2: [[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]]
        // This swaps |01> <-> |10> (single-particle hopping).
        let n_sites = 4usize;
        let hop_mat = array![
            [
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0)
            ],
            [
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0)
            ],
            [
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0)
            ],
            [
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0)
            ],
        ];
        let bonds: Vec<(u32, u32)> = (0..n_sites as u32 - 1).map(|i| (i, i + 1)).collect();
        let term = BondTerm {
            cindex: 0u8,
            matrix: hop_mat,
            bonds,
        };
        let ham = BondOperatorInner::Ham8(BondOperator::new(vec![term]).unwrap());

        let mut basis = FermionBasis::new(n_sites, SpaceKind::Sub).unwrap();
        // Seed: state 0b0011 (sites 0 and 1 occupied = 2-particle sector).
        let seed = vec![1u8, 1, 0, 0];
        basis.build_bond(&ham, &[seed]).unwrap();

        // 2-particle sector of 4 sites: C(4,2) = 6 states.
        assert_eq!(basis.inner.size(), 6);
    }
}
