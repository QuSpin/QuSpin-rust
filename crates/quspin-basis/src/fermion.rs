/// Fermionic basis type [`FermionBasis`].
use super::dispatch::SpaceInner;
use crate::spin::SpaceKind;
use num_complex::Complex;
use quspin_bitbasis::StateGraph;
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

    /// Build the subspace reachable from `seeds` under the connectivity
    /// described by `graph`. Requires `graph.lhss() == 2`.
    ///
    /// # Errors
    /// - Called on a [`SpaceKind::Full`] basis
    /// - Basis is already built
    /// - `graph.lhss() != 2`
    pub fn build<G: StateGraph>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        super::build_inner(&mut self.inner, graph, seeds)
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
        use quspin_operator::fermion::{
            FermionOp, FermionOpEntry, FermionOperator, FermionOperatorInner,
        };
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
        basis.build(&ham, &[seed]).unwrap();

        // 2-particle sector of 4 sites: C(4,2) = 6 states.
        assert_eq!(basis.inner.size(), 6);
    }

    #[test]
    fn fermion_basis_build_bond() {
        use ndarray::array;
        use quspin_operator::bond::{BondOperator, BondOperatorInner, BondTerm};

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
        basis.build(&ham, &[seed]).unwrap();

        // 2-particle sector of 4 sites: C(4,2) = 6 states.
        assert_eq!(basis.inner.size(), 6);
    }
}
