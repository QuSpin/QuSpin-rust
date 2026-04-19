/// Bosonic basis type [`BosonBasis`].
use super::dispatch::SpaceInner;
use crate::spin::SpaceKind;
use num_complex::Complex;
use quspin_bitbasis::StateTransitions;
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// BosonBasis
// ---------------------------------------------------------------------------

/// A unified bosonic basis combining space-kind selection, symmetry group
/// building, and basis construction into one type.
///
/// - [`SpaceKind::Full`]  — full Hilbert space; no build step required.
/// - [`SpaceKind::Sub`]   — particle-number / energy subspace built by BFS.
/// - [`SpaceKind::Symm`]  — symmetry-reduced subspace; add lattice symmetry
///   elements with [`add_lattice`](BosonBasis::add_lattice) before calling a
///   `build_*` method.
pub struct BosonBasis {
    pub inner: SpaceInner,
}

impl BosonBasis {
    /// Construct a new bosonic basis.
    ///
    /// Bosons are never fermionic (`fermionic = false`).
    ///
    /// # Errors
    /// - `lhss < 2`
    /// - [`SpaceKind::Full`] with more than 64 bits required
    /// - [`SpaceKind::Sub`] / [`SpaceKind::Symm`] with more than 8192 bits
    pub fn new(n_sites: usize, lhss: usize, space_kind: SpaceKind) -> Result<Self, QuSpinError> {
        let inner = super::make_space_inner(n_sites, lhss, space_kind, false)?;
        Ok(BosonBasis { inner })
    }

    /// The [`SpaceKind`] this basis was constructed with.
    pub fn space_kind(&self) -> SpaceKind {
        self.inner.space_kind()
    }

    /// Add a lattice (site-permutation) symmetry element.
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
    /// described by `graph`.
    ///
    /// Seeds are per-site occupation byte slices. For lhss=2 they are
    /// bit-encoded; for lhss>2 they are dit-encoded.
    ///
    /// # Errors
    /// - Called on a [`SpaceKind::Full`] basis
    /// - Basis is already built
    /// - `graph.lhss() != self.lhss`
    pub fn build<G: StateTransitions>(
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
    fn boson_basis_new_full_lhss2_ok() {
        let basis = BosonBasis::new(4, 2, SpaceKind::Full).unwrap();
        // Full 4-site, lhss=2: 2^4 = 16 states.
        assert_eq!(basis.inner.size(), 16);
        assert!(basis.inner.is_built());
    }

    #[test]
    fn boson_basis_new_sub_ok() {
        let basis = BosonBasis::new(4, 3, SpaceKind::Sub).unwrap();
        assert!(!basis.inner.is_built());
        assert_eq!(basis.inner.n_sites(), 4);
        assert_eq!(basis.inner.lhss(), 3);
    }

    #[test]
    fn boson_basis_new_symm_lhss2_ok() {
        let basis = BosonBasis::new(4, 2, SpaceKind::Symm).unwrap();
        assert!(!basis.inner.is_built());
        assert_eq!(basis.inner.lhss(), 2);
        assert_eq!(basis.inner.n_sites(), 4);
    }

    #[test]
    fn boson_basis_new_symm_lhss3_ok() {
        let basis = BosonBasis::new(4, 3, SpaceKind::Symm).unwrap();
        assert!(!basis.inner.is_built());
        assert_eq!(basis.inner.lhss(), 3);
        assert_eq!(basis.inner.n_sites(), 4);
    }

    #[test]
    fn boson_basis_lhss1_errors() {
        assert!(BosonBasis::new(4, 1, SpaceKind::Sub).is_err());
        assert!(BosonBasis::new(4, 1, SpaceKind::Full).is_err());
        assert!(BosonBasis::new(4, 1, SpaceKind::Symm).is_err());
    }

    #[test]
    fn boson_basis_add_lattice_non_symm_errors() {
        let mut basis = BosonBasis::new(4, 2, SpaceKind::Sub).unwrap();
        let result = basis.add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 3, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn boson_basis_build_boson_lhss2() {
        use quspin_operator::boson::{BosonOp, BosonOpEntry, BosonOperator, BosonOperatorInner};
        use smallvec::smallvec;

        // H = a†_0 a_1 + a_0 a†_1  (hopping), lhss=2, 4 sites
        let n_sites = 4usize;
        let lhss = 2;
        let mut terms = vec![];
        for i in 0..n_sites as u32 - 1 {
            terms.push(BosonOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(BosonOp::Plus, i), (BosonOp::Minus, i + 1)],
            ));
            terms.push(BosonOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(BosonOp::Minus, i), (BosonOp::Plus, i + 1)],
            ));
        }
        let ham = BosonOperatorInner::Ham8(BosonOperator::new(terms, lhss));

        let mut basis = BosonBasis::new(n_sites, lhss, SpaceKind::Sub).unwrap();
        // Seed: 2 bosons, sites 0 and 1 occupied.
        let seed = vec![1u8, 1, 0, 0];
        basis.build(&ham, &[seed]).unwrap();

        // 2-particle (lhss=2 → hard-core boson) sector of 4 sites: C(4,2) = 6
        assert_eq!(basis.inner.size(), 6);
    }

    #[test]
    fn boson_basis_build_bond_lhss3() {
        use ndarray::Array2;
        use quspin_operator::bond::{BondOperator, BondOperatorInner, BondTerm};

        // Hopping matrix for lhss=3 (9x9): swaps |1,0> <-> |0,1> at two sites.
        // Two-site local Hilbert space dimension: 3^2 = 9.
        // Encode: state index = dit_0 + lhss * dit_1.
        // |1,0> = index 1, |0,1> = index 3.
        let lhss = 3usize;
        let n_sites = 4usize;
        let dim = lhss * lhss; // 9

        let mut mat = Array2::from_elem((dim, dim), Complex::new(0.0f64, 0.0));
        // |1,0> <-> |0,1>: mat[3,1] = 1, mat[1,3] = 1 (hopping)
        mat[[3, 1]] = Complex::new(1.0, 0.0);
        mat[[1, 3]] = Complex::new(1.0, 0.0);

        let bonds: Vec<(u32, u32)> = (0..n_sites as u32 - 1).map(|i| (i, i + 1)).collect();
        let term = BondTerm {
            cindex: 0u8,
            matrix: mat,
            bonds,
        };
        let ham = BondOperatorInner::Ham8(BondOperator::new(vec![term]).unwrap());

        let mut basis = BosonBasis::new(n_sites, lhss, SpaceKind::Sub).unwrap();
        // Seed: 1 boson at site 0 (dit=1), all others empty (dit=0).
        // Encodes single-particle sector.
        use quspin_bitbasis::manip::DynamicDitManip;
        let manip = DynamicDitManip::new(lhss);
        let mut seed_state: u32 = 0;
        seed_state = manip.set_dit(seed_state, 1, 0);
        // Convert to byte seed (per-site occupation).
        let seed: Vec<u8> = (0..n_sites)
            .map(|i| manip.get_dit(seed_state, i) as u8)
            .collect();

        basis.build(&ham, &[seed]).unwrap();

        // Single-particle sector on 4 sites with lhss=3: 4 states (boson on each site).
        assert_eq!(basis.inner.size(), 4);
    }
}
