//! Fermionic basis type [`FermionBasis`].
use crate::generic::GenericBasis;
use crate::spin::SpaceKind;
use num_complex::Complex;
use quspin_bitbasis::StateTransitions;
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// FermionBasis
// ---------------------------------------------------------------------------

/// Fermionic basis. Thin wrapper that pins `fermionic = true, lhss = 2`
/// on top of [`GenericBasis`]; Jordan-Wigner sign tracking is enabled
/// automatically.
///
/// - [`SpaceKind::Full`]  — full Hilbert space; no build step required.
/// - [`SpaceKind::Sub`]   — particle-number sector subspace built by BFS.
/// - [`SpaceKind::Symm`]  — symmetry-reduced subspace; add lattice symmetry
///   elements with [`add_lattice`](Self::add_lattice) before
///   [`build`](Self::build).
pub struct FermionBasis {
    pub inner: GenericBasis,
}

impl FermionBasis {
    /// Construct a new fermionic basis. `lhss` is fixed at 2 (one bit
    /// per orbital) and `fermionic` at true.
    ///
    /// # Errors
    /// - [`SpaceKind::Full`] with `n_sites > 64`
    /// - [`SpaceKind::Sub`] / [`SpaceKind::Symm`] with `n_sites > 8192`
    pub fn new(n_sites: usize, space_kind: SpaceKind) -> Result<Self, QuSpinError> {
        Ok(Self {
            inner: GenericBasis::new(n_sites, 2, space_kind, true)?,
        })
    }

    /// The [`SpaceKind`] this basis was constructed with.
    #[inline]
    pub fn space_kind(&self) -> SpaceKind {
        self.inner.space_kind()
    }

    /// Add a lattice (site-permutation) symmetry element.
    /// Jordan-Wigner sign tracking is applied automatically by the
    /// underlying [`GenericBasis`].
    pub fn add_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        self.inner.add_lattice(grp_char, perm)
    }

    /// Build the subspace reachable from `seeds` under `graph`.
    /// Requires `graph.lhss() == 2`.
    ///
    /// # Errors
    /// - Called on a [`SpaceKind::Full`] basis
    /// - Basis is already built
    /// - `graph.lhss() != 2`
    pub fn build<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        self.inner.build(graph, seeds)
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
        assert!(FermionBasis::new(65, SpaceKind::Full).is_err());
    }

    #[test]
    fn fermion_basis_add_lattice_on_non_symm_errors() {
        let mut basis = FermionBasis::new(4, SpaceKind::Sub).unwrap();
        let result = basis.add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 3, 0]);
        assert!(result.is_err());
    }
}
