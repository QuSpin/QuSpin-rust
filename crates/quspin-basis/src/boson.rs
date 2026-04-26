//! Bosonic basis type [`BosonBasis`].
use crate::dispatch::GenericBasis;
use crate::spin::SpaceKind;
use num_complex::Complex;
use quspin_bitbasis::StateTransitions;
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// BosonBasis
// ---------------------------------------------------------------------------

/// Bosonic basis (`fermionic = false`). Thin wrapper that adds
/// boson-specific conventions on top of [`GenericBasis`].
///
/// - [`SpaceKind::Full`]  — full Hilbert space; no build step required.
/// - [`SpaceKind::Sub`]   — particle-number / energy subspace built by BFS.
/// - [`SpaceKind::Symm`]  — symmetry-reduced subspace; add lattice symmetry
///   elements with [`add_lattice`](Self::add_lattice) before
///   [`build`](Self::build).
pub struct BosonBasis {
    pub inner: GenericBasis,
}

impl BosonBasis {
    /// Construct a new bosonic basis (always `fermionic = false`).
    ///
    /// # Errors
    /// - `lhss < 2`
    /// - [`SpaceKind::Full`] with more than 64 bits required
    /// - [`SpaceKind::Sub`] / [`SpaceKind::Symm`] with more than 8192 bits
    pub fn new(n_sites: usize, lhss: usize, space_kind: SpaceKind) -> Result<Self, QuSpinError> {
        Ok(Self {
            inner: GenericBasis::new(n_sites, lhss, space_kind, false)?,
        })
    }

    /// The [`SpaceKind`] this basis was constructed with.
    #[inline]
    pub fn space_kind(&self) -> SpaceKind {
        self.inner.space_kind()
    }

    /// Add a lattice (site-permutation) symmetry element.
    ///
    /// Valid only for [`SpaceKind::Symm`] bases before [`build`](Self::build)
    /// is called.
    pub fn add_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        self.inner.add_lattice(grp_char, perm)
    }

    /// Build the subspace reachable from `seeds` under the connectivity
    /// described by `graph`.
    ///
    /// Seeds are per-site occupation byte slices. For `lhss = 2` they
    /// are bit-encoded; for `lhss > 2` they are dit-encoded.
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
    fn boson_basis_new_full_lhss2_ok() {
        let basis = BosonBasis::new(4, 2, SpaceKind::Full).unwrap();
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
}
