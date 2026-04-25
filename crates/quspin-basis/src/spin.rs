//! Spin basis type [`SpinBasis`].
use crate::generic::GenericBasis;
use num_complex::Complex;
use quspin_bitbasis::StateTransitions;
use quspin_types::QuSpinError;

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

/// Spin basis (`fermionic = false`). Thin wrapper that adds
/// spin-specific conventions on top of [`GenericBasis`].
///
/// - [`SpaceKind::Full`]  — full Hilbert space; no build step required.
/// - [`SpaceKind::Sub`]   — particle-number / energy subspace built by BFS.
/// - [`SpaceKind::Symm`]  — symmetry-reduced subspace; add symmetry elements
///   with [`add_lattice`](Self::add_lattice) /
///   [`add_inv`](Self::add_inv) before calling [`build`](Self::build).
pub struct SpinBasis {
    pub inner: GenericBasis,
}

impl SpinBasis {
    /// Construct a new spin basis (always `fermionic = false`).
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

    /// Add a spin-inversion symmetry element.
    ///
    /// - LHSS = 2: XOR bit-flip (`v → 1 − v`) at each site in `locs`.
    /// - LHSS > 2: value inversion (`v → lhss − v − 1`) at each site in `locs`.
    ///
    /// `locs = None` applies the operation to all sites.
    pub fn add_inv(&mut self, locs: Option<Vec<u32>>) -> Result<(), QuSpinError> {
        let n_sites = self.inner.inner.n_sites();
        let lhss = self.inner.inner.lhss();
        let locs_u32 = locs.unwrap_or_else(|| (0..n_sites as u32).collect());
        let locs_usize: Vec<usize> = locs_u32.iter().map(|&v| v as usize).collect();
        let char = Complex::new(-1.0, 0.0);
        if lhss == 2 {
            self.inner.add_inv(char, locs_usize)
        } else {
            let perm_vals: Vec<u8> = (0..lhss).rev().map(|v| v as u8).collect();
            self.inner.add_local(char, perm_vals, locs_usize)
        }
    }

    /// Build the subspace reachable from `seeds` under the connectivity
    /// described by `graph`.
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
    fn spin_basis_new_full_ok() {
        let basis = SpinBasis::new(4, 2, SpaceKind::Full).unwrap();
        assert_eq!(basis.inner.inner.size(), 16);
        assert!(basis.inner.inner.is_built());
    }

    #[test]
    fn spin_basis_new_sub_ok() {
        let basis = SpinBasis::new(4, 2, SpaceKind::Sub).unwrap();
        assert!(!basis.inner.inner.is_built());
    }

    #[test]
    fn spin_basis_new_symm_lhss2_ok() {
        let basis = SpinBasis::new(4, 2, SpaceKind::Symm).unwrap();
        assert!(!basis.inner.inner.is_built());
        assert_eq!(basis.inner.inner.lhss(), 2);
        assert_eq!(basis.inner.inner.n_sites(), 4);
    }

    #[test]
    fn spin_basis_new_symm_lhss3_ok() {
        let basis = SpinBasis::new(4, 3, SpaceKind::Symm).unwrap();
        assert!(!basis.inner.inner.is_built());
        assert_eq!(basis.inner.inner.lhss(), 3);
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
}
