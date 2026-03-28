/// Fermionic symmetry group type.
///
/// Extends the hardcore (LHSS=2) basis with Jordan-Wigner permutation signs.
/// Each lattice element includes the fermionic sign of the permutation acting
/// on the pre-image state.
use super::sym_grp::{HardcoreGrpInner, SymmetryGrpInner};
use crate::error::QuSpinError;
use num_complex::Complex;

// ---------------------------------------------------------------------------
// FermionicSymGrp
// ---------------------------------------------------------------------------

/// A lattice symmetry group for fermionic systems.
///
/// Extends the hardcore (LHSS=2) basis with Jordan-Wigner permutation signs.
/// Each lattice element includes the fermionic sign of the permutation
/// acting on the state.
///
/// Use [`SpinSymGrp`](super::SpinSymGrp) for bosonic systems.
#[derive(Clone)]
pub struct FermionicSymGrp {
    n_sites: usize,
    inner: SymmetryGrpInner,
}

impl FermionicSymGrp {
    /// Construct an empty fermionic symmetry group for `n_sites` sites.
    ///
    /// Returns `Err` if `n_sites > 8192`.
    pub fn new(n_sites: usize) -> Result<Self, QuSpinError> {
        let inner = crate::select_b_for_n_sites!(
            n_sites,
            B,
            return Err(QuSpinError::ValueError(format!(
                "n_sites={n_sites} exceeds the maximum supported value of 8192"
            ))),
            { SymmetryGrpInner::from(HardcoreGrpInner::<B>::new_empty(2, n_sites, true)) }
        );
        Ok(FermionicSymGrp { n_sites, inner })
    }

    /// The number of lattice sites.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Add a lattice (site-permutation) symmetry element with fermionic sign
    /// tracking.
    ///
    /// `perm[src] = dst` maps source site `src` to destination `dst`.
    /// The Jordan-Wigner sign is automatically included in the group character
    /// when computing orbit representatives.
    pub fn add_lattice(&mut self, grp_char: Complex<f64>, perm: Vec<usize>) {
        self.inner.push_lattice(grp_char, &perm);
    }

    /// Access the inner hardcore dispatch type.
    ///
    /// Used by `quspin-py` to construct `SymmetricSubspace<HardcoreGrpInner<B>>`.
    pub fn as_hardcore(&self) -> Option<&SymmetryGrpInner> {
        Some(&self.inner)
    }

    /// Access the inner dispatch type directly.
    pub fn inner(&self) -> &SymmetryGrpInner {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fermionic_sym_grp_new_4sites() {
        let mut grp = FermionicSymGrp::new(4).unwrap();
        // Add a translation symmetry with character 1.
        grp.add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 3, 0]);
        assert_eq!(grp.n_sites(), 4);
        assert!(grp.as_hardcore().is_some());
    }

    #[test]
    fn fermionic_sym_grp_n_sites_too_large_errors() {
        assert!(FermionicSymGrp::new(8193).is_err());
    }

    #[test]
    fn fermionic_sym_grp_get_refstate() {
        use crate::bitbasis::BitInt;
        use crate::with_sym_grp;
        let mut grp = FermionicSymGrp::new(4).unwrap();
        // Identity permutation, character 1.
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1, 2, 3]);

        let hc = grp.as_hardcore().unwrap();
        with_sym_grp!(hc, B, g, {
            let state = B::from_u64(0b0001);
            let (ref_s, _coeff) = g.get_refstate(state);
            // With only identity, representative = state itself.
            assert_eq!(ref_s, state);
        });
    }
}
