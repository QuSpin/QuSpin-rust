/// Spin-symmetry group types.
///
/// The public type is [`SpinSymGrp`]. B-type dispatch lives in [`super::dispatch`].
use super::dispatch::{SymGrpInner, SymmetryGrpInner};
use crate::error::QuSpinError;
use num_complex::Complex;

// ---------------------------------------------------------------------------
// SpinSymGrp — public type
// ---------------------------------------------------------------------------

/// A symmetry group combining lattice permutations with spin-inversion operations.
///
/// - For LHSS = 2: local operations are value-permutations `v → lhss − v − 1`
///   (same as XOR bit-flip for 2-valued spins).
/// - For LHSS > 2: local operations map `v → lhss − v − 1` (spin inversion).
///
/// Use [`DitSymGrp`](super::DitSymGrp) for arbitrary local value-permutation
/// symmetries (LHSS > 2). Mixing both op types in the same group is not
/// supported because the orbit computation would be incomplete.
#[derive(Clone)]
pub struct SpinSymGrp {
    lhss: usize,
    n_sites: usize,
    inner: SymmetryGrpInner,
}

impl SpinSymGrp {
    /// Construct an empty spin-symmetry group.
    ///
    /// Returns `Err` if the required bit width exceeds 8192 bits.
    pub fn new(lhss: usize, n_sites: usize) -> Result<Self, QuSpinError> {
        let bits_per_dit = if lhss <= 2 {
            1
        } else {
            (usize::BITS - (lhss - 1).leading_zeros()) as usize
        };
        let n_bits = n_sites * bits_per_dit;
        let inner = crate::select_b_for_n_sites!(
            n_bits,
            B,
            return Err(QuSpinError::ValueError(format!(
                "n_sites={n_sites} with lhss={lhss} requires {n_bits} bits, exceeding the 8192-bit maximum"
            ))),
            { SymmetryGrpInner::from(SymGrpInner::<B>::new_empty(lhss, n_sites, false)) }
        );
        Ok(SpinSymGrp {
            lhss,
            n_sites,
            inner,
        })
    }

    /// The local Hilbert-space size for this group.
    pub fn lhss(&self) -> usize {
        self.lhss
    }

    /// The number of lattice sites.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Add a lattice (site-permutation) symmetry element.
    ///
    /// `perm[src] = dst` maps source site `src` to destination `dst`.
    pub fn add_lattice(&mut self, grp_char: Complex<f64>, perm: Vec<usize>) {
        self.inner.push_lattice(grp_char, &perm);
    }

    /// Add a spin-inversion / bit-flip symmetry element.
    ///
    /// For LHSS = 2: maps `v → 1 − v` (bit-flip) at the specified sites.
    /// For LHSS > 2: maps `v → lhss − v − 1` at the specified sites.
    pub fn add_inverse(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        self.inner.push_spin_inv(grp_char, locs);
    }

    /// Access the inner dispatch type.
    ///
    /// Used by `quspin-py` to construct `SymmetricSubspace<SymGrpInner<B>>` via `with_sym_grp!`.
    /// Returns `None` for LHSS > 2 groups where the inner type does not support
    /// a hardcore (LHSS=2) subspace — currently returns `Some` for all groups
    /// since the unified `SymGrpInner` is used for all LHSS values.
    pub fn as_hardcore(&self) -> Option<&SymmetryGrpInner> {
        // Only LHSS=2 is currently wired to the hardcore subspace builder in quspin-py.
        if self.lhss == 2 {
            Some(&self.inner)
        } else {
            None
        }
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
    fn spin_sym_bitflip_get_refstate() {
        let mut grp = SpinSymGrp::new(2, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_inverse(Complex::new(1.0, 0.0), vec![0, 1]);

        let grp_inner = grp.as_hardcore().unwrap();
        match grp_inner {
            SymmetryGrpInner::Sym32(g) => {
                let (ref_s, _) = g.get_refstate(0b01u32);
                assert_eq!(ref_s, 0b10u32);
            }
            _ => panic!("expected Sym32"),
        }
    }

    #[test]
    fn spin_sym_translation() {
        let mut grp = SpinSymGrp::new(2, 3).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 0]);

        let grp_inner = grp.as_hardcore().unwrap();
        match grp_inner {
            SymmetryGrpInner::Sym32(g) => {
                let (ref_s, _) = g.get_refstate(0b001u32);
                assert_eq!(ref_s, 0b010u32);
            }
            _ => panic!("expected Sym32"),
        }
    }

    #[test]
    fn spin_sym_n_sites_too_large_errors() {
        assert!(SpinSymGrp::new(2, 8193).is_err());
    }

    #[test]
    fn spin_sym_higher_spin_inversion() {
        use crate::bitbasis::DynamicDitManip;
        let mut grp = SpinSymGrp::new(3, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_inverse(Complex::new(1.0, 0.0), vec![0, 1]);

        assert_eq!(grp.n_sites(), 2);
        assert_eq!(grp.lhss(), 3);

        // n_sites=2, lhss=3 => bits_per_dit=2, n_bits=4 => Sym32
        let manip = DynamicDitManip::new(3);
        let state: u32 = manip.set_dit(manip.set_dit(0u32, 1, 0), 0, 1);
        match grp.inner() {
            SymmetryGrpInner::Sym32(inner) => {
                let (ref_s, _) = inner.get_refstate(state);
                assert!(ref_s >= state);
            }
            _ => panic!("expected Sym32 variant for n_bits=4"),
        }
    }

    #[test]
    fn spin_sym_lhss_dyn() {
        let mut grp = SpinSymGrp::new(6, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_inverse(Complex::new(1.0, 0.0), vec![0, 1]);
        assert_eq!(grp.lhss(), 6);
    }
}
