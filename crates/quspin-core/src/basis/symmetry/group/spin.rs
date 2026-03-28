/// Spin-symmetry group types.
///
/// The public type is [`SpinSymGrp`]. B-type dispatch lives in [`super::dispatch`].
use super::dispatch::SymmetryGrpInner;
use super::dispatch::{DitSymGrpInner, DitSymGrpInnerEnum};
use crate::error::QuSpinError;
use num_complex::Complex;

// Re-export inner types so that `pub use spin::{HardcoreGrpElement, HardcoreSymmetryGrp, SpinSymGrp}`
// in mod.rs continues to work unchanged.
pub use super::dispatch::HardcoreGrpElement;
pub use super::dispatch::HardcoreSymmetryGrp;

// ---------------------------------------------------------------------------
// SpinSymGrp — public type
// ---------------------------------------------------------------------------

/// A symmetry group combining lattice permutations with spin-inversion operations.
///
/// - For LHSS = 2: local operations are XOR bit-flips (Z₂ symmetry).
/// - For LHSS > 2: local operations map `v → lhss − v − 1` (spin inversion).
///
/// Use [`DitSymGrp`](super::DitSymGrp) for local value-permutation
/// symmetries (LHSS > 2). Mixing both op types in the same group is not
/// supported because the orbit computation would be incomplete.
#[derive(Clone)]
pub struct SpinSymGrp {
    lhss: usize,
    n_sites: usize,
    inner: SpinSymGrpInner,
}

#[derive(Clone)]
enum SpinSymGrpInner {
    /// LHSS = 2: concrete `B` resolved from `n_sites` at construction.
    Hardcore(SymmetryGrpInner),
    /// LHSS > 2: spin-inversion ops; `B` resolved at construction.
    Dit(DitSymGrpInnerEnum),
}

impl SpinSymGrp {
    /// Construct an empty spin-symmetry group.
    ///
    /// Returns `Err` if `lhss == 2` and `n_sites > 8192`.
    pub fn new(lhss: usize, n_sites: usize) -> Result<Self, QuSpinError> {
        let inner = if lhss == 2 {
            let hc = crate::select_b_for_n_sites!(
                n_sites,
                B,
                return Err(QuSpinError::ValueError(format!(
                    "n_sites={n_sites} exceeds the maximum supported value of 8192"
                ))),
                { SymmetryGrpInner::from(HardcoreSymmetryGrp::<B>::new_empty(n_sites)) }
            );
            SpinSymGrpInner::Hardcore(hc)
        } else {
            let bits_per_dit = if lhss <= 1 {
                1
            } else {
                (usize::BITS - (lhss - 1).leading_zeros()) as usize
            };
            let n_bits = n_sites * bits_per_dit;
            let dit = crate::select_b_for_n_sites!(
                n_bits,
                B,
                return Err(QuSpinError::ValueError(format!(
                    "n_sites={n_sites} with lhss={lhss} requires {n_bits} bits, exceeding the 8192-bit maximum"
                ))),
                { DitSymGrpInnerEnum::from(DitSymGrpInner::<B>::new_empty(lhss, n_sites)) }
            );
            SpinSymGrpInner::Dit(dit)
        };
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
        match &mut self.inner {
            SpinSymGrpInner::Hardcore(hc) => hc.push_lattice(grp_char, &perm, false),
            SpinSymGrpInner::Dit(dit) => dit.push_lattice(grp_char, &perm),
        }
    }

    /// Add a spin-inversion / bit-flip symmetry element.
    ///
    /// For LHSS = 2: XOR-flips the bits at the specified site indices.
    /// For LHSS > 2: maps `v → lhss − v − 1` at the specified sites.
    pub fn add_inverse(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        match &mut self.inner {
            SpinSymGrpInner::Hardcore(hc) => hc.push_inverse(grp_char, &locs),
            SpinSymGrpInner::Dit(dit) => dit.push_spin_inv(grp_char, locs),
        }
    }

    /// Access the hardcore (LHSS=2) inner dispatch type.
    ///
    /// Used by `quspin-py` to construct `SymmetricSubspace<HardcoreSymmetryGrp<B>>` via `with_sym_grp!`.
    /// Returns `None` for LHSS > 2 groups.
    pub fn as_hardcore(&self) -> Option<&SymmetryGrpInner> {
        match &self.inner {
            SpinSymGrpInner::Hardcore(hc) => Some(hc),
            SpinSymGrpInner::Dit(_) => None,
        }
    }

    /// Access the dit (LHSS>2) inner dispatch type.
    ///
    /// Returns `None` for LHSS=2 groups.
    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn as_dit(&self) -> Option<&DitSymGrpInnerEnum> {
        match &self.inner {
            SpinSymGrpInner::Dit(dit) => Some(dit),
            SpinSymGrpInner::Hardcore(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::dispatch::DitSymGrpInnerEnum;
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

        let dit = grp.as_dit().unwrap();
        // n_sites=2, lhss=3 => bits_per_dit=2, n_bits=4 => B32
        let manip = DynamicDitManip::new(3);
        let state: u32 = manip.set_dit(manip.set_dit(0u32, 1, 0), 0, 1);
        match dit {
            DitSymGrpInnerEnum::B32(inner) => {
                let (ref_s, _) = inner.get_refstate(state);
                assert!(ref_s >= state);
            }
            _ => panic!("expected B32 variant for n_bits=4"),
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
