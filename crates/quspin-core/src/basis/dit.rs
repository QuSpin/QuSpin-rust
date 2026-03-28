/// Dit symmetry group types.
///
/// The public type is [`DitSymGrp`]. Inner types live in [`super::sym_grp`].
use super::sym_grp::{DitGrpInner, SymmetryGrpInner};
use crate::error::QuSpinError;
use num_complex::Complex;

// ---------------------------------------------------------------------------
// DitSymGrp — public type
// ---------------------------------------------------------------------------

/// A symmetry group combining lattice permutations with local value-permutation ops.
///
/// Only supported for LHSS > 2. Use [`SpinSymGrp`](super::SpinSymGrp) for LHSS = 2
/// or for spin-inversion symmetries (`v → lhss − v − 1`).
///
/// Mixing value-permutation and spin-inversion ops in the same group is not
/// supported because the orbit computation would be incomplete.
#[derive(Clone)]
pub struct DitSymGrp {
    lhss: usize,
    n_sites: usize,
    inner: SymmetryGrpInner,
}

impl DitSymGrp {
    /// Construct an empty dit symmetry group.
    ///
    /// Returns `Err` if `lhss < 3` (use [`SpinSymGrp`](super::SpinSymGrp) for `lhss = 2`).
    pub fn new(lhss: usize, n_sites: usize) -> Result<Self, QuSpinError> {
        if lhss < 3 {
            return Err(QuSpinError::ValueError(format!(
                "DitSymGrp requires lhss >= 3; use SpinSymGrp for lhss={lhss}"
            )));
        }
        let bits_per_dit = if lhss <= 1 {
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
            { SymmetryGrpInner::from(DitGrpInner::<B>::new_empty(lhss, n_sites, false)) }
        );
        Ok(DitSymGrp {
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

    /// Add an on-site value-permutation symmetry element.
    ///
    /// `perm[v] = w` maps local occupation `v` to `w` at each site in `locs`.
    /// The length of `perm` must equal `self.lhss`.
    pub fn add_local_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        self.inner.push_local_perm(grp_char, perm, locs);
    }

    /// Access the inner dispatch type.
    pub fn as_dit(&self) -> &SymmetryGrpInner {
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
    fn dit_sym_basic() {
        let mut grp = DitSymGrp::new(3, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_local_perm(Complex::new(1.0, 0.0), vec![2, 1, 0], vec![0, 1]);
        assert_eq!(grp.lhss(), 3);
        assert_eq!(grp.n_sites(), 2);
    }

    #[test]
    fn dit_sym_rejects_lhss2() {
        assert!(DitSymGrp::new(2, 4).is_err());
    }

    #[test]
    fn dit_sym_lhss_dyn() {
        // LHSS=6: bits_per_dit=3, n_bits=6 → Sym32
        let mut grp = DitSymGrp::new(6, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_local_perm(Complex::new(1.0, 0.0), vec![5, 4, 3, 2, 1, 0], vec![0, 1]);
        assert_eq!(grp.lhss(), 6);
    }
}
