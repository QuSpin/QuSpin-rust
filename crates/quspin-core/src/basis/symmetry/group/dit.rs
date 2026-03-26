/// Dit symmetry group types.
///
/// Covers LHSS > 2 only:
/// - LHSS 3–5: [`DitValueGrpInner<LHSS>`] — compile-time-LHSS value-perm hot-path.
/// - LHSS ≥ 6: [`DitValueGrpDyn`] — runtime-LHSS value-perm fallback.
///
/// The public type is [`DitSymGrp`].
use super::LatticeElement;
use crate::bitbasis::{BitInt, DynamicPermDitValues, PermDitValues};
use crate::error::QuSpinError;
use num_complex::Complex;

// ---------------------------------------------------------------------------
// DitValueGrpInner — compile-time-LHSS value-permutation (LHSS 3–5)
// ---------------------------------------------------------------------------

// Dit value-perm infrastructure is forward-looking (no dit basis yet). Allow
// dead_code until DitSymmetricSubspace is implemented.
#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) struct DitValueGrpInner<const LHSS: usize> {
    n_sites: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<(Complex<f64>, PermDitValues<LHSS>)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl<const LHSS: usize> DitValueGrpInner<LHSS> {
    fn new_empty(n_sites: usize) -> Self {
        DitValueGrpInner {
            n_sites,
            lattice: Vec::new(),
            local: Vec::new(),
        }
    }

    fn push_lattice(&mut self, el: LatticeElement) {
        self.lattice.push(el);
    }

    fn push_dit_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        let arr: [u8; LHSS] = perm.try_into().expect("perm length must match LHSS");
        self.local
            .push((grp_char, PermDitValues::<LHSS>::new(arr, locs)));
    }

    fn n_sites(&self) -> usize {
        self.n_sites
    }

    fn get_refstate<B: BitInt>(&self, state: B) -> (B, Complex<f64>) {
        super::get_refstate(&self.lattice, &self.local, state)
    }

    fn check_refstate<B: BitInt>(&self, state: B) -> (B, f64) {
        super::check_refstate(&self.lattice, &self.local, state)
    }
}

// ---------------------------------------------------------------------------
// DitValueGrpDyn — runtime-LHSS value-permutation fallback (LHSS ≥ 6)
// ---------------------------------------------------------------------------

#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) struct DitValueGrpDyn {
    n_sites: usize,
    lhss: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<(Complex<f64>, DynamicPermDitValues)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl DitValueGrpDyn {
    fn new_empty(lhss: usize, n_sites: usize) -> Self {
        DitValueGrpDyn {
            n_sites,
            lhss,
            lattice: Vec::new(),
            local: Vec::new(),
        }
    }

    fn push_lattice(&mut self, el: LatticeElement) {
        self.lattice.push(el);
    }

    fn push_dit_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        self.local
            .push((grp_char, DynamicPermDitValues::new(self.lhss, perm, locs)));
    }

    fn n_sites(&self) -> usize {
        self.n_sites
    }

    fn lhss(&self) -> usize {
        self.lhss
    }

    fn get_refstate<B: BitInt>(&self, state: B) -> (B, Complex<f64>) {
        super::get_refstate(&self.lattice, &self.local, state)
    }

    fn check_refstate<B: BitInt>(&self, state: B) -> (B, f64) {
        super::check_refstate(&self.lattice, &self.local, state)
    }
}

// ---------------------------------------------------------------------------
// DitSymGrpInner — LHSS dispatch enum for value-permutation (LHSS > 2)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum DitSymGrpInner {
    Lhss3(DitValueGrpInner<3>),
    Lhss4(DitValueGrpInner<4>),
    Lhss5(DitValueGrpInner<5>),
    LhssDyn(DitValueGrpDyn),
}

impl DitSymGrpInner {
    fn new_empty(lhss: usize, n_sites: usize) -> Self {
        match lhss {
            3 => DitSymGrpInner::Lhss3(DitValueGrpInner::<3>::new_empty(n_sites)),
            4 => DitSymGrpInner::Lhss4(DitValueGrpInner::<4>::new_empty(n_sites)),
            5 => DitSymGrpInner::Lhss5(DitValueGrpInner::<5>::new_empty(n_sites)),
            _ => DitSymGrpInner::LhssDyn(DitValueGrpDyn::new_empty(lhss, n_sites)),
        }
    }

    pub(crate) fn push_lattice(&mut self, el: LatticeElement) {
        match self {
            DitSymGrpInner::Lhss3(g) => g.push_lattice(el),
            DitSymGrpInner::Lhss4(g) => g.push_lattice(el),
            DitSymGrpInner::Lhss5(g) => g.push_lattice(el),
            DitSymGrpInner::LhssDyn(g) => g.push_lattice(el),
        }
    }

    pub(crate) fn push_dit_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) {
        match self {
            DitSymGrpInner::Lhss3(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInner::Lhss4(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInner::Lhss5(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInner::LhssDyn(g) => g.push_dit_perm(grp_char, perm, locs),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn n_sites(&self) -> usize {
        match self {
            DitSymGrpInner::Lhss3(g) => g.n_sites(),
            DitSymGrpInner::Lhss4(g) => g.n_sites(),
            DitSymGrpInner::Lhss5(g) => g.n_sites(),
            DitSymGrpInner::LhssDyn(g) => g.n_sites(),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn lhss(&self) -> usize {
        match self {
            DitSymGrpInner::Lhss3(_) => 3,
            DitSymGrpInner::Lhss4(_) => 4,
            DitSymGrpInner::Lhss5(_) => 5,
            DitSymGrpInner::LhssDyn(g) => g.lhss(),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn get_refstate<B: BitInt>(&self, state: B) -> (B, Complex<f64>) {
        match self {
            DitSymGrpInner::Lhss3(g) => g.get_refstate(state),
            DitSymGrpInner::Lhss4(g) => g.get_refstate(state),
            DitSymGrpInner::Lhss5(g) => g.get_refstate(state),
            DitSymGrpInner::LhssDyn(g) => g.get_refstate(state),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn check_refstate<B: BitInt>(&self, state: B) -> (B, f64) {
        match self {
            DitSymGrpInner::Lhss3(g) => g.check_refstate(state),
            DitSymGrpInner::Lhss4(g) => g.check_refstate(state),
            DitSymGrpInner::Lhss5(g) => g.check_refstate(state),
            DitSymGrpInner::LhssDyn(g) => g.check_refstate(state),
        }
    }
}

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
    inner: DitSymGrpInner,
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
        Ok(DitSymGrp {
            lhss,
            n_sites,
            inner: DitSymGrpInner::new_empty(lhss, n_sites),
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
        use crate::bitbasis::PermDitLocations;
        let el = LatticeElement::new(
            grp_char,
            PermDitLocations::new(self.lhss, &perm),
            self.n_sites,
        );
        self.inner.push_lattice(el);
    }

    /// Add an on-site value-permutation symmetry element.
    ///
    /// `perm[v] = w` maps local occupation `v` to `w` at each site in `locs`.
    /// The length of `perm` must equal `self.lhss`.
    pub fn add_local_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        self.inner.push_dit_perm(grp_char, perm, locs);
    }

    /// Access the inner dispatch type.
    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn as_dit(&self) -> &DitSymGrpInner {
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
        // LHSS=6 falls back to DitValueGrpDyn path.
        let mut grp = DitSymGrp::new(6, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_local_perm(Complex::new(1.0, 0.0), vec![5, 4, 3, 2, 1, 0], vec![0, 1]);
        assert_eq!(grp.lhss(), 6);
    }
}
