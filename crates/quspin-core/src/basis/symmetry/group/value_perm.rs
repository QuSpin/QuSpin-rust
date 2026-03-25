/// Value-permutation symmetry group types.
///
/// Covers LHSS > 2 only:
/// - LHSS 3–5: [`DitValueGrpInner<LHSS>`] — compile-time-LHSS value-perm hot-path.
/// - LHSS ≥ 6: [`DitValueGrpDyn`] — runtime-LHSS value-perm fallback.
///
/// The public type is [`ValuePermSymGrp`].
use super::LatticeElement;
use crate::bitbasis::{BitInt, BitStateOp, DynamicPermDitValues, PermDitValues};
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

    fn push_value_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        let arr: [u8; LHSS] = perm.try_into().expect("perm length must match LHSS");
        self.local
            .push((grp_char, PermDitValues::<LHSS>::new(arr, locs)));
    }

    fn n_sites(&self) -> usize {
        self.n_sites
    }

    fn iter_images<B: BitInt>(&self, state: B) -> Vec<(B, Complex<f64>)> {
        let one = Complex::new(1.0, 0.0);
        let mut images = Vec::with_capacity(self.lattice.len() * (1 + self.local.len()));
        for lat in &self.lattice {
            images.push(lat.apply(state, one));
        }
        for (char_, op) in &self.local {
            let loc_state = op.apply(state);
            for lat in &self.lattice {
                images.push(lat.apply(loc_state, *char_));
            }
        }
        images
    }

    fn get_refstate<B: BitInt>(&self, state: B) -> (B, Complex<f64>) {
        let mut best = state;
        let mut best_coeff = Complex::new(1.0, 0.0);
        for (s, c) in self.iter_images(state) {
            if s > best {
                best = s;
                best_coeff = c;
            }
        }
        (best, best_coeff)
    }

    fn check_refstate<B: BitInt>(&self, state: B) -> (B, f64) {
        let mut ref_state = state;
        let mut norm = 0.0_f64;
        for (s, _) in self.iter_images(state) {
            if s > ref_state {
                ref_state = s;
            }
            if s == state {
                norm += 1.0;
            }
        }
        (ref_state, norm)
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

    fn push_value_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        self.local
            .push((grp_char, DynamicPermDitValues::new(self.lhss, perm, locs)));
    }

    fn n_sites(&self) -> usize {
        self.n_sites
    }

    fn lhss(&self) -> usize {
        self.lhss
    }

    fn iter_images<B: BitInt>(&self, state: B) -> Vec<(B, Complex<f64>)> {
        let one = Complex::new(1.0, 0.0);
        let mut images = Vec::with_capacity(self.lattice.len() * (1 + self.local.len()));
        for lat in &self.lattice {
            images.push(lat.apply(state, one));
        }
        for (char_, op) in &self.local {
            let loc_state = op.apply(state);
            for lat in &self.lattice {
                images.push(lat.apply(loc_state, *char_));
            }
        }
        images
    }

    fn get_refstate<B: BitInt>(&self, state: B) -> (B, Complex<f64>) {
        let mut best = state;
        let mut best_coeff = Complex::new(1.0, 0.0);
        for (s, c) in self.iter_images(state) {
            if s > best {
                best = s;
                best_coeff = c;
            }
        }
        (best, best_coeff)
    }

    fn check_refstate<B: BitInt>(&self, state: B) -> (B, f64) {
        let mut ref_state = state;
        let mut norm = 0.0_f64;
        for (s, _) in self.iter_images(state) {
            if s > ref_state {
                ref_state = s;
            }
            if s == state {
                norm += 1.0;
            }
        }
        (ref_state, norm)
    }
}

// ---------------------------------------------------------------------------
// DitValuePermSymGrpInner — LHSS dispatch enum for value-permutation (LHSS > 2)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum DitValuePermSymGrpInner {
    Lhss3(DitValueGrpInner<3>),
    Lhss4(DitValueGrpInner<4>),
    Lhss5(DitValueGrpInner<5>),
    LhssDyn(DitValueGrpDyn),
}

impl DitValuePermSymGrpInner {
    fn new_empty(lhss: usize, n_sites: usize) -> Self {
        match lhss {
            3 => DitValuePermSymGrpInner::Lhss3(DitValueGrpInner::<3>::new_empty(n_sites)),
            4 => DitValuePermSymGrpInner::Lhss4(DitValueGrpInner::<4>::new_empty(n_sites)),
            5 => DitValuePermSymGrpInner::Lhss5(DitValueGrpInner::<5>::new_empty(n_sites)),
            _ => DitValuePermSymGrpInner::LhssDyn(DitValueGrpDyn::new_empty(lhss, n_sites)),
        }
    }

    pub(crate) fn push_lattice(&mut self, el: LatticeElement) {
        match self {
            DitValuePermSymGrpInner::Lhss3(g) => g.push_lattice(el),
            DitValuePermSymGrpInner::Lhss4(g) => g.push_lattice(el),
            DitValuePermSymGrpInner::Lhss5(g) => g.push_lattice(el),
            DitValuePermSymGrpInner::LhssDyn(g) => g.push_lattice(el),
        }
    }

    pub(crate) fn push_value_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) {
        match self {
            DitValuePermSymGrpInner::Lhss3(g) => g.push_value_perm(grp_char, perm, locs),
            DitValuePermSymGrpInner::Lhss4(g) => g.push_value_perm(grp_char, perm, locs),
            DitValuePermSymGrpInner::Lhss5(g) => g.push_value_perm(grp_char, perm, locs),
            DitValuePermSymGrpInner::LhssDyn(g) => g.push_value_perm(grp_char, perm, locs),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn n_sites(&self) -> usize {
        match self {
            DitValuePermSymGrpInner::Lhss3(g) => g.n_sites(),
            DitValuePermSymGrpInner::Lhss4(g) => g.n_sites(),
            DitValuePermSymGrpInner::Lhss5(g) => g.n_sites(),
            DitValuePermSymGrpInner::LhssDyn(g) => g.n_sites(),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn lhss(&self) -> usize {
        match self {
            DitValuePermSymGrpInner::Lhss3(_) => 3,
            DitValuePermSymGrpInner::Lhss4(_) => 4,
            DitValuePermSymGrpInner::Lhss5(_) => 5,
            DitValuePermSymGrpInner::LhssDyn(g) => g.lhss(),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn get_refstate<B: BitInt>(&self, state: B) -> (B, Complex<f64>) {
        match self {
            DitValuePermSymGrpInner::Lhss3(g) => g.get_refstate(state),
            DitValuePermSymGrpInner::Lhss4(g) => g.get_refstate(state),
            DitValuePermSymGrpInner::Lhss5(g) => g.get_refstate(state),
            DitValuePermSymGrpInner::LhssDyn(g) => g.get_refstate(state),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn check_refstate<B: BitInt>(&self, state: B) -> (B, f64) {
        match self {
            DitValuePermSymGrpInner::Lhss3(g) => g.check_refstate(state),
            DitValuePermSymGrpInner::Lhss4(g) => g.check_refstate(state),
            DitValuePermSymGrpInner::Lhss5(g) => g.check_refstate(state),
            DitValuePermSymGrpInner::LhssDyn(g) => g.check_refstate(state),
        }
    }
}

// ---------------------------------------------------------------------------
// ValuePermSymGrp — public type
// ---------------------------------------------------------------------------

/// A symmetry group combining lattice permutations with local value-permutation ops.
///
/// Only supported for LHSS > 2. Use [`SpinSymGrp`](super::SpinSymGrp) for LHSS = 2
/// or for spin-inversion symmetries (`v → lhss − v − 1`).
///
/// Mixing value-permutation and spin-inversion ops in the same group is not
/// supported because the orbit computation would be incomplete.
#[derive(Clone)]
pub struct ValuePermSymGrp {
    lhss: usize,
    n_sites: usize,
    inner: DitValuePermSymGrpInner,
}

impl ValuePermSymGrp {
    /// Construct an empty value-permutation symmetry group.
    ///
    /// Returns `Err` if `lhss < 3` (use [`SpinSymGrp`](super::SpinSymGrp) for `lhss = 2`).
    pub fn new(lhss: usize, n_sites: usize) -> Result<Self, QuSpinError> {
        if lhss < 3 {
            return Err(QuSpinError::ValueError(format!(
                "ValuePermSymGrp requires lhss >= 3; use SpinSymGrp for lhss={lhss}"
            )));
        }
        Ok(ValuePermSymGrp {
            lhss,
            n_sites,
            inner: DitValuePermSymGrpInner::new_empty(lhss, n_sites),
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
        self.inner.push_value_perm(grp_char, perm, locs);
    }

    /// Access the dit inner dispatch type.
    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn as_dit(&self) -> &DitValuePermSymGrpInner {
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
    fn value_perm_sym_basic() {
        let mut grp = ValuePermSymGrp::new(3, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_local_perm(Complex::new(1.0, 0.0), vec![2, 1, 0], vec![0, 1]);
        assert_eq!(grp.lhss(), 3);
        assert_eq!(grp.n_sites(), 2);
    }

    #[test]
    fn value_perm_sym_rejects_lhss2() {
        assert!(ValuePermSymGrp::new(2, 4).is_err());
    }

    #[test]
    fn value_perm_sym_lhss_dyn() {
        // LHSS=6 falls back to DitValueGrpDyn path.
        let mut grp = ValuePermSymGrp::new(6, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_local_perm(Complex::new(1.0, 0.0), vec![5, 4, 3, 2, 1, 0], vec![0, 1]);
        assert_eq!(grp.lhss(), 6);
    }
}
