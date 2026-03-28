use super::LatticeElement;
/// Spin-symmetry group types.
///
/// Covers the full LHSS range:
/// - LHSS = 2: [`HardcoreSymmetryGrp<B>`] — XOR bit-flip ops, `B` selected at construction.
/// - LHSS 3–5: [`DitSpinGrpInner<LHSS>`] — compile-time-LHSS spin-inversion hot-path.
/// - LHSS ≥ 6: [`DitSpinGrpDyn`] — runtime-LHSS spin-inversion fallback.
///
/// The public type is [`SpinSymGrp`]. B-type dispatch for LHSS = 2 lives in
/// [`super::dispatch`].
use super::dispatch::SymmetryGrpInner;
use crate::basis::traits::SymGrp;
use crate::bitbasis::{BitInt, BitStateOp, DynamicHigherSpinInv, HigherSpinInv, PermDitMask};
use crate::error::QuSpinError;
use num_complex::Complex;

// ---------------------------------------------------------------------------
// HardcoreGrpElement / HardcoreSymmetryGrp  (LHSS = 2)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct HardcoreGrpElement<B: BitInt> {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub op: PermDitMask<B>,
}

impl<B: BitInt> HardcoreGrpElement<B> {
    pub fn new(grp_char: Complex<f64>, op: PermDitMask<B>, n_sites: usize) -> Self {
        HardcoreGrpElement {
            grp_char,
            n_sites,
            op,
        }
    }

    #[inline]
    pub fn apply(&self, state: B, coeff: Complex<f64>) -> (B, Complex<f64>) {
        (self.op.apply(state), coeff * self.grp_char)
    }
}

impl<B: BitInt> super::LocalOpItem<B> for HardcoreGrpElement<B> {
    #[inline]
    fn apply_local(&self, state: B) -> (B, Complex<f64>) {
        (self.op.apply(state), self.grp_char)
    }
}

#[derive(Clone)]
pub struct HardcoreSymmetryGrp<B: BitInt> {
    n_sites: usize,
    lattice: Vec<LatticeElement>,
    pub local: Vec<HardcoreGrpElement<B>>,
}

impl<B: BitInt> HardcoreSymmetryGrp<B> {
    pub fn new_empty(n_sites: usize) -> Self {
        HardcoreSymmetryGrp {
            n_sites,
            lattice: Vec::new(),
            local: Vec::new(),
        }
    }

    pub fn push_lattice(&mut self, el: LatticeElement) {
        self.lattice.push(el);
    }

    pub fn push_inverse(&mut self, grp_char: Complex<f64>, locs: &[usize]) {
        let mask = locs.iter().fold(B::from_u64(0), |acc, &site| {
            if site < B::BITS as usize {
                acc | (B::from_u64(1) << site)
            } else {
                acc
            }
        });
        self.local.push(HardcoreGrpElement::new(
            grp_char,
            PermDitMask::new(mask),
            self.n_sites,
        ));
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    pub fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        super::get_refstate(&self.lattice, &self.local, state)
    }

    pub fn check_refstate(&self, state: B) -> (B, f64) {
        super::check_refstate(&self.lattice, &self.local, state)
    }

    /// Batch variant: computes `check_refstate` for every element of `states`
    /// in a single pass with loop order optimised for auto-vectorisation.
    pub fn check_refstate_batch(&self, states: &[B], out: &mut [(B, f64)]) {
        super::orbit::check_refstate_batch(&self.lattice, &self.local, states, out);
    }
}

// ---------------------------------------------------------------------------
// SymGrp impl for HardcoreSymmetryGrp<B>
// ---------------------------------------------------------------------------

impl<B: BitInt> SymGrp for HardcoreSymmetryGrp<B> {
    type State = B;

    fn n_sites(&self) -> usize {
        HardcoreSymmetryGrp::n_sites(self)
    }

    fn get_refstate(&self, state: B) -> (B, num_complex::Complex<f64>) {
        HardcoreSymmetryGrp::get_refstate(self, state)
    }

    fn check_refstate(&self, state: B) -> (B, f64) {
        HardcoreSymmetryGrp::check_refstate(self, state)
    }

    fn check_refstate_batch(&self, states: &[B], out: &mut [(B, f64)]) {
        HardcoreSymmetryGrp::check_refstate_batch(self, states, out);
    }
}

// ---------------------------------------------------------------------------
// DitSpinGrpInner — compile-time-LHSS spin-inversion (LHSS 3–5)
// ---------------------------------------------------------------------------

// Dit spin infrastructure is forward-looking (no dit basis yet). Allow
// dead_code until DitSymmetricSubspace is implemented.
#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) struct DitSpinGrpInner<const LHSS: usize> {
    n_sites: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<(Complex<f64>, HigherSpinInv<LHSS>)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl<const LHSS: usize> DitSpinGrpInner<LHSS> {
    fn new_empty(n_sites: usize) -> Self {
        DitSpinGrpInner {
            n_sites,
            lattice: Vec::new(),
            local: Vec::new(),
        }
    }

    fn push_lattice(&mut self, el: LatticeElement) {
        self.lattice.push(el);
    }

    fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        self.local
            .push((grp_char, HigherSpinInv::<LHSS>::new(locs)));
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
// DitSpinGrpDyn — runtime-LHSS spin-inversion fallback (LHSS ≥ 6)
// ---------------------------------------------------------------------------

#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) struct DitSpinGrpDyn {
    n_sites: usize,
    lhss: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<(Complex<f64>, DynamicHigherSpinInv)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl DitSpinGrpDyn {
    fn new_empty(lhss: usize, n_sites: usize) -> Self {
        DitSpinGrpDyn {
            n_sites,
            lhss,
            lattice: Vec::new(),
            local: Vec::new(),
        }
    }

    fn push_lattice(&mut self, el: LatticeElement) {
        self.lattice.push(el);
    }

    fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        self.local
            .push((grp_char, DynamicHigherSpinInv::new(self.lhss, locs)));
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
// DitSpinSymGrpInner — LHSS dispatch enum for spin-inversion (LHSS > 2)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum DitSpinSymGrpInner {
    Lhss3(DitSpinGrpInner<3>),
    Lhss4(DitSpinGrpInner<4>),
    Lhss5(DitSpinGrpInner<5>),
    LhssDyn(DitSpinGrpDyn),
}

impl DitSpinSymGrpInner {
    fn new_empty(lhss: usize, n_sites: usize) -> Self {
        match lhss {
            3 => DitSpinSymGrpInner::Lhss3(DitSpinGrpInner::<3>::new_empty(n_sites)),
            4 => DitSpinSymGrpInner::Lhss4(DitSpinGrpInner::<4>::new_empty(n_sites)),
            5 => DitSpinSymGrpInner::Lhss5(DitSpinGrpInner::<5>::new_empty(n_sites)),
            _ => DitSpinSymGrpInner::LhssDyn(DitSpinGrpDyn::new_empty(lhss, n_sites)),
        }
    }

    pub(crate) fn push_lattice(&mut self, el: LatticeElement) {
        match self {
            DitSpinSymGrpInner::Lhss3(g) => g.push_lattice(el),
            DitSpinSymGrpInner::Lhss4(g) => g.push_lattice(el),
            DitSpinSymGrpInner::Lhss5(g) => g.push_lattice(el),
            DitSpinSymGrpInner::LhssDyn(g) => g.push_lattice(el),
        }
    }

    pub(crate) fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        match self {
            DitSpinSymGrpInner::Lhss3(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInner::Lhss4(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInner::Lhss5(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInner::LhssDyn(g) => g.push_spin_inv(grp_char, locs),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn n_sites(&self) -> usize {
        match self {
            DitSpinSymGrpInner::Lhss3(g) => g.n_sites(),
            DitSpinSymGrpInner::Lhss4(g) => g.n_sites(),
            DitSpinSymGrpInner::Lhss5(g) => g.n_sites(),
            DitSpinSymGrpInner::LhssDyn(g) => g.n_sites(),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn lhss(&self) -> usize {
        match self {
            DitSpinSymGrpInner::Lhss3(_) => 3,
            DitSpinSymGrpInner::Lhss4(_) => 4,
            DitSpinSymGrpInner::Lhss5(_) => 5,
            DitSpinSymGrpInner::LhssDyn(g) => g.lhss(),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn get_refstate<B: BitInt>(&self, state: B) -> (B, Complex<f64>) {
        match self {
            DitSpinSymGrpInner::Lhss3(g) => g.get_refstate(state),
            DitSpinSymGrpInner::Lhss4(g) => g.get_refstate(state),
            DitSpinSymGrpInner::Lhss5(g) => g.get_refstate(state),
            DitSpinSymGrpInner::LhssDyn(g) => g.get_refstate(state),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn check_refstate<B: BitInt>(&self, state: B) -> (B, f64) {
        match self {
            DitSpinSymGrpInner::Lhss3(g) => g.check_refstate(state),
            DitSpinSymGrpInner::Lhss4(g) => g.check_refstate(state),
            DitSpinSymGrpInner::Lhss5(g) => g.check_refstate(state),
            DitSpinSymGrpInner::LhssDyn(g) => g.check_refstate(state),
        }
    }
}

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
    /// LHSS > 2: spin-inversion ops; `B` resolved at call sites.
    Dit(DitSpinSymGrpInner),
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
            SpinSymGrpInner::Dit(DitSpinSymGrpInner::new_empty(lhss, n_sites))
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
        use crate::bitbasis::PermDitLocations;
        let el = LatticeElement::new(
            grp_char,
            PermDitLocations::new(self.lhss, &perm),
            self.n_sites,
        );
        match &mut self.inner {
            SpinSymGrpInner::Hardcore(hc) => hc.push_lattice(el),
            SpinSymGrpInner::Dit(dit) => dit.push_lattice(el),
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
    pub(crate) fn as_dit(&self) -> Option<&DitSpinSymGrpInner> {
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
        let manip = DynamicDitManip::new(3);
        let state: u32 = manip.set_dit(manip.set_dit(0u32, 1, 0), 0, 1);
        let (ref_s, _) = dit.get_refstate(state);
        assert!(ref_s >= state);
    }

    #[test]
    fn spin_sym_lhss_dyn() {
        let mut grp = SpinSymGrp::new(6, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_inverse(Complex::new(1.0, 0.0), vec![0, 1]);
        assert_eq!(grp.lhss(), 6);
    }
}
