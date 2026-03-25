use crate::bitbasis::{
    BitInt, BitStateOp, DynamicHigherSpinInv, DynamicPermDitValues, HigherSpinInv,
    PermDitLocations, PermDitMask, PermDitValues,
};
use crate::error::QuSpinError;
use num_complex::Complex;

use super::dispatch::SymmetryGrpInner;

// ---------------------------------------------------------------------------
// LatticeElement
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LatticeElement {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub op: PermDitLocations,
}

impl LatticeElement {
    pub fn new(grp_char: Complex<f64>, op: PermDitLocations, n_sites: usize) -> Self {
        LatticeElement {
            grp_char,
            n_sites,
            op,
        }
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    #[inline]
    pub fn apply<B: BitInt>(&self, state: B, coeff: Complex<f64>) -> (B, Complex<f64>) {
        (self.op.apply(state), coeff * self.grp_char)
    }
}

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

#[derive(Clone)]
pub struct HardcoreSymmetryGrp<B: BitInt> {
    n_sites: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<HardcoreGrpElement<B>>,
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

    pub fn push_local_inv(&mut self, grp_char: Complex<f64>, locs: &[usize]) {
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

    fn iter_images(&self, state: B) -> impl Iterator<Item = (B, Complex<f64>)> + '_ {
        let one = Complex::new(1.0, 0.0);
        let lattice_images = self.lattice.iter().map(move |el| el.apply(state, one));
        let local_then_lattice = self.local.iter().flat_map(move |loc| {
            let (loc_state, loc_coeff) = loc.apply(state, one);
            self.lattice
                .iter()
                .map(move |lat| lat.apply(loc_state, loc_coeff))
        });
        lattice_images.chain(local_then_lattice)
    }

    pub fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
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

    pub fn check_refstate(&self, state: B) -> (B, f64) {
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
// DitGrpInner — private, compile-time-LHSS hot-path storage
// ---------------------------------------------------------------------------

// Dit infrastructure is forward-looking (no dit basis yet). Allow dead_code
// until the dit basis and DitSymmetricSubspace are implemented.
#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) struct DitGrpInner<const LHSS: usize> {
    pub(crate) n_sites: usize,
    pub(crate) lattice: Vec<LatticeElement>,
    pub(crate) value_local: Vec<(Complex<f64>, PermDitValues<LHSS>)>,
    pub(crate) spin_inv_local: Vec<(Complex<f64>, HigherSpinInv<LHSS>)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl<const LHSS: usize> DitGrpInner<LHSS> {
    fn new_empty(n_sites: usize) -> Self {
        DitGrpInner {
            n_sites,
            lattice: Vec::new(),
            value_local: Vec::new(),
            spin_inv_local: Vec::new(),
        }
    }

    fn push_lattice(&mut self, el: LatticeElement) {
        self.lattice.push(el);
    }

    fn push_value_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        let arr: [u8; LHSS] = perm.try_into().expect("perm length must match LHSS");
        self.value_local
            .push((grp_char, PermDitValues::<LHSS>::new(arr, locs)));
    }

    fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        self.spin_inv_local
            .push((grp_char, HigherSpinInv::<LHSS>::new(locs)));
    }

    fn n_sites(&self) -> usize {
        self.n_sites
    }

    fn iter_images<B: BitInt>(&self, state: B) -> Vec<(B, Complex<f64>)> {
        let one = Complex::new(1.0, 0.0);
        let n_local = self.value_local.len() + self.spin_inv_local.len();
        let mut images = Vec::with_capacity(self.lattice.len() * (1 + n_local));
        for lat in &self.lattice {
            images.push(lat.apply(state, one));
        }
        for (char_, op) in &self.value_local {
            let loc_state = op.apply(state);
            for lat in &self.lattice {
                images.push(lat.apply(loc_state, *char_));
            }
        }
        for (char_, op) in &self.spin_inv_local {
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
// DitGrpDyn — private, runtime-LHSS fallback (LHSS ≥ 6)
// ---------------------------------------------------------------------------

/// Local operation for the dynamic (LHSS ≥ 6) fallback path.
#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
enum DynLocalOp {
    Value(DynamicPermDitValues),
    SpinInv(DynamicHigherSpinInv),
}

#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) struct DitGrpDyn {
    n_sites: usize,
    lhss: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<(Complex<f64>, DynLocalOp)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl DitGrpDyn {
    fn new_empty(lhss: usize, n_sites: usize) -> Self {
        DitGrpDyn {
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
        self.local.push((
            grp_char,
            DynLocalOp::Value(DynamicPermDitValues::new(self.lhss, perm, locs)),
        ));
    }

    fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        self.local.push((
            grp_char,
            DynLocalOp::SpinInv(DynamicHigherSpinInv::new(self.lhss, locs)),
        ));
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
            let loc_state = match op {
                DynLocalOp::Value(v) => v.apply(state),
                DynLocalOp::SpinInv(s) => s.apply(state),
            };
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
// DitSymGrpInner — LHSS dispatch enum (LHSS > 2)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum DitSymGrpInner {
    Lhss3(DitGrpInner<3>),
    Lhss4(DitGrpInner<4>),
    Lhss5(DitGrpInner<5>),
    LhssDyn(DitGrpDyn),
}

impl DitSymGrpInner {
    fn new_empty(lhss: usize, n_sites: usize) -> Self {
        match lhss {
            3 => DitSymGrpInner::Lhss3(DitGrpInner::<3>::new_empty(n_sites)),
            4 => DitSymGrpInner::Lhss4(DitGrpInner::<4>::new_empty(n_sites)),
            5 => DitSymGrpInner::Lhss5(DitGrpInner::<5>::new_empty(n_sites)),
            _ => DitSymGrpInner::LhssDyn(DitGrpDyn::new_empty(lhss, n_sites)),
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

    pub(crate) fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        match self {
            DitSymGrpInner::Lhss3(g) => g.push_spin_inv(grp_char, locs),
            DitSymGrpInner::Lhss4(g) => g.push_spin_inv(grp_char, locs),
            DitSymGrpInner::Lhss5(g) => g.push_spin_inv(grp_char, locs),
            DitSymGrpInner::LhssDyn(g) => g.push_spin_inv(grp_char, locs),
        }
    }

    pub(crate) fn push_value_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) {
        match self {
            DitSymGrpInner::Lhss3(g) => g.push_value_perm(grp_char, perm, locs),
            DitSymGrpInner::Lhss4(g) => g.push_value_perm(grp_char, perm, locs),
            DitSymGrpInner::Lhss5(g) => g.push_value_perm(grp_char, perm, locs),
            DitSymGrpInner::LhssDyn(g) => g.push_value_perm(grp_char, perm, locs),
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
// SymmetryGrp — public unified builder
// ---------------------------------------------------------------------------

/// A symmetry group that resolves both the basis integer type `B` (from
/// `n_sites`) and the local Hilbert-space size `lhss` at construction time.
///
/// Elements are added with the mutable builder methods:
/// - [`add_lattice`](SymmetryGrp::add_lattice) — site-permutation symmetry
/// - [`add_local_inv`](SymmetryGrp::add_local_inv) — spin inversion / bit-flip
/// - [`add_local_perm`](SymmetryGrp::add_local_perm) — on-site value permutation (LHSS > 2 only)
///
/// The group is immediately usable after construction; there is no separate
/// `.build()` step.
#[derive(Clone)]
pub struct SymmetryGrp {
    lhss: usize,
    n_sites: usize,
    inner: SymGrpInner,
}

#[derive(Clone)]
enum SymGrpInner {
    /// LHSS = 2 path: concrete `B` resolved from `n_sites` at construction.
    Hardcore(SymmetryGrpInner),
    /// LHSS > 2 path: `B` is only needed at `get_refstate` / `check_refstate`
    /// call sites; LHSS dispatch happens once at construction.
    Dit(DitSymGrpInner),
}

impl SymmetryGrp {
    /// Construct an empty symmetry group.
    ///
    /// Both `B` (selected from `n_sites`) and the LHSS dispatch (from `lhss`)
    /// are resolved here, once, at construction time.
    ///
    /// Elements are added after construction via [`add_lattice`](Self::add_lattice),
    /// [`add_local_inv`](Self::add_local_inv), and [`add_local_perm`](Self::add_local_perm).
    ///
    /// Returns `Err` if `n_sites > 8192`.
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
            SymGrpInner::Hardcore(hc)
        } else {
            SymGrpInner::Dit(DitSymGrpInner::new_empty(lhss, n_sites))
        };
        Ok(SymmetryGrp {
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
    /// The length of `perm` must equal `self.n_sites`.
    pub fn add_lattice(&mut self, grp_char: Complex<f64>, perm: Vec<usize>) {
        let el = LatticeElement::new(
            grp_char,
            PermDitLocations::new(self.lhss, &perm),
            self.n_sites,
        );
        match &mut self.inner {
            SymGrpInner::Hardcore(hc) => hc.push_lattice(el),
            SymGrpInner::Dit(dit) => dit.push_lattice(el),
        }
    }

    /// Add a spin-inversion / bit-flip symmetry element.
    ///
    /// For LHSS = 2: XOR-flips the bits at the specified site indices.
    /// For LHSS > 2: maps `v → lhss − v − 1` at the specified sites.
    ///
    /// `locs` is the list of site indices on which the operation acts.
    pub fn add_local_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        match &mut self.inner {
            SymGrpInner::Hardcore(hc) => hc.push_local_inv(grp_char, &locs),
            SymGrpInner::Dit(dit) => dit.push_spin_inv(grp_char, locs),
        }
    }

    /// Add an on-site value-permutation symmetry element (LHSS > 2 only).
    ///
    /// `perm[v] = w` maps local occupation `v` to `w` at each site in `locs`.
    /// The length of `perm` must equal `self.lhss`.
    ///
    /// Returns `Err` if `self.lhss == 2` (no value permutation for two-valued dits).
    pub fn add_local_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        match &mut self.inner {
            SymGrpInner::Hardcore(_) => Err(QuSpinError::ValueError(
                "value permutation is not supported for LHSS=2 (hardcore) groups".to_string(),
            )),
            SymGrpInner::Dit(dit) => {
                dit.push_value_perm(grp_char, perm, locs);
                Ok(())
            }
        }
    }

    /// Access the hardcore (LHSS=2) inner dispatch type.
    ///
    /// Used by `quspin-py` to construct `SymmetricSubspace<B>` via `with_sym_grp!`.
    /// Returns `None` for dit groups.
    pub fn as_hardcore(&self) -> Option<&SymmetryGrpInner> {
        match &self.inner {
            SymGrpInner::Hardcore(hc) => Some(hc),
            SymGrpInner::Dit(_) => None,
        }
    }

    /// Access the dit (LHSS>2) inner dispatch type.
    ///
    /// Returns `None` for hardcore groups.
    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn as_dit(&self) -> Option<&DitSymGrpInner> {
        match &self.inner {
            SymGrpInner::Dit(dit) => Some(dit),
            SymGrpInner::Hardcore(_) => None,
        }
    }
}

// `Clone` on SymGrpKind requires Clone on the variants.
// SymmetryGrpInner already derives Clone; DitSymGrpInner does too.
// SymGrpInner is defined above with Clone.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SymmetryGrp builder (LHSS=2) ---

    #[test]
    fn hardcore_bitflip_get_refstate() {
        // 2-site chain: translation (identity) + full bit-flip.
        let mut grp = SymmetryGrp::new(2, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]); // identity translation
        grp.add_local_inv(Complex::new(1.0, 0.0), vec![0, 1]); // flip both sites

        // state |01⟩=1: image under flip = |10⟩=2 → ref = 2
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
    fn hardcore_translation() {
        // 3-site translation: perm [1,2,0]
        let mut grp = SymmetryGrp::new(2, 3).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 0]);

        let grp_inner = grp.as_hardcore().unwrap();
        match grp_inner {
            SymmetryGrpInner::Sym32(g) => {
                // |001⟩=1, T(|001⟩)=|010⟩=2 → ref = 2
                let (ref_s, _) = g.get_refstate(0b001u32);
                assert_eq!(ref_s, 0b010u32);
            }
            _ => panic!("expected Sym32"),
        }
    }

    #[test]
    fn hardcore_n_sites_too_large_errors() {
        assert!(SymmetryGrp::new(2, 8193).is_err());
    }

    #[test]
    fn hardcore_add_local_perm_errors() {
        let mut grp = SymmetryGrp::new(2, 4).unwrap();
        assert!(
            grp.add_local_perm(Complex::new(1.0, 0.0), vec![1, 0], vec![0])
                .is_err()
        );
    }

    // --- SymmetryGrp builder (LHSS>2) ---

    #[test]
    fn dit_spin_inversion_get_refstate() {
        use crate::bitbasis::DynamicDitManip;
        // LHSS=3 spin inversion on a 2-site chain.
        let mut grp = SymmetryGrp::new(3, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]); // identity
        grp.add_local_inv(Complex::new(1.0, 0.0), vec![0, 1]); // spin inv

        assert_eq!(grp.n_sites(), 2);
        assert_eq!(grp.lhss(), 3);

        let dit = grp.as_dit().unwrap();
        let manip = DynamicDitManip::new(3);
        let state: u32 = manip.set_dit(manip.set_dit(0u32, 1, 0), 0, 1);
        let (ref_s, _) = dit.get_refstate(state);
        assert!(ref_s >= state);
    }

    #[test]
    fn dit_value_perm() {
        let mut grp = SymmetryGrp::new(3, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        assert!(
            grp.add_local_perm(Complex::new(1.0, 0.0), vec![2, 1, 0], vec![0, 1])
                .is_ok()
        );
        assert_eq!(grp.lhss(), 3);
    }

    #[test]
    fn dit_lhss_dyn() {
        // LHSS=6 falls back to DynLocalOp path.
        let mut grp = SymmetryGrp::new(6, 2).unwrap();
        grp.add_lattice(Complex::new(1.0, 0.0), vec![0, 1]);
        grp.add_local_inv(Complex::new(1.0, 0.0), vec![0, 1]);
        assert_eq!(grp.lhss(), 6);
    }
}
