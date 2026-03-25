use crate::bitbasis::{
    BitInt, BitStateOp, DynamicHigherSpinInv, DynamicPermDitValues, HigherSpinInv,
    PermDitLocations, PermDitMask, PermDitValues,
};
use crate::error::QuSpinError;
use num_complex::Complex;

// ---------------------------------------------------------------------------
// LatticeElement — shared by both hardcore and dit groups
// ---------------------------------------------------------------------------

/// A lattice symmetry element (translation, reflection, …).
///
/// Shared between `HardcoreSymmetryGrp` and `DitSymmetryGrp`: site
/// permutations are the same regardless of the local Hilbert-space size.
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

/// A local symmetry element for LHSS=2 (hardcore bosons / spin-1/2).
///
/// The only local operation for a two-valued dit is an XOR mask (bit-flip);
/// no enum is needed.
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

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    #[inline]
    pub fn apply(&self, state: B, coeff: Complex<f64>) -> (B, Complex<f64>) {
        (self.op.apply(state), coeff * self.grp_char)
    }
}

/// A symmetry group for LHSS=2 basis states.
///
/// Local operations are restricted to `PermDitMask<B>` (XOR-based bit-flip),
/// which is the only meaningful local symmetry for a two-valued dit.
/// `B` is fixed on the struct because `PermDitMask<B>` is monomorphic.
///
/// Mirrors `grp<grp_result_t, lattice_t, local_t>` from `grp.hpp` for the
/// spin-1/2 case.
#[derive(Clone)]
pub struct HardcoreSymmetryGrp<B: BitInt> {
    n_sites: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<HardcoreGrpElement<B>>,
}

impl<B: BitInt> HardcoreSymmetryGrp<B> {
    /// Construct, validating that all elements agree on `n_sites`.
    pub fn new(
        lattice: Vec<LatticeElement>,
        local: Vec<HardcoreGrpElement<B>>,
    ) -> Result<Self, QuSpinError> {
        let mut n_sites_opt: Option<usize> = None;
        for el in lattice
            .iter()
            .map(|e| e.n_sites)
            .chain(local.iter().map(|e| e.n_sites))
        {
            match n_sites_opt {
                None => n_sites_opt = Some(el),
                Some(existing) if existing != el => {
                    return Err(QuSpinError::ValueError(format!(
                        "n_sites mismatch in symmetry group: element has {el} but expected {existing}"
                    )));
                }
                _ => {}
            }
        }
        Ok(HardcoreSymmetryGrp {
            n_sites: n_sites_opt.unwrap_or(0),
            lattice,
            local,
        })
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
// DitLocalOp / DitGrpElement  (construction / description types)
// ---------------------------------------------------------------------------

/// Local symmetry operation for a dit basis with LHSS > 2.
///
/// This type is the **construction input** to `DitSymmetryGrp::new`.
/// At group construction time the LHSS is resolved once and the op is
/// moved into typed hot-path storage; this enum is not in the critical path.
///
/// LHSS ∈ {3, 4, 5} produce compile-time-monomorphised variants;
/// all other values fall back to the dynamic (runtime-LHSS) variants.
#[derive(Clone)]
pub enum DitLocalOp {
    // --- compile-time LHSS ---
    Value3(PermDitValues<3>),
    Value4(PermDitValues<4>),
    Value5(PermDitValues<5>),
    SpinInv3(HigherSpinInv<3>),
    SpinInv4(HigherSpinInv<4>),
    SpinInv5(HigherSpinInv<5>),
    // --- runtime LHSS ---
    ValueDyn(DynamicPermDitValues),
    SpinInvDyn(DynamicHigherSpinInv),
}

impl DitLocalOp {
    pub(crate) fn new_value(lhss: usize, perm: Vec<u8>, locs: Vec<usize>) -> Self {
        match lhss {
            3 => DitLocalOp::Value3(PermDitValues::<3>::new(perm.try_into().unwrap(), locs)),
            4 => DitLocalOp::Value4(PermDitValues::<4>::new(perm.try_into().unwrap(), locs)),
            5 => DitLocalOp::Value5(PermDitValues::<5>::new(perm.try_into().unwrap(), locs)),
            _ => DitLocalOp::ValueDyn(DynamicPermDitValues::new(lhss, perm, locs)),
        }
    }

    pub(crate) fn new_spin_inv(lhss: usize, locs: Vec<usize>) -> Self {
        match lhss {
            3 => DitLocalOp::SpinInv3(HigherSpinInv::<3>::new(locs)),
            4 => DitLocalOp::SpinInv4(HigherSpinInv::<4>::new(locs)),
            5 => DitLocalOp::SpinInv5(HigherSpinInv::<5>::new(locs)),
            _ => DitLocalOp::SpinInvDyn(DynamicHigherSpinInv::new(lhss, locs)),
        }
    }
}

impl<I: BitInt> BitStateOp<I> for DitLocalOp {
    #[inline]
    fn apply(&self, state: I) -> I {
        match self {
            DitLocalOp::Value3(op) => op.apply(state),
            DitLocalOp::Value4(op) => op.apply(state),
            DitLocalOp::Value5(op) => op.apply(state),
            DitLocalOp::SpinInv3(op) => op.apply(state),
            DitLocalOp::SpinInv4(op) => op.apply(state),
            DitLocalOp::SpinInv5(op) => op.apply(state),
            DitLocalOp::ValueDyn(op) => op.apply(state),
            DitLocalOp::SpinInvDyn(op) => op.apply(state),
        }
    }
}

/// A local symmetry element for a dit basis with LHSS > 2.
///
/// Used as construction input to `DitSymmetryGrp::new`, which decomposes it
/// into typed hot-path storage at construction time.
#[derive(Clone)]
pub struct DitGrpElement {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub lhss: usize,
    pub op: DitLocalOp,
}

impl DitGrpElement {
    pub fn new(grp_char: Complex<f64>, n_sites: usize, lhss: usize, op: DitLocalOp) -> Self {
        DitGrpElement {
            grp_char,
            n_sites,
            lhss,
            op,
        }
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    pub fn lhss(&self) -> usize {
        self.lhss
    }

    #[inline]
    pub fn apply<B: BitInt>(&self, state: B, coeff: Complex<f64>) -> (B, Complex<f64>) {
        (self.op.apply(state), coeff * self.grp_char)
    }
}

// ---------------------------------------------------------------------------
// DitGrpInner — private, compile-time-LHSS hot-path storage
// ---------------------------------------------------------------------------

/// Inner storage for a `DitSymmetryGrp` when LHSS is known at compile time.
///
/// Value and spin-inversion ops are kept in separate vecs so the iteration
/// loop contains no enum dispatch at all — just plain method calls on
/// monomorphised types.
#[derive(Clone)]
struct DitGrpInner<const LHSS: usize> {
    n_sites: usize,
    lattice: Vec<LatticeElement>,
    value_local: Vec<(Complex<f64>, PermDitValues<LHSS>)>,
    spin_inv_local: Vec<(Complex<f64>, HigherSpinInv<LHSS>)>,
}

impl<const LHSS: usize> DitGrpInner<LHSS> {
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

/// Consume a `Vec<DitGrpElement>` and split it into the two typed vecs for
/// `DitGrpInner<LHSS>`.  Panics if any element carries the wrong variant.
macro_rules! build_dit_inner {
    ($lhss:literal, $value_var:ident, $spin_inv_var:ident,
     $n_sites:expr, $lattice:expr, $local:expr) => {{
        let mut value_local = Vec::new();
        let mut spin_inv_local = Vec::new();
        for el in $local {
            match el.op {
                DitLocalOp::$value_var(op) => value_local.push((el.grp_char, op)),
                DitLocalOp::$spin_inv_var(op) => spin_inv_local.push((el.grp_char, op)),
                _ => panic!(
                    "DitSymmetryGrp: expected LHSS={} op but element carries a different variant",
                    $lhss
                ),
            }
        }
        DitGrpInner::<$lhss> {
            n_sites: $n_sites,
            lattice: $lattice,
            value_local,
            spin_inv_local,
        }
    }};
}

// ---------------------------------------------------------------------------
// DitGrpDyn — private, runtime-LHSS fallback (LHSS ≥ 6)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct DitGrpDyn {
    n_sites: usize,
    lhss: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<DitGrpElement>,
}

impl DitGrpDyn {
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
        for loc in &self.local {
            let (loc_state, loc_coeff) = loc.apply(state, one);
            for lat in &self.lattice {
                images.push(lat.apply(loc_state, loc_coeff));
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
// DitSymmetryGrp — public
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum DitSymGrpInner {
    Lhss3(DitGrpInner<3>),
    Lhss4(DitGrpInner<4>),
    Lhss5(DitGrpInner<5>),
    LhssDyn(DitGrpDyn),
}

/// A symmetry group for dit bases with LHSS > 2.
///
/// Unlike `HardcoreSymmetryGrp<B>`, this type carries **no `B` type
/// parameter**: the integer width is only needed when `get_refstate` and
/// `check_refstate` are called.
///
/// LHSS dispatch happens **once** at construction time.  For LHSS ∈ {3, 4, 5}
/// the group is stored in a fully-typed inner struct (`DitGrpInner<LHSS>`)
/// that keeps value and spin-inversion ops in separate vecs — the hot path
/// contains no enum dispatch at all.  LHSS ≥ 6 falls back to a dynamic
/// runtime representation.
///
/// # Panics
/// `new` panics if elements have mismatched `n_sites` or `lhss`, or if an
/// element's `DitLocalOp` variant does not match the resolved `lhss`.
#[derive(Clone)]
pub struct DitSymmetryGrp(DitSymGrpInner);

impl DitSymmetryGrp {
    pub fn new(lattice: Vec<LatticeElement>, local: Vec<DitGrpElement>) -> Self {
        // validate n_sites
        let mut n_sites_opt: Option<usize> = None;
        for n in lattice
            .iter()
            .map(|l| l.n_sites())
            .chain(local.iter().map(|e| e.n_sites))
        {
            match n_sites_opt {
                None => n_sites_opt = Some(n),
                Some(existing) => assert_eq!(
                    existing, n,
                    "DitSymmetryGrp: n_sites mismatch ({existing} vs {n})"
                ),
            }
        }
        let n_sites = n_sites_opt.unwrap_or(0);

        // validate lhss consistency
        let mut lhss_opt: Option<usize> = None;
        for l in local.iter().map(|e| e.lhss) {
            match lhss_opt {
                None => lhss_opt = Some(l),
                Some(existing) => assert_eq!(
                    existing, l,
                    "DitSymmetryGrp: lhss mismatch ({existing} vs {l})"
                ),
            }
        }
        let lhss = lhss_opt.unwrap_or(3);

        let inner = match lhss {
            3 => DitSymGrpInner::Lhss3(build_dit_inner!(
                3, Value3, SpinInv3, n_sites, lattice, local
            )),
            4 => DitSymGrpInner::Lhss4(build_dit_inner!(
                4, Value4, SpinInv4, n_sites, lattice, local
            )),
            5 => DitSymGrpInner::Lhss5(build_dit_inner!(
                5, Value5, SpinInv5, n_sites, lattice, local
            )),
            _ => DitSymGrpInner::LhssDyn(DitGrpDyn {
                n_sites,
                lhss,
                lattice,
                local,
            }),
        };
        DitSymmetryGrp(inner)
    }

    pub fn n_sites(&self) -> usize {
        match &self.0 {
            DitSymGrpInner::Lhss3(g) => g.n_sites(),
            DitSymGrpInner::Lhss4(g) => g.n_sites(),
            DitSymGrpInner::Lhss5(g) => g.n_sites(),
            DitSymGrpInner::LhssDyn(g) => g.n_sites(),
        }
    }

    pub fn lhss(&self) -> usize {
        match &self.0 {
            DitSymGrpInner::Lhss3(_) => 3,
            DitSymGrpInner::Lhss4(_) => 4,
            DitSymGrpInner::Lhss5(_) => 5,
            DitSymGrpInner::LhssDyn(g) => g.lhss(),
        }
    }

    pub fn get_refstate<B: BitInt>(&self, state: B) -> (B, Complex<f64>) {
        match &self.0 {
            DitSymGrpInner::Lhss3(g) => g.get_refstate(state),
            DitSymGrpInner::Lhss4(g) => g.get_refstate(state),
            DitSymGrpInner::Lhss5(g) => g.get_refstate(state),
            DitSymGrpInner::LhssDyn(g) => g.get_refstate(state),
        }
    }

    pub fn check_refstate<B: BitInt>(&self, state: B) -> (B, f64) {
        match &self.0 {
            DitSymGrpInner::Lhss3(g) => g.check_refstate(state),
            DitSymGrpInner::Lhss4(g) => g.check_refstate(state),
            DitSymGrpInner::Lhss5(g) => g.check_refstate(state),
            DitSymGrpInner::LhssDyn(g) => g.check_refstate(state),
        }
    }
}

// ---------------------------------------------------------------------------
// GrpOpDesc — type-erased description, independent of B
// ---------------------------------------------------------------------------

/// Stores the data for a local symmetry operation before the basis integer
/// type `B` is known.
///
/// Used by the Python FFI layer: elements are described at Python call time,
/// then converted to concrete typed elements inside `PySymmetryGrp::new`
/// once `n_sites` (and therefore `B`) is resolved.
#[derive(Clone, Debug)]
pub enum GrpOpDesc {
    /// XOR with a fixed mask (Z₂ bit-flip, LHSS=2 only).
    Bitflip {
        n_sites: usize,
        /// Site indices to flip. `None` = all `n_sites` bits.
        locs: Option<Vec<usize>>,
    },
    /// Uniform value permutation on a subset of sites (LHSS > 2).
    LocalValue {
        n_sites: usize,
        lhss: usize,
        perm: Vec<u8>,
        locs: Vec<usize>,
    },
    /// Spin inversion `v → lhss − v − 1` on a subset of sites (LHSS > 2).
    SpinInversion {
        n_sites: usize,
        lhss: usize,
        locs: Vec<usize>,
    },
}

impl GrpOpDesc {
    pub fn n_sites(&self) -> usize {
        match self {
            GrpOpDesc::Bitflip { n_sites, .. } => *n_sites,
            GrpOpDesc::LocalValue { n_sites, .. } => *n_sites,
            GrpOpDesc::SpinInversion { n_sites, .. } => *n_sites,
        }
    }

    /// Convert a `Bitflip` desc into a `HardcoreGrpElement<B>`.
    ///
    /// Returns `Err` if this desc is not a `Bitflip`.
    pub fn into_hardcore_element<B: BitInt>(
        self,
        grp_char: Complex<f64>,
    ) -> Result<HardcoreGrpElement<B>, QuSpinError> {
        match self {
            GrpOpDesc::Bitflip { n_sites, locs } => {
                let effective: Vec<usize> = match locs {
                    Some(l) => l,
                    None => (0..n_sites).collect(),
                };
                let mask = effective.iter().fold(B::from_u64(0), |acc, &site| {
                    if site < B::BITS as usize {
                        acc | (B::from_u64(1) << site)
                    } else {
                        acc
                    }
                });
                Ok(HardcoreGrpElement::new(
                    grp_char,
                    PermDitMask::new(mask),
                    n_sites,
                ))
            }
            _ => Err(QuSpinError::ValueError(
                "expected a Bitflip op for a hardcore (LHSS=2) symmetry group".to_string(),
            )),
        }
    }

    /// Convert a `LocalValue` or `SpinInversion` desc into a `DitGrpElement`.
    ///
    /// Returns `Err` if this desc is a `Bitflip`.
    pub fn into_dit_element(self, grp_char: Complex<f64>) -> Result<DitGrpElement, QuSpinError> {
        match self {
            GrpOpDesc::LocalValue {
                n_sites,
                lhss,
                perm,
                locs,
            } => Ok(DitGrpElement::new(
                grp_char,
                n_sites,
                lhss,
                DitLocalOp::new_value(lhss, perm, locs),
            )),
            GrpOpDesc::SpinInversion {
                n_sites,
                lhss,
                locs,
            } => Ok(DitGrpElement::new(
                grp_char,
                n_sites,
                lhss,
                DitLocalOp::new_spin_inv(lhss, locs),
            )),
            GrpOpDesc::Bitflip { .. } => Err(QuSpinError::ValueError(
                "expected a LocalValue or SpinInversion op for a dit (LHSS>2) symmetry group"
                    .to_string(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitbasis::PermDitLocations;

    fn id_lattice(n_sites: usize) -> LatticeElement {
        let locs: Vec<usize> = (0..n_sites).collect();
        LatticeElement::new(
            Complex::new(1.0, 0.0),
            PermDitLocations::new(2, &locs),
            n_sites,
        )
    }

    // --- HardcoreSymmetryGrp ---

    #[test]
    fn hardcore_bitflip_get_refstate() {
        let mask: u32 = 0b11;
        let el = HardcoreGrpElement::new(Complex::new(1.0, 0.0), PermDitMask::new(mask), 2);
        let grp = HardcoreSymmetryGrp::new(vec![id_lattice(2)], vec![el]).unwrap();
        // |01⟩=1: images include flip(1)=2. ref = 2.
        let (ref_s, _) = grp.get_refstate(0b01u32);
        assert_eq!(ref_s, 0b10u32);
    }

    #[test]
    fn hardcore_check_refstate_norm() {
        let mask: u32 = 0b11;
        let el = HardcoreGrpElement::new(Complex::new(1.0, 0.0), PermDitMask::new(mask), 2);
        let grp = HardcoreSymmetryGrp::new(vec![id_lattice(2)], vec![el]).unwrap();
        let (_, norm) = grp.check_refstate(0b01u32);
        assert_eq!(norm, 1.0);
    }

    #[test]
    fn hardcore_translation() {
        // 3-site translation: site 0→1, 1→2, 2→0 (forward perm = [1, 2, 0])
        let lat = LatticeElement::new(
            Complex::new(1.0, 0.0),
            PermDitLocations::new(2, &[1, 2, 0]),
            3,
        );
        let grp = HardcoreSymmetryGrp::<u32>::new(vec![lat], vec![]).unwrap();
        // state |001⟩=1, T(|001⟩)=|010⟩=2 → ref = 2
        let (ref_s, _) = grp.get_refstate(0b001u32);
        assert_eq!(ref_s, 0b010u32);
    }

    #[test]
    fn hardcore_n_sites_mismatch_errors() {
        let lat3 = id_lattice(3);
        let el2 = HardcoreGrpElement::new(
            Complex::new(1.0, 0.0),
            PermDitMask::new(0b11u32),
            2, // disagrees with lat3
        );
        assert!(HardcoreSymmetryGrp::new(vec![lat3], vec![el2]).is_err());
    }

    // --- DitSymmetryGrp ---

    #[test]
    fn dit_spin_inversion_get_refstate() {
        use crate::bitbasis::DynamicDitManip;
        // LHSS=3 spin inversion on a 2-site chain.
        let id = LatticeElement::new(Complex::new(1.0, 0.0), PermDitLocations::new(3, &[0, 1]), 2);
        let inv_el = DitGrpElement::new(
            Complex::new(1.0, 0.0),
            2,
            3,
            DitLocalOp::new_spin_inv(3, vec![0, 1]),
        );
        let grp = DitSymmetryGrp::new(vec![id], vec![inv_el]);
        assert_eq!(grp.n_sites(), 2);
        assert_eq!(grp.lhss(), 3);

        // State with site0=1, site1=0 (LHSS=3, 2 bits/site).
        let manip = DynamicDitManip::new(3);
        let state: u32 = manip.set_dit(manip.set_dit(0u32, 1, 0), 0, 1);
        let (ref_s, _) = grp.get_refstate(state);
        // Image: site0=1→1, site1=0→2 — should be >= state
        assert!(ref_s >= state);
    }

    #[test]
    #[should_panic(expected = "lhss mismatch")]
    fn dit_lhss_mismatch_panics() {
        let el3 = DitGrpElement::new(
            Complex::new(1.0, 0.0),
            2,
            3,
            DitLocalOp::new_spin_inv(3, vec![0]),
        );
        let el4 = DitGrpElement::new(
            Complex::new(1.0, 0.0),
            2,
            4, // disagrees with el3
            DitLocalOp::new_spin_inv(4, vec![1]),
        );
        DitSymmetryGrp::new(vec![], vec![el3, el4]);
    }

    // --- GrpOpDesc ---

    #[test]
    fn grp_op_desc_bitflip_into_hardcore() {
        let desc = GrpOpDesc::Bitflip {
            n_sites: 4,
            locs: None,
        };
        let el: HardcoreGrpElement<u32> =
            desc.into_hardcore_element(Complex::new(1.0, 0.0)).unwrap();
        assert_eq!(el.n_sites, 4);
    }

    #[test]
    fn grp_op_desc_local_value_into_dit() {
        let desc = GrpOpDesc::LocalValue {
            n_sites: 3,
            lhss: 3,
            perm: vec![2, 1, 0],
            locs: vec![0, 1, 2],
        };
        let el = desc.into_dit_element(Complex::new(1.0, 0.0)).unwrap();
        assert_eq!(el.n_sites, 3);
        assert_eq!(el.lhss, 3);
    }

    #[test]
    fn grp_op_desc_wrong_variant_errors() {
        let desc = GrpOpDesc::SpinInversion {
            n_sites: 2,
            lhss: 3,
            locs: vec![0],
        };
        assert!(
            desc.into_hardcore_element::<u32>(Complex::new(1.0, 0.0))
                .is_err()
        );
    }
}
