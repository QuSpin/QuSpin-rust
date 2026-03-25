use crate::bitbasis::{
    BitInt, BitStateOp, DynamicHigherSpinInv, DynamicPermDitValues, PermDitLocations, PermDitMask,
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
// DitLocalOp / DitGrpElement / DitSymmetryGrp  (LHSS > 2)
// ---------------------------------------------------------------------------

/// Local symmetry operation for a dit basis with LHSS > 2.
///
/// Does **not** carry a `B` type parameter: the dynamic operation types
/// (`DynamicPermDitValues`, `DynamicHigherSpinInv`) are independent of the
/// basis integer width, so `B` is deferred to call time.
#[derive(Clone)]
pub enum DitLocalOp {
    /// Permutation of dit values: `v → perm[v]` at specified sites.
    Value(DynamicPermDitValues),
    /// Spin inversion: `v → lhss - v - 1` at specified sites.
    SpinInversion(DynamicHigherSpinInv),
}

impl<I: BitInt> BitStateOp<I> for DitLocalOp {
    #[inline]
    fn apply(&self, state: I) -> I {
        match self {
            DitLocalOp::Value(op) => op.apply(state),
            DitLocalOp::SpinInversion(op) => op.apply(state),
        }
    }
}

/// A local symmetry element for a dit basis with LHSS > 2.
///
/// `B` does not appear on this type — it is only needed when `apply` is called.
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

/// A symmetry group for dit bases with LHSS > 2.
///
/// Unlike `HardcoreSymmetryGrp<B>`, this type carries **no `B` type
/// parameter**: the integer width is only needed when `get_refstate` and
/// `check_refstate` are called.  This makes `DitSymmetryGrp` easy to store
/// and pass around without monomorphisation overhead at the group level.
#[derive(Clone)]
pub struct DitSymmetryGrp {
    n_sites: usize,
    lhss: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<DitGrpElement>,
}

impl DitSymmetryGrp {
    /// Construct, validating that all elements agree on `n_sites` and `lhss`.
    pub fn new(
        lattice: Vec<LatticeElement>,
        local: Vec<DitGrpElement>,
    ) -> Result<Self, QuSpinError> {
        let mut n_sites_opt: Option<usize> = None;
        let mut lhss_opt: Option<usize> = None;

        for n in lattice.iter().map(|e| e.n_sites) {
            match n_sites_opt {
                None => n_sites_opt = Some(n),
                Some(existing) if existing != n => {
                    return Err(QuSpinError::ValueError(format!(
                        "n_sites mismatch: lattice element has {n} but expected {existing}"
                    )));
                }
                _ => {}
            }
        }
        for el in &local {
            match n_sites_opt {
                None => n_sites_opt = Some(el.n_sites),
                Some(existing) if existing != el.n_sites => {
                    return Err(QuSpinError::ValueError(format!(
                        "n_sites mismatch: local element has {} but expected {existing}",
                        el.n_sites
                    )));
                }
                _ => {}
            }
            match lhss_opt {
                None => lhss_opt = Some(el.lhss),
                Some(existing) if existing != el.lhss => {
                    return Err(QuSpinError::ValueError(format!(
                        "lhss mismatch: local element has {} but expected {existing}",
                        el.lhss
                    )));
                }
                _ => {}
            }
        }

        Ok(DitSymmetryGrp {
            n_sites: n_sites_opt.unwrap_or(0),
            lhss: lhss_opt.unwrap_or(2),
            lattice,
            local,
        })
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    pub fn lhss(&self) -> usize {
        self.lhss
    }

    fn iter_images<B: BitInt>(&self, state: B) -> Vec<(B, Complex<f64>)> {
        let one = Complex::new(1.0, 0.0);
        let mut images =
            Vec::with_capacity(self.lattice.len() + self.lattice.len() * self.local.len());
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

    pub fn get_refstate<B: BitInt>(&self, state: B) -> (B, Complex<f64>) {
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

    pub fn check_refstate<B: BitInt>(&self, state: B) -> (B, f64) {
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
                DitLocalOp::Value(DynamicPermDitValues::new(lhss, perm, locs)),
            )),
            GrpOpDesc::SpinInversion {
                n_sites,
                lhss,
                locs,
            } => Ok(DitGrpElement::new(
                grp_char,
                n_sites,
                lhss,
                DitLocalOp::SpinInversion(DynamicHigherSpinInv::new(lhss, locs)),
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
            DitLocalOp::SpinInversion(DynamicHigherSpinInv::new(3, vec![0, 1])),
        );
        let grp = DitSymmetryGrp::new(vec![id], vec![inv_el]).unwrap();
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
    fn dit_lhss_mismatch_errors() {
        let el3 = DitGrpElement::new(
            Complex::new(1.0, 0.0),
            2,
            3,
            DitLocalOp::SpinInversion(DynamicHigherSpinInv::new(3, vec![0])),
        );
        let el4 = DitGrpElement::new(
            Complex::new(1.0, 0.0),
            2,
            4, // disagrees with el3
            DitLocalOp::SpinInversion(DynamicHigherSpinInv::new(4, vec![1])),
        );
        assert!(DitSymmetryGrp::new(vec![], vec![el3, el4]).is_err());
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
