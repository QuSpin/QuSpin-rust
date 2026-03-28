/// Inner types for the spin symmetry group.
///
/// Covers the full LHSS range:
/// - LHSS = 2: [`HardcoreGrpElement`] / [`HardcoreSymmetryGrp<B>`] — XOR bit-flip ops.
/// - LHSS 3–5: [`DitSpinGrpInner<B, LHSS>`] — compile-time-LHSS spin-inversion hot-path.
/// - LHSS ≥ 6: [`DitSpinGrpDyn<B>`] — runtime-LHSS spin-inversion fallback.
/// - [`DitSpinSymGrpInner<B>`] — LHSS dispatch enum (LHSS > 2), generic over B.
/// - [`DitSpinSymGrpInnerEnum`] — B-erased enum over [`DitSpinSymGrpInner<B>`] variants.
use super::BenesLatticeElement;
use crate::basis::traits::SymGrp;
use crate::bitbasis::{
    BenesPermDitLocations, BitInt, BitStateOp, DynamicHigherSpinInv, HigherSpinInv, PermDitMask,
};
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
    lattice: Vec<BenesLatticeElement<B>>,
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

    /// Add a lattice (site-permutation) symmetry element backed by a Benes network.
    ///
    /// `fermionic=true` enables Jordan-Wigner sign tracking.
    pub fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize], fermionic: bool) {
        let op = BenesPermDitLocations::<B>::new(2, perm, fermionic);
        self.lattice
            .push(BenesLatticeElement::new(grp_char, op, self.n_sites));
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

    /// Batch variant: maps every element of `states` to its orbit
    /// representative and accumulated group character in a single pass with
    /// the orbit loop amortised across the batch.
    pub fn get_refstate_batch(&self, states: &[B], out: &mut [(B, Complex<f64>)]) {
        super::orbit::get_refstate_batch(&self.lattice, &self.local, states, out);
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

    fn get_refstate_batch(&self, states: &[B], out: &mut [(B, Complex<f64>)]) {
        HardcoreSymmetryGrp::get_refstate_batch(self, states, out);
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
pub(crate) struct DitSpinGrpInner<B: BitInt, const LHSS: usize> {
    n_sites: usize,
    lattice: Vec<BenesLatticeElement<B>>,
    local: Vec<(Complex<f64>, HigherSpinInv<LHSS>)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl<B: BitInt, const LHSS: usize> DitSpinGrpInner<B, LHSS> {
    pub(crate) fn new_empty(n_sites: usize) -> Self {
        DitSpinGrpInner {
            n_sites,
            lattice: Vec::new(),
            local: Vec::new(),
        }
    }

    pub(crate) fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        let op = BenesPermDitLocations::<B>::new(LHSS, perm, false);
        self.lattice
            .push(BenesLatticeElement::new(grp_char, op, self.n_sites));
    }

    pub(crate) fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        self.local
            .push((grp_char, HigherSpinInv::<LHSS>::new(locs)));
    }

    pub(crate) fn n_sites(&self) -> usize {
        self.n_sites
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        super::get_refstate(&self.lattice, &self.local, state)
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn check_refstate(&self, state: B) -> (B, f64) {
        super::check_refstate(&self.lattice, &self.local, state)
    }
}

// ---------------------------------------------------------------------------
// DitSpinGrpDyn — runtime-LHSS spin-inversion fallback (LHSS ≥ 6)
// ---------------------------------------------------------------------------

#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) struct DitSpinGrpDyn<B: BitInt> {
    n_sites: usize,
    lhss: usize,
    lattice: Vec<BenesLatticeElement<B>>,
    local: Vec<(Complex<f64>, DynamicHigherSpinInv)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl<B: BitInt> DitSpinGrpDyn<B> {
    pub(crate) fn new_empty(lhss: usize, n_sites: usize) -> Self {
        DitSpinGrpDyn {
            n_sites,
            lhss,
            lattice: Vec::new(),
            local: Vec::new(),
        }
    }

    pub(crate) fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        let op = BenesPermDitLocations::<B>::new(self.lhss, perm, false);
        self.lattice
            .push(BenesLatticeElement::new(grp_char, op, self.n_sites));
    }

    pub(crate) fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        self.local
            .push((grp_char, DynamicHigherSpinInv::new(self.lhss, locs)));
    }

    pub(crate) fn n_sites(&self) -> usize {
        self.n_sites
    }

    pub(crate) fn lhss(&self) -> usize {
        self.lhss
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        super::get_refstate(&self.lattice, &self.local, state)
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn check_refstate(&self, state: B) -> (B, f64) {
        super::check_refstate(&self.lattice, &self.local, state)
    }
}

// ---------------------------------------------------------------------------
// DitSpinSymGrpInner — LHSS dispatch enum for spin-inversion (LHSS > 2), generic over B
// ---------------------------------------------------------------------------

#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) enum DitSpinSymGrpInner<B: BitInt> {
    Lhss3(DitSpinGrpInner<B, 3>),
    Lhss4(DitSpinGrpInner<B, 4>),
    Lhss5(DitSpinGrpInner<B, 5>),
    LhssDyn(DitSpinGrpDyn<B>),
}

#[allow(dead_code)] // dit basis not yet implemented
impl<B: BitInt> DitSpinSymGrpInner<B> {
    pub(crate) fn new_empty(lhss: usize, n_sites: usize) -> Self {
        match lhss {
            3 => DitSpinSymGrpInner::Lhss3(DitSpinGrpInner::<B, 3>::new_empty(n_sites)),
            4 => DitSpinSymGrpInner::Lhss4(DitSpinGrpInner::<B, 4>::new_empty(n_sites)),
            5 => DitSpinSymGrpInner::Lhss5(DitSpinGrpInner::<B, 5>::new_empty(n_sites)),
            _ => DitSpinSymGrpInner::LhssDyn(DitSpinGrpDyn::<B>::new_empty(lhss, n_sites)),
        }
    }

    pub(crate) fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        match self {
            DitSpinSymGrpInner::Lhss3(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInner::Lhss4(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInner::Lhss5(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInner::LhssDyn(g) => g.push_lattice(grp_char, perm),
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

    pub(crate) fn n_sites(&self) -> usize {
        match self {
            DitSpinSymGrpInner::Lhss3(g) => g.n_sites(),
            DitSpinSymGrpInner::Lhss4(g) => g.n_sites(),
            DitSpinSymGrpInner::Lhss5(g) => g.n_sites(),
            DitSpinSymGrpInner::LhssDyn(g) => g.n_sites(),
        }
    }

    pub(crate) fn lhss(&self) -> usize {
        match self {
            DitSpinSymGrpInner::Lhss3(_) => 3,
            DitSpinSymGrpInner::Lhss4(_) => 4,
            DitSpinSymGrpInner::Lhss5(_) => 5,
            DitSpinSymGrpInner::LhssDyn(g) => g.lhss(),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        match self {
            DitSpinSymGrpInner::Lhss3(g) => g.get_refstate(state),
            DitSpinSymGrpInner::Lhss4(g) => g.get_refstate(state),
            DitSpinSymGrpInner::Lhss5(g) => g.get_refstate(state),
            DitSpinSymGrpInner::LhssDyn(g) => g.get_refstate(state),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn check_refstate(&self, state: B) -> (B, f64) {
        match self {
            DitSpinSymGrpInner::Lhss3(g) => g.check_refstate(state),
            DitSpinSymGrpInner::Lhss4(g) => g.check_refstate(state),
            DitSpinSymGrpInner::Lhss5(g) => g.check_refstate(state),
            DitSpinSymGrpInner::LhssDyn(g) => g.check_refstate(state),
        }
    }
}

// ---------------------------------------------------------------------------
// DitSpinSymGrpInnerEnum — B-erased enum over DitSpinSymGrpInner<B> variants
// ---------------------------------------------------------------------------

use super::dispatch::{B128, B256, B512, B1024, B2048, B4096, B8192};

#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) enum DitSpinSymGrpInnerEnum {
    B32(DitSpinSymGrpInner<u32>),
    B64(DitSpinSymGrpInner<u64>),
    B128(DitSpinSymGrpInner<B128>),
    B256(DitSpinSymGrpInner<B256>),
    B512(DitSpinSymGrpInner<B512>),
    B1024(DitSpinSymGrpInner<B1024>),
    B2048(DitSpinSymGrpInner<B2048>),
    B4096(DitSpinSymGrpInner<B4096>),
    B8192(DitSpinSymGrpInner<B8192>),
}

macro_rules! impl_from_dit_spin_sym_grp_inner {
    ($B:ty, $variant:ident) => {
        impl From<DitSpinSymGrpInner<$B>> for DitSpinSymGrpInnerEnum {
            #[inline]
            fn from(g: DitSpinSymGrpInner<$B>) -> Self {
                DitSpinSymGrpInnerEnum::$variant(g)
            }
        }
    };
}

impl_from_dit_spin_sym_grp_inner!(u32, B32);
impl_from_dit_spin_sym_grp_inner!(u64, B64);
impl_from_dit_spin_sym_grp_inner!(B128, B128);
impl_from_dit_spin_sym_grp_inner!(B256, B256);
impl_from_dit_spin_sym_grp_inner!(B512, B512);
impl_from_dit_spin_sym_grp_inner!(B1024, B1024);
impl_from_dit_spin_sym_grp_inner!(B2048, B2048);
impl_from_dit_spin_sym_grp_inner!(B4096, B4096);
impl_from_dit_spin_sym_grp_inner!(B8192, B8192);

#[allow(dead_code)] // dit basis not yet implemented
impl DitSpinSymGrpInnerEnum {
    pub(crate) fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        match self {
            DitSpinSymGrpInnerEnum::B32(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInnerEnum::B64(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInnerEnum::B128(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInnerEnum::B256(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInnerEnum::B512(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInnerEnum::B1024(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInnerEnum::B2048(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInnerEnum::B4096(g) => g.push_lattice(grp_char, perm),
            DitSpinSymGrpInnerEnum::B8192(g) => g.push_lattice(grp_char, perm),
        }
    }

    pub(crate) fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        match self {
            DitSpinSymGrpInnerEnum::B32(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInnerEnum::B64(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInnerEnum::B128(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInnerEnum::B256(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInnerEnum::B512(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInnerEnum::B1024(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInnerEnum::B2048(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInnerEnum::B4096(g) => g.push_spin_inv(grp_char, locs),
            DitSpinSymGrpInnerEnum::B8192(g) => g.push_spin_inv(grp_char, locs),
        }
    }

    pub(crate) fn n_sites(&self) -> usize {
        match self {
            DitSpinSymGrpInnerEnum::B32(g) => g.n_sites(),
            DitSpinSymGrpInnerEnum::B64(g) => g.n_sites(),
            DitSpinSymGrpInnerEnum::B128(g) => g.n_sites(),
            DitSpinSymGrpInnerEnum::B256(g) => g.n_sites(),
            DitSpinSymGrpInnerEnum::B512(g) => g.n_sites(),
            DitSpinSymGrpInnerEnum::B1024(g) => g.n_sites(),
            DitSpinSymGrpInnerEnum::B2048(g) => g.n_sites(),
            DitSpinSymGrpInnerEnum::B4096(g) => g.n_sites(),
            DitSpinSymGrpInnerEnum::B8192(g) => g.n_sites(),
        }
    }

    pub(crate) fn lhss(&self) -> usize {
        match self {
            DitSpinSymGrpInnerEnum::B32(g) => g.lhss(),
            DitSpinSymGrpInnerEnum::B64(g) => g.lhss(),
            DitSpinSymGrpInnerEnum::B128(g) => g.lhss(),
            DitSpinSymGrpInnerEnum::B256(g) => g.lhss(),
            DitSpinSymGrpInnerEnum::B512(g) => g.lhss(),
            DitSpinSymGrpInnerEnum::B1024(g) => g.lhss(),
            DitSpinSymGrpInnerEnum::B2048(g) => g.lhss(),
            DitSpinSymGrpInnerEnum::B4096(g) => g.lhss(),
            DitSpinSymGrpInnerEnum::B8192(g) => g.lhss(),
        }
    }
}
