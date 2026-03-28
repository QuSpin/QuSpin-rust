use super::BenesLatticeElement;
use crate::basis::traits::SymGrp;
use crate::bitbasis::{
    BenesPermDitLocations, BitInt, BitStateOp, DynamicHigherSpinInv, DynamicPermDitValues,
    HigherSpinInv, PermDitMask, PermDitValues,
};
/// B-type dispatch for the LHSS = 2 (hardcore) symmetry group path.
///
/// - [`SymmetryGrpInner`]: type-erased enum over 9 concrete
///   `HardcoreSymmetryGrp<B>` types, selected by `n_sites` at construction.
/// - [`with_sym_grp!`]: match macro that injects the concrete `B` type alias
///   and a binding to the inner group, used by call-sites that need to be
///   generic over `B`.
use num_complex::Complex;

// ---------------------------------------------------------------------------
// ruint type aliases
// ---------------------------------------------------------------------------

pub(crate) type B128 = ruint::Uint<128, 2>;
pub(crate) type B256 = ruint::Uint<256, 4>;
pub(crate) type B512 = ruint::Uint<512, 8>;
pub(crate) type B1024 = ruint::Uint<1024, 16>;
pub(crate) type B2048 = ruint::Uint<2048, 32>;
pub(crate) type B4096 = ruint::Uint<4096, 64>;
pub(crate) type B8192 = ruint::Uint<8192, 128>;

// ---------------------------------------------------------------------------
// SymmetryGrpInner
// ---------------------------------------------------------------------------

/// Type-erased symmetry group: one of 9 concrete `HardcoreSymmetryGrp<B>` types.
///
/// The concrete `B` is selected based on `n_sites` at construction time inside
/// [`SpinSymGrp::new`](super::spin::SpinSymGrp::new).
#[derive(Clone)]
pub enum SymmetryGrpInner {
    Sym32(HardcoreSymmetryGrp<u32>),
    Sym64(HardcoreSymmetryGrp<u64>),
    Sym128(HardcoreSymmetryGrp<B128>),
    Sym256(HardcoreSymmetryGrp<B256>),
    Sym512(HardcoreSymmetryGrp<B512>),
    Sym1024(HardcoreSymmetryGrp<B1024>),
    Sym2048(HardcoreSymmetryGrp<B2048>),
    Sym4096(HardcoreSymmetryGrp<B4096>),
    Sym8192(HardcoreSymmetryGrp<B8192>),
}

macro_rules! impl_from_hardcore_grp {
    ($B:ty, $variant:ident) => {
        impl From<HardcoreSymmetryGrp<$B>> for SymmetryGrpInner {
            #[inline]
            fn from(g: HardcoreSymmetryGrp<$B>) -> Self {
                SymmetryGrpInner::$variant(g)
            }
        }
    };
}

impl_from_hardcore_grp!(u32, Sym32);
impl_from_hardcore_grp!(u64, Sym64);
impl_from_hardcore_grp!(B128, Sym128);
impl_from_hardcore_grp!(B256, Sym256);
impl_from_hardcore_grp!(B512, Sym512);
impl_from_hardcore_grp!(B1024, Sym1024);
impl_from_hardcore_grp!(B2048, Sym2048);
impl_from_hardcore_grp!(B4096, Sym4096);
impl_from_hardcore_grp!(B8192, Sym8192);

impl SymmetryGrpInner {
    pub fn n_sites(&self) -> usize {
        match self {
            SymmetryGrpInner::Sym32(g) => g.n_sites(),
            SymmetryGrpInner::Sym64(g) => g.n_sites(),
            SymmetryGrpInner::Sym128(g) => g.n_sites(),
            SymmetryGrpInner::Sym256(g) => g.n_sites(),
            SymmetryGrpInner::Sym512(g) => g.n_sites(),
            SymmetryGrpInner::Sym1024(g) => g.n_sites(),
            SymmetryGrpInner::Sym2048(g) => g.n_sites(),
            SymmetryGrpInner::Sym4096(g) => g.n_sites(),
            SymmetryGrpInner::Sym8192(g) => g.n_sites(),
        }
    }

    pub(crate) fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize], fermionic: bool) {
        match self {
            SymmetryGrpInner::Sym32(g) => g.push_lattice(grp_char, perm, fermionic),
            SymmetryGrpInner::Sym64(g) => g.push_lattice(grp_char, perm, fermionic),
            SymmetryGrpInner::Sym128(g) => g.push_lattice(grp_char, perm, fermionic),
            SymmetryGrpInner::Sym256(g) => g.push_lattice(grp_char, perm, fermionic),
            SymmetryGrpInner::Sym512(g) => g.push_lattice(grp_char, perm, fermionic),
            SymmetryGrpInner::Sym1024(g) => g.push_lattice(grp_char, perm, fermionic),
            SymmetryGrpInner::Sym2048(g) => g.push_lattice(grp_char, perm, fermionic),
            SymmetryGrpInner::Sym4096(g) => g.push_lattice(grp_char, perm, fermionic),
            SymmetryGrpInner::Sym8192(g) => g.push_lattice(grp_char, perm, fermionic),
        }
    }

    pub(crate) fn push_inverse(&mut self, grp_char: Complex<f64>, locs: &[usize]) {
        match self {
            SymmetryGrpInner::Sym32(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Sym64(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Sym128(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Sym256(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Sym512(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Sym1024(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Sym2048(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Sym4096(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Sym8192(g) => g.push_inverse(grp_char, locs),
        }
    }
}

// ---------------------------------------------------------------------------
// with_sym_grp! macro
// ---------------------------------------------------------------------------

/// Match on a [`SymmetryGrpInner`] reference, injecting a type alias `$B` and
/// binding `$grp` to the inner `HardcoreSymmetryGrp<B>` reference.
#[macro_export]
macro_rules! with_sym_grp {
    ($inner:expr, $B:ident, $grp:ident, $body:block) => {
        match $inner {
            $crate::basis::symmetry::group::dispatch::SymmetryGrpInner::Sym32($grp) => {
                type $B = u32;
                $body
            }
            $crate::basis::symmetry::group::dispatch::SymmetryGrpInner::Sym64($grp) => {
                type $B = u64;
                $body
            }
            $crate::basis::symmetry::group::dispatch::SymmetryGrpInner::Sym128($grp) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::symmetry::group::dispatch::SymmetryGrpInner::Sym256($grp) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::symmetry::group::dispatch::SymmetryGrpInner::Sym512($grp) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::symmetry::group::dispatch::SymmetryGrpInner::Sym1024($grp) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::symmetry::group::dispatch::SymmetryGrpInner::Sym2048($grp) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::symmetry::group::dispatch::SymmetryGrpInner::Sym4096($grp) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::symmetry::group::dispatch::SymmetryGrpInner::Sym8192($grp) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
        }
    };
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

// ---------------------------------------------------------------------------
// DitValueGrpInner — compile-time-LHSS value-permutation (LHSS 3–5)
// ---------------------------------------------------------------------------

// Dit value-perm infrastructure is forward-looking (no dit basis yet). Allow
// dead_code until DitSymmetricSubspace is implemented.
#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) struct DitValueGrpInner<B: BitInt, const LHSS: usize> {
    n_sites: usize,
    lattice: Vec<BenesLatticeElement<B>>,
    local: Vec<(Complex<f64>, PermDitValues<LHSS>)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl<B: BitInt, const LHSS: usize> DitValueGrpInner<B, LHSS> {
    pub(crate) fn new_empty(n_sites: usize) -> Self {
        DitValueGrpInner {
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

    pub(crate) fn push_dit_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) {
        let arr: [u8; LHSS] = perm.try_into().expect("perm length must match LHSS");
        self.local
            .push((grp_char, PermDitValues::<LHSS>::new(arr, locs)));
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
// DitValueGrpDyn — runtime-LHSS value-permutation fallback (LHSS ≥ 6)
// ---------------------------------------------------------------------------

#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) struct DitValueGrpDyn<B: BitInt> {
    n_sites: usize,
    lhss: usize,
    lattice: Vec<BenesLatticeElement<B>>,
    local: Vec<(Complex<f64>, DynamicPermDitValues)>,
}

#[allow(dead_code)] // dit basis not yet implemented
impl<B: BitInt> DitValueGrpDyn<B> {
    pub(crate) fn new_empty(lhss: usize, n_sites: usize) -> Self {
        DitValueGrpDyn {
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

    pub(crate) fn push_dit_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) {
        self.local
            .push((grp_char, DynamicPermDitValues::new(self.lhss, perm, locs)));
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
// DitSymGrpInner — LHSS dispatch enum for value-permutation (LHSS > 2), generic over B
// ---------------------------------------------------------------------------

#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) enum DitSymGrpInner<B: BitInt> {
    Lhss3(DitValueGrpInner<B, 3>),
    Lhss4(DitValueGrpInner<B, 4>),
    Lhss5(DitValueGrpInner<B, 5>),
    LhssDyn(DitValueGrpDyn<B>),
}

#[allow(dead_code)] // dit basis not yet implemented
impl<B: BitInt> DitSymGrpInner<B> {
    pub(crate) fn new_empty(lhss: usize, n_sites: usize) -> Self {
        match lhss {
            3 => DitSymGrpInner::Lhss3(DitValueGrpInner::<B, 3>::new_empty(n_sites)),
            4 => DitSymGrpInner::Lhss4(DitValueGrpInner::<B, 4>::new_empty(n_sites)),
            5 => DitSymGrpInner::Lhss5(DitValueGrpInner::<B, 5>::new_empty(n_sites)),
            _ => DitSymGrpInner::LhssDyn(DitValueGrpDyn::<B>::new_empty(lhss, n_sites)),
        }
    }

    pub(crate) fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        match self {
            DitSymGrpInner::Lhss3(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInner::Lhss4(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInner::Lhss5(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInner::LhssDyn(g) => g.push_lattice(grp_char, perm),
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

    pub(crate) fn n_sites(&self) -> usize {
        match self {
            DitSymGrpInner::Lhss3(g) => g.n_sites(),
            DitSymGrpInner::Lhss4(g) => g.n_sites(),
            DitSymGrpInner::Lhss5(g) => g.n_sites(),
            DitSymGrpInner::LhssDyn(g) => g.n_sites(),
        }
    }

    pub(crate) fn lhss(&self) -> usize {
        match self {
            DitSymGrpInner::Lhss3(_) => 3,
            DitSymGrpInner::Lhss4(_) => 4,
            DitSymGrpInner::Lhss5(_) => 5,
            DitSymGrpInner::LhssDyn(g) => g.lhss(),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        match self {
            DitSymGrpInner::Lhss3(g) => g.get_refstate(state),
            DitSymGrpInner::Lhss4(g) => g.get_refstate(state),
            DitSymGrpInner::Lhss5(g) => g.get_refstate(state),
            DitSymGrpInner::LhssDyn(g) => g.get_refstate(state),
        }
    }

    #[allow(dead_code)] // dit basis not yet implemented
    pub(crate) fn check_refstate(&self, state: B) -> (B, f64) {
        match self {
            DitSymGrpInner::Lhss3(g) => g.check_refstate(state),
            DitSymGrpInner::Lhss4(g) => g.check_refstate(state),
            DitSymGrpInner::Lhss5(g) => g.check_refstate(state),
            DitSymGrpInner::LhssDyn(g) => g.check_refstate(state),
        }
    }
}

// ---------------------------------------------------------------------------
// DitSymGrpInnerEnum — B-erased enum over DitSymGrpInner<B> variants
// ---------------------------------------------------------------------------

#[allow(dead_code)] // dit basis not yet implemented
#[derive(Clone)]
pub(crate) enum DitSymGrpInnerEnum {
    B32(DitSymGrpInner<u32>),
    B64(DitSymGrpInner<u64>),
    B128(DitSymGrpInner<B128>),
    B256(DitSymGrpInner<B256>),
    B512(DitSymGrpInner<B512>),
    B1024(DitSymGrpInner<B1024>),
    B2048(DitSymGrpInner<B2048>),
    B4096(DitSymGrpInner<B4096>),
    B8192(DitSymGrpInner<B8192>),
}

macro_rules! impl_from_dit_sym_grp_inner {
    ($B:ty, $variant:ident) => {
        impl From<DitSymGrpInner<$B>> for DitSymGrpInnerEnum {
            #[inline]
            fn from(g: DitSymGrpInner<$B>) -> Self {
                DitSymGrpInnerEnum::$variant(g)
            }
        }
    };
}

impl_from_dit_sym_grp_inner!(u32, B32);
impl_from_dit_sym_grp_inner!(u64, B64);
impl_from_dit_sym_grp_inner!(B128, B128);
impl_from_dit_sym_grp_inner!(B256, B256);
impl_from_dit_sym_grp_inner!(B512, B512);
impl_from_dit_sym_grp_inner!(B1024, B1024);
impl_from_dit_sym_grp_inner!(B2048, B2048);
impl_from_dit_sym_grp_inner!(B4096, B4096);
impl_from_dit_sym_grp_inner!(B8192, B8192);

#[allow(dead_code)] // dit basis not yet implemented
impl DitSymGrpInnerEnum {
    pub(crate) fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        match self {
            DitSymGrpInnerEnum::B32(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInnerEnum::B64(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInnerEnum::B128(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInnerEnum::B256(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInnerEnum::B512(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInnerEnum::B1024(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInnerEnum::B2048(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInnerEnum::B4096(g) => g.push_lattice(grp_char, perm),
            DitSymGrpInnerEnum::B8192(g) => g.push_lattice(grp_char, perm),
        }
    }

    pub(crate) fn push_dit_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) {
        match self {
            DitSymGrpInnerEnum::B32(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInnerEnum::B64(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInnerEnum::B128(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInnerEnum::B256(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInnerEnum::B512(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInnerEnum::B1024(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInnerEnum::B2048(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInnerEnum::B4096(g) => g.push_dit_perm(grp_char, perm, locs),
            DitSymGrpInnerEnum::B8192(g) => g.push_dit_perm(grp_char, perm, locs),
        }
    }

    pub(crate) fn n_sites(&self) -> usize {
        match self {
            DitSymGrpInnerEnum::B32(g) => g.n_sites(),
            DitSymGrpInnerEnum::B64(g) => g.n_sites(),
            DitSymGrpInnerEnum::B128(g) => g.n_sites(),
            DitSymGrpInnerEnum::B256(g) => g.n_sites(),
            DitSymGrpInnerEnum::B512(g) => g.n_sites(),
            DitSymGrpInnerEnum::B1024(g) => g.n_sites(),
            DitSymGrpInnerEnum::B2048(g) => g.n_sites(),
            DitSymGrpInnerEnum::B4096(g) => g.n_sites(),
            DitSymGrpInnerEnum::B8192(g) => g.n_sites(),
        }
    }

    pub(crate) fn lhss(&self) -> usize {
        match self {
            DitSymGrpInnerEnum::B32(g) => g.lhss(),
            DitSymGrpInnerEnum::B64(g) => g.lhss(),
            DitSymGrpInnerEnum::B128(g) => g.lhss(),
            DitSymGrpInnerEnum::B256(g) => g.lhss(),
            DitSymGrpInnerEnum::B512(g) => g.lhss(),
            DitSymGrpInnerEnum::B1024(g) => g.lhss(),
            DitSymGrpInnerEnum::B2048(g) => g.lhss(),
            DitSymGrpInnerEnum::B4096(g) => g.lhss(),
            DitSymGrpInnerEnum::B8192(g) => g.lhss(),
        }
    }
}
