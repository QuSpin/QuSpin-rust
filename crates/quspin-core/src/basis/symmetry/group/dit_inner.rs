/// Inner types for the dit symmetry group.
///
/// Covers LHSS > 2 only:
/// - LHSS 3–5: [`DitValueGrpInner<B, LHSS>`] — compile-time-LHSS value-perm hot-path.
/// - LHSS ≥ 6: [`DitValueGrpDyn<B>`] — runtime-LHSS value-perm fallback.
/// - [`DitSymGrpInner<B>`] — LHSS dispatch enum for value-permutation, generic over B.
/// - [`DitSymGrpInnerEnum`] — B-erased enum over [`DitSymGrpInner<B>`] variants.
use super::BenesLatticeElement;
use super::dispatch::{B128, B256, B512, B1024, B2048, B4096, B8192};
use crate::bitbasis::{BenesPermDitLocations, BitInt, DynamicPermDitValues, PermDitValues};
use num_complex::Complex;

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
