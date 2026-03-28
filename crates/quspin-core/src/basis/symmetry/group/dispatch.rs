/// B-type dispatch for the LHSS = 2 (hardcore) symmetry group path.
///
/// - [`SymmetryGrpInner`]: type-erased enum over 9 concrete
///   `HardcoreSymmetryGrp<B>` types, selected by `n_sites` at construction.
/// - [`with_sym_grp!`]: match macro that injects the concrete `B` type alias
///   and a binding to the inner group, used by call-sites that need to be
///   generic over `B`.
use super::spin::HardcoreSymmetryGrp;
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
