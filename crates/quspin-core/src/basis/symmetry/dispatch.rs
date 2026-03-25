/// Type-erased `SymmetryGrpInner` and the `with_sym_grp!` dispatch macro.
///
/// `SymmetryGrpInner` wraps a `HardcoreSymmetryGrp<B>` for each supported basis
/// integer width, selected at construction time from `n_sites`.
use crate::basis::symmetry::group::{HardcoreSymmetryGrp, LatticeElement};
use num_complex::Complex;

type B128 = ruint::Uint<128, 2>;
type B256 = ruint::Uint<256, 4>;
type B512 = ruint::Uint<512, 8>;
type B1024 = ruint::Uint<1024, 16>;
type B2048 = ruint::Uint<2048, 32>;
type B4096 = ruint::Uint<4096, 64>;
type B8192 = ruint::Uint<8192, 128>;

// ---------------------------------------------------------------------------
// SymmetryGrpInner
// ---------------------------------------------------------------------------

/// Type-erased symmetry group: one of 9 concrete `HardcoreSymmetryGrp<B>` types.
///
/// The concrete `B` is selected based on `n_sites` at construction time.
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

// ---------------------------------------------------------------------------
// From impls — wrap a concrete SymmetryGrp<B> without naming the variant
// ---------------------------------------------------------------------------

macro_rules! impl_from_sym_grp {
    ($B:ty, $variant:ident) => {
        impl From<HardcoreSymmetryGrp<$B>> for SymmetryGrpInner {
            #[inline]
            fn from(g: HardcoreSymmetryGrp<$B>) -> Self {
                SymmetryGrpInner::$variant(g)
            }
        }
    };
}

impl_from_sym_grp!(u32, Sym32);
impl_from_sym_grp!(u64, Sym64);
impl_from_sym_grp!(B128, Sym128);
impl_from_sym_grp!(B256, Sym256);
impl_from_sym_grp!(B512, Sym512);
impl_from_sym_grp!(B1024, Sym1024);
impl_from_sym_grp!(B2048, Sym2048);
impl_from_sym_grp!(B4096, Sym4096);
impl_from_sym_grp!(B8192, Sym8192);

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

    pub(crate) fn push_lattice(&mut self, el: LatticeElement) {
        match self {
            SymmetryGrpInner::Sym32(g) => g.push_lattice(el),
            SymmetryGrpInner::Sym64(g) => g.push_lattice(el),
            SymmetryGrpInner::Sym128(g) => g.push_lattice(el),
            SymmetryGrpInner::Sym256(g) => g.push_lattice(el),
            SymmetryGrpInner::Sym512(g) => g.push_lattice(el),
            SymmetryGrpInner::Sym1024(g) => g.push_lattice(el),
            SymmetryGrpInner::Sym2048(g) => g.push_lattice(el),
            SymmetryGrpInner::Sym4096(g) => g.push_lattice(el),
            SymmetryGrpInner::Sym8192(g) => g.push_lattice(el),
        }
    }

    pub(crate) fn push_local_inv(&mut self, grp_char: Complex<f64>, locs: &[usize]) {
        match self {
            SymmetryGrpInner::Sym32(g) => g.push_local_inv(grp_char, locs),
            SymmetryGrpInner::Sym64(g) => g.push_local_inv(grp_char, locs),
            SymmetryGrpInner::Sym128(g) => g.push_local_inv(grp_char, locs),
            SymmetryGrpInner::Sym256(g) => g.push_local_inv(grp_char, locs),
            SymmetryGrpInner::Sym512(g) => g.push_local_inv(grp_char, locs),
            SymmetryGrpInner::Sym1024(g) => g.push_local_inv(grp_char, locs),
            SymmetryGrpInner::Sym2048(g) => g.push_local_inv(grp_char, locs),
            SymmetryGrpInner::Sym4096(g) => g.push_local_inv(grp_char, locs),
            SymmetryGrpInner::Sym8192(g) => g.push_local_inv(grp_char, locs),
        }
    }
}

// ---------------------------------------------------------------------------
// with_sym_grp! macro
// ---------------------------------------------------------------------------

/// Match on a `SymmetryGrpInner` reference, injecting a type alias `$B` and
/// binding `$grp` to the inner `SymmetryGrp<B>` reference.
#[macro_export]
macro_rules! with_sym_grp {
    ($inner:expr, $B:ident, $grp:ident, $body:block) => {
        match $inner {
            $crate::basis::symmetry::dispatch::SymmetryGrpInner::Sym32($grp) => {
                type $B = u32;
                $body
            }
            $crate::basis::symmetry::dispatch::SymmetryGrpInner::Sym64($grp) => {
                type $B = u64;
                $body
            }
            $crate::basis::symmetry::dispatch::SymmetryGrpInner::Sym128($grp) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::symmetry::dispatch::SymmetryGrpInner::Sym256($grp) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::symmetry::dispatch::SymmetryGrpInner::Sym512($grp) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::symmetry::dispatch::SymmetryGrpInner::Sym1024($grp) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::symmetry::dispatch::SymmetryGrpInner::Sym2048($grp) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::symmetry::dispatch::SymmetryGrpInner::Sym4096($grp) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::symmetry::dispatch::SymmetryGrpInner::Sym8192($grp) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
        }
    };
}
