use super::BenesLatticeElement;
use crate::basis::traits::SymGrp;
use crate::bitbasis::{BenesPermDitLocations, BitInt, DynamicPermDitValues};
/// B-type dispatch for the unified symmetry group path.
///
/// - [`SymGrpInner<B>`]: single concrete generic type backing all group
///   variants (hardcore, dit, fermionic).  Carries `lhss`, `fermionic`, and
///   two element lists.
/// - [`SymmetryGrpInner`]: type-erased enum over 9 concrete `SymGrpInner<B>`
///   types, selected by `n_bits` at construction.
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
// SymGrpInner<B> — unified inner group type
// ---------------------------------------------------------------------------

/// Unified inner symmetry group.
///
/// Replaces `HardcoreSymmetryGrp<B>`, `DitValueGrpInner<B,LHSS>`,
/// `DitValueGrpDyn<B>`, `DitSymGrpInner<B>`, and `DitSymGrpInnerEnum`.
///
/// The `lhss` field distinguishes hardcore (2) from dit (≥3).
/// The `fermionic` flag enables Jordan-Wigner sign tracking for lattice ops.
#[derive(Clone)]
pub struct SymGrpInner<B: BitInt> {
    lhss: usize,
    fermionic: bool,
    n_sites: usize,
    lattice: Vec<BenesLatticeElement<B>>,
    local: Vec<(Complex<f64>, DynamicPermDitValues)>,
}

impl<B: BitInt> SymGrpInner<B> {
    /// Construct an empty group.
    pub fn new_empty(lhss: usize, n_sites: usize, fermionic: bool) -> Self {
        SymGrpInner {
            lhss,
            fermionic,
            n_sites,
            lattice: Vec::new(),
            local: Vec::new(),
        }
    }

    /// Number of lattice sites.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Local Hilbert-space size.
    pub fn lhss(&self) -> usize {
        self.lhss
    }

    /// Add a lattice (site-permutation) symmetry element backed by a Benes network.
    ///
    /// The `fermionic` flag stored in `self` propagates to the underlying
    /// `BenesPermDitLocations`, enabling Jordan-Wigner sign tracking.
    pub fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        let op = BenesPermDitLocations::<B>::new(self.lhss, perm, self.fermionic);
        self.lattice
            .push(BenesLatticeElement::new(grp_char, op, self.n_sites));
    }

    /// Add a local value-permutation element.
    ///
    /// `perm[v] = w` maps local occupation `v` to `w` at each site in `locs`.
    pub fn push_local_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        self.local
            .push((grp_char, DynamicPermDitValues::new(self.lhss, perm, locs)));
    }

    /// Add a spin-inversion element: maps `v → lhss − v − 1` at each site in `locs`.
    ///
    /// For LHSS = 2 this is the same as a bit-flip.
    pub fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        let perm: Vec<u8> = (0..self.lhss).rev().map(|v| v as u8).collect();
        self.push_local_perm(grp_char, perm, locs);
    }

    /// Add a spin-inversion element at all sites in `locs`.
    ///
    /// For LHSS = 2 this matches the old `push_inverse` bit-XOR semantics but
    /// is implemented via `DynamicPermDitValues` for uniformity.
    #[allow(dead_code)]
    pub fn push_inverse(&mut self, grp_char: Complex<f64>, locs: &[usize]) {
        self.push_spin_inv(grp_char, locs.to_vec());
    }

    pub fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        super::get_refstate(&self.lattice, &self.local, state)
    }

    pub fn check_refstate(&self, state: B) -> (B, f64) {
        super::check_refstate(&self.lattice, &self.local, state)
    }

    /// Batch variant: maps every element of `states` to its orbit
    /// representative and accumulated group character in a single pass.
    pub fn get_refstate_batch(&self, states: &[B], out: &mut [(B, Complex<f64>)]) {
        super::orbit::get_refstate_batch(&self.lattice, &self.local, states, out);
    }

    /// Batch variant: computes `check_refstate` for every element of `states`.
    pub fn check_refstate_batch(&self, states: &[B], out: &mut [(B, f64)]) {
        super::orbit::check_refstate_batch(&self.lattice, &self.local, states, out);
    }
}

// ---------------------------------------------------------------------------
// SymGrp impl for SymGrpInner<B>
// ---------------------------------------------------------------------------

impl<B: BitInt> SymGrp for SymGrpInner<B> {
    type State = B;

    fn n_sites(&self) -> usize {
        SymGrpInner::n_sites(self)
    }

    fn get_refstate(&self, state: B) -> (B, num_complex::Complex<f64>) {
        SymGrpInner::get_refstate(self, state)
    }

    fn get_refstate_batch(&self, states: &[B], out: &mut [(B, Complex<f64>)]) {
        SymGrpInner::get_refstate_batch(self, states, out);
    }

    fn check_refstate(&self, state: B) -> (B, f64) {
        SymGrpInner::check_refstate(self, state)
    }

    fn check_refstate_batch(&self, states: &[B], out: &mut [(B, f64)]) {
        SymGrpInner::check_refstate_batch(self, states, out);
    }
}

// ---------------------------------------------------------------------------
// SymmetryGrpInner
// ---------------------------------------------------------------------------

/// Type-erased symmetry group: one of 9 concrete `SymGrpInner<B>` types.
///
/// The concrete `B` is selected based on `n_bits` (= `n_sites * bits_per_dit`)
/// at construction time inside the public wrappers (`SpinSymGrp`, `DitSymGrp`,
/// `FermionicSymGrp`).
#[derive(Clone)]
pub enum SymmetryGrpInner {
    Sym32(SymGrpInner<u32>),
    Sym64(SymGrpInner<u64>),
    Sym128(SymGrpInner<B128>),
    Sym256(SymGrpInner<B256>),
    Sym512(SymGrpInner<B512>),
    Sym1024(SymGrpInner<B1024>),
    Sym2048(SymGrpInner<B2048>),
    Sym4096(SymGrpInner<B4096>),
    Sym8192(SymGrpInner<B8192>),
}

macro_rules! impl_from_sym_grp_inner {
    ($B:ty, $variant:ident) => {
        impl From<SymGrpInner<$B>> for SymmetryGrpInner {
            #[inline]
            fn from(g: SymGrpInner<$B>) -> Self {
                SymmetryGrpInner::$variant(g)
            }
        }
    };
}

impl_from_sym_grp_inner!(u32, Sym32);
impl_from_sym_grp_inner!(u64, Sym64);
impl_from_sym_grp_inner!(B128, Sym128);
impl_from_sym_grp_inner!(B256, Sym256);
impl_from_sym_grp_inner!(B512, Sym512);
impl_from_sym_grp_inner!(B1024, Sym1024);
impl_from_sym_grp_inner!(B2048, Sym2048);
impl_from_sym_grp_inner!(B4096, Sym4096);
impl_from_sym_grp_inner!(B8192, Sym8192);

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

    pub(crate) fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        match self {
            SymmetryGrpInner::Sym32(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Sym64(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Sym128(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Sym256(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Sym512(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Sym1024(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Sym2048(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Sym4096(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Sym8192(g) => g.push_lattice(grp_char, perm),
        }
    }

    pub(crate) fn push_local_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) {
        match self {
            SymmetryGrpInner::Sym32(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Sym64(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Sym128(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Sym256(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Sym512(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Sym1024(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Sym2048(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Sym4096(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Sym8192(g) => g.push_local_perm(grp_char, perm, locs),
        }
    }

    pub(crate) fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        match self {
            SymmetryGrpInner::Sym32(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Sym64(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Sym128(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Sym256(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Sym512(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Sym1024(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Sym2048(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Sym4096(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Sym8192(g) => g.push_spin_inv(grp_char, locs),
        }
    }

    #[allow(dead_code)]
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
/// binding `$grp` to the inner `SymGrpInner<B>` reference.
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
