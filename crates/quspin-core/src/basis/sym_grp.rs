use super::lattice::BenesLatticeElement;
use crate::bitbasis::{
    BenesPermDitLocations, BitInt, BitStateOp, DynamicPermDitValues, PermDitMask,
};
/// B-type dispatch for the unified symmetry group path.
///
/// - [`SymGrpBase<B, L>`]: single concrete generic struct backing all group
///   variants.  `L` is the local-op type: [`PermDitMask<B>`] for LHSS=2
///   (hardcore/fermionic), [`DynamicPermDitValues`] for LHSS≥3 (dit/boson).
/// - [`HardcoreGrpInner<B>`]: type alias for LHSS=2 groups.
/// - [`DitGrpInner<B>`]: type alias for LHSS≥3 groups.
/// - [`SymmetryGrpInner`]: type-erased enum over 18 concrete inner types
///   (9 `Hc*` + 9 `Dit*`), selected at construction time based on LHSS and
///   `n_bits`.
/// - [`with_sym_grp!`]: match macro for `Hc*` variants; injects a concrete
///   `B` type alias and a binding to the inner group.
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
// SymGrpBase<B, L> — generic inner group (no runtime dispatch on local ops)
// ---------------------------------------------------------------------------

/// Generic inner symmetry group.
///
/// The local-op type `L` is monomorphised at the [`SymmetryGrpInner`]
/// construction site:
/// - `L = PermDitMask<B>` → [`HardcoreGrpInner<B>`] — LHSS=2, XOR bit-flip.
/// - `L = DynamicPermDitValues` → [`DitGrpInner<B>`] — LHSS≥3, value perm.
///
/// No runtime `match` on the local-op type occurs in the hot path.
#[derive(Clone)]
pub struct SymGrpBase<B: BitInt, L> {
    pub(crate) lhss: usize,
    pub(crate) fermionic: bool,
    pub(crate) n_sites: usize,
    pub(crate) lattice: Vec<BenesLatticeElement<B>>,
    pub(crate) local: Vec<(Complex<f64>, L)>,
}

/// Inner group type for LHSS=2 (hardcore bosons / spinless fermions).
///
/// Local operations are XOR bit-flip masks ([`PermDitMask<B>`]).
pub type HardcoreGrpInner<B> = SymGrpBase<B, PermDitMask<B>>;

/// Inner group type for LHSS≥3 (bosons with occupancy ≥3 / higher spin).
///
/// Local operations are value permutations ([`DynamicPermDitValues`]).
pub type DitGrpInner<B> = SymGrpBase<B, DynamicPermDitValues>;

/// Backward-compatible alias: `SymGrpInner<B>` is [`HardcoreGrpInner<B>`].
pub type SymGrpInner<B> = HardcoreGrpInner<B>;

// ---------------------------------------------------------------------------
// SymGrpBase: common methods (no L bound needed)
// ---------------------------------------------------------------------------

impl<B: BitInt, L> SymGrpBase<B, L> {
    /// Construct an empty group.
    pub fn new_empty(lhss: usize, n_sites: usize, fermionic: bool) -> Self {
        SymGrpBase {
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
}

// ---------------------------------------------------------------------------
// SymGrpBase: orbit methods (require L: BitStateOp<B> for LocalOpItem blanket)
// ---------------------------------------------------------------------------

impl<B: BitInt, L: BitStateOp<B>> SymGrpBase<B, L> {
    pub fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        super::orbit::get_refstate(&self.lattice, &self.local, state)
    }

    pub fn check_refstate(&self, state: B) -> (B, f64) {
        super::orbit::check_refstate(&self.lattice, &self.local, state)
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
// HardcoreGrpInner-specific methods (LHSS=2, XOR path)
// ---------------------------------------------------------------------------

impl<B: BitInt> SymGrpBase<B, PermDitMask<B>> {
    /// Add a spin-inversion / bit-flip element for LHSS = 2.
    ///
    /// Builds an XOR mask from `locs` and stores it as a [`PermDitMask<B>`],
    /// which applies via a single XOR instruction rather than a loop over sites.
    pub fn push_inverse(&mut self, grp_char: Complex<f64>, locs: &[usize]) {
        let mask = locs.iter().fold(B::from_u64(0), |acc, &site| {
            if site < B::BITS as usize {
                acc | (B::from_u64(1) << site)
            } else {
                acc
            }
        });
        self.local.push((grp_char, PermDitMask::new(mask)));
    }
}

// ---------------------------------------------------------------------------
// DitGrpInner-specific methods (LHSS≥3, value-permutation path)
// ---------------------------------------------------------------------------

impl<B: BitInt> SymGrpBase<B, DynamicPermDitValues> {
    /// Add a local value-permutation element (LHSS ≥ 3).
    ///
    /// `perm[v] = w` maps local occupation `v` to `w` at each site in `locs`.
    pub fn push_local_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        self.local
            .push((grp_char, DynamicPermDitValues::new(self.lhss, perm, locs)));
    }

    /// Add a spin-inversion element: maps `v → lhss − v − 1` at each site in `locs`.
    pub fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        let perm: Vec<u8> = (0..self.lhss).rev().map(|v| v as u8).collect();
        self.push_local_perm(grp_char, perm, locs);
    }
}

// ---------------------------------------------------------------------------
// SymmetryGrpInner — 18-variant type-erased enum
// ---------------------------------------------------------------------------

/// Type-erased symmetry group: one of 18 concrete inner types.
///
/// - `Hc*` variants hold [`HardcoreGrpInner<B>`] — LHSS=2 (hardcore / fermionic).
/// - `Dit*` variants hold [`DitGrpInner<B>`] — LHSS≥3 (dit / boson / higher spin).
///
/// The concrete variant is selected based on LHSS and `n_bits`
/// (= `n_sites * bits_per_dit`) at construction time inside the public
/// wrappers (`SpinSymGrp`, `DitSymGrp`, `FermionicSymGrp`).
#[derive(Clone)]
pub enum SymmetryGrpInner {
    // LHSS=2 — hardcore bosons / fermions
    Hc32(HardcoreGrpInner<u32>),
    Hc64(HardcoreGrpInner<u64>),
    Hc128(HardcoreGrpInner<B128>),
    Hc256(HardcoreGrpInner<B256>),
    Hc512(HardcoreGrpInner<B512>),
    Hc1024(HardcoreGrpInner<B1024>),
    Hc2048(HardcoreGrpInner<B2048>),
    Hc4096(HardcoreGrpInner<B4096>),
    Hc8192(HardcoreGrpInner<B8192>),
    // LHSS≥3 — dit bosons / higher spin
    Dit32(DitGrpInner<u32>),
    Dit64(DitGrpInner<u64>),
    Dit128(DitGrpInner<B128>),
    Dit256(DitGrpInner<B256>),
    Dit512(DitGrpInner<B512>),
    Dit1024(DitGrpInner<B1024>),
    Dit2048(DitGrpInner<B2048>),
    Dit4096(DitGrpInner<B4096>),
    Dit8192(DitGrpInner<B8192>),
}

macro_rules! impl_from_hc {
    ($B:ty, $variant:ident) => {
        impl From<HardcoreGrpInner<$B>> for SymmetryGrpInner {
            #[inline]
            fn from(g: HardcoreGrpInner<$B>) -> Self {
                SymmetryGrpInner::$variant(g)
            }
        }
    };
}

macro_rules! impl_from_dit {
    ($B:ty, $variant:ident) => {
        impl From<DitGrpInner<$B>> for SymmetryGrpInner {
            #[inline]
            fn from(g: DitGrpInner<$B>) -> Self {
                SymmetryGrpInner::$variant(g)
            }
        }
    };
}

impl_from_hc!(u32, Hc32);
impl_from_hc!(u64, Hc64);
impl_from_hc!(B128, Hc128);
impl_from_hc!(B256, Hc256);
impl_from_hc!(B512, Hc512);
impl_from_hc!(B1024, Hc1024);
impl_from_hc!(B2048, Hc2048);
impl_from_hc!(B4096, Hc4096);
impl_from_hc!(B8192, Hc8192);

impl_from_dit!(u32, Dit32);
impl_from_dit!(u64, Dit64);
impl_from_dit!(B128, Dit128);
impl_from_dit!(B256, Dit256);
impl_from_dit!(B512, Dit512);
impl_from_dit!(B1024, Dit1024);
impl_from_dit!(B2048, Dit2048);
impl_from_dit!(B4096, Dit4096);
impl_from_dit!(B8192, Dit8192);

impl SymmetryGrpInner {
    pub fn n_sites(&self) -> usize {
        match self {
            SymmetryGrpInner::Hc32(g) => g.n_sites(),
            SymmetryGrpInner::Hc64(g) => g.n_sites(),
            SymmetryGrpInner::Hc128(g) => g.n_sites(),
            SymmetryGrpInner::Hc256(g) => g.n_sites(),
            SymmetryGrpInner::Hc512(g) => g.n_sites(),
            SymmetryGrpInner::Hc1024(g) => g.n_sites(),
            SymmetryGrpInner::Hc2048(g) => g.n_sites(),
            SymmetryGrpInner::Hc4096(g) => g.n_sites(),
            SymmetryGrpInner::Hc8192(g) => g.n_sites(),
            SymmetryGrpInner::Dit32(g) => g.n_sites(),
            SymmetryGrpInner::Dit64(g) => g.n_sites(),
            SymmetryGrpInner::Dit128(g) => g.n_sites(),
            SymmetryGrpInner::Dit256(g) => g.n_sites(),
            SymmetryGrpInner::Dit512(g) => g.n_sites(),
            SymmetryGrpInner::Dit1024(g) => g.n_sites(),
            SymmetryGrpInner::Dit2048(g) => g.n_sites(),
            SymmetryGrpInner::Dit4096(g) => g.n_sites(),
            SymmetryGrpInner::Dit8192(g) => g.n_sites(),
        }
    }

    pub(crate) fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        match self {
            SymmetryGrpInner::Hc32(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Hc64(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Hc128(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Hc256(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Hc512(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Hc1024(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Hc2048(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Hc4096(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Hc8192(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Dit32(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Dit64(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Dit128(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Dit256(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Dit512(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Dit1024(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Dit2048(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Dit4096(g) => g.push_lattice(grp_char, perm),
            SymmetryGrpInner::Dit8192(g) => g.push_lattice(grp_char, perm),
        }
    }

    /// Add a spin-inversion / bit-flip element.
    ///
    /// Only valid for `Hc*` (LHSS=2) variants.  Panics if called on a `Dit*` variant.
    pub(crate) fn push_inverse(&mut self, grp_char: Complex<f64>, locs: &[usize]) {
        match self {
            SymmetryGrpInner::Hc32(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Hc64(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Hc128(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Hc256(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Hc512(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Hc1024(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Hc2048(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Hc4096(g) => g.push_inverse(grp_char, locs),
            SymmetryGrpInner::Hc8192(g) => g.push_inverse(grp_char, locs),
            _ => panic!("push_inverse called on a LHSS≥3 (Dit*) symmetry group"),
        }
    }

    /// Add a local value-permutation element (LHSS ≥ 3).
    ///
    /// Only valid for `Dit*` variants.  Panics if called on a `Hc*` variant.
    pub(crate) fn push_local_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) {
        match self {
            SymmetryGrpInner::Dit32(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Dit64(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Dit128(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Dit256(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Dit512(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Dit1024(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Dit2048(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Dit4096(g) => g.push_local_perm(grp_char, perm, locs),
            SymmetryGrpInner::Dit8192(g) => g.push_local_perm(grp_char, perm, locs),
            _ => panic!("push_local_perm called on a LHSS=2 (Hc*) symmetry group"),
        }
    }

    /// Add a spin-inversion element (v → lhss − v − 1) for LHSS ≥ 3.
    ///
    /// Only valid for `Dit*` variants.  Panics if called on a `Hc*` variant.
    pub(crate) fn push_spin_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        match self {
            SymmetryGrpInner::Dit32(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Dit64(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Dit128(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Dit256(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Dit512(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Dit1024(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Dit2048(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Dit4096(g) => g.push_spin_inv(grp_char, locs),
            SymmetryGrpInner::Dit8192(g) => g.push_spin_inv(grp_char, locs),
            _ => panic!("push_spin_inv called on a LHSS=2 (Hc*) symmetry group"),
        }
    }
}

// ---------------------------------------------------------------------------
// with_sym_grp! macro
// ---------------------------------------------------------------------------

/// Match on a [`SymmetryGrpInner`] reference for `Hc*` (LHSS=2) variants,
/// injecting a type alias `$B` (the basis integer type) and `$N` (the norm
/// storage type), and binding `$grp` to the inner `HardcoreGrpInner<B>`
/// reference.
///
/// The B→N pairing is:
/// - Hc32  → B=u32,  N=u8
/// - Hc64  → B=u64,  N=u16
/// - Hc128..Hc8192 → N=u32
///
/// # Panics
///
/// Panics (via `unreachable!`) if called with a `Dit*` (LHSS≥3) variant.
/// Only pass groups obtained via `as_hardcore()` (which returns `None` for
/// LHSS≥3 groups).
#[macro_export]
macro_rules! with_sym_grp {
    ($inner:expr, $B:ident, $N:ident, $grp:ident, $body:block) => {
        match $inner {
            $crate::basis::sym_grp::SymmetryGrpInner::Hc32($grp) => {
                type $B = u32;
                type $N = u8;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Hc64($grp) => {
                type $B = u64;
                type $N = u16;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Hc128($grp) => {
                type $B = ::ruint::Uint<128, 2>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Hc256($grp) => {
                type $B = ::ruint::Uint<256, 4>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Hc512($grp) => {
                type $B = ::ruint::Uint<512, 8>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Hc1024($grp) => {
                type $B = ::ruint::Uint<1024, 16>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Hc2048($grp) => {
                type $B = ::ruint::Uint<2048, 32>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Hc4096($grp) => {
                type $B = ::ruint::Uint<4096, 64>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Hc8192($grp) => {
                type $B = ::ruint::Uint<8192, 128>;
                type $N = u32;
                $body
            }
            _ => {
                unreachable!("with_sym_grp! requires a LHSS=2 (hardcore/fermionic) symmetry group")
            }
        }
    };
}

// ---------------------------------------------------------------------------
// with_dit_sym_grp! macro
// ---------------------------------------------------------------------------

/// Match on a [`SymmetryGrpInner`] reference for `Dit*` (LHSS≥3) variants,
/// injecting a type alias `$B` (the basis integer type) and `$N` (the norm
/// storage type), and binding `$grp` to the inner `DitGrpInner<B>` reference.
///
/// The B→N pairing is:
/// - Dit32  → B=u32,  N=u8
/// - Dit64  → B=u64,  N=u16
/// - Dit128..Dit8192 → N=u32
///
/// # Panics
///
/// Panics (via `unreachable!`) if called with a `Hc*` (LHSS=2) variant.
/// Only pass groups obtained via `as_dit()`.
#[macro_export]
macro_rules! with_dit_sym_grp {
    ($inner:expr, $B:ident, $N:ident, $grp:ident, $body:block) => {
        match $inner {
            $crate::basis::sym_grp::SymmetryGrpInner::Dit32($grp) => {
                type $B = u32;
                type $N = u8;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Dit64($grp) => {
                type $B = u64;
                type $N = u16;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Dit128($grp) => {
                type $B = ::ruint::Uint<128, 2>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Dit256($grp) => {
                type $B = ::ruint::Uint<256, 4>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Dit512($grp) => {
                type $B = ::ruint::Uint<512, 8>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Dit1024($grp) => {
                type $B = ::ruint::Uint<1024, 16>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Dit2048($grp) => {
                type $B = ::ruint::Uint<2048, 32>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Dit4096($grp) => {
                type $B = ::ruint::Uint<4096, 64>;
                type $N = u32;
                $body
            }
            $crate::basis::sym_grp::SymmetryGrpInner::Dit8192($grp) => {
                type $B = ::ruint::Uint<8192, 128>;
                type $N = u32;
                $body
            }
            _ => {
                unreachable!("with_dit_sym_grp! requires a LHSS≥3 (dit) symmetry group")
            }
        }
    };
}
