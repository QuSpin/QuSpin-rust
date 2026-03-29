/// Type-erased `SpaceInner` and its dispatch macros.
///
/// ## Supported basis sizes
///
/// | Variant suffix | Rust type                  | Bit width |
/// |----------------|----------------------------|-----------|
/// | `32`           | `u32`                      | 32        |
/// | `64`           | `u64`                      | 64        |
/// | `128`          | `ruint::Uint<128,  2>`     | 128       |
/// | `256`          | `ruint::Uint<256,  4>`     | 256       |
/// | `512`          | `ruint::Uint<512,  8>`     | 512       |
/// | `1024`         | `ruint::Uint<1024, 16>`    | 1024      |
/// | `2048`         | `ruint::Uint<2048, 32>`    | 2048      |
/// | `4096`         | `ruint::Uint<4096, 64>`    | 4096      |
/// | `8192`         | `ruint::Uint<8192, 128>`   | 8192      |
///
/// `FullSpace` is only instantiated for `u32` and `u64`; larger full spaces
/// are not physically meaningful.
use crate::basis::{
    BasisSpace,
    seed::{seed_from_bytes, state_to_str},
    space::{FullSpace, Subspace},
    sym::SymBasis,
};
use crate::bitbasis::{BitInt, DynamicPermDitValues, PermDitMask};
use crate::error::QuSpinError;
use num_complex::Complex;

type B128 = ruint::Uint<128, 2>;
type B256 = ruint::Uint<256, 4>;
type B512 = ruint::Uint<512, 8>;
type B1024 = ruint::Uint<1024, 16>;
type B2048 = ruint::Uint<2048, 32>;
type B4096 = ruint::Uint<4096, 64>;
type B8192 = ruint::Uint<8192, 128>;

// ---------------------------------------------------------------------------
// SpaceInner
// ---------------------------------------------------------------------------

/// Type-erased wrapper for all basis-space variants over all supported
/// integer widths.
///
/// 29 variants total:
/// - 2 `Full` variants (u32, u64)
/// - 9 `Sub` variants (u32, u64, and 128–8192 bit ruint integers)
/// - 9 `Sym` variants — LHSS=2 symmetric (hardcore bosons / spin-½ / fermions)
/// - 9 `DitSym` variants — LHSS≥3 symmetric (bosons / higher spin)
pub enum SpaceInner {
    // Full Hilbert spaces (small n_sites only).
    Full32(FullSpace<u32>),
    Full64(FullSpace<u64>),

    // Subspaces (particle-number or energy sector).
    Sub32(Subspace<u32>),
    Sub64(Subspace<u64>),
    Sub128(Subspace<B128>),
    Sub256(Subspace<B256>),
    Sub512(Subspace<B512>),
    Sub1024(Subspace<B1024>),
    Sub2048(Subspace<B2048>),
    Sub4096(Subspace<B4096>),
    Sub8192(Subspace<B8192>),

    // LHSS=2 symmetry-reduced subspaces (hardcore bosons / spin-½ / fermions).
    Sym32(SymBasis<u32, PermDitMask<u32>, u8>),
    Sym64(SymBasis<u64, PermDitMask<u64>, u16>),
    Sym128(SymBasis<B128, PermDitMask<B128>, u32>),
    Sym256(SymBasis<B256, PermDitMask<B256>, u32>),
    Sym512(SymBasis<B512, PermDitMask<B512>, u32>),
    Sym1024(SymBasis<B1024, PermDitMask<B1024>, u32>),
    Sym2048(SymBasis<B2048, PermDitMask<B2048>, u32>),
    Sym4096(SymBasis<B4096, PermDitMask<B4096>, u32>),
    Sym8192(SymBasis<B8192, PermDitMask<B8192>, u32>),

    // LHSS≥3 symmetry-reduced subspaces (bosons / higher spin).
    DitSym32(SymBasis<u32, DynamicPermDitValues, u8>),
    DitSym64(SymBasis<u64, DynamicPermDitValues, u16>),
    DitSym128(SymBasis<B128, DynamicPermDitValues, u32>),
    DitSym256(SymBasis<B256, DynamicPermDitValues, u32>),
    DitSym512(SymBasis<B512, DynamicPermDitValues, u32>),
    DitSym1024(SymBasis<B1024, DynamicPermDitValues, u32>),
    DitSym2048(SymBasis<B2048, DynamicPermDitValues, u32>),
    DitSym4096(SymBasis<B4096, DynamicPermDitValues, u32>),
    DitSym8192(SymBasis<B8192, DynamicPermDitValues, u32>),
}

impl SpaceInner {
    /// Number of lattice sites.
    pub fn n_sites(&self) -> usize {
        match self {
            SpaceInner::Full32(b) => b.n_sites(),
            SpaceInner::Full64(b) => b.n_sites(),
            SpaceInner::Sub32(b) => b.n_sites(),
            SpaceInner::Sub64(b) => b.n_sites(),
            SpaceInner::Sub128(b) => b.n_sites(),
            SpaceInner::Sub256(b) => b.n_sites(),
            SpaceInner::Sub512(b) => b.n_sites(),
            SpaceInner::Sub1024(b) => b.n_sites(),
            SpaceInner::Sub2048(b) => b.n_sites(),
            SpaceInner::Sub4096(b) => b.n_sites(),
            SpaceInner::Sub8192(b) => b.n_sites(),
            SpaceInner::Sym32(b) => b.n_sites(),
            SpaceInner::Sym64(b) => b.n_sites(),
            SpaceInner::Sym128(b) => b.n_sites(),
            SpaceInner::Sym256(b) => b.n_sites(),
            SpaceInner::Sym512(b) => b.n_sites(),
            SpaceInner::Sym1024(b) => b.n_sites(),
            SpaceInner::Sym2048(b) => b.n_sites(),
            SpaceInner::Sym4096(b) => b.n_sites(),
            SpaceInner::Sym8192(b) => b.n_sites(),
            SpaceInner::DitSym32(b) => b.n_sites(),
            SpaceInner::DitSym64(b) => b.n_sites(),
            SpaceInner::DitSym128(b) => b.n_sites(),
            SpaceInner::DitSym256(b) => b.n_sites(),
            SpaceInner::DitSym512(b) => b.n_sites(),
            SpaceInner::DitSym1024(b) => b.n_sites(),
            SpaceInner::DitSym2048(b) => b.n_sites(),
            SpaceInner::DitSym4096(b) => b.n_sites(),
            SpaceInner::DitSym8192(b) => b.n_sites(),
        }
    }

    /// Local Hilbert-space size (number of states per site).
    pub fn lhss(&self) -> usize {
        match self {
            SpaceInner::Full32(b) => b.lhss(),
            SpaceInner::Full64(b) => b.lhss(),
            SpaceInner::Sub32(b) => b.lhss(),
            SpaceInner::Sub64(b) => b.lhss(),
            SpaceInner::Sub128(b) => b.lhss(),
            SpaceInner::Sub256(b) => b.lhss(),
            SpaceInner::Sub512(b) => b.lhss(),
            SpaceInner::Sub1024(b) => b.lhss(),
            SpaceInner::Sub2048(b) => b.lhss(),
            SpaceInner::Sub4096(b) => b.lhss(),
            SpaceInner::Sub8192(b) => b.lhss(),
            SpaceInner::Sym32(b) => b.lhss(),
            SpaceInner::Sym64(b) => b.lhss(),
            SpaceInner::Sym128(b) => b.lhss(),
            SpaceInner::Sym256(b) => b.lhss(),
            SpaceInner::Sym512(b) => b.lhss(),
            SpaceInner::Sym1024(b) => b.lhss(),
            SpaceInner::Sym2048(b) => b.lhss(),
            SpaceInner::Sym4096(b) => b.lhss(),
            SpaceInner::Sym8192(b) => b.lhss(),
            SpaceInner::DitSym32(b) => b.lhss(),
            SpaceInner::DitSym64(b) => b.lhss(),
            SpaceInner::DitSym128(b) => b.lhss(),
            SpaceInner::DitSym256(b) => b.lhss(),
            SpaceInner::DitSym512(b) => b.lhss(),
            SpaceInner::DitSym1024(b) => b.lhss(),
            SpaceInner::DitSym2048(b) => b.lhss(),
            SpaceInner::DitSym4096(b) => b.lhss(),
            SpaceInner::DitSym8192(b) => b.lhss(),
        }
    }

    /// Number of basis states.
    pub fn size(&self) -> usize {
        match self {
            SpaceInner::Full32(b) => b.size(),
            SpaceInner::Full64(b) => b.size(),
            SpaceInner::Sub32(b) => b.size(),
            SpaceInner::Sub64(b) => b.size(),
            SpaceInner::Sub128(b) => b.size(),
            SpaceInner::Sub256(b) => b.size(),
            SpaceInner::Sub512(b) => b.size(),
            SpaceInner::Sub1024(b) => b.size(),
            SpaceInner::Sub2048(b) => b.size(),
            SpaceInner::Sub4096(b) => b.size(),
            SpaceInner::Sub8192(b) => b.size(),
            SpaceInner::Sym32(b) => b.size(),
            SpaceInner::Sym64(b) => b.size(),
            SpaceInner::Sym128(b) => b.size(),
            SpaceInner::Sym256(b) => b.size(),
            SpaceInner::Sym512(b) => b.size(),
            SpaceInner::Sym1024(b) => b.size(),
            SpaceInner::Sym2048(b) => b.size(),
            SpaceInner::Sym4096(b) => b.size(),
            SpaceInner::Sym8192(b) => b.size(),
            SpaceInner::DitSym32(b) => b.size(),
            SpaceInner::DitSym64(b) => b.size(),
            SpaceInner::DitSym128(b) => b.size(),
            SpaceInner::DitSym256(b) => b.size(),
            SpaceInner::DitSym512(b) => b.size(),
            SpaceInner::DitSym1024(b) => b.size(),
            SpaceInner::DitSym2048(b) => b.size(),
            SpaceInner::DitSym4096(b) => b.size(),
            SpaceInner::DitSym8192(b) => b.size(),
        }
    }

    /// Return the `i`-th basis state as a bit-string (site 0 = index 0).
    pub fn state_at_str(&self, i: usize) -> String {
        match self {
            SpaceInner::Full32(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Full64(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sub32(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sub64(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sub128(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sub256(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sub512(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sub1024(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sub2048(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sub4096(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sub8192(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sym32(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sym64(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sym128(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sym256(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sym512(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sym1024(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sym2048(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sym4096(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::Sym8192(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::DitSym32(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::DitSym64(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::DitSym128(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::DitSym256(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::DitSym512(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::DitSym1024(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::DitSym2048(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::DitSym4096(b) => state_to_str(b.state_at(i), b.n_sites()),
            SpaceInner::DitSym8192(b) => state_to_str(b.state_at(i), b.n_sites()),
        }
    }

    /// Look up the index of the state encoded as a site-occupation byte slice.
    ///
    /// Returns `None` if the state is not in the basis.
    pub fn index_of_bytes(&self, bytes: &[u8]) -> Option<usize> {
        match self {
            SpaceInner::Full32(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Full64(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sub32(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sub64(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sub128(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sub256(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sub512(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sub1024(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sub2048(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sub4096(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sub8192(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sym32(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sym64(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sym128(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sym256(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sym512(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sym1024(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sym2048(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sym4096(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::Sym8192(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::DitSym32(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::DitSym64(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::DitSym128(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::DitSym256(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::DitSym512(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::DitSym1024(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::DitSym2048(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::DitSym4096(b) => b.index(seed_from_bytes(bytes)),
            SpaceInner::DitSym8192(b) => b.index(seed_from_bytes(bytes)),
        }
    }

    /// One of `"full"`, `"subspace"`, or `"symmetric"`.
    pub fn kind(&self) -> &'static str {
        match self {
            SpaceInner::Full32(_) | SpaceInner::Full64(_) => "full",
            SpaceInner::Sub32(_)
            | SpaceInner::Sub64(_)
            | SpaceInner::Sub128(_)
            | SpaceInner::Sub256(_)
            | SpaceInner::Sub512(_)
            | SpaceInner::Sub1024(_)
            | SpaceInner::Sub2048(_)
            | SpaceInner::Sub4096(_)
            | SpaceInner::Sub8192(_) => "subspace",
            SpaceInner::Sym32(_)
            | SpaceInner::Sym64(_)
            | SpaceInner::Sym128(_)
            | SpaceInner::Sym256(_)
            | SpaceInner::Sym512(_)
            | SpaceInner::Sym1024(_)
            | SpaceInner::Sym2048(_)
            | SpaceInner::Sym4096(_)
            | SpaceInner::Sym8192(_)
            | SpaceInner::DitSym32(_)
            | SpaceInner::DitSym64(_)
            | SpaceInner::DitSym128(_)
            | SpaceInner::DitSym256(_)
            | SpaceInner::DitSym512(_)
            | SpaceInner::DitSym1024(_)
            | SpaceInner::DitSym2048(_)
            | SpaceInner::DitSym4096(_)
            | SpaceInner::DitSym8192(_) => "symmetric",
        }
    }

    /// Returns `true` once `build` has been called on the inner basis.
    ///
    /// - `Full*` → always `true` (no build step required)
    /// - `Sub*`  → `subspace.is_built()`
    /// - `Sym*` / `DitSym*` → `sym_basis.is_built()`
    pub fn is_built(&self) -> bool {
        match self {
            SpaceInner::Full32(_) | SpaceInner::Full64(_) => true,
            SpaceInner::Sub32(b) => b.is_built(),
            SpaceInner::Sub64(b) => b.is_built(),
            SpaceInner::Sub128(b) => b.is_built(),
            SpaceInner::Sub256(b) => b.is_built(),
            SpaceInner::Sub512(b) => b.is_built(),
            SpaceInner::Sub1024(b) => b.is_built(),
            SpaceInner::Sub2048(b) => b.is_built(),
            SpaceInner::Sub4096(b) => b.is_built(),
            SpaceInner::Sub8192(b) => b.is_built(),
            SpaceInner::Sym32(b) => b.is_built(),
            SpaceInner::Sym64(b) => b.is_built(),
            SpaceInner::Sym128(b) => b.is_built(),
            SpaceInner::Sym256(b) => b.is_built(),
            SpaceInner::Sym512(b) => b.is_built(),
            SpaceInner::Sym1024(b) => b.is_built(),
            SpaceInner::Sym2048(b) => b.is_built(),
            SpaceInner::Sym4096(b) => b.is_built(),
            SpaceInner::Sym8192(b) => b.is_built(),
            SpaceInner::DitSym32(b) => b.is_built(),
            SpaceInner::DitSym64(b) => b.is_built(),
            SpaceInner::DitSym128(b) => b.is_built(),
            SpaceInner::DitSym256(b) => b.is_built(),
            SpaceInner::DitSym512(b) => b.is_built(),
            SpaceInner::DitSym1024(b) => b.is_built(),
            SpaceInner::DitSym2048(b) => b.is_built(),
            SpaceInner::DitSym4096(b) => b.is_built(),
            SpaceInner::DitSym8192(b) => b.is_built(),
        }
    }

    /// Add a lattice (site-permutation) symmetry element.
    ///
    /// Valid only on `Sym*` and `DitSym*` variants; errors on `Full*` / `Sub*`.
    pub fn push_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
    ) -> Result<(), QuSpinError> {
        match self {
            SpaceInner::Sym32(b) => b.push_lattice(grp_char, perm),
            SpaceInner::Sym64(b) => b.push_lattice(grp_char, perm),
            SpaceInner::Sym128(b) => b.push_lattice(grp_char, perm),
            SpaceInner::Sym256(b) => b.push_lattice(grp_char, perm),
            SpaceInner::Sym512(b) => b.push_lattice(grp_char, perm),
            SpaceInner::Sym1024(b) => b.push_lattice(grp_char, perm),
            SpaceInner::Sym2048(b) => b.push_lattice(grp_char, perm),
            SpaceInner::Sym4096(b) => b.push_lattice(grp_char, perm),
            SpaceInner::Sym8192(b) => b.push_lattice(grp_char, perm),
            SpaceInner::DitSym32(b) => b.push_lattice(grp_char, perm),
            SpaceInner::DitSym64(b) => b.push_lattice(grp_char, perm),
            SpaceInner::DitSym128(b) => b.push_lattice(grp_char, perm),
            SpaceInner::DitSym256(b) => b.push_lattice(grp_char, perm),
            SpaceInner::DitSym512(b) => b.push_lattice(grp_char, perm),
            SpaceInner::DitSym1024(b) => b.push_lattice(grp_char, perm),
            SpaceInner::DitSym2048(b) => b.push_lattice(grp_char, perm),
            SpaceInner::DitSym4096(b) => b.push_lattice(grp_char, perm),
            SpaceInner::DitSym8192(b) => b.push_lattice(grp_char, perm),
            _ => {
                return Err(QuSpinError::ValueError(
                    "push_lattice requires a symmetric (Sym* or DitSym*) basis".into(),
                ));
            }
        }
        Ok(())
    }

    /// Add a local XOR-mask symmetry element for LHSS=2 (`Sym*`) bases.
    ///
    /// `locs` is the list of site indices to include in the mask.
    /// Errors on `DitSym*`, `Full*`, and `Sub*` variants.
    pub fn push_local_mask(
        &mut self,
        grp_char: Complex<f64>,
        locs: &[usize],
    ) -> Result<(), QuSpinError> {
        macro_rules! build_mask_and_push {
            ($basis:expr, $B:ty) => {{
                let mask = locs.iter().fold(<$B>::from_u64(0), |acc, &site| {
                    if site < <$B>::BITS as usize {
                        acc | (<$B>::from_u64(1) << site)
                    } else {
                        acc
                    }
                });
                $basis.push_local(grp_char, PermDitMask::new(mask));
            }};
        }
        match self {
            SpaceInner::Sym32(b) => build_mask_and_push!(b, u32),
            SpaceInner::Sym64(b) => build_mask_and_push!(b, u64),
            SpaceInner::Sym128(b) => build_mask_and_push!(b, B128),
            SpaceInner::Sym256(b) => build_mask_and_push!(b, B256),
            SpaceInner::Sym512(b) => build_mask_and_push!(b, B512),
            SpaceInner::Sym1024(b) => build_mask_and_push!(b, B1024),
            SpaceInner::Sym2048(b) => build_mask_and_push!(b, B2048),
            SpaceInner::Sym4096(b) => build_mask_and_push!(b, B4096),
            SpaceInner::Sym8192(b) => build_mask_and_push!(b, B8192),
            SpaceInner::DitSym32(_)
            | SpaceInner::DitSym64(_)
            | SpaceInner::DitSym128(_)
            | SpaceInner::DitSym256(_)
            | SpaceInner::DitSym512(_)
            | SpaceInner::DitSym1024(_)
            | SpaceInner::DitSym2048(_)
            | SpaceInner::DitSym4096(_)
            | SpaceInner::DitSym8192(_) => {
                return Err(QuSpinError::ValueError(
                    "push_local_mask requires an LHSS=2 (Sym*) basis".into(),
                ));
            }
            _ => {
                return Err(QuSpinError::ValueError(
                    "push_local_mask requires a symmetric basis".into(),
                ));
            }
        }
        Ok(())
    }

    /// Add a local value-permutation symmetry element for LHSS≥3 (`DitSym*`) bases.
    ///
    /// `perm[v] = w` maps local occupation `v` to `w` at each site in `locs`.
    /// Errors on `Sym*`, `Full*`, and `Sub*` variants.
    pub fn push_local_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        macro_rules! push_perm {
            ($basis:expr) => {
                $basis.push_local(grp_char, DynamicPermDitValues::new($basis.lhss, perm, locs))
            };
        }
        match self {
            SpaceInner::DitSym32(b) => push_perm!(b),
            SpaceInner::DitSym64(b) => push_perm!(b),
            SpaceInner::DitSym128(b) => push_perm!(b),
            SpaceInner::DitSym256(b) => push_perm!(b),
            SpaceInner::DitSym512(b) => push_perm!(b),
            SpaceInner::DitSym1024(b) => push_perm!(b),
            SpaceInner::DitSym2048(b) => push_perm!(b),
            SpaceInner::DitSym4096(b) => push_perm!(b),
            SpaceInner::DitSym8192(b) => push_perm!(b),
            SpaceInner::Sym32(_)
            | SpaceInner::Sym64(_)
            | SpaceInner::Sym128(_)
            | SpaceInner::Sym256(_)
            | SpaceInner::Sym512(_)
            | SpaceInner::Sym1024(_)
            | SpaceInner::Sym2048(_)
            | SpaceInner::Sym4096(_)
            | SpaceInner::Sym8192(_) => {
                return Err(QuSpinError::ValueError(
                    "push_local_perm requires an LHSS≥3 (DitSym*) basis".into(),
                ));
            }
            _ => {
                return Err(QuSpinError::ValueError(
                    "push_local_perm requires a symmetric basis".into(),
                ));
            }
        }
        Ok(())
    }

    /// Returns `true` for `Sym*` and `DitSym*` variants (symmetry-reduced subspaces).
    pub fn is_symmetric(&self) -> bool {
        matches!(
            self,
            SpaceInner::Sym32(_)
                | SpaceInner::Sym64(_)
                | SpaceInner::Sym128(_)
                | SpaceInner::Sym256(_)
                | SpaceInner::Sym512(_)
                | SpaceInner::Sym1024(_)
                | SpaceInner::Sym2048(_)
                | SpaceInner::Sym4096(_)
                | SpaceInner::Sym8192(_)
                | SpaceInner::DitSym32(_)
                | SpaceInner::DitSym64(_)
                | SpaceInner::DitSym128(_)
                | SpaceInner::DitSym256(_)
                | SpaceInner::DitSym512(_)
                | SpaceInner::DitSym1024(_)
                | SpaceInner::DitSym2048(_)
                | SpaceInner::DitSym4096(_)
                | SpaceInner::DitSym8192(_)
        )
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

const DISPLAY_HEAD: usize = 25;
const DISPLAY_TAIL: usize = 25;

impl std::fmt::Display for SpaceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = self.size();
        let sym_display = if self.is_symmetric() {
            "[symmetric]"
        } else {
            "[]"
        };
        let index_width = size.saturating_sub(1).to_string().len();

        write!(
            f,
            "{}(n_sites={}, size={}, symmetries={}):",
            self.kind(),
            self.n_sites(),
            size,
            sym_display,
        )?;

        let truncate = size > DISPLAY_HEAD + DISPLAY_TAIL;
        let indices: Box<dyn Iterator<Item = usize>> = if truncate {
            Box::new((0..DISPLAY_HEAD).chain(size - DISPLAY_TAIL..size))
        } else {
            Box::new(0..size)
        };

        let mut prev: Option<usize> = None;
        for i in indices {
            if truncate
                && let Some(p) = prev
                && i > p + 1
            {
                write!(f, "\n  {:>width$}", "...", width = index_width + 1)?;
            }
            write!(
                f,
                "\n  {:>width$}. |{}>",
                i,
                self.state_at_str(i),
                width = index_width,
            )?;
            prev = Some(i);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// From impls — wrap a concrete basis space without naming the variant
// ---------------------------------------------------------------------------

macro_rules! impl_from_basis_spaces {
    ($B:ty, $N:ty, $sub_variant:ident, $sym_variant:ident, $dit_sym_variant:ident) => {
        impl From<Subspace<$B>> for SpaceInner {
            #[inline]
            fn from(b: Subspace<$B>) -> Self {
                SpaceInner::$sub_variant(b)
            }
        }
        impl From<SymBasis<$B, PermDitMask<$B>, $N>> for SpaceInner {
            #[inline]
            fn from(b: SymBasis<$B, PermDitMask<$B>, $N>) -> Self {
                SpaceInner::$sym_variant(b)
            }
        }
        impl From<SymBasis<$B, DynamicPermDitValues, $N>> for SpaceInner {
            #[inline]
            fn from(b: SymBasis<$B, DynamicPermDitValues, $N>) -> Self {
                SpaceInner::$dit_sym_variant(b)
            }
        }
    };
}

impl_from_basis_spaces!(u32, u8, Sub32, Sym32, DitSym32);
impl_from_basis_spaces!(u64, u16, Sub64, Sym64, DitSym64);
impl_from_basis_spaces!(B128, u32, Sub128, Sym128, DitSym128);
impl_from_basis_spaces!(B256, u32, Sub256, Sym256, DitSym256);
impl_from_basis_spaces!(B512, u32, Sub512, Sym512, DitSym512);
impl_from_basis_spaces!(B1024, u32, Sub1024, Sym1024, DitSym1024);
impl_from_basis_spaces!(B2048, u32, Sub2048, Sym2048, DitSym2048);
impl_from_basis_spaces!(B4096, u32, Sub4096, Sym4096, DitSym4096);
impl_from_basis_spaces!(B8192, u32, Sub8192, Sym8192, DitSym8192);

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------

/// Match on a [`SpaceInner`] reference, injecting a type alias `$B` for
/// the concrete `BitInt` type and binding `$basis` to the inner basis reference.
///
/// Covers all 29 variants (Full*, Sub*, Sym*, DitSym*).
#[macro_export]
macro_rules! with_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::Full32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Full64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
        }
    };
}

/// Like `with_basis!` but restricted to Full* and Sub* (non-symmetric) variants.
#[macro_export]
macro_rules! with_plain_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::Full32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Full64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_plain_basis! called on a symmetric variant"),
        }
    };
}

/// Like `with_basis!` but restricted to `Sym*` (LHSS=2 symmetric) variants.
///
/// Panics if called on a `DitSym*` variant.
#[macro_export]
macro_rules! with_sym_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_sym_basis! called on a non-Sym variant"),
        }
    };
}

/// Like `with_basis!` but restricted to `DitSym*` (LHSS≥3 symmetric) variants.
///
/// Panics if called on a `Sym*` or non-symmetric variant.
#[macro_export]
macro_rules! with_dit_sym_basis {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::DitSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_dit_sym_basis! called on a non-DitSym variant"),
        }
    };
}

/// Like `with_sym_basis!` but binds `$basis` as `&mut`.
#[macro_export]
macro_rules! with_sym_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::Sym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_sym_basis_mut! called on a non-Sym variant"),
        }
    };
}

/// Like `with_dit_sym_basis!` but binds `$basis` as `&mut`.
#[macro_export]
macro_rules! with_dit_sym_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::DitSym32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::DitSym8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_dit_sym_basis_mut! called on a non-DitSym variant"),
        }
    };
}

/// Like `with_plain_basis!` but restricted to `Sub*` variants and binds `$basis` as `&mut`.
///
/// Does not match `Full*` — full spaces are always built and cannot be mutated.
#[macro_export]
macro_rules! with_sub_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            $crate::basis::dispatch::SpaceInner::Sub32($basis) => {
                type $B = u32;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub64($basis) => {
                type $B = u64;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub128($basis) => {
                type $B = ::ruint::Uint<128, 2>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub256($basis) => {
                type $B = ::ruint::Uint<256, 4>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub512($basis) => {
                type $B = ::ruint::Uint<512, 8>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub1024($basis) => {
                type $B = ::ruint::Uint<1024, 16>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub2048($basis) => {
                type $B = ::ruint::Uint<2048, 32>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub4096($basis) => {
                type $B = ::ruint::Uint<4096, 64>;
                $body
            }
            $crate::basis::dispatch::SpaceInner::Sub8192($basis) => {
                type $B = ::ruint::Uint<8192, 128>;
                $body
            }
            _ => unreachable!("with_sub_basis_mut! called on a non-Sub variant"),
        }
    };
}

/// Select the smallest `B: BitInt` that fits `$n_sites` site indices, inject
/// it as a local type alias `$B`, and evaluate `$body`.
///
/// The ladder is: ≤32 → `u32`, ≤64 → `u64`, ≤128 → `Uint<128,2>`, …,
/// ≤8192 → `Uint<8192,128>`.
///
/// `$on_overflow` is evaluated (and must diverge or return) when
/// `n_sites > 8192`.  Each FFI consumer supplies its own expression:
///
/// ```rust,ignore
/// // quspin-py
/// select_b_for_n_sites!(n, B,
///     return Err(pyo3::exceptions::PyValueError::new_err("n_sites > 8192")),
///     { ... }
/// );
///
/// // quspin-c
/// select_b_for_n_sites!(n, B,
///     return write_error(err, QuSpinError::ValueError("n_sites > 8192".into())),
///     { ... }
/// );
/// ```
#[macro_export]
macro_rules! select_b_for_n_sites {
    ($n_sites:expr, $B:ident, $on_overflow:expr, $body:block) => {
        if $n_sites <= 32 {
            type $B = u32;
            $body
        } else if $n_sites <= 64 {
            type $B = u64;
            $body
        } else if $n_sites <= 128 {
            type $B = ::ruint::Uint<128, 2>;
            $body
        } else if $n_sites <= 256 {
            type $B = ::ruint::Uint<256, 4>;
            $body
        } else if $n_sites <= 512 {
            type $B = ::ruint::Uint<512, 8>;
            $body
        } else if $n_sites <= 1024 {
            type $B = ::ruint::Uint<1024, 16>;
            $body
        } else if $n_sites <= 2048 {
            type $B = ::ruint::Uint<2048, 32>;
            $body
        } else if $n_sites <= 4096 {
            type $B = ::ruint::Uint<4096, 64>;
            $body
        } else if $n_sites <= 8192 {
            type $B = ::ruint::Uint<8192, 128>;
            $body
        } else {
            $on_overflow
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::space::{FullSpace, Subspace};

    #[test]
    fn display_full_space() {
        let inner = SpaceInner::Full32(FullSpace::new(2, 2, false));
        let s = inner.to_string();
        assert!(s.starts_with("full(n_sites=2, size=4, symmetries=[]):"));
        assert!(s.contains("|11>"));
        assert!(s.contains("|00>"));
    }

    #[test]
    fn display_subspace() {
        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0b01u32, |s| {
            vec![(num_complex::Complex::new(1.0, 0.0), s ^ 0b11, 0u8)]
        });
        let inner = SpaceInner::Sub32(sub);
        let s = inner.to_string();
        assert!(s.starts_with("subspace(n_sites=2, size="));
        assert!(s.contains("symmetries=[]"));
    }

    #[test]
    fn display_index_alignment() {
        // 16 states → indices 0-15, width 2; row 9 and 10 should be right-aligned
        let inner = SpaceInner::Full32(FullSpace::new(2, 4, false));
        let s = inner.to_string();
        assert!(s.contains("  9."));
        assert!(s.contains(" 10."));
    }

    #[test]
    fn display_truncation() {
        // 64 states > 50 → should truncate with "..."
        let inner = SpaceInner::Full32(FullSpace::new(2, 6, false));
        let s = inner.to_string();
        assert!(s.contains("..."), "expected truncation marker");
        // First 25 rows present (index 0 and 24)
        assert!(s.contains("\n   0."), "expected row 0");
        assert!(s.contains("\n  24."), "expected row 24");
        // Row 25 should be absent (truncated)
        assert!(!s.contains("\n  25."), "row 25 should be truncated");
        // Last 25 rows present (index 39 and 63)
        assert!(s.contains("\n  39."), "expected row 39");
        assert!(s.contains("\n  63."), "expected row 63");
    }
}
