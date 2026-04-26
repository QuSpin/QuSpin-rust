//! Type-erased basis dispatch.
//!
//! Two-level umbrella over four LHSS-family enums, with [`BitBasis`]
//! exposed directly for fermion code paths so it doesn't pull in
//! dit-family monomorphizations.
//!
//! ```text
//! GenericBasis        2-arm umbrella (used by spin / boson / generic)
//! ├── BitBasis        ──→ BitBasisDefault, BitBasisLargeInt        (LHSS = 2)
//! └── DitBasis        3-arm dit umbrella (LHSS > 2)
//!     ├── TritBasis   ──→ TritBasisDefault, TritBasisLargeInt      (LHSS = 3)
//!     ├── QuatBasis   ──→ QuatBasisDefault, QuatBasisLargeInt      (LHSS = 4)
//!     └── DynDitBasis ──→ DynDitBasisDefault, DynDitBasisLargeInt  (LHSS ≥ 5)
//! ```
//!
//! [`FermionBasis`](crate::FermionBasis) wraps [`BitBasis`] directly to
//! avoid monomorphizing the dit families. [`SpinBasis`](crate::SpinBasis)
//! and [`BosonBasis`](crate::BosonBasis) wrap [`GenericBasis`].
//!
//! ## Match-arm shape
//!
//! - Per-size inner enums have at most 10 arms (Full*/Sub*/Sym* over
//!   the four widths in their tier) and contain no `#[cfg]` attributes —
//!   the `*LargeInt` enums are themselves feature-gated whole types.
//! - Family enums have 1 (`large-int` off) or 2 (on) arms. Match arms
//!   for the `LargeInt` variant carry `#[cfg(feature = "large-int")]`
//!   because the variant itself is cfg-gated; arm and variant vanish
//!   together so the match stays exhaustive in either configuration.
//! - Umbrella enums ([`GenericBasis`], [`DitBasis`]) have 2 / 3 arms
//!   and no `#[cfg]` arms.
//!
//! ## Validation
//!
//! The dispatch enums (`GenericBasis`, `DitBasis`, family enums) only
//! emit "method not supported on this variant" errors (e.g. `add_inv`
//! on the `Dit` arm of `GenericBasis`, `add_lattice` on a `Full*` /
//! `Sub*` variant of an inner enum). All argument-shape and
//! lifecycle validation lives in the concrete impls:
//!
//! - [`SymBasis::add_symmetry`](crate::sym::SymBasis::add_symmetry)
//!   validates `is_built` and the supplied permutation
//!   (length / range / bijection).
//! - [`SymBasis::build`](crate::sym::SymBasis::build) and
//!   [`Subspace::build`](crate::space::Subspace::build) validate the
//!   graph's LHSS against the basis.
//! - The per-family inner enums' `add_local` / `add_inv` validate
//!   `perm_vals` and `locs` at the level where the typed local op is
//!   constructed.
//!
//! ## Supported integer widths
//!
//! | Variant suffix | Rust type                | Bit width | Feature |
//! |----------------|--------------------------|-----------|---------|
//! | `32`           | `u32`                    | 32        | always  |
//! | `64`           | `u64`                    | 64        | always  |
//! | `128`          | `ruint::Uint<128, 2>`    | 128       | always  |
//! | `256`          | `ruint::Uint<256, 4>`    | 256       | always  |
//! | `512`          | `ruint::Uint<512, 8>`    | 512       | `large-int` |
//! | `1024`         | `ruint::Uint<1024, 16>`  | 1024      | `large-int` |
//! | `2048`         | `ruint::Uint<2048, 32>`  | 2048      | `large-int` |
//! | `4096`         | `ruint::Uint<4096, 64>`  | 4096      | `large-int` |
//! | `8192`         | `ruint::Uint<8192, 128>` | 8192      | `large-int` |

pub mod bit;
pub mod dit;
pub(crate) mod macros;
pub mod quat;
pub mod trit;
pub(crate) mod types;
pub(crate) mod validate;

pub use bit::{BitBasis, BitBasisDefault};
pub use dit::{DynDitBasis, DynDitBasisDefault};
pub use quat::{QuatBasis, QuatBasisDefault};
pub use trit::{TritBasis, TritBasisDefault};

#[cfg(feature = "large-int")]
pub use bit::BitBasisLargeInt;
#[cfg(feature = "large-int")]
pub use dit::DynDitBasisLargeInt;
#[cfg(feature = "large-int")]
pub use quat::QuatBasisLargeInt;
#[cfg(feature = "large-int")]
pub use trit::TritBasisLargeInt;

use crate::seed::{dit_state_to_str, state_to_str};
use crate::space::{FullSpace, Subspace};
use crate::spin::SpaceKind;
use crate::sym::SymBasis;
use num_complex::Complex;
use quspin_bitbasis::manip::{BITS_TABLE, DynamicDitManip};
use quspin_bitbasis::{DynamicPermDitValues, PermDitMask, PermDitValues, StateTransitions};
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// fmt_state — used by per-family inner enums to format basis states.
// ---------------------------------------------------------------------------

/// Format a basis state as a string, using bit-encoding for LHSS = 2 and
/// decimal dit-encoding for LHSS ≥ 3.
#[inline]
pub(crate) fn fmt_state<B: quspin_bitbasis::BitInt>(
    state: B,
    n_sites: usize,
    lhss: usize,
) -> String {
    if lhss == 2 {
        state_to_str(state, n_sites)
    } else {
        let manip = DynamicDitManip::new(lhss);
        dit_state_to_str(state, n_sites, &manip)
    }
}

// ---------------------------------------------------------------------------
// DitBasis — 3-arm umbrella for LHSS > 2 (Trit / Quat / Dyn)
// ---------------------------------------------------------------------------

/// Type-erased dit basis (LHSS > 2).
///
/// Three-arm enum: [`Trit`](Self::Trit) (LHSS = 3) and
/// [`Quat`](Self::Quat) (LHSS = 4) keep their const-generic
/// stack-allocated permutation arrays via
/// [`PermDitValues<3>`](quspin_bitbasis::PermDitValues) /
/// [`PermDitValues<4>`](quspin_bitbasis::PermDitValues), while
/// [`Dyn`](Self::Dyn) (LHSS ≥ 5) uses the heap-allocated
/// [`DynamicPermDitValues`](quspin_bitbasis::DynamicPermDitValues).
pub enum DitBasis {
    /// LHSS = 3 — spin-1.
    Trit(TritBasis),
    /// LHSS = 4 — spin-3/2.
    Quat(QuatBasis),
    /// LHSS ≥ 5 — bosons / higher spin.
    Dyn(DynDitBasis),
}

impl DitBasis {
    /// Construct a new dit basis. `lhss` must be ≥ 3.
    pub fn new(n_sites: usize, lhss: usize, space_kind: SpaceKind) -> Result<Self, QuSpinError> {
        if lhss < 3 {
            return Err(QuSpinError::ValueError(format!(
                "DitBasis requires lhss >= 3, got {lhss}"
            )));
        }
        match space_kind {
            SpaceKind::Full => build_dit_full(n_sites, lhss),
            SpaceKind::Sub => build_dit_sub(n_sites, lhss),
            SpaceKind::Symm => build_dit_symm(n_sites, lhss),
        }
    }

    /// Number of lattice sites.
    #[inline]
    pub fn n_sites(&self) -> usize {
        match self {
            Self::Trit(b) => b.n_sites(),
            Self::Quat(b) => b.n_sites(),
            Self::Dyn(b) => b.n_sites(),
        }
    }

    /// Local Hilbert-space size (LHSS).
    #[inline]
    pub fn lhss(&self) -> usize {
        match self {
            Self::Trit(b) => b.lhss(),
            Self::Quat(b) => b.lhss(),
            Self::Dyn(b) => b.lhss(),
        }
    }

    /// Number of basis states.
    #[inline]
    pub fn size(&self) -> usize {
        match self {
            Self::Trit(b) => b.size(),
            Self::Quat(b) => b.size(),
            Self::Dyn(b) => b.size(),
        }
    }

    /// `true` once the basis has been built (or always for `Full*`).
    #[inline]
    pub fn is_built(&self) -> bool {
        match self {
            Self::Trit(b) => b.is_built(),
            Self::Quat(b) => b.is_built(),
            Self::Dyn(b) => b.is_built(),
        }
    }

    /// Which kind of space this basis represents.
    #[inline]
    pub fn space_kind(&self) -> SpaceKind {
        match self {
            Self::Trit(b) => b.space_kind(),
            Self::Quat(b) => b.space_kind(),
            Self::Dyn(b) => b.space_kind(),
        }
    }

    /// One of `"full"`, `"subspace"`, or `"symmetric"`.
    #[inline]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Trit(b) => b.kind(),
            Self::Quat(b) => b.kind(),
            Self::Dyn(b) => b.kind(),
        }
    }

    /// `true` for `Sym*` variants (symmetry-reduced subspaces).
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        match self {
            Self::Trit(b) => b.is_symmetric(),
            Self::Quat(b) => b.is_symmetric(),
            Self::Dyn(b) => b.is_symmetric(),
        }
    }

    /// Look up the index of a basis state given as a per-site
    /// occupation byte slice.
    #[inline]
    pub fn index_of_bytes(&self, bytes: &[u8]) -> Option<usize> {
        match self {
            Self::Trit(b) => b.index_of_bytes(bytes),
            Self::Quat(b) => b.index_of_bytes(bytes),
            Self::Dyn(b) => b.index_of_bytes(bytes),
        }
    }

    /// `i`-th basis state formatted as a dit string.
    #[inline]
    pub fn state_at_str(&self, i: usize) -> String {
        match self {
            Self::Trit(b) => b.state_at_str(i),
            Self::Quat(b) => b.state_at_str(i),
            Self::Dyn(b) => b.state_at_str(i),
        }
    }

    /// `i`-th basis state formatted as a decimal integer string.
    #[inline]
    pub fn state_at_decimal_str(&self, i: usize) -> String {
        match self {
            Self::Trit(b) => b.state_at_decimal_str(i),
            Self::Quat(b) => b.state_at_decimal_str(i),
            Self::Dyn(b) => b.state_at_decimal_str(i),
        }
    }

    /// Add a lattice (site-permutation) symmetry element. Validation
    /// happens in [`SymBasis::add_symmetry`] downstream.
    pub fn add_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
    ) -> Result<(), QuSpinError> {
        match self {
            Self::Trit(b) => b.add_lattice(grp_char, perm),
            Self::Quat(b) => b.add_lattice(grp_char, perm),
            Self::Dyn(b) => b.add_lattice(grp_char, perm),
        }
    }

    /// Add a local dit-permutation symmetry element. Validation of
    /// `perm_vals` / `locs` happens in the per-family inner enum, where
    /// the typed local op is actually constructed.
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        match self {
            Self::Trit(b) => b.add_local(grp_char, perm_vals, locs),
            Self::Quat(b) => b.add_local(grp_char, perm_vals, locs),
            Self::Dyn(b) => b.add_local(grp_char, perm_vals, locs),
        }
    }

    /// Build the basis subspace from `seeds` under `graph`.
    pub fn build_seeds<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        match self {
            Self::Trit(b) => b.build_seeds(graph, seeds),
            Self::Quat(b) => b.build_seeds(graph, seeds),
            Self::Dyn(b) => b.build_seeds(graph, seeds),
        }
    }
}

// ---------------------------------------------------------------------------
// GenericBasis — 2-arm umbrella over BitBasis / DitBasis
// ---------------------------------------------------------------------------

/// Type-erased basis-space wrapper used by spin / boson / generic Python
/// bindings. [`FermionBasis`](crate::FermionBasis) bypasses this in
/// favour of [`BitBasis`] directly so its compile path doesn't pull in
/// dit-family code.
pub enum GenericBasis {
    /// LHSS = 2 — hard-core bosons / spin-½ / fermions.
    Bit(BitBasis),
    /// LHSS > 2 — three-arm dit umbrella.
    Dit(DitBasis),
}

impl GenericBasis {
    /// Construct a new generic basis.
    ///
    /// `fermionic = true` is only meaningful for `lhss == 2`; passing
    /// `true` with a different `lhss` returns an error.
    ///
    /// # Errors
    /// - `lhss < 2`
    /// - `fermionic == true && lhss != 2`
    /// - [`SpaceKind::Full`] with more than 64 bits required
    /// - [`SpaceKind::Sub`] / [`SpaceKind::Symm`] with more than 8192 bits
    pub fn new(
        n_sites: usize,
        lhss: usize,
        space_kind: SpaceKind,
        fermionic: bool,
    ) -> Result<Self, QuSpinError> {
        if lhss < 2 {
            return Err(QuSpinError::ValueError(format!(
                "lhss must be >= 2, got {lhss}"
            )));
        }
        if fermionic && lhss != 2 {
            return Err(QuSpinError::ValueError(format!(
                "fermionic=true requires lhss=2, got lhss={lhss}"
            )));
        }
        if lhss == 2 {
            Ok(Self::Bit(BitBasis::new(n_sites, space_kind, fermionic)?))
        } else {
            Ok(Self::Dit(DitBasis::new(n_sites, lhss, space_kind)?))
        }
    }

    /// Number of lattice sites.
    #[inline]
    pub fn n_sites(&self) -> usize {
        match self {
            Self::Bit(b) => b.n_sites(),
            Self::Dit(b) => b.n_sites(),
        }
    }

    /// Local Hilbert-space size (LHSS).
    #[inline]
    pub fn lhss(&self) -> usize {
        match self {
            Self::Bit(b) => b.lhss(),
            Self::Dit(b) => b.lhss(),
        }
    }

    /// Number of basis states.
    #[inline]
    pub fn size(&self) -> usize {
        match self {
            Self::Bit(b) => b.size(),
            Self::Dit(b) => b.size(),
        }
    }

    /// `true` once the basis has been built (or always for `Full*`).
    #[inline]
    pub fn is_built(&self) -> bool {
        match self {
            Self::Bit(b) => b.is_built(),
            Self::Dit(b) => b.is_built(),
        }
    }

    /// Which kind of space this basis represents.
    #[inline]
    pub fn space_kind(&self) -> SpaceKind {
        match self {
            Self::Bit(b) => b.space_kind(),
            Self::Dit(b) => b.space_kind(),
        }
    }

    /// One of `"full"`, `"subspace"`, or `"symmetric"`.
    #[inline]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Bit(b) => b.kind(),
            Self::Dit(b) => b.kind(),
        }
    }

    /// `true` for `Sym*` variants (symmetry-reduced subspaces).
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        match self {
            Self::Bit(b) => b.is_symmetric(),
            Self::Dit(b) => b.is_symmetric(),
        }
    }

    /// Look up the index of a basis state given as a per-site
    /// occupation byte slice.
    #[inline]
    pub fn index_of_bytes(&self, bytes: &[u8]) -> Option<usize> {
        match self {
            Self::Bit(b) => b.index_of_bytes(bytes),
            Self::Dit(b) => b.index_of_bytes(bytes),
        }
    }

    /// `i`-th basis state formatted as a bit / dit string.
    #[inline]
    pub fn state_at_str(&self, i: usize) -> String {
        match self {
            Self::Bit(b) => b.state_at_str(i),
            Self::Dit(b) => b.state_at_str(i),
        }
    }

    /// `i`-th basis state formatted as a decimal integer string.
    #[inline]
    pub fn state_at_decimal_str(&self, i: usize) -> String {
        match self {
            Self::Bit(b) => b.state_at_decimal_str(i),
            Self::Dit(b) => b.state_at_decimal_str(i),
        }
    }

    /// Add a lattice (site-permutation) symmetry element. Pure
    /// delegation; all validation lives in the concrete impl
    /// ([`SymBasis::add_symmetry`]).
    pub fn add_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        match self {
            Self::Bit(b) => b.add_lattice(grp_char, &perm),
            Self::Dit(b) => b.add_lattice(grp_char, &perm),
        }
    }

    /// Add an inversion (XOR bit-flip) symmetry element. LHSS = 2 only —
    /// the `Dit` arm's lack of `add_inv` is the only error this method
    /// generates; `locs` validation happens in the bit-family inner enum.
    pub fn add_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) -> Result<(), QuSpinError> {
        match self {
            Self::Bit(b) => b.add_inv(grp_char, &locs),
            Self::Dit(_) => Err(QuSpinError::ValueError(format!(
                "add_inv requires lhss=2, got lhss={}",
                self.lhss()
            ))),
        }
    }

    /// Add a local dit-permutation symmetry element. Pure delegation;
    /// `perm_vals` / `locs` validation happens in the per-family inner
    /// enum, where the typed local op is actually constructed.
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        match self {
            Self::Bit(b) => b.add_local(grp_char, perm_vals, locs),
            Self::Dit(b) => b.add_local(grp_char, perm_vals, locs),
        }
    }

    /// Build the basis subspace from `seeds` under the connectivity
    /// described by `graph`. Validation (already-built, graph LHSS
    /// mismatch, `Full` variant) happens in the concrete `build_seeds`
    /// implementations downstream.
    pub fn build<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        match self {
            Self::Bit(b) => b.build(graph, seeds),
            Self::Dit(b) => b.build(graph, seeds),
        }
    }
}

// ---------------------------------------------------------------------------
// From impls
// ---------------------------------------------------------------------------

impl From<BitBasis> for GenericBasis {
    #[inline]
    fn from(b: BitBasis) -> Self {
        Self::Bit(b)
    }
}

impl From<DitBasis> for GenericBasis {
    #[inline]
    fn from(b: DitBasis) -> Self {
        Self::Dit(b)
    }
}

impl From<TritBasis> for DitBasis {
    #[inline]
    fn from(b: TritBasis) -> Self {
        Self::Trit(b)
    }
}

impl From<QuatBasis> for DitBasis {
    #[inline]
    fn from(b: QuatBasis) -> Self {
        Self::Quat(b)
    }
}

impl From<DynDitBasis> for DitBasis {
    #[inline]
    fn from(b: DynDitBasis) -> Self {
        Self::Dyn(b)
    }
}

impl From<TritBasis> for GenericBasis {
    #[inline]
    fn from(b: TritBasis) -> Self {
        Self::Dit(DitBasis::Trit(b))
    }
}

impl From<QuatBasis> for GenericBasis {
    #[inline]
    fn from(b: QuatBasis) -> Self {
        Self::Dit(DitBasis::Quat(b))
    }
}

impl From<DynDitBasis> for GenericBasis {
    #[inline]
    fn from(b: DynDitBasis) -> Self {
        Self::Dit(DitBasis::Dyn(b))
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

const DISPLAY_HEAD: usize = 25;
const DISPLAY_TAIL: usize = 25;

/// Trait abstracting the introspection methods needed to render a basis
/// as the standard "reference states" table used by every umbrella
/// dispatch type's `Display` impl.
trait DisplayBasis {
    fn size(&self) -> usize;
    fn lhss(&self) -> usize;
    fn n_sites(&self) -> usize;
    fn is_symmetric(&self) -> bool;
    fn state_at_str(&self, i: usize) -> String;
    fn state_at_decimal_str(&self, i: usize) -> String;
}

impl DisplayBasis for BitBasis {
    fn size(&self) -> usize {
        self.size()
    }
    fn lhss(&self) -> usize {
        self.lhss()
    }
    fn n_sites(&self) -> usize {
        self.n_sites()
    }
    fn is_symmetric(&self) -> bool {
        self.is_symmetric()
    }
    fn state_at_str(&self, i: usize) -> String {
        self.state_at_str(i)
    }
    fn state_at_decimal_str(&self, i: usize) -> String {
        self.state_at_decimal_str(i)
    }
}

impl DisplayBasis for DitBasis {
    fn size(&self) -> usize {
        self.size()
    }
    fn lhss(&self) -> usize {
        self.lhss()
    }
    fn n_sites(&self) -> usize {
        self.n_sites()
    }
    fn is_symmetric(&self) -> bool {
        self.is_symmetric()
    }
    fn state_at_str(&self, i: usize) -> String {
        self.state_at_str(i)
    }
    fn state_at_decimal_str(&self, i: usize) -> String {
        self.state_at_decimal_str(i)
    }
}

impl DisplayBasis for GenericBasis {
    fn size(&self) -> usize {
        self.size()
    }
    fn lhss(&self) -> usize {
        self.lhss()
    }
    fn n_sites(&self) -> usize {
        self.n_sites()
    }
    fn is_symmetric(&self) -> bool {
        self.is_symmetric()
    }
    fn state_at_str(&self, i: usize) -> String {
        self.state_at_str(i)
    }
    fn state_at_decimal_str(&self, i: usize) -> String {
        self.state_at_decimal_str(i)
    }
}

fn fmt_display_basis<B: DisplayBasis>(b: &B, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let size = b.size();
    let lhss = b.lhss();
    let n_sites = b.n_sites();

    writeln!(f, "reference states:")?;
    writeln!(f, "array index   /   Fock state   /   integer repr.")?;

    if size == 0 {
        if b.is_symmetric() {
            write!(
                f,
                "\nThe states printed do NOT correspond to the physical states: \
                 see review arXiv:1101.3281 for more details about reference \
                 states for symmetry-reduced blocks."
            )?;
        }
        return Ok(());
    }

    let w_idx = (size - 1).to_string().len();
    let n_space = (lhss - 1).to_string().len().max(1);
    let fock_w = 1 + n_sites * n_space + n_sites.saturating_sub(1) + 1;
    let w_int = b.state_at_decimal_str(size - 1).len();

    macro_rules! write_row {
        ($i:expr) => {{
            let compact = b.state_at_str($i);
            let spaced: String = compact
                .chars()
                .map(|c| format!("{:>n_space$}", c))
                .collect::<Vec<_>>()
                .join(" ");
            let fock = format!("|{}>", spaced);
            let int_str = b.state_at_decimal_str($i);
            writeln!(
                f,
                " {:>w_idx$}.  {:<fock_w$}  {:>w_int$}",
                $i, fock, int_str,
            )?;
        }};
    }

    let truncate = size > DISPLAY_HEAD + DISPLAY_TAIL;
    if !truncate {
        for i in 0..size {
            write_row!(i);
        }
    } else {
        for i in 0..DISPLAY_HEAD {
            write_row!(i);
        }
        let pipe_pos = 1 + w_idx + 3;
        let colon_col = pipe_pos + fock_w / 2;
        writeln!(f, "{:>colon_col$}", ":")?;
        for i in size - DISPLAY_TAIL..size {
            write_row!(i);
        }
    }

    if b.is_symmetric() {
        write!(
            f,
            "\nThe states printed do NOT correspond to the physical states: \
             see review arXiv:1101.3281 for more details about reference \
             states for symmetry-reduced blocks."
        )?;
    }
    Ok(())
}

impl std::fmt::Display for BitBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_display_basis(self, f)
    }
}

impl std::fmt::Display for DitBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_display_basis(self, f)
    }
}

impl std::fmt::Display for GenericBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_display_basis(self, f)
    }
}

// ---------------------------------------------------------------------------
// Construction helpers — pick the right family + width at build time.
// ---------------------------------------------------------------------------

/// Number of bits required to encode `n_sites` dits with `lhss`
/// possible values per site.
#[inline]
fn n_bits_for(n_sites: usize, lhss: usize) -> usize {
    n_sites * BITS_TABLE[lhss]
}

fn overflow_err(n_sites: usize, lhss: usize, n_bits: usize) -> QuSpinError {
    QuSpinError::ValueError(format!(
        "n_sites={n_sites} with lhss={lhss} requires {n_bits} bits, \
         exceeding the 8192-bit maximum"
    ))
}

// --- BitBasis constructors -------------------------------------------------

impl BitBasis {
    /// Construct a new bit-family basis (LHSS = 2). `fermionic = true`
    /// enables Jordan-Wigner sign tracking.
    pub fn new(
        n_sites: usize,
        space_kind: SpaceKind,
        fermionic: bool,
    ) -> Result<Self, QuSpinError> {
        match space_kind {
            SpaceKind::Full => build_bit_full(n_sites, fermionic),
            SpaceKind::Sub => build_bit_sub(n_sites, fermionic),
            SpaceKind::Symm => build_bit_symm(n_sites, fermionic),
        }
    }

    /// Build the basis subspace from `seeds` under `graph`. Pure
    /// delegation to `build_seeds`, which calls the concrete
    /// `Subspace::build` / `SymBasis::build`; those run all
    /// per-build validation.
    pub fn build<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        self.build_seeds(graph, seeds)
    }
}

impl DitBasis {
    /// Build the basis subspace from `seeds` under `graph`. Pure
    /// delegation to `build_seeds`.
    pub fn build<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        self.build_seeds(graph, seeds)
    }
}

fn build_bit_full(n_sites: usize, fermionic: bool) -> Result<BitBasis, QuSpinError> {
    let lhss = 2;
    let n_bits = n_bits_for(n_sites, lhss);
    if n_bits > 64 {
        return Err(QuSpinError::ValueError(format!(
            "Full basis requires n_bits <= 64, but n_sites={n_sites} with \
             lhss={lhss} needs {n_bits} bits"
        )));
    }
    if n_bits <= 32 {
        let s = FullSpace::<u32>::new(lhss, n_sites, fermionic);
        Ok(BitBasis::Default(BitBasisDefault::Full32(s)))
    } else {
        let s = FullSpace::<u64>::new(lhss, n_sites, fermionic);
        Ok(BitBasis::Default(BitBasisDefault::Full64(s)))
    }
}

fn build_bit_sub(n_sites: usize, fermionic: bool) -> Result<BitBasis, QuSpinError> {
    let lhss = 2;
    let n_bits = n_bits_for(n_sites, lhss);
    macro_rules! sub_default {
        ($variant:ident, $B:ty) => {{
            let s = Subspace::<$B>::new_empty(lhss, n_sites, fermionic);
            BitBasis::Default(BitBasisDefault::$variant(s))
        }};
    }
    #[cfg(feature = "large-int")]
    macro_rules! sub_largeint {
        ($variant:ident, $B:ty) => {{
            let s = Subspace::<$B>::new_empty(lhss, n_sites, fermionic);
            BitBasis::LargeInt(BitBasisLargeInt::$variant(s))
        }};
    }
    let inner = match n_bits {
        0..=32 => sub_default!(Sub32, u32),
        33..=64 => sub_default!(Sub64, u64),
        65..=128 => sub_default!(Sub128, ::ruint::Uint<128, 2>),
        129..=256 => sub_default!(Sub256, ::ruint::Uint<256, 4>),
        #[cfg(feature = "large-int")]
        257..=512 => sub_largeint!(Sub512, ::ruint::Uint<512, 8>),
        #[cfg(feature = "large-int")]
        513..=1024 => sub_largeint!(Sub1024, ::ruint::Uint<1024, 16>),
        #[cfg(feature = "large-int")]
        1025..=2048 => sub_largeint!(Sub2048, ::ruint::Uint<2048, 32>),
        #[cfg(feature = "large-int")]
        2049..=4096 => sub_largeint!(Sub4096, ::ruint::Uint<4096, 64>),
        #[cfg(feature = "large-int")]
        4097..=8192 => sub_largeint!(Sub8192, ::ruint::Uint<8192, 128>),
        _ => return Err(overflow_err(n_sites, lhss, n_bits)),
    };
    Ok(inner)
}

fn build_bit_symm(n_sites: usize, fermionic: bool) -> Result<BitBasis, QuSpinError> {
    type L<B> = PermDitMask<B>;
    let lhss = 2;
    let n_bits = n_bits_for(n_sites, lhss);
    match n_bits {
        0..=32 => {
            let s = SymBasis::<u32, L<u32>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(BitBasis::Default(BitBasisDefault::Sym32(s)))
        }
        33..=64 => {
            let s = SymBasis::<u64, L<u64>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(BitBasis::Default(BitBasisDefault::Sym64(s)))
        }
        65..=128 => {
            type B = ::ruint::Uint<128, 2>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(BitBasis::Default(BitBasisDefault::Sym128(s)))
        }
        129..=256 => {
            type B = ::ruint::Uint<256, 4>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(BitBasis::Default(BitBasisDefault::Sym256(s)))
        }
        #[cfg(feature = "large-int")]
        257..=512 => {
            type B = ::ruint::Uint<512, 8>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(BitBasis::LargeInt(BitBasisLargeInt::Sym512(s)))
        }
        #[cfg(feature = "large-int")]
        513..=1024 => {
            type B = ::ruint::Uint<1024, 16>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(BitBasis::LargeInt(BitBasisLargeInt::Sym1024(s)))
        }
        #[cfg(feature = "large-int")]
        1025..=2048 => {
            type B = ::ruint::Uint<2048, 32>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(BitBasis::LargeInt(BitBasisLargeInt::Sym2048(s)))
        }
        #[cfg(feature = "large-int")]
        2049..=4096 => {
            type B = ::ruint::Uint<4096, 64>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(BitBasis::LargeInt(BitBasisLargeInt::Sym4096(s)))
        }
        #[cfg(feature = "large-int")]
        4097..=8192 => {
            type B = ::ruint::Uint<8192, 128>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(BitBasis::LargeInt(BitBasisLargeInt::Sym8192(s)))
        }
        _ => Err(overflow_err(n_sites, lhss, n_bits)),
    }
}

// --- DitBasis constructors -------------------------------------------------

fn build_dit_full(n_sites: usize, lhss: usize) -> Result<DitBasis, QuSpinError> {
    let n_bits = n_bits_for(n_sites, lhss);
    if n_bits > 64 {
        return Err(QuSpinError::ValueError(format!(
            "Full basis requires n_bits <= 64, but n_sites={n_sites} with \
             lhss={lhss} needs {n_bits} bits"
        )));
    }
    let fermionic = false;
    macro_rules! mk_full {
        ($B:ty, $variant:ident, $family_var:ident, $family_ty:ident, $default_ty:ident) => {{
            let s = FullSpace::<$B>::new(lhss, n_sites, fermionic);
            DitBasis::$family_var($family_ty::Default($default_ty::$variant(s)))
        }};
    }
    let inner = match (lhss, n_bits) {
        (3, 0..=32) => mk_full!(u32, Full32, Trit, TritBasis, TritBasisDefault),
        (3, _) => mk_full!(u64, Full64, Trit, TritBasis, TritBasisDefault),
        (4, 0..=32) => mk_full!(u32, Full32, Quat, QuatBasis, QuatBasisDefault),
        (4, _) => mk_full!(u64, Full64, Quat, QuatBasis, QuatBasisDefault),
        (_, 0..=32) => mk_full!(u32, Full32, Dyn, DynDitBasis, DynDitBasisDefault),
        (_, _) => mk_full!(u64, Full64, Dyn, DynDitBasis, DynDitBasisDefault),
    };
    Ok(inner)
}

fn build_dit_sub(n_sites: usize, lhss: usize) -> Result<DitBasis, QuSpinError> {
    let n_bits = n_bits_for(n_sites, lhss);
    let fermionic = false;
    macro_rules! sub_default {
        ($variant:ident, $B:ty, $family_var:ident, $family_ty:ident, $default_ty:ident) => {{
            let s = Subspace::<$B>::new_empty(lhss, n_sites, fermionic);
            DitBasis::$family_var($family_ty::Default($default_ty::$variant(s)))
        }};
    }
    #[cfg(feature = "large-int")]
    macro_rules! sub_largeint {
        ($variant:ident, $B:ty, $family_var:ident, $family_ty:ident, $largeint_ty:ident) => {{
            let s = Subspace::<$B>::new_empty(lhss, n_sites, fermionic);
            DitBasis::$family_var($family_ty::LargeInt($largeint_ty::$variant(s)))
        }};
    }
    macro_rules! family_for_lhss_default {
        (3, $variant:ident, $B:ty) => {
            sub_default!($variant, $B, Trit, TritBasis, TritBasisDefault)
        };
        (4, $variant:ident, $B:ty) => {
            sub_default!($variant, $B, Quat, QuatBasis, QuatBasisDefault)
        };
    }
    let inner = match (lhss, n_bits) {
        (3, 0..=32) => family_for_lhss_default!(3, Sub32, u32),
        (3, 33..=64) => family_for_lhss_default!(3, Sub64, u64),
        (3, 65..=128) => family_for_lhss_default!(3, Sub128, ::ruint::Uint<128, 2>),
        (3, 129..=256) => family_for_lhss_default!(3, Sub256, ::ruint::Uint<256, 4>),
        (4, 0..=32) => family_for_lhss_default!(4, Sub32, u32),
        (4, 33..=64) => family_for_lhss_default!(4, Sub64, u64),
        (4, 65..=128) => family_for_lhss_default!(4, Sub128, ::ruint::Uint<128, 2>),
        (4, 129..=256) => family_for_lhss_default!(4, Sub256, ::ruint::Uint<256, 4>),
        (_, 0..=32) => sub_default!(Sub32, u32, Dyn, DynDitBasis, DynDitBasisDefault),
        (_, 33..=64) => sub_default!(Sub64, u64, Dyn, DynDitBasis, DynDitBasisDefault),
        (_, 65..=128) => sub_default!(
            Sub128,
            ::ruint::Uint<128, 2>,
            Dyn,
            DynDitBasis,
            DynDitBasisDefault
        ),
        (_, 129..=256) => sub_default!(
            Sub256,
            ::ruint::Uint<256, 4>,
            Dyn,
            DynDitBasis,
            DynDitBasisDefault
        ),
        #[cfg(feature = "large-int")]
        (3, 257..=512) => sub_largeint!(
            Sub512,
            ::ruint::Uint<512, 8>,
            Trit,
            TritBasis,
            TritBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (3, 513..=1024) => sub_largeint!(
            Sub1024,
            ::ruint::Uint<1024, 16>,
            Trit,
            TritBasis,
            TritBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (3, 1025..=2048) => sub_largeint!(
            Sub2048,
            ::ruint::Uint<2048, 32>,
            Trit,
            TritBasis,
            TritBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (3, 2049..=4096) => sub_largeint!(
            Sub4096,
            ::ruint::Uint<4096, 64>,
            Trit,
            TritBasis,
            TritBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (3, 4097..=8192) => sub_largeint!(
            Sub8192,
            ::ruint::Uint<8192, 128>,
            Trit,
            TritBasis,
            TritBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (4, 257..=512) => sub_largeint!(
            Sub512,
            ::ruint::Uint<512, 8>,
            Quat,
            QuatBasis,
            QuatBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (4, 513..=1024) => sub_largeint!(
            Sub1024,
            ::ruint::Uint<1024, 16>,
            Quat,
            QuatBasis,
            QuatBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (4, 1025..=2048) => sub_largeint!(
            Sub2048,
            ::ruint::Uint<2048, 32>,
            Quat,
            QuatBasis,
            QuatBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (4, 2049..=4096) => sub_largeint!(
            Sub4096,
            ::ruint::Uint<4096, 64>,
            Quat,
            QuatBasis,
            QuatBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (4, 4097..=8192) => sub_largeint!(
            Sub8192,
            ::ruint::Uint<8192, 128>,
            Quat,
            QuatBasis,
            QuatBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (_, 257..=512) => sub_largeint!(
            Sub512,
            ::ruint::Uint<512, 8>,
            Dyn,
            DynDitBasis,
            DynDitBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (_, 513..=1024) => sub_largeint!(
            Sub1024,
            ::ruint::Uint<1024, 16>,
            Dyn,
            DynDitBasis,
            DynDitBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (_, 1025..=2048) => sub_largeint!(
            Sub2048,
            ::ruint::Uint<2048, 32>,
            Dyn,
            DynDitBasis,
            DynDitBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (_, 2049..=4096) => sub_largeint!(
            Sub4096,
            ::ruint::Uint<4096, 64>,
            Dyn,
            DynDitBasis,
            DynDitBasisLargeInt
        ),
        #[cfg(feature = "large-int")]
        (_, 4097..=8192) => sub_largeint!(
            Sub8192,
            ::ruint::Uint<8192, 128>,
            Dyn,
            DynDitBasis,
            DynDitBasisLargeInt
        ),
        _ => return Err(overflow_err(n_sites, lhss, n_bits)),
    };
    Ok(inner)
}

fn build_dit_symm(n_sites: usize, lhss: usize) -> Result<DitBasis, QuSpinError> {
    match lhss {
        3 => build_symm_trit(n_sites, lhss),
        4 => build_symm_quat(n_sites, lhss),
        _ => build_symm_dyn(n_sites, lhss),
    }
}

macro_rules! mk_symm_dit {
    (
        $n_sites:expr, $lhss:expr,
        $L:ty,
        $family_var:ident,
        $family_ty:ident,
        $default_ty:ident,
        $largeint_ty:ident
        $(,)?
    ) => {{
        let n_sites = $n_sites;
        let lhss = $lhss;
        let fermionic = false;
        let n_bits = n_bits_for(n_sites, lhss);
        match n_bits {
            0..=32 => {
                let s = SymBasis::<u32, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(DitBasis::$family_var($family_ty::Default(
                    $default_ty::Sym32(s),
                )))
            }
            33..=64 => {
                let s = SymBasis::<u64, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(DitBasis::$family_var($family_ty::Default(
                    $default_ty::Sym64(s),
                )))
            }
            65..=128 => {
                let s =
                    SymBasis::<::ruint::Uint<128, 2>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(DitBasis::$family_var($family_ty::Default(
                    $default_ty::Sym128(s),
                )))
            }
            129..=256 => {
                let s =
                    SymBasis::<::ruint::Uint<256, 4>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(DitBasis::$family_var($family_ty::Default(
                    $default_ty::Sym256(s),
                )))
            }
            #[cfg(feature = "large-int")]
            257..=512 => {
                let s =
                    SymBasis::<::ruint::Uint<512, 8>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(DitBasis::$family_var($family_ty::LargeInt(
                    $largeint_ty::Sym512(s),
                )))
            }
            #[cfg(feature = "large-int")]
            513..=1024 => {
                let s =
                    SymBasis::<::ruint::Uint<1024, 16>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(DitBasis::$family_var($family_ty::LargeInt(
                    $largeint_ty::Sym1024(s),
                )))
            }
            #[cfg(feature = "large-int")]
            1025..=2048 => {
                let s =
                    SymBasis::<::ruint::Uint<2048, 32>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(DitBasis::$family_var($family_ty::LargeInt(
                    $largeint_ty::Sym2048(s),
                )))
            }
            #[cfg(feature = "large-int")]
            2049..=4096 => {
                let s =
                    SymBasis::<::ruint::Uint<4096, 64>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(DitBasis::$family_var($family_ty::LargeInt(
                    $largeint_ty::Sym4096(s),
                )))
            }
            #[cfg(feature = "large-int")]
            4097..=8192 => {
                let s = SymBasis::<::ruint::Uint<8192, 128>, $L, _>::new_empty(
                    lhss, n_sites, fermionic,
                );
                Ok(DitBasis::$family_var($family_ty::LargeInt(
                    $largeint_ty::Sym8192(s),
                )))
            }
            _ => Err(overflow_err(n_sites, lhss, n_bits)),
        }
    }};
}

fn build_symm_trit(n_sites: usize, lhss: usize) -> Result<DitBasis, QuSpinError> {
    mk_symm_dit!(
        n_sites,
        lhss,
        PermDitValues<3>,
        Trit,
        TritBasis,
        TritBasisDefault,
        TritBasisLargeInt,
    )
}

fn build_symm_quat(n_sites: usize, lhss: usize) -> Result<DitBasis, QuSpinError> {
    mk_symm_dit!(
        n_sites,
        lhss,
        PermDitValues<4>,
        Quat,
        QuatBasis,
        QuatBasisDefault,
        QuatBasisLargeInt,
    )
}

fn build_symm_dyn(n_sites: usize, lhss: usize) -> Result<DitBasis, QuSpinError> {
    mk_symm_dit!(
        n_sites,
        lhss,
        DynamicPermDitValues,
        Dyn,
        DynDitBasis,
        DynDitBasisDefault,
        DynDitBasisLargeInt,
    )
}

// ---------------------------------------------------------------------------
// Width-selection macros (unchanged from the pre-split layout)
// ---------------------------------------------------------------------------

/// Pick the narrowest `BitInt` type that fits `$n_sites` bits and run
/// `$body` with `$B` bound to that type. Falls back to `$on_overflow`
/// when even the widest available integer (currently 8192 bits when
/// the `large-int` feature is on, 256 bits otherwise) is not enough.
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
        } else {
            $crate::select_b_for_large_n_sites!($n_sites, $B, $on_overflow, $body)
        }
    };
}

/// Extension of [`select_b_for_n_sites!`] for >256-bit integers.
#[cfg(not(feature = "large-int"))]
#[macro_export]
macro_rules! select_b_for_large_n_sites {
    ($n_sites:expr, $B:ident, $on_overflow:expr, $body:block) => {
        $on_overflow
    };
}

/// Extension of [`select_b_for_n_sites!`] for >256-bit integers
/// (`large-int` enabled).
#[cfg(feature = "large-int")]
#[macro_export]
macro_rules! select_b_for_large_n_sites {
    ($n_sites:expr, $B:ident, $on_overflow:expr, $body:block) => {
        if $n_sites <= 512 {
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
    use num_complex::Complex;

    #[test]
    fn generic_basis_full_lhss3() {
        let basis = GenericBasis::new(2, 3, SpaceKind::Full, false).unwrap();
        assert_eq!(basis.size(), 9);
        assert!(basis.is_built());
    }

    #[test]
    fn generic_basis_lhss1_errors() {
        assert!(GenericBasis::new(4, 1, SpaceKind::Sub, false).is_err());
    }

    #[test]
    fn generic_basis_fermionic_with_lhss3_errors() {
        assert!(GenericBasis::new(4, 3, SpaceKind::Sub, true).is_err());
    }

    #[test]
    fn generic_basis_add_lattice_on_non_symm_errors() {
        let mut basis = GenericBasis::new(4, 2, SpaceKind::Sub, false).unwrap();
        let result = basis.add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 3, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn generic_basis_build_lhss2_bit_seeds() {
        let n_sites = 4;
        let mut basis = GenericBasis::new(n_sites, 2, SpaceKind::Sub, false).unwrap();
        let graph = quspin_bitbasis::test_graphs::XAllSites::new(n_sites as u32);
        let seed = vec![0u8, 0, 0, 0];
        basis.build(&graph, &[seed]).unwrap();
        assert!(basis.is_built());
        assert_eq!(basis.size(), 1 << n_sites);
    }

    #[test]
    fn generic_basis_build_lhss3_dit_seeds() {
        use num_complex::Complex;
        use quspin_bitbasis::{BitInt, StateTransitions};

        struct NoNeighborsLhss3;
        impl StateTransitions for NoNeighborsLhss3 {
            fn lhss(&self) -> usize {
                3
            }
            fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, _state: B, _visit: F) {}
        }

        let n_sites = 3;
        let mut basis = GenericBasis::new(n_sites, 3, SpaceKind::Sub, false).unwrap();
        let seed = vec![0u8, 0, 0];
        basis.build(&NoNeighborsLhss3, &[seed]).unwrap();
        assert!(basis.is_built());
        assert_eq!(basis.size(), 1);
    }

    #[test]
    fn generic_basis_build_full_errors() {
        let mut basis = GenericBasis::new(2, 3, SpaceKind::Full, false).unwrap();
        let graph = quspin_bitbasis::test_graphs::XAllSites::new(4);
        let result = basis.build(&graph, &[vec![0u8, 0]]);
        assert!(result.is_err());
    }

    #[test]
    fn generic_basis_double_build_idempotent() {
        // Subspace::build is idempotent on duplicate seeds; the dispatch
        // layer does no built-flag check (validation lives in concrete
        // impls). A second build with the same seed is a no-op.
        let mut basis = GenericBasis::new(3, 2, SpaceKind::Sub, false).unwrap();
        let graph = quspin_bitbasis::test_graphs::XAllSites::new(3);
        basis.build(&graph, &[vec![0u8, 0, 0]]).unwrap();
        let size = basis.size();
        basis.build(&graph, &[vec![0u8, 0, 0]]).unwrap();
        assert_eq!(basis.size(), size);
    }
}
