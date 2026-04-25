//! Three-level type-erased basis dispatch.
//!
//! Replaces the previous monolithic 47-variant `SpaceInner` enum with a
//! per-LHSS-family hierarchy:
//!
//! ```text
//! SpaceInner               4-arm outer (family selector)
//! ├── SpaceInnerBit        ──→ SpaceInnerBitDefault, SpaceInnerBitLargeInt   (LHSS=2)
//! ├── SpaceInnerTrit       ──→ SpaceInnerTritDefault, SpaceInnerTritLargeInt (LHSS=3)
//! ├── SpaceInnerQuat       ──→ SpaceInnerQuatDefault, SpaceInnerQuatLargeInt (LHSS=4)
//! └── SpaceInnerDit        ──→ SpaceInnerDitDefault, SpaceInnerDitLargeInt   (LHSS≥5)
//! ```
//!
//! Each per-size inner enum has at most 10 arms; each family enum has 1
//! or 2 arms (Default + optional LargeInt); the outer enum has 4. No
//! match anywhere exceeds 10 arms, and `#[cfg(feature = "large-int")]`
//! gates whole types — never appears inside a match expression.
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

pub use bit::{SpaceInnerBit, SpaceInnerBitDefault};
pub use dit::{SpaceInnerDit, SpaceInnerDitDefault};
pub use quat::{SpaceInnerQuat, SpaceInnerQuatDefault};
pub use trit::{SpaceInnerTrit, SpaceInnerTritDefault};

#[cfg(feature = "large-int")]
pub use bit::SpaceInnerBitLargeInt;
#[cfg(feature = "large-int")]
pub use dit::SpaceInnerDitLargeInt;
#[cfg(feature = "large-int")]
pub use quat::SpaceInnerQuatLargeInt;
#[cfg(feature = "large-int")]
pub use trit::SpaceInnerTritLargeInt;

use crate::seed::{dit_state_to_str, state_to_str};
use quspin_bitbasis::manip::DynamicDitManip;

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
// Outer SpaceInner
// ---------------------------------------------------------------------------

/// Type-erased basis-space wrapper.
///
/// The four arms select which LHSS family the basis lives in. Methods
/// on this type are 4-arm matches that delegate to the family enum,
/// which in turn delegates to its `Default` / `LargeInt` sub-enum.
pub enum SpaceInner {
    /// LHSS = 2 — hard-core bosons / spin-½ / fermions.
    Bit(SpaceInnerBit),
    /// LHSS = 3 — spin-1.
    Trit(SpaceInnerTrit),
    /// LHSS = 4 — spin-3/2.
    Quat(SpaceInnerQuat),
    /// LHSS ≥ 5 — bosons / higher spin.
    Dit(SpaceInnerDit),
}

impl SpaceInner {
    /// Number of lattice sites.
    #[inline]
    pub fn n_sites(&self) -> usize {
        match self {
            Self::Bit(b) => b.n_sites(),
            Self::Trit(b) => b.n_sites(),
            Self::Quat(b) => b.n_sites(),
            Self::Dit(b) => b.n_sites(),
        }
    }

    /// Local Hilbert-space size (LHSS).
    #[inline]
    pub fn lhss(&self) -> usize {
        match self {
            Self::Bit(b) => b.lhss(),
            Self::Trit(b) => b.lhss(),
            Self::Quat(b) => b.lhss(),
            Self::Dit(b) => b.lhss(),
        }
    }

    /// Number of basis states.
    #[inline]
    pub fn size(&self) -> usize {
        match self {
            Self::Bit(b) => b.size(),
            Self::Trit(b) => b.size(),
            Self::Quat(b) => b.size(),
            Self::Dit(b) => b.size(),
        }
    }

    /// `true` once the basis has been built (or always for `Full*`).
    #[inline]
    pub fn is_built(&self) -> bool {
        match self {
            Self::Bit(b) => b.is_built(),
            Self::Trit(b) => b.is_built(),
            Self::Quat(b) => b.is_built(),
            Self::Dit(b) => b.is_built(),
        }
    }

    /// Which kind of space this basis represents.
    #[inline]
    pub fn space_kind(&self) -> crate::spin::SpaceKind {
        match self {
            Self::Bit(b) => b.space_kind(),
            Self::Trit(b) => b.space_kind(),
            Self::Quat(b) => b.space_kind(),
            Self::Dit(b) => b.space_kind(),
        }
    }

    /// One of `"full"`, `"subspace"`, or `"symmetric"`.
    #[inline]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Bit(b) => b.kind(),
            Self::Trit(b) => b.kind(),
            Self::Quat(b) => b.kind(),
            Self::Dit(b) => b.kind(),
        }
    }

    /// `true` for `Sym*` variants (symmetry-reduced subspaces).
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        match self {
            Self::Bit(b) => b.is_symmetric(),
            Self::Trit(b) => b.is_symmetric(),
            Self::Quat(b) => b.is_symmetric(),
            Self::Dit(b) => b.is_symmetric(),
        }
    }

    /// Look up the index of a basis state given as a per-site
    /// occupation byte slice.
    #[inline]
    pub fn index_of_bytes(&self, bytes: &[u8]) -> Option<usize> {
        match self {
            Self::Bit(b) => b.index_of_bytes(bytes),
            Self::Trit(b) => b.index_of_bytes(bytes),
            Self::Quat(b) => b.index_of_bytes(bytes),
            Self::Dit(b) => b.index_of_bytes(bytes),
        }
    }

    /// `i`-th basis state formatted as a bit / dit string.
    #[inline]
    pub fn state_at_str(&self, i: usize) -> String {
        match self {
            Self::Bit(b) => b.state_at_str(i),
            Self::Trit(b) => b.state_at_str(i),
            Self::Quat(b) => b.state_at_str(i),
            Self::Dit(b) => b.state_at_str(i),
        }
    }

    /// `i`-th basis state formatted as a decimal integer string.
    #[inline]
    pub fn state_at_decimal_str(&self, i: usize) -> String {
        match self {
            Self::Bit(b) => b.state_at_decimal_str(i),
            Self::Trit(b) => b.state_at_decimal_str(i),
            Self::Quat(b) => b.state_at_decimal_str(i),
            Self::Dit(b) => b.state_at_decimal_str(i),
        }
    }

    /// Add a lattice (site-permutation) symmetry element.
    ///
    /// Errors on `Full*` and `Sub*` variants (any family). Caller is
    /// responsible for upstream argument validation (perm length /
    /// contents); this only routes to the per-family dispatcher.
    pub fn add_lattice(
        &mut self,
        grp_char: num_complex::Complex<f64>,
        perm: &[usize],
    ) -> Result<(), quspin_types::QuSpinError> {
        match self {
            Self::Bit(b) => b.add_lattice(grp_char, perm),
            Self::Trit(b) => b.add_lattice(grp_char, perm),
            Self::Quat(b) => b.add_lattice(grp_char, perm),
            Self::Dit(b) => b.add_lattice(grp_char, perm),
        }
    }

    /// Add an inversion (XOR bit-flip) symmetry element. LHSS = 2 only —
    /// errors on `Trit` / `Quat` / `Dit` families.
    pub fn add_inv(
        &mut self,
        grp_char: num_complex::Complex<f64>,
        locs: &[usize],
    ) -> Result<(), quspin_types::QuSpinError> {
        match self {
            Self::Bit(b) => b.add_inv(grp_char, locs),
            _ => Err(quspin_types::QuSpinError::ValueError(
                "add_inv requires a Bit family (LHSS=2) basis".into(),
            )),
        }
    }

    /// Add a local dit-permutation symmetry element. The required shape
    /// of `perm_vals` depends on the family's LHSS:
    ///
    /// - Bit (LHSS=2): `perm_vals` must be `[1, 0]`.
    /// - Trit / Quat: `perm_vals.len()` must be 3 / 4.
    /// - Dit: `perm_vals.len()` must equal the basis's runtime LHSS.
    pub fn add_local(
        &mut self,
        grp_char: num_complex::Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), quspin_types::QuSpinError> {
        match self {
            Self::Bit(b) => b.add_local(grp_char, perm_vals, locs),
            Self::Trit(b) => b.add_local(grp_char, perm_vals, locs),
            Self::Quat(b) => b.add_local(grp_char, perm_vals, locs),
            Self::Dit(b) => b.add_local(grp_char, perm_vals, locs),
        }
    }

    /// Build the basis subspace from `seeds` under `graph`.
    ///
    /// Family-specific seed decoding (bit-string for Bit, dit-encoded
    /// for the others) lives in each family's `build_seeds` method.
    pub fn build_seeds<G: quspin_bitbasis::StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), quspin_types::QuSpinError> {
        match self {
            Self::Bit(b) => b.build_seeds(graph, seeds),
            Self::Trit(b) => b.build_seeds(graph, seeds),
            Self::Quat(b) => b.build_seeds(graph, seeds),
            Self::Dit(b) => b.build_seeds(graph, seeds),
        }
    }
}

// ---------------------------------------------------------------------------
// From impls — outer level (family selector)
// ---------------------------------------------------------------------------

impl From<SpaceInnerBit> for SpaceInner {
    #[inline]
    fn from(b: SpaceInnerBit) -> Self {
        Self::Bit(b)
    }
}

impl From<SpaceInnerTrit> for SpaceInner {
    #[inline]
    fn from(b: SpaceInnerTrit) -> Self {
        Self::Trit(b)
    }
}

impl From<SpaceInnerQuat> for SpaceInner {
    #[inline]
    fn from(b: SpaceInnerQuat) -> Self {
        Self::Quat(b)
    }
}

impl From<SpaceInnerDit> for SpaceInner {
    #[inline]
    fn from(b: SpaceInnerDit) -> Self {
        Self::Dit(b)
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
        let lhss = self.lhss();
        let n_sites = self.n_sites();

        writeln!(f, "reference states:")?;
        writeln!(f, "array index   /   Fock state   /   integer repr.")?;

        if size == 0 {
            if self.is_symmetric() {
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
        let w_int = self.state_at_decimal_str(size - 1).len();

        macro_rules! write_row {
            ($i:expr) => {{
                let compact = self.state_at_str($i);
                let spaced: String = compact
                    .chars()
                    .map(|c| format!("{:>n_space$}", c))
                    .collect::<Vec<_>>()
                    .join(" ");
                let fock = format!("|{}>", spaced);
                let int_str = self.state_at_decimal_str($i);
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

        if self.is_symmetric() {
            write!(
                f,
                "\nThe states printed do NOT correspond to the physical states: \
                 see review arXiv:1101.3281 for more details about reference \
                 states for symmetry-reduced blocks."
            )?;
        }
        Ok(())
    }
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
///
/// When the `large-int` feature is disabled this immediately
/// evaluates `$on_overflow`; when enabled it continues the ladder up
/// to 8192 bits.
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
