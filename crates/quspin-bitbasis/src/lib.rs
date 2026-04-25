//! Bit-level integer manipulation for the QuSpin workspace.
//!
//! Holds the Benes permutation network, dit-manipulation helpers, and
//! state-op transforms. The `BitInt` and `StateTransitions` traits
//! (and the three `impl BitInt` blocks for `u32` / `u64` / `ruint::Uint`)
//! live in `quspin-types` — re-exported here so existing
//! `quspin_bitbasis::{BitInt, StateTransitions}` import paths keep resolving.

pub mod benes;
pub mod manip;
#[cfg(feature = "test-graphs")]
pub mod test_graphs;
pub mod transform;

pub use benes::{BenesNetwork, benes_fwd, gen_benes};
pub use manip::{DitManip, DynamicDitManip};
pub use quspin_types::{BitInt, StateTransitions};
pub use transform::{
    BenesPermDitLocations, BitStateOp, Compose, DynamicPermDitValues, FermionicBitStateOp,
    PermDitLocations, PermDitMask, PermDitValues, SignedPermDitMask,
};
