//! Bit-level integer manipulation for the QuSpin workspace.

pub mod benes;
pub mod int;
pub mod manip;
pub mod transform;

pub use benes::{BenesNetwork, benes_fwd, gen_benes};
pub use int::BitInt;
pub use manip::{DitManip, DynamicDitManip};
pub use transform::{
    BenesPermDitLocations, BitStateOp, DynamicPermDitValues, PermDitLocations, PermDitMask,
    PermDitValues,
};
