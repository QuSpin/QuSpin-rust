// Forwarding shim — expm algorithm lives in `quspin-expm`.
// compute.rs lives in `quspin-types` (already a shim).
pub mod compute;
pub use quspin_expm::*;
