//! Matrix-exponential action: `exp(a·A) · v`.
//!
//! Implements the partitioned Taylor/Padé algorithm of
//! Al-Mohy & Higham (2011), porting `ExpmMultiplyParallel` from
//! [`parallel-sparse-tools`](https://github.com/QuSpin/parallel-sparse-tools).
//!
//! Generic over `impl LinearOperator<V>` (trait in `quspin-types`) — does not
//! depend on `quspin-matrix`. Concrete wrappers like `QMatrixOperator` live in
//! `quspin-matrix`.

pub mod algorithm;
pub mod expm_op;
pub mod norm_est;
pub mod params;
mod shifted_op;

pub use algorithm::PAR_THRESHOLD;
pub use expm_op::{ExpmOp, ExpmWorker, ExpmWorker2, compute_expm_params};
pub use params::{LazyNormInfo, fragment_3_1};
pub use quspin_types::{
    AtomicAccum, DynLinearOperator, ExpmComputation, FnLinearOperator, LinearOperator,
};
