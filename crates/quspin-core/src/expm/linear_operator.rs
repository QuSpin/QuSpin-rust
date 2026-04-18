// Re-export from the crate-level `linear_operator` module.
// Internal `expm` sub-modules (`algorithm`, `params`, `norm_est`) use
// `super::linear_operator::LinearOperator`, which resolves here.
pub use crate::linear_operator::{LinearOperator, QMatrixOperator};
