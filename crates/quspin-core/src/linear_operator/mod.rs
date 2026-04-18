// Forwarding shim — trait + `FnLinearOperator` definitions live in `quspin-types`.
// `QMatrixOperator` remains here because it depends on `QMatrix`, which
// is still in `quspin-core`. It will move to `quspin-matrix` in PR 5.
pub use quspin_types::linear_operator::*;

mod qmatrix_op;
pub use qmatrix_op::QMatrixOperator;
