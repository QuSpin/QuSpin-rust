//! Matrix construction and operator application.
//!
//! The "glue" crate — the only one that depends on both quspin-operator
//! and quspin-basis. Contains:
//! - [`QMatrix`] and [`Hamiltonian`] types
//! - [`QMatrixOperator`] adapter implementing `LinearOperator<V>`
//! - [`apply_and_project_to`](apply::apply_and_project_to) generic function
//! - [`OperatorDispatch`] extension trait on `*OperatorInner` enums

pub mod apply;
pub mod dispatch;
pub mod hamiltonian;
mod owned_qmatrix_op;
pub mod qmatrix;
mod qmatrix_helpers;
mod qmatrix_op;

pub use apply::apply_and_project_to;
pub use dispatch::OperatorDispatch;
pub use hamiltonian::{Hamiltonian, HamiltonianInner, IntoHamiltonianInner, SchrodingerEq};
pub use owned_qmatrix_op::OwnedQMatrixOperator;
pub use qmatrix::{QMatrix, QMatrixInner};
pub use qmatrix_op::QMatrixOperator;
