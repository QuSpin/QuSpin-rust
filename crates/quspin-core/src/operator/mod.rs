//! Operator types and `*Inner` dispatch enums.
//!
//! The type definitions (`SpinOperator<C>`, `BondOperator<C>`, etc.) and the
//! `Operator` / `ParseOp` traits live in `quspin-operator`. The `*Inner`
//! dispatch enums and `apply` / `apply_and_project_to` methods remain here
//! because they depend on `crate::basis::dispatch::SpaceInner`. They will
//! move to `quspin-matrix` in PR 5.

pub mod apply;
pub mod bond;
pub mod boson;
pub mod fermion;
pub mod monomial;
pub mod pauli;
pub mod spin;

pub use bond::{BondOperator, BondOperatorInner, BondTerm};
pub use boson::{BosonOp, BosonOpEntry, BosonOperator, BosonOperatorInner};
pub use fermion::{FermionOp, FermionOpEntry, FermionOperator, FermionOperatorInner};
pub use monomial::{MonomialOperator, MonomialOperatorInner, MonomialTerm};
pub use quspin_operator::{Operator, ParseOp};
pub use spin::{SpinOp, SpinOpEntry, SpinOperator, SpinOperatorInner};
