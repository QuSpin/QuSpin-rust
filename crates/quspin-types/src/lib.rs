//! Foundation types shared across the QuSpin workspace.
//!
//! Pure abstractions — no physics logic. Every other `quspin-*` crate depends
//! on this one.

pub mod compute;
pub mod dtype;
pub mod error;
pub mod linear_operator;
pub mod primitive;

pub use compute::{
    AtomicAccum, AtomicComplex32, AtomicComplex64, AtomicF32, AtomicF64, ExpmComputation,
};
pub use dtype::{CIndexDType, ValueDType};
pub use error::QuSpinError;
pub use linear_operator::{
    DynLinearOperator, FnLinearOperator, FnLinearOperatorBuilder, LinearOperator,
};
pub use primitive::Primitive;
