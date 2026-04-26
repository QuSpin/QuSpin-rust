//! Pure re-export facade.
//!
//! All logic lives in the `quspin-*` sub-crates:
//!
//! - [`quspin-types`](quspin_types)     — foundation (errors, primitives, dtype, compute, LinearOperator trait)
//! - [`quspin-bitbasis`](quspin_bitbasis) — bit-level integer manipulation
//! - [`quspin-operator`](quspin_operator) — operator types and traits
//! - [`quspin-basis`](quspin_basis)      — Hilbert-space enumeration
//! - [`quspin-matrix`](quspin_matrix)    — `QMatrix`, `Hamiltonian`, apply/dispatch glue
//! - [`quspin-expm`](quspin_expm)        — Taylor-series matrix exponential
//! - [`quspin-krylov`](quspin_krylov)    — Lanczos, FTLM, LTLM

pub use quspin_bitbasis as bitbasis_crate;
pub use quspin_types::{compute, dtype, error, primitive};

// Flat re-export at crate root
pub use quspin_basis::*;
pub use quspin_bitbasis::*;
pub use quspin_expm::{
    expm_multiply, expm_multiply_auto, expm_multiply_auto_into, expm_multiply_many,
    expm_multiply_many_auto, expm_multiply_many_auto_into, expm_multiply_par,
};
pub use quspin_krylov::*;
pub use quspin_matrix::{
    Hamiltonian, HamiltonianInner, IntoHamiltonianInner, OperatorDispatch, QMatrix, QMatrixInner,
    QMatrixOperator, SchrodingerEq, apply_and_project_to,
};
pub use quspin_operator::{
    BondOperator, BondOperatorInner, BondTerm, BosonOp, BosonOpEntry, BosonOperator,
    BosonOperatorInner, FermionOp, FermionOpEntry, FermionOperator, FermionOperatorInner,
    HardcoreOp, HardcoreOperator, HardcoreOperatorInner, MonomialOperator, MonomialOperatorInner,
    MonomialTerm, OpEntry, Operator, ParseOp, SpinOp, SpinOpEntry, SpinOperator, SpinOperatorInner,
};
pub use quspin_types::{
    AtomicAccum, AtomicComplex32, AtomicComplex64, AtomicF32, AtomicF64, CIndexDType,
    DynLinearOperator, ExpmComputation, FnLinearOperator, FnLinearOperatorBuilder, LinearOperator,
    Primitive, QuSpinError, ValueDType,
};

// Modules reachable as paths (for quspin-py's deep imports like
// `quspin_core::operator::bond::BondOperator` or
// `quspin_core::basis::dispatch::GenericBasis`):
pub mod basis {
    pub use quspin_basis::*;
}
pub mod bitbasis {
    pub use quspin_bitbasis::*;
}
pub mod operator {
    pub use quspin_operator::*;

    pub mod bond {
        pub use quspin_operator::bond::*;
    }
    pub mod boson {
        pub use quspin_operator::boson::*;
    }
    pub mod fermion {
        pub use quspin_operator::fermion::*;
    }
    pub mod monomial {
        pub use quspin_operator::monomial::*;
    }
    pub mod pauli {
        pub use quspin_operator::pauli::*;
    }
    pub mod spin {
        pub use quspin_operator::spin::*;
    }
}
pub mod qmatrix {
    pub use quspin_matrix::qmatrix::*;
}
pub mod hamiltonian {
    pub use quspin_matrix::hamiltonian::*;
}
pub mod krylov {
    pub use quspin_krylov::*;
}
pub mod expm {
    pub use quspin_expm::*;
}
pub mod linear_operator {
    pub use quspin_matrix::QMatrixOperator;
    pub use quspin_types::linear_operator::*;
}
