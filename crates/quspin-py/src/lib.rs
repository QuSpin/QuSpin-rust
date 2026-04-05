pub mod basis;
pub mod dtype;
pub mod error;
pub mod hamiltonian;
pub mod krylov;
pub mod operator;
pub mod qmatrix;
pub mod schrodinger;

use basis::{PyBosonBasis, PyFermionBasis, PyGenericBasis, PySpinBasis};
use hamiltonian::{PyHamiltonian, PyStatic};
use krylov::{PyEigSolver, PyFTLM, PyFTLMDynamic, PyLTLM};
use operator::{
    PyBondOperator, PyBosonOperator, PyFermionOperator, PyMonomialOperator, PyPauliOperator,
};
use pyo3::prelude::*;
use qmatrix::PyQMatrix;
use schrodinger::PySchrodingerEq;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Basis types
    m.add_class::<PySpinBasis>()?;
    m.add_class::<PyFermionBasis>()?;
    m.add_class::<PyBosonBasis>()?;
    m.add_class::<PyGenericBasis>()?;
    // Operator types
    m.add_class::<PyPauliOperator>()?;
    m.add_class::<PyBondOperator>()?;
    m.add_class::<PyBosonOperator>()?;
    m.add_class::<PyFermionOperator>()?;
    m.add_class::<PyMonomialOperator>()?;
    // Matrix, Hamiltonian, and integrator types
    m.add_class::<PyQMatrix>()?;
    m.add_class::<PyStatic>()?;
    m.add_class::<PyHamiltonian>()?;
    m.add_class::<PySchrodingerEq>()?;
    // Krylov subspace methods
    m.add_class::<PyEigSolver>()?;
    m.add_class::<PyFTLM>()?;
    m.add_class::<PyLTLM>()?;
    m.add_class::<PyFTLMDynamic>()?;
    Ok(())
}
