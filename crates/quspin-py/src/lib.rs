pub mod basis;
pub mod dtype;
pub mod error;
pub mod hamiltonian;
pub mod macros;
pub mod operator;
pub mod qmatrix;

use basis::{PyBosonBasis, PyFermionBasis, PySpinBasis};
use operator::{PyBondOperator, PyBosonOperator, PyFermionOperator, PyPauliOperator};
use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Basis types
    m.add_class::<PySpinBasis>()?;
    m.add_class::<PyFermionBasis>()?;
    m.add_class::<PyBosonBasis>()?;
    // Operator types
    m.add_class::<PyPauliOperator>()?;
    m.add_class::<PyBondOperator>()?;
    m.add_class::<PyBosonOperator>()?;
    m.add_class::<PyFermionOperator>()?;
    Ok(())
}
