pub mod basis;
pub mod dtype;
pub mod error;
pub mod hamiltonian;
pub mod macros;
pub mod qmatrix;

use basis::{PyBosonBasis, PyFermionBasis, PySpinBasis};
use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpinBasis>()?;
    m.add_class::<PyFermionBasis>()?;
    m.add_class::<PyBosonBasis>()?;
    Ok(())
}
