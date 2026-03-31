pub mod basis;
pub mod dtype;
pub mod error;
pub mod hamiltonian;
pub mod macros;
pub mod qmatrix;

use pyo3::prelude::*;

#[pymodule]
fn _rs(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
