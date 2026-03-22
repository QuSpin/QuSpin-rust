pub mod basis;
pub mod dispatch;
pub mod dtype;
pub mod error;
pub mod hamiltonian;
pub mod symmetry;

use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<hamiltonian::PyPauliHamiltonian>()?;
    Ok(())
}
