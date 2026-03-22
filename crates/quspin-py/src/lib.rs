pub mod dispatch;
pub mod dtype;
pub mod error;
pub mod hamiltonian;

use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<hamiltonian::PyPauliHamiltonian>()?;
    Ok(())
}
