pub mod basis;
pub mod dispatch;
pub mod dtype;
pub mod error;
pub mod hamiltonian;
pub mod qmatrix;
pub mod symmetry;

use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<symmetry::PyLatticeElement>()?;
    m.add_class::<symmetry::PyGrpElement>()?;
    m.add_class::<symmetry::PySymmetryGrp>()?;
    m.add_class::<hamiltonian::PyPauliHamiltonian>()?;
    m.add_class::<basis::PyHardcoreBasis>()?;
    m.add_class::<qmatrix::PyQMatrix>()?;
    Ok(())
}
