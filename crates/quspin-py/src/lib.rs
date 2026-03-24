pub mod basis;
pub mod dtype;
pub mod error;
pub mod hamiltonian;
pub mod macros;
pub mod qmatrix;

use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<basis::PyLatticeElement>()?;
    m.add_class::<basis::PyGrpElement>()?;
    m.add_class::<basis::PySymmetryGrp>()?;
    m.add_class::<hamiltonian::PyHardcoreHamiltonian>()?;
    m.add_class::<hamiltonian::PyBondTerm>()?;
    m.add_class::<hamiltonian::PyBondHamiltonian>()?;
    m.add_class::<basis::PyHardcoreBasis>()?;
    m.add_class::<qmatrix::PyQMatrix>()?;
    Ok(())
}
