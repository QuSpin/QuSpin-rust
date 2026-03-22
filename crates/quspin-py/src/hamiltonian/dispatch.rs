/// Type-erased `PauliHamiltonianInner`.
///
/// Selects between `PauliHamiltonian<u8>` (≤ 255 cindices / site indices) and
/// `PauliHamiltonian<u16>` (larger).  The cindex type is chosen at
/// construction time by `PyPauliHamiltonian::new`.
use quspin_core::operator::PauliHamiltonian;

/// Type-erased `PauliHamiltonian`: either u8 or u16 cindex type.
pub enum PauliHamiltonianInner {
    Ham8(PauliHamiltonian<u8>),
    Ham16(PauliHamiltonian<u16>),
}

impl PauliHamiltonianInner {
    pub fn n_sites(&self) -> usize {
        match self {
            PauliHamiltonianInner::Ham8(h) => h.n_sites(),
            PauliHamiltonianInner::Ham16(h) => h.n_sites(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            PauliHamiltonianInner::Ham8(h) => h.num_terms(),
            PauliHamiltonianInner::Ham16(h) => h.num_terms(),
        }
    }
}
