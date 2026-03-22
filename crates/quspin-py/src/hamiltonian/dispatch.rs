/// Type-erased `HardcoreHamiltonianInner`.
///
/// Selects between `HardcoreHamiltonian<u8>` (≤ 255 cindices / site indices) and
/// `HardcoreHamiltonian<u16>` (larger).  The cindex type is chosen at
/// construction time by `PyHardcoreHamiltonian::new`.
use quspin_core::hamiltonian::hardcore::HardcoreHamiltonian;

/// Type-erased `HardcoreHamiltonian`: either u8 or u16 cindex type.
pub enum HardcoreHamiltonianInner {
    Ham8(HardcoreHamiltonian<u8>),
    Ham16(HardcoreHamiltonian<u16>),
}

impl HardcoreHamiltonianInner {
    pub fn n_sites(&self) -> usize {
        match self {
            HardcoreHamiltonianInner::Ham8(h) => h.n_sites(),
            HardcoreHamiltonianInner::Ham16(h) => h.n_sites(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            HardcoreHamiltonianInner::Ham8(h) => h.num_terms(),
            HardcoreHamiltonianInner::Ham16(h) => h.num_terms(),
        }
    }
}
