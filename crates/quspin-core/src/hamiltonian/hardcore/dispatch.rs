/// Type-erased `HardcoreHamiltonianInner`.
///
/// Selects between `HardcoreHamiltonian<u8>` (≤ 255 cindices / site indices) and
/// `HardcoreHamiltonian<u16>` (larger).  The cindex type is chosen at
/// construction time by `PyHardcoreHamiltonian::new`.
use super::HardcoreHamiltonian;

/// Type-erased `HardcoreHamiltonian`: either u8 or u16 cindex type.
pub enum HardcoreHamiltonianInner {
    Ham8(HardcoreHamiltonian<u8>),
    Ham16(HardcoreHamiltonian<u16>),
}

impl HardcoreHamiltonianInner {
    pub fn max_site(&self) -> usize {
        match self {
            HardcoreHamiltonianInner::Ham8(h) => h.max_site(),
            HardcoreHamiltonianInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            HardcoreHamiltonianInner::Ham8(h) => h.num_cindices(),
            HardcoreHamiltonianInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub const fn lhss(&self) -> usize {
        2
    }
}
