use super::BondHamiltonian;
use crate::hamiltonian::Hamiltonian;

/// Type-erased `BondHamiltonian`: either u8 or u16 cindex type.
///
/// The cindex type is chosen at construction time based on the number of
/// distinct cindices and the maximum site index.
pub enum BondHamiltonianInner {
    Ham8(BondHamiltonian<u8>),
    Ham16(BondHamiltonian<u16>),
}

impl BondHamiltonianInner {
    pub fn n_sites(&self) -> usize {
        match self {
            BondHamiltonianInner::Ham8(h) => h.n_sites(),
            BondHamiltonianInner::Ham16(h) => h.n_sites(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            BondHamiltonianInner::Ham8(h) => h.num_cindices(),
            BondHamiltonianInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub fn lhss(&self) -> usize {
        match self {
            BondHamiltonianInner::Ham8(h) => h.lhss(),
            BondHamiltonianInner::Ham16(h) => h.lhss(),
        }
    }
}
