/// Type-erased `FermionHamiltonianInner`.
///
/// Selects between `FermionHamiltonian<u8>` (≤ 255 cindices / site indices)
/// and `FermionHamiltonian<u16>` (larger), matching the cindex types supported
/// by `QMatrixInner`.
use super::FermionHamiltonian;

pub enum FermionHamiltonianInner {
    Ham8(FermionHamiltonian<u8>),
    Ham16(FermionHamiltonian<u16>),
}

impl FermionHamiltonianInner {
    pub fn max_site(&self) -> usize {
        match self {
            FermionHamiltonianInner::Ham8(h) => h.max_site(),
            FermionHamiltonianInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            FermionHamiltonianInner::Ham8(h) => h.num_cindices(),
            FermionHamiltonianInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub const fn lhss(&self) -> usize {
        2
    }
}
