use super::SpinHamiltonian;

/// Type-erased `SpinHamiltonian`: either u8 or u16 cindex type.
pub enum SpinHamiltonianInner {
    Ham8(SpinHamiltonian<u8>),
    Ham16(SpinHamiltonian<u16>),
}

impl SpinHamiltonianInner {
    pub fn max_site(&self) -> usize {
        match self {
            SpinHamiltonianInner::Ham8(h) => h.max_site(),
            SpinHamiltonianInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            SpinHamiltonianInner::Ham8(h) => h.num_cindices(),
            SpinHamiltonianInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub fn lhss(&self) -> usize {
        match self {
            SpinHamiltonianInner::Ham8(h) => h.lhss(),
            SpinHamiltonianInner::Ham16(h) => h.lhss(),
        }
    }
}
