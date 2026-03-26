use super::BosonHamiltonian;

/// Type-erased `BosonHamiltonian`: either u8 or u16 cindex type.
pub enum BosonHamiltonianInner {
    Ham8(BosonHamiltonian<u8>),
    Ham16(BosonHamiltonian<u16>),
}

impl BosonHamiltonianInner {
    pub fn max_site(&self) -> usize {
        match self {
            BosonHamiltonianInner::Ham8(h) => h.max_site(),
            BosonHamiltonianInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            BosonHamiltonianInner::Ham8(h) => h.num_cindices(),
            BosonHamiltonianInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub fn lhss(&self) -> usize {
        match self {
            BosonHamiltonianInner::Ham8(h) => h.lhss(),
            BosonHamiltonianInner::Ham16(h) => h.lhss(),
        }
    }
}
