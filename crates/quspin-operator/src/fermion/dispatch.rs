use super::FermionOperator;

/// Type-erased `FermionOperator`: either u8 or u16 cindex type.
pub enum FermionOperatorInner {
    Ham8(FermionOperator<u8>),
    Ham16(FermionOperator<u16>),
}

impl FermionOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            FermionOperatorInner::Ham8(h) => h.max_site(),
            FermionOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            FermionOperatorInner::Ham8(h) => h.num_cindices(),
            FermionOperatorInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub const fn lhss(&self) -> usize {
        2
    }
}
