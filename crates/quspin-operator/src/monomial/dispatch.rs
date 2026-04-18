use super::MonomialOperator;

/// Type-erased `MonomialOperator`: either u8 or u16 cindex type.
pub enum MonomialOperatorInner {
    Ham8(MonomialOperator<u8>),
    Ham16(MonomialOperator<u16>),
}

impl MonomialOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            MonomialOperatorInner::Ham8(h) => h.max_site(),
            MonomialOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            MonomialOperatorInner::Ham8(h) => h.num_cindices(),
            MonomialOperatorInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub fn lhss(&self) -> usize {
        match self {
            MonomialOperatorInner::Ham8(h) => h.lhss(),
            MonomialOperatorInner::Ham16(h) => h.lhss(),
        }
    }
}
