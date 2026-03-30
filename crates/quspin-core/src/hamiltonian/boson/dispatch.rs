use super::BosonOperator;

/// Type-erased `BosonOperator`: either u8 or u16 cindex type.
pub enum BosonOperatorInner {
    Ham8(BosonOperator<u8>),
    Ham16(BosonOperator<u16>),
}

impl BosonOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            BosonOperatorInner::Ham8(h) => h.max_site(),
            BosonOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            BosonOperatorInner::Ham8(h) => h.num_cindices(),
            BosonOperatorInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub fn lhss(&self) -> usize {
        match self {
            BosonOperatorInner::Ham8(h) => h.lhss(),
            BosonOperatorInner::Ham16(h) => h.lhss(),
        }
    }
}
