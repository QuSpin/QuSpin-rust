use super::SpinOperator;

/// Type-erased `SpinOperator`: either u8 or u16 cindex type.
pub enum SpinOperatorInner {
    Ham8(SpinOperator<u8>),
    Ham16(SpinOperator<u16>),
}

impl SpinOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            SpinOperatorInner::Ham8(h) => h.max_site(),
            SpinOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            SpinOperatorInner::Ham8(h) => h.num_cindices(),
            SpinOperatorInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub fn lhss(&self) -> usize {
        match self {
            SpinOperatorInner::Ham8(h) => h.lhss(),
            SpinOperatorInner::Ham16(h) => h.lhss(),
        }
    }
}
