use super::HardcoreOperator;

/// Type-erased `HardcoreOperator`: either u8 or u16 cindex type.
pub enum HardcoreOperatorInner {
    Ham8(HardcoreOperator<u8>),
    Ham16(HardcoreOperator<u16>),
}

impl HardcoreOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            HardcoreOperatorInner::Ham8(h) => h.max_site(),
            HardcoreOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            HardcoreOperatorInner::Ham8(h) => h.num_cindices(),
            HardcoreOperatorInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub const fn lhss(&self) -> usize {
        2
    }
}
