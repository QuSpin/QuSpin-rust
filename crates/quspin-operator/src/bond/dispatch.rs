use super::BondOperator;
use crate::Operator;

/// Type-erased `BondOperator`: either u8 or u16 cindex type.
///
/// The cindex type is chosen at construction time based on the number of
/// distinct cindices and the maximum site index.
pub enum BondOperatorInner {
    Ham8(BondOperator<u8>),
    Ham16(BondOperator<u16>),
}

impl BondOperatorInner {
    pub fn max_site(&self) -> usize {
        match self {
            BondOperatorInner::Ham8(h) => h.max_site(),
            BondOperatorInner::Ham16(h) => h.max_site(),
        }
    }

    pub fn num_cindices(&self) -> usize {
        match self {
            BondOperatorInner::Ham8(h) => h.num_cindices(),
            BondOperatorInner::Ham16(h) => h.num_cindices(),
        }
    }

    pub fn lhss(&self) -> usize {
        match self {
            BondOperatorInner::Ham8(h) => h.lhss(),
            BondOperatorInner::Ham16(h) => h.lhss(),
        }
    }
}
