/// Type-erased `FermionOperatorInner`.
///
/// Selects between `FermionOperator<u8>` (≤ 255 cindices / site indices)
/// and `FermionOperator<u16>` (larger), matching the cindex types supported
/// by `QMatrixInner`.
use super::FermionOperator;

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
