//! [`StateGraph`] impls for every operator type in this crate.
//!
//! Operators implement [`StateGraph`] by delegating to [`Operator::apply`]
//! and dropping the cindex argument — BFS uses the amplitude (for
//! cancellation detection) but ignores cindex.

use num_complex::Complex;
use quspin_bitbasis::{BitInt, StateGraph};

use crate::Operator;
use crate::{
    BondOperator, BondOperatorInner, BosonOperator, BosonOperatorInner, FermionOperator,
    FermionOperatorInner, HardcoreOperator, HardcoreOperatorInner, MonomialOperator,
    MonomialOperatorInner, SpinOperator, SpinOperatorInner,
};

// ---------------------------------------------------------------------------
// Per-cindex generic operator types
// ---------------------------------------------------------------------------

macro_rules! impl_state_graph_for_operator {
    ($op:ident) => {
        impl<C: Copy + Ord + Send + Sync> StateGraph for $op<C> {
            #[inline]
            fn lhss(&self) -> usize {
                <Self as Operator<C>>::lhss(self)
            }

            #[inline]
            fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, mut visit: F) {
                <Self as Operator<C>>::apply::<B, _>(self, state, |_c, amp, ns| visit(amp, ns));
            }
        }
    };
}

impl_state_graph_for_operator!(SpinOperator);
impl_state_graph_for_operator!(BondOperator);
impl_state_graph_for_operator!(BosonOperator);
impl_state_graph_for_operator!(FermionOperator);
impl_state_graph_for_operator!(HardcoreOperator);
impl_state_graph_for_operator!(MonomialOperator);

// ---------------------------------------------------------------------------
// Dispatch enums
// ---------------------------------------------------------------------------

macro_rules! impl_state_graph_for_inner {
    ($inner:ident) => {
        impl StateGraph for $inner {
            #[inline]
            fn lhss(&self) -> usize {
                match self {
                    Self::Ham8(h) => StateGraph::lhss(h),
                    Self::Ham16(h) => StateGraph::lhss(h),
                }
            }

            #[inline]
            fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, visit: F) {
                match self {
                    Self::Ham8(h) => h.neighbors(state, visit),
                    Self::Ham16(h) => h.neighbors(state, visit),
                }
            }
        }
    };
}

impl_state_graph_for_inner!(SpinOperatorInner);
impl_state_graph_for_inner!(BondOperatorInner);
impl_state_graph_for_inner!(BosonOperatorInner);
impl_state_graph_for_inner!(FermionOperatorInner);
impl_state_graph_for_inner!(HardcoreOperatorInner);
impl_state_graph_for_inner!(MonomialOperatorInner);
