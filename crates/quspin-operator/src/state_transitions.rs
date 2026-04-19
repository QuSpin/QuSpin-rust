//! [`StateTransitions`] impls for every operator type in this crate.
//!
//! Each impl delegates to [`Operator::apply`] and drops the cindex argument —
//! basis BFS does not need it. Amplitudes are forwarded because BFS uses
//! them for symbolic-cancellation detection.

use num_complex::Complex;
use quspin_bitbasis::{BitInt, StateTransitions};

use crate::Operator;
use crate::{
    BondOperator, BondOperatorInner, BosonOperator, BosonOperatorInner, FermionOperator,
    FermionOperatorInner, HardcoreOperator, HardcoreOperatorInner, MonomialOperator,
    MonomialOperatorInner, SpinOperator, SpinOperatorInner,
};

// ---------------------------------------------------------------------------
// Per-cindex generic operator types
// ---------------------------------------------------------------------------

macro_rules! impl_state_transitions_for_operator {
    ($op:ident) => {
        // `Send + Sync` on C is required because `StateTransitions` inherits
        // those from its supertrait bound. In practice C is always u8 or u16,
        // so this is a no-op for every real caller. See the trait docs for
        // the rationale on why the supertrait bound stays in place.
        impl<C: Copy + Ord + Send + Sync> StateTransitions for $op<C> {
            #[inline]
            fn lhss(&self) -> usize {
                Operator::<C>::lhss(self)
            }

            #[inline]
            fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, mut visit: F) {
                self.apply::<B, _>(state, |_c, amp, ns| visit(amp, ns));
            }
        }
    };
}

impl_state_transitions_for_operator!(SpinOperator);
impl_state_transitions_for_operator!(BondOperator);
impl_state_transitions_for_operator!(BosonOperator);
impl_state_transitions_for_operator!(FermionOperator);
impl_state_transitions_for_operator!(HardcoreOperator);
impl_state_transitions_for_operator!(MonomialOperator);

// ---------------------------------------------------------------------------
// Dispatch enums
// ---------------------------------------------------------------------------

macro_rules! impl_state_transitions_for_inner {
    ($inner:ident) => {
        impl StateTransitions for $inner {
            #[inline]
            fn lhss(&self) -> usize {
                match self {
                    Self::Ham8(h) => StateTransitions::lhss(h),
                    Self::Ham16(h) => StateTransitions::lhss(h),
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

impl_state_transitions_for_inner!(SpinOperatorInner);
impl_state_transitions_for_inner!(BondOperatorInner);
impl_state_transitions_for_inner!(BosonOperatorInner);
impl_state_transitions_for_inner!(FermionOperatorInner);
impl_state_transitions_for_inner!(HardcoreOperatorInner);
impl_state_transitions_for_inner!(MonomialOperatorInner);
