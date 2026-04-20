//! [`StateTransitions`]: operator-side input to basis BFS.
//!
//! A type implementing [`StateTransitions`] describes how one application of
//! the operator maps a basis state to its reachable neighbours, together with
//! each contribution's complex amplitude. Amplitudes are a required part of
//! the contract (not just connectivity) because the basis needs them to
//! detect symbolic cancellations — e.g. `XX + YY` applied to `|00⟩` emits
//! two terms that cancel exactly, and the target must not enter the sector.
//!
//! This is a superset of pure graph connectivity and a subset of
//! `quspin_operator::Operator`: the cindex tag is dropped because basis
//! enumeration doesn't need it. The trait lives here in `quspin-types`
//! rather than in `quspin-bitbasis` because it is a workspace-level
//! abstraction; it has no bit-manipulation implementation.

use num_complex::Complex;

use crate::bit_int::BitInt;

/// State-to-neighbour mapping used by basis BFS, with amplitudes.
///
/// The callback `visit(amplitude, new_state)` is invoked once for every
/// non-zero term produced by applying `self` to `state`. Duplicate targets
/// are permitted — the basis accumulates per-target contributions and
/// discards states whose summed amplitudes cancel below tolerance.
///
/// The trait has method-level type parameters (`B: BitInt`, `F: FnMut`) so
/// a single operator value can drive BFS for any supported bit-integer
/// width. This makes the trait non-`dyn`-compatible; callers use static
/// dispatch.
///
/// `Send + Sync` is a supertrait so parallel BFS paths can take
/// `&impl StateTransitions` without additional bounds at each call site.
/// All in-tree implementations (built on `Operator<C>` with `C` being `u8`
/// or `u16`) trivially satisfy it.
pub trait StateTransitions: Send + Sync {
    /// Local Hilbert-space size this operator acts on.
    ///
    /// The basis checks `transitions.lhss() == basis.lhss()` before building.
    fn lhss(&self) -> usize;

    /// Call `visit(amplitude, new_state)` once for each non-zero term
    /// produced by applying `self` to `state`.
    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, visit: F);
}
