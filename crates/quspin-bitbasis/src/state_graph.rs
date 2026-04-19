//! Connectivity abstraction used by basis BFS.
//!
//! A type implementing [`StateGraph`] describes how a state maps to its
//! reachable neighbours under one application, along with each contribution's
//! complex amplitude. Amplitudes are required so the basis can detect
//! symbolic cancellations (e.g. `XX + YY` exactly cancels on `|00⟩`), which
//! determines sector membership correctly. The cindex tag is **not** needed
//! for basis enumeration, which is why this trait is narrower than
//! [`quspin_operator::Operator`](https://docs.rs/quspin-operator).

use num_complex::Complex;

use crate::int::BitInt;

/// Abstract connectivity oracle used by basis BFS.
///
/// The callback receives `(amplitude, new_state)` for every non-zero term
/// produced by one application of `self` to `state`. Duplicates are
/// permitted — the basis accumulates per-target contributions and discards
/// states whose summed amplitudes cancel below tolerance.
///
/// The trait has method-level type parameters (`B: BitInt`, `F: FnMut`) so
/// a single operator value can drive BFS for any supported bit-integer
/// width. This makes the trait non-`dyn`-compatible; callers use static
/// dispatch.
pub trait StateGraph: Send + Sync {
    /// Local Hilbert space size this operator acts on.
    ///
    /// The basis checks `graph.lhss() == basis.lhss()` before building.
    fn lhss(&self) -> usize;

    /// Call `visit(amplitude, new_state)` once for each non-zero term
    /// produced by applying `self` to `state`.
    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, visit: F);
}
