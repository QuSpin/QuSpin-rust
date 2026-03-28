use crate::bitbasis::BitInt;
use num_complex::Complex;

/// Uniform interface over `FullSpace`, `Subspace`, and `SymmetricSubspace`.
///
/// Mirrors the ad-hoc duck-typed interface used across `space.hpp`,
/// `subspace::build()`, and `qmatrix` construction in the C++.
pub trait BasisSpace<B: BitInt> {
    /// Number of lattice sites.
    fn n_sites(&self) -> usize;

    /// Number of basis states.
    fn size(&self) -> usize;

    /// The `i`-th basis state (0-indexed, ascending order).
    fn state_at(&self, i: usize) -> B;

    /// Return the index of `state`, or `None` if it is not in this space.
    fn index(&self, state: B) -> Option<usize>;
}

/// Interface for symmetry groups over a fixed `BitInt` state type.
///
/// Implemented by [`HardcoreSymmetryGrp<B>`](crate::basis::symmetry::group::HardcoreSymmetryGrp)
/// for hardcore (LHSS = 2) spin bases.  Future implementations will cover
/// higher-spin bases once `DitSymmetricSubspace` is added.
pub trait SymGrp {
    type State: BitInt;

    fn n_sites(&self) -> usize;

    /// Return the representative state (largest orbit element) and the
    /// accumulated group character for `state`.
    fn get_refstate(&self, state: Self::State) -> (Self::State, Complex<f64>);

    /// Return the representative state and the orbit norm (number of images
    /// equal to `state` under the full group action).
    fn check_refstate(&self, state: Self::State) -> (Self::State, f64);

    /// Batch variant of [`get_refstate`].
    ///
    /// Writes `(representative, grp_char)` for each element of `states` into
    /// the corresponding element of `out`.  The default implementation calls
    /// the scalar [`get_refstate`](Self::get_refstate) for each state;
    /// implementations may override this with an amortised version.
    ///
    /// # Panics
    ///
    /// Panics if `states.len() != out.len()`.
    fn get_refstate_batch(&self, states: &[Self::State], out: &mut [(Self::State, Complex<f64>)]) {
        assert_eq!(states.len(), out.len());
        for (state, o) in states.iter().zip(out.iter_mut()) {
            *o = self.get_refstate(*state);
        }
    }

    /// Batch variant of [`check_refstate`].
    ///
    /// Writes `(representative, norm)` for each element of `states` into the
    /// corresponding element of `out`.  The default implementation calls the
    /// scalar [`check_refstate`](Self::check_refstate) for each state;
    /// implementations may override this with a SIMD-optimised version.
    ///
    /// # Panics
    ///
    /// Panics if `states.len() != out.len()`.
    fn check_refstate_batch(&self, states: &[Self::State], out: &mut [(Self::State, f64)]) {
        assert_eq!(states.len(), out.len());
        for (state, o) in states.iter().zip(out.iter_mut()) {
            *o = self.check_refstate(*state);
        }
    }
}
