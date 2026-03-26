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
}
