use crate::bitbasis::BitInt;

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
