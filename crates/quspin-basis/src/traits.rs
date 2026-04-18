use quspin_bitbasis::BitInt;

/// Uniform interface over `FullSpace`, `Subspace`, and `SymBasis`.
///
/// Mirrors the ad-hoc duck-typed interface used across `space.hpp`,
/// `subspace::build()`, and `qmatrix` construction in the C++.
pub trait BasisSpace<B: BitInt> {
    /// Number of lattice sites.
    fn n_sites(&self) -> usize;

    /// Local Hilbert-space size (number of states per site).
    fn lhss(&self) -> usize;

    /// Whether Jordan-Wigner signs are tracked (fermionic basis).
    fn fermionic(&self) -> bool;

    /// Number of basis states.
    fn size(&self) -> usize;

    /// The `i`-th basis state (0-indexed, ascending order).
    fn state_at(&self, i: usize) -> B;

    /// Return the index of `state`, or `None` if it is not in this space.
    fn index(&self, state: B) -> Option<usize>;
}

/// Interface for symmetry groups over a fixed `BitInt` state type.
///
/// Orbit computation is provided directly by [`SymBasis<B, L, N>`](crate::sym_basis::SymBasis)
/// via inherent methods rather than through this trait.  This trait retains
/// only the metadata accessor needed for construction-time validation.
pub trait SymGrp {
    type State: BitInt;

    fn n_sites(&self) -> usize;
}
