/// Full-space expansion of subspace and symmetric-subspace vectors.
///
/// [`get_full_vector`] maps a coefficient vector in a reduced basis back to
/// the full Hilbert space by iterating over orbit images and accumulating
/// symmetry-weighted contributions.
///
/// Full-space indices are resolved via a caller-supplied [`BasisSpace`]
/// reference, which correctly handles both the lhss=2 (bit-packed) and
/// lhss>2 (dit-packed) cases where raw state values are not contiguous
/// dense indices.
use super::{
    orbit::iter_images,
    space::Subspace,
    sym_basis::{NormInt, SymBasis},
    traits::BasisSpace,
};
use crate::bitbasis::{BitInt, BitStateOp};
use num_complex::Complex;
use std::ops::{AddAssign, Div, Mul};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Expand a single representative state from a subspace into `(state, value)`
/// pairs representing its contribution to the full-space vector.
///
/// The caller is responsible for mapping each returned state to a full-space
/// index (e.g. via [`BasisSpace::index`]) and accumulating into the output.
pub trait ExpandRefState<B: BitInt, T, O> {
    /// Yields `(full_space_state, weighted_coefficient)` pairs for the `i`-th
    /// basis state, given the subspace coefficient `coeff`.
    fn expand_ref_state_iter(&self, i: usize, coeff: &T) -> impl Iterator<Item = (B, O)>;
}

// ---------------------------------------------------------------------------
// Top-level function
// ---------------------------------------------------------------------------

/// Expand a subspace coefficient vector `vec` into the full-space vector
/// `out`.
///
/// For each basis state, calls [`ExpandRefState::expand_ref_state_iter`] and
/// accumulates contributions at the indices given by `full_space`.  `out`
/// must be pre-zeroed by the caller and have length `full_space.size()`.
///
/// # Panics
///
/// Panics (debug only) if `vec.len() != space.size()`.
pub fn get_full_vector<B, T, O, E, FS>(space: &E, full_space: &FS, vec: &[T], out: &mut [O])
where
    B: BitInt,
    E: BasisSpace<B> + ExpandRefState<B, T, O>,
    FS: BasisSpace<B>,
    O: AddAssign,
{
    debug_assert_eq!(vec.len(), space.size());
    for (i, coeff) in vec.iter().enumerate() {
        for (state, val) in space.expand_ref_state_iter(i, coeff) {
            if let Some(idx) = full_space.index(state) {
                out[idx] += val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Subspace impl
// ---------------------------------------------------------------------------

/// For a plain subspace every state is its own representative with norm 1.
/// Yields a single `(state, O::from(coeff))` pair.
impl<B, T, O> ExpandRefState<B, T, O> for Subspace<B>
where
    B: BitInt,
    T: Copy,
    O: From<T>,
{
    fn expand_ref_state_iter(&self, i: usize, coeff: &T) -> impl Iterator<Item = (B, O)> {
        std::iter::once((self.state_at(i), O::from(*coeff)))
    }
}

// ---------------------------------------------------------------------------
// SymBasis impl
// ---------------------------------------------------------------------------

/// For a symmetric basis, yields one `(state, value)` pair per orbit image.
///
/// For the `i`-th representative state `r` with orbit norm `n`, each image
/// `s = g(r)` with group character `χ_g` contributes:
/// `(s,  (coeff / n) * χ_g)`
impl<B, L, N, T, O> ExpandRefState<B, T, O> for SymBasis<B, L, N>
where
    B: BitInt,
    L: BitStateOp<B>,
    N: NormInt,
    T: Copy,
    O: Copy + From<T> + Div<f64, Output = O> + Mul<Complex<f64>, Output = O>,
{
    fn expand_ref_state_iter(&self, i: usize, coeff: &T) -> impl Iterator<Item = (B, O)> {
        let (ref_state, norm) = self.entry(i);
        let val = O::from(*coeff) / norm;
        iter_images(&self.lattice, &self.local, ref_state)
            .into_iter()
            .map(move |(s, char_g)| (s, val * char_g))
    }
}
