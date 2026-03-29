/// Full-space expansion of subspace and symmetric-subspace vectors.
///
/// [`get_full_vector`] maps a coefficient vector in a reduced basis back to
/// the full Hilbert space by iterating over orbit images and accumulating
/// symmetry-weighted contributions.
///
/// # Indexing convention
///
/// `out` is indexed by the raw (unsigned integer) value of each basis state,
/// i.e. `out[state.to_usize()]`.  Callers are responsible for any reordering
/// required by a specific full-space basis convention.
use super::{
    orbit::iter_images,
    space::Subspace,
    sym_basis::{NormInt, SymBasis},
    traits::BasisSpace,
};
use crate::bitbasis::{BitInt, BitStateOp};
use num_complex::Complex;
use std::ops::{AddAssign, Mul};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Expand a single representative state from a subspace into a full-space
/// coefficient vector.
///
/// # Safety
///
/// Implementations reinterpret the raw pointer `coeff: *const T` as
/// `*const O` via [`std::ptr::read`].  The caller must guarantee that `T` and
/// `O` are layout-compatible (same size and alignment).  The valid `(T, O)`
/// pairs for each dynamic-dispatch path will be validated separately.
pub unsafe trait ExpandRefState<B: BitInt, T, O> {
    /// Accumulate the contribution of the `i`-th basis state into `out`.
    ///
    /// `coeff` points to the subspace coefficient for state `i` and will be
    /// unsafely reinterpreted as `O`.
    ///
    /// # Safety
    /// * `coeff` must be a valid, aligned, non-null pointer whose pointee is
    ///   layout-compatible with `O`.
    /// * `out` must be large enough to hold every full-space index produced by
    ///   the expansion (at minimum `lhss^n_sites` elements).
    unsafe fn expand_ref_state(&self, i: usize, coeff: *const T, out: &mut [O]);
}

// ---------------------------------------------------------------------------
// Top-level function
// ---------------------------------------------------------------------------

/// Expand a subspace coefficient vector `vec` into the full-space vector
/// `out`.
///
/// For each basis state `i`, calls [`ExpandRefState::expand_ref_state`] with
/// a pointer into `vec`.
///
/// # Safety
///
/// Inherits the safety requirements of the underlying [`ExpandRefState`]
/// implementation: the `T`/`O` pair must be layout-compatible.
///
/// # Panics
///
/// Panics (debug only) if `vec.len() != space.size()`.
pub unsafe fn get_full_vector<B, T, O, E>(space: &E, vec: &[T], out: &mut [O])
where
    B: BitInt,
    E: BasisSpace<B> + ExpandRefState<B, T, O>,
{
    debug_assert_eq!(vec.len(), space.size());
    for i in 0..space.size() {
        // SAFETY: delegated to the ExpandRefState impl.
        unsafe { space.expand_ref_state(i, vec.as_ptr().add(i), out) };
    }
}

// ---------------------------------------------------------------------------
// Subspace impl
// ---------------------------------------------------------------------------

/// For a plain subspace every state is its own representative with norm 1.
/// The coefficient is reinterpreted as `O` and added at `out[state.to_usize()]`.
unsafe impl<B, T, O> ExpandRefState<B, T, O> for Subspace<B>
where
    B: BitInt,
    O: Copy + AddAssign,
{
    unsafe fn expand_ref_state(&self, i: usize, coeff: *const T, out: &mut [O]) {
        let idx = self.state_at(i).to_usize();
        // SAFETY: caller guarantees T and O are layout-compatible.
        let val: O = unsafe { std::ptr::read(coeff as *const O) };
        out[idx] += val;
    }
}

// ---------------------------------------------------------------------------
// SymBasis impl
// ---------------------------------------------------------------------------

/// For a symmetric basis, all orbit images of the representative state are
/// enumerated. Each image at state `s` with group character `χ_g` contributes
/// `(χ_g / norm) * coeff` to `out[s.to_usize()]`.
unsafe impl<B, L, N, T, O> ExpandRefState<B, T, O> for SymBasis<B, L, N>
where
    B: BitInt,
    L: BitStateOp<B>,
    N: NormInt,
    O: Copy + AddAssign + Mul<Complex<f64>, Output = O>,
{
    unsafe fn expand_ref_state(&self, i: usize, coeff: *const T, out: &mut [O]) {
        let (ref_state, norm) = self.entry(i);
        // SAFETY: caller guarantees T and O are layout-compatible.
        let val: O = unsafe { std::ptr::read(coeff as *const O) };
        let inv_norm = 1.0 / norm;

        for (s, char_g) in iter_images(&self.lattice, &self.local, ref_state) {
            let idx = s.to_usize();
            out[idx] += val * (char_g * inv_norm);
        }
    }
}
