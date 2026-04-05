/// Full-space expansion of subspace and symmetric-subspace vectors.
///
/// [`expand_to_map`] is the core primitive: it expands a reduced-basis
/// coefficient vector into a sparse `HashMap<B, Complex<f64>>` whose keys are
/// full-space basis states and whose values are the accumulated complex
/// amplitudes.  Memory scales with the number of states that actually carry
/// non-zero weight, not with the full Hilbert-space dimension — important for
/// particle-number-conserving subspaces where the subspace is much smaller
/// than the full space.
///
/// [`get_full_vector`] wraps [`expand_to_map`] and scatters the sparse result
/// into a pre-allocated dense output slice indexed by a caller-supplied
/// [`BasisSpace`].
///
/// [`reduced_density_matrix`] uses the same sparse map to compute the reduced
/// density matrix for an arbitrary subsystem without ever materialising the
/// full-space vector.
use super::{
    orbit::iter_images,
    space::{FullSpace, Subspace},
    sym::{NormInt, SymBasis},
    traits::BasisSpace,
};
use crate::bitbasis::{BenesPermDitLocations, BitInt, BitStateOp, manip::DynamicDitManip};
use ndarray::{Array2, ArrayBase, ArrayView1, Data, Ix1, aview1};
use num_complex::Complex;
use std::collections::HashMap;
use std::ops::AddAssign;

// ---------------------------------------------------------------------------
// AsStateVec — blanket conversion to ArrayView1
// ---------------------------------------------------------------------------

/// Convert a coefficient vector into an [`ArrayView1`].
///
/// Implemented for:
/// - `[T]` — covers `&[T]` and slice expressions via `?Sized` function bounds
/// - `Vec<T>`
/// - `ArrayBase<S, Ix1>` — covers `Array1<T>`, `ArrayView1<'_, T>`, etc.
pub trait ToView<T> {
    fn to_view(&self) -> ArrayView1<'_, T>;
}

impl<T> ToView<T> for [T] {
    fn to_view(&self) -> ArrayView1<'_, T> {
        aview1(self)
    }
}

impl<T> ToView<T> for Vec<T> {
    fn to_view(&self) -> ArrayView1<'_, T> {
        aview1(self.as_slice())
    }
}

impl<T, S: Data<Elem = T>> ToView<T> for ArrayBase<S, Ix1> {
    fn to_view(&self) -> ArrayView1<'_, T> {
        self.view()
    }
}

// ---------------------------------------------------------------------------
// ExpandRefState trait
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
// Core sparse expansion
// ---------------------------------------------------------------------------

/// Expand `vec` into a sparse map from full-space state → accumulated amplitude.
///
/// Calls [`ExpandRefState::expand_ref_state_iter`] for each basis state and
/// accumulates contributions by state key.  States that cancel exactly are
/// retained with zero amplitude; callers that need filtering can check
/// `amp.norm()` themselves.
///
/// Memory is proportional to the number of distinct full-space states reached,
/// not to `lhss^n_sites`.  This makes it efficient for particle-number-
/// conserving subspaces and other sparse sectors.
pub fn expand_to_map<B, T, E, V>(space: &E, vec: &V) -> HashMap<B, Complex<f64>>
where
    B: BitInt,
    V: ToView<T> + ?Sized,
    E: BasisSpace<B> + ExpandRefState<B, T, Complex<f64>>,
{
    let vec = vec.to_view();
    debug_assert_eq!(vec.len(), space.size());
    let mut map: HashMap<B, Complex<f64>> = HashMap::new();
    for (i, coeff) in vec.iter().enumerate() {
        for (state, val) in space.expand_ref_state_iter(i, coeff) {
            *map.entry(state).or_insert(Complex::new(0.0, 0.0)) += val;
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Dense full-space vector
// ---------------------------------------------------------------------------

/// Expand a subspace coefficient vector `vec` into the dense full-space vector
/// `out`.
///
/// Internally calls [`expand_to_map`] then scatters the sparse result into
/// `out` using `full_space.index`.  `out` must be pre-zeroed by the caller
/// and have length `full_space.size()`.
///
/// # Panics
///
/// Panics (debug only) if `vec.len() != space.size()`.
pub fn get_full_vector<B, T, E, FS, V>(
    space: &E,
    full_space: &FS,
    vec: &V,
    out: &mut [Complex<f64>],
) where
    B: BitInt,
    V: ToView<T> + ?Sized,
    E: BasisSpace<B> + ExpandRefState<B, T, Complex<f64>>,
    FS: BasisSpace<B>,
    Complex<f64>: AddAssign,
{
    let map = expand_to_map(space, vec);
    for (state, val) in map {
        if let Some(idx) = full_space.index(state) {
            out[idx] += val;
        }
    }
}

// ---------------------------------------------------------------------------
// Reduced density matrix
// ---------------------------------------------------------------------------

/// Compute the reduced density matrix (RDM) for the subsystem defined by
/// `sites_a`, tracing out all other sites.
///
/// Given a pure state `|ψ⟩ = Σ_i c_i |i⟩` the RDM is:
///
/// ```text
/// ρ_A[α, β] = Σ_{s_B}  ψ[α, s_B] · conj(ψ[β, s_B])
/// ```
///
/// # Algorithm
///
/// 1. [`expand_to_map`] produces a sparse map of full-space states → amplitudes.
/// 2. Each state is decomposed into `(s_A, s_B)` using a [`BenesPermDitLocations`]
///    permutation built once with `sites_a[i] → i` and `sites_b[j] → n_a + j`.
///    This works for any `lhss` (multi-bit dits, `bits_per_dit = ceil(log2(lhss))`).
///    After `op.apply(state)`, A-subsystem dits are at positions `0..n_a` and
///    B-subsystem dits are at positions `n_a..n_a+n_b`.  `op.fermionic_sign(state)`
///    gives the Jordan-Wigner exchange sign η; it returns `1.0` for bosons since
///    sign masks are only populated when `fermionic=true AND lhss=2`.  Because
///    `sites_a` and `sites_b` are both sorted ascending, the A–A and B–B sublists
///    contribute no permutation inversions, so `fermionic_sign` counts only the
///    physical A–B exchange pairs.
/// 3. States sharing the same `s_B` are grouped; their outer products are
///    accumulated into `out`.
///
/// Memory is proportional to the number of non-zero states in the expanded
/// map — never `lhss^n_sites`.
///
/// # Parameters
///
/// - `space`   — the reduced basis the coefficient vector lives in
/// - `vec`     — coefficient vector of length `space.size()`
/// - `sites_a` — site indices of subsystem A.  Sorted ascending internally;
///   the caller's order does not affect the output.
///
/// # Parameters
///
/// - `out` — a `dim_a × dim_a` matrix where `dim_a = lhss^sites_a.len()`.
///   Accumulated into (not zeroed); caller is responsible for zeroing if needed.
///
/// # Panics
///
/// Panics (debug only) if `vec.len() != space.size()` or
/// `out` does not have shape `(dim_a, dim_a)`.
pub fn reduced_density_matrix<B, T, E, V>(
    space: &E,
    vec: &V,
    sites_a: &[usize],
    out: &mut Array2<Complex<f64>>,
) where
    B: BitInt,
    V: ToView<T> + ?Sized,
    E: BasisSpace<B> + ExpandRefState<B, T, Complex<f64>>,
{
    let vec = vec.to_view();
    debug_assert_eq!(vec.len(), space.size());
    let lhss = space.lhss();
    let n_sites = space.n_sites();
    let n_a = sites_a.len();
    let dim_a = lhss.saturating_pow(n_a as u32);
    debug_assert_eq!(out.shape(), &[dim_a, dim_a]);

    let fermionic = space.fermionic();

    // Sort sites_a so that no A–A or B–B permutation inversions arise; the
    // BenesPermDitLocations sign computation then counts only A–B exchange pairs.
    let mut sites_a_sorted = sites_a.to_vec();
    sites_a_sorted.sort_unstable();
    let sites_b: Vec<usize> = (0..n_sites)
        .filter(|s| !sites_a_sorted.contains(s))
        .collect();
    let n_b = sites_b.len();

    // Step 1: sparse expansion.
    let map = expand_to_map(space, &vec);

    // Build permutation once: sites_a_sorted[i] → i,  sites_b[j] → n_a + j.
    // BenesPermDitLocations handles any lhss via multi-bit dits
    // (bits_per_dit = ceil(log2(lhss))).  Sign masks are only populated when
    // fermionic=true AND lhss=2; fermionic_sign returns 1.0 otherwise.
    let mut perm = vec![0usize; n_sites];
    for (i, &site) in sites_a_sorted.iter().enumerate() {
        perm[site] = i;
    }
    for (j, &site) in sites_b.iter().enumerate() {
        perm[site] = n_a + j;
    }
    let benes_op = BenesPermDitLocations::<B>::new(lhss, &perm, fermionic);

    // After benes_op.apply(state), A-subsystem dits occupy positions 0..n_a-1
    // and B-subsystem dits occupy positions n_a..n_a+n_b-1.
    let manip = DynamicDitManip::new(lhss);
    let n_a_bits = n_a * manip.bits;
    // Positions of A-subsystem dits in the permuted state.  Pre-computed once
    // so get_sub_state can be called without allocating inside the loop.
    let a_positions: Vec<usize> = (0..n_a).collect();

    // Step 2: group by s_B (B-typed key for correctness with wide integers).
    let mut groups: HashMap<B, Vec<(usize, Complex<f64>)>> = HashMap::new();

    for (state, amp) in &map {
        let s_perm = benes_op.apply(*state);

        // sa: mixed-radix index for the A subsystem extracted from positions 0..n_a.
        let sa = manip.get_sub_state(s_perm, &a_positions);

        // sb: B-subsystem dits shifted down to start at bit 0.
        // Guard against shift-by-B::BITS when n_b == 0.
        let sb: B = if n_b == 0 {
            B::from_u64(0)
        } else {
            s_perm >> n_a_bits
        };

        let sign = benes_op.fermionic_sign(*state);
        groups.entry(sb).or_default().push((sa, amp * sign));
    }

    // Step 3: accumulate outer products into the RDM.
    for group in groups.values() {
        for &(sa_i, c_i) in group {
            for &(sa_j, c_j) in group {
                out[[sa_i, sa_j]] += c_i * c_j.conj();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ExpandRefState impls
// ---------------------------------------------------------------------------

/// For a full space every state is its own representative with norm 1.
/// Yields a single `(state, coeff)` pair.
impl<B, T> ExpandRefState<B, T, Complex<f64>> for FullSpace<B>
where
    B: BitInt,
    T: Copy,
    Complex<f64>: From<T>,
{
    fn expand_ref_state_iter(
        &self,
        i: usize,
        coeff: &T,
    ) -> impl Iterator<Item = (B, Complex<f64>)> {
        std::iter::once((self.state_at(i), Complex::<f64>::from(*coeff)))
    }
}

/// For a plain subspace every state is its own representative with norm 1.
/// Yields a single `(state, coeff)` pair.
impl<B, T> ExpandRefState<B, T, Complex<f64>> for Subspace<B>
where
    B: BitInt,
    T: Copy,
    Complex<f64>: From<T>,
{
    fn expand_ref_state_iter(
        &self,
        i: usize,
        coeff: &T,
    ) -> impl Iterator<Item = (B, Complex<f64>)> {
        std::iter::once((self.state_at(i), Complex::<f64>::from(*coeff)))
    }
}

/// For a symmetric basis, yields one `(state, value)` pair per orbit image.
///
/// For the `i`-th representative state `r` with orbit norm `n`, each image
/// `s = g(r)` with group character `χ_g` contributes:
/// `(s,  (coeff / n) * χ_g)`
impl<B, L, N, T> ExpandRefState<B, T, Complex<f64>> for SymBasis<B, L, N>
where
    B: BitInt,
    L: BitStateOp<B>,
    N: NormInt,
    T: Copy,
    Complex<f64>: From<T>,
{
    fn expand_ref_state_iter(
        &self,
        i: usize,
        coeff: &T,
    ) -> impl Iterator<Item = (B, Complex<f64>)> {
        let (ref_state, norm) = self.entry(i);
        let val = Complex::<f64>::from(*coeff) / norm;
        iter_images(&self.lattice, &self.local, ref_state)
            .into_iter()
            .map(move |(s, char_g)| (s, val * char_g))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::space::{FullSpace, Subspace};
    use ndarray::Array2;

    /// For a 2-site spin-1/2 system in the product state |↓↑⟩ (site 0 = 0,
    /// site 1 = 1), stored as state integer 0b10 = 2, tracing out site 1
    /// should give the pure-state RDM ρ_A = |↓⟩⟨↓| = [[1,0],[0,0]] for site 0.
    ///
    /// Bit encoding: bit i = site i (LSB = site 0).
    /// State 2 = 0b10: site 0 = 0 (↓), site 1 = 1 (↑).
    #[test]
    fn rdm_product_state_site0() {
        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0u32, |s: u32| {
            (0..2).map(move |i| (Complex::new(1.0, 0.0), s ^ (1 << i), 0u8))
        });

        // psi = |↓↑⟩ = state 2 = 0b10.
        let idx = sub.index(0b10u32).unwrap();
        let mut vec = vec![Complex::new(0.0, 0.0); sub.size()];
        vec[idx] = Complex::new(1.0, 0.0);

        // Trace out site 1, keep site 0.
        // site 0 is in state 0 (↓) → ρ_A = [[1,0],[0,0]].
        let mut rdm = Array2::<Complex<f64>>::zeros((2, 2));
        reduced_density_matrix(&sub, &vec, &[0], &mut rdm);

        let tol = 1e-14;
        assert!(
            (rdm[[0, 0]].re - 1.0).abs() < tol,
            "rdm[0,0] = {}",
            rdm[[0, 0]]
        );
        assert!(rdm[[0, 1]].norm() < tol, "rdm[0,1] = {}", rdm[[0, 1]]);
        assert!(rdm[[1, 0]].norm() < tol, "rdm[1,0] = {}", rdm[[1, 0]]);
        assert!(rdm[[1, 1]].norm() < tol, "rdm[1,1] = {}", rdm[[1, 1]]);
    }

    /// Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2.
    /// Tracing out either site gives the maximally mixed state ρ = I/2.
    #[test]
    fn rdm_bell_state_maximally_mixed() {
        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0u32, |s: u32| {
            (0..2).map(move |i| (Complex::new(1.0, 0.0), s ^ (1 << i), 0u8))
        });

        let inv_sqrt2 = Complex::new(1.0 / 2f64.sqrt(), 0.0);
        let idx_00 = sub.index(0b00u32).unwrap();
        let idx_11 = sub.index(0b11u32).unwrap();
        let mut vec = vec![Complex::new(0.0, 0.0); sub.size()];
        vec[idx_00] = inv_sqrt2;
        vec[idx_11] = inv_sqrt2;

        // Trace out site 1.
        let mut rdm = Array2::<Complex<f64>>::zeros((2, 2));
        reduced_density_matrix(&sub, &vec, &[0], &mut rdm);

        let tol = 1e-14;
        // ρ_A = [[0.5, 0], [0, 0.5]]
        assert!(
            (rdm[[0, 0]].re - 0.5).abs() < tol,
            "rdm[0,0] = {}",
            rdm[[0, 0]]
        );
        assert!(rdm[[0, 1]].norm() < tol, "rdm[0,1] = {}", rdm[[0, 1]]);
        assert!(rdm[[1, 0]].norm() < tol, "rdm[1,0] = {}", rdm[[1, 0]]);
        assert!(
            (rdm[[1, 1]].re - 0.5).abs() < tol,
            "rdm[1,1] = {}",
            rdm[[1, 1]]
        );
    }

    /// Trace is preserved: Tr(ρ_A) = 1 for a normalised state vector.
    #[test]
    fn rdm_trace_is_one() {
        // Random-ish state on 4 sites.
        let mut sub = Subspace::<u32>::new(2, 4, false);
        sub.build(0u32, |s: u32| {
            (0..4).map(move |i| (Complex::new(1.0, 0.0), s ^ (1 << i), 0u8))
        });

        // Uniform superposition over all 16 states.
        let amp = Complex::new(1.0 / 4.0, 0.0);
        let vec: Vec<Complex<f64>> = (0..sub.size()).map(|_| amp).collect();

        // Trace out sites 2 and 3, keep sites 0 and 1 (dim_a = 4).
        let mut rdm = Array2::<Complex<f64>>::zeros((4, 4));
        reduced_density_matrix(&sub, &vec, &[0, 1], &mut rdm);

        let trace: f64 = (0..4).map(|i| rdm[[i, i]].re).sum();
        assert!((trace - 1.0).abs() < 1e-13, "trace = {trace}");
    }

    /// Hermiticity: ρ[i,j] = conj(ρ[j,i]).
    #[test]
    fn rdm_is_hermitian() {
        let mut sub = Subspace::<u32>::new(2, 4, false);
        sub.build(0u32, |s: u32| {
            (0..4).map(move |i| (Complex::new(1.0, 0.0), s ^ (1 << i), 0u8))
        });

        // Non-uniform state with a complex coefficient.
        let mut vec = vec![Complex::new(0.0, 0.0); sub.size()];
        let idx0 = sub.index(0b0000).unwrap();
        let idx1 = sub.index(0b0001).unwrap();
        let idx2 = sub.index(0b0010).unwrap();
        vec[idx0] = Complex::new(1.0 / 2f64.sqrt(), 0.0);
        vec[idx1] = Complex::new(0.0, 1.0 / 2.0);
        vec[idx2] = Complex::new(1.0 / 2.0, 0.0);

        let mut rdm = Array2::<Complex<f64>>::zeros((2, 2));
        reduced_density_matrix(&sub, &vec, &[0], &mut rdm);

        let tol = 1e-14;
        for i in 0..2 {
            for j in 0..2 {
                let diff = rdm[[i, j]] - rdm[[j, i]].conj();
                assert!(diff.norm() < tol, "rdm[{i},{j}] not hermitian: {diff}");
            }
        }
    }

    /// get_full_vector still works correctly after the refactor to use expand_to_map.
    #[test]
    fn get_full_vector_roundtrip() {
        let mut sub = Subspace::<u32>::new(2, 3, false);
        sub.build(0u32, |s: u32| {
            (0..3).map(move |i| (Complex::new(1.0, 0.0), s ^ (1 << i), 0u8))
        });
        let fs = FullSpace::<u32>::new(2, 3, false);

        let amp = Complex::new(1.0 / (8f64.sqrt()), 0.0);
        let vec: Vec<Complex<f64>> = (0..sub.size()).map(|_| amp).collect();

        let mut out = vec![Complex::new(0.0, 0.0); fs.size()];
        get_full_vector(&sub, &fs, &vec, &mut out);

        // Every full-space slot should be filled with `amp`.
        let tol = 1e-14;
        for (i, v) in out.iter().enumerate() {
            assert!((v.re - amp.re).abs() < tol, "out[{i}] = {v}");
        }
    }
}
