/// Merged symmetry-group + basis type: [`SymBasis<B, L, N>`].
///
/// Replaces the two-type hierarchy (`SymGrpBase<B,L>` + `SymmetricSubspace<G,N>`)
/// with a single flat struct. `N` is an explicit type parameter — the B→N
/// pairing is encoded in `SpaceInner` variant definitions, not a runtime enum.
use super::bfs::{AMP_CANCEL_TOL, PARALLEL_FRONTIER_THRESHOLD};
use super::lattice::BenesLatticeElement;
use super::traits::BasisSpace;
use num_complex::Complex;
use quspin_bitbasis::{
    BenesPermDitLocations, BitInt, Compose, FermionicBitStateOp, StateTransitions,
};
use quspin_types::QuSpinError;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Minimum batch size before switching to parallel orbit computation.
const PARALLEL_BATCH_THRESHOLD: usize = 256;

// ---------------------------------------------------------------------------
// NormInt
// ---------------------------------------------------------------------------

/// Storage type for orbit norms inside [`SymBasis`].
///
/// The orbit norm is a small non-negative integer bounded by the symmetry
/// group order. Using a narrower integer than `f64` reduces memory by up to
/// 8×; the `f64` value is only materialised at the [`entry`](SymBasis::entry)
/// API boundary.
///
/// The B→N pairing policy is:
/// - B = u32  → N = u8  (n_sites ≤ 32, max group order fits in u8)
/// - B = u64  → N = u16
/// - B = Uint<128..8192> → N = u32
///
/// This pairing is enforced statically via [`SpaceInner`](crate::dispatch::SpaceInner)
/// variant definitions and the `with_sym_grp!` macro.
pub trait NormInt: Copy + Send + Sync + 'static {
    fn from_norm(norm: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl NormInt for u8 {
    #[inline]
    fn from_norm(norm: f64) -> Self {
        norm as u8
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl NormInt for u16 {
    #[inline]
    fn from_norm(norm: f64) -> Self {
        norm as u16
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl NormInt for u32 {
    #[inline]
    fn from_norm(norm: f64) -> Self {
        norm as u32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

// ---------------------------------------------------------------------------
// SymBasis
// ---------------------------------------------------------------------------

/// Merged symmetry group + symmetric subspace.
///
/// Combines the group data previously in `SymGrpBase<B, L>` (lattice ops,
/// local ops, metadata) with the basis data previously in
/// `SymmetricSubspace<G, N>` (representative states, index map) into a single
/// flat struct. `N` is an explicit type parameter — no runtime enum dispatch
/// occurs in `get_refstate_batch` or `check_refstate_batch`.
///
/// # Type parameters
/// - `B` — basis integer type ([`BitInt`])
/// - `L` — local-op type: [`PermDitMask<B>`] for LHSS=2, [`DynamicPermDitValues`] for LHSS≥3
/// - `N` — norm storage type ([`NormInt`]): `u8`, `u16`, or `u32`
///
/// [`PermDitMask<B>`]: quspin_bitbasis::PermDitMask
/// [`DynamicPermDitValues`]: quspin_bitbasis::DynamicPermDitValues
pub struct SymBasis<B: BitInt, L, N: NormInt> {
    // Group data (was SymGrpBase<B, L>)
    pub(crate) lhss: usize,
    pub(crate) fermionic: bool,
    pub(crate) n_sites: usize,
    /// Pure-lattice group elements: site permutation only, no local op.
    pub(crate) lattice_only: Vec<BenesLatticeElement<B>>,
    /// Pure-local group elements: local op only, no site permutation.
    pub(crate) local_only: Vec<(Complex<f64>, L)>,
    /// Composite group elements: site permutation + local op applied
    /// atomically as one element with a single character. The character
    /// lives on the `BenesLatticeElement`.
    pub(crate) composite: Vec<(BenesLatticeElement<B>, L)>,
    // Basis data (was SymmetricSubspace<G, N>)
    /// `(representative_state, orbit_norm)` pairs, sorted ascending by state.
    states: Vec<(B, N)>,
    /// Maps representative state → index in `states`.
    index_map: HashMap<B, usize>,
    /// Set to `true` at the start of `build()`, never reset.
    /// A seed whose orbit norm is zero produces no entries in `states`, so
    /// `states.is_empty()` is not a reliable built indicator.
    built: bool,
}

// ---------------------------------------------------------------------------
// SymBasis: construction and non-orbit methods (no L bound)
// ---------------------------------------------------------------------------

impl<B: BitInt, L, N: NormInt> SymBasis<B, L, N> {
    /// Construct an empty basis with no group elements and no states.
    ///
    /// Call [`add_symmetry`](Self::add_symmetry) (or
    /// [`add_cyclic`](Self::add_cyclic) as sugar for a cyclic subgroup) to
    /// add explicit group elements, then [`build`](Self::build) to populate
    /// states. The identity element is always implicit and should not be
    /// added.
    pub fn new_empty(lhss: usize, n_sites: usize, fermionic: bool) -> Self {
        SymBasis {
            lhss,
            fermionic,
            n_sites,
            lattice_only: Vec::new(),
            local_only: Vec::new(),
            composite: Vec::new(),
            states: Vec::new(),
            index_map: HashMap::new(),
            built: false,
        }
    }

    /// Add an explicit group element with its representation character.
    ///
    /// The user enumerates every non-identity element of the group; the
    /// identity is implicit with `χ(I) = 1`. Empty `SymElement`s
    /// (both `perm` and `local` absent) are rejected — the identity
    /// must not be added explicitly.
    ///
    /// The walker dispatches each element into one of three typed
    /// storage vectors at insert time based on which components are
    /// present, so the orbit hot loop has zero variant-branching.
    pub fn add_symmetry(
        &mut self,
        grp_char: Complex<f64>,
        element: crate::SymElement<L>,
    ) -> Result<(), QuSpinError> {
        let (perm, local) = element.into_parts();
        match (perm, local) {
            (Some(p), None) => {
                let op = BenesPermDitLocations::<B>::new(self.lhss, &p, self.fermionic);
                self.lattice_only
                    .push(BenesLatticeElement::new(grp_char, op, self.n_sites));
                Ok(())
            }
            (None, Some(l)) => {
                self.local_only.push((grp_char, l));
                Ok(())
            }
            (Some(p), Some(l)) => {
                let op = BenesPermDitLocations::<B>::new(self.lhss, &p, self.fermionic);
                self.composite
                    .push((BenesLatticeElement::new(grp_char, op, self.n_sites), l));
                Ok(())
            }
            (None, None) => Err(QuSpinError::ValueError(
                "empty symmetry element: identity is implicit, do not add it".into(),
            )),
        }
    }

    /// Returns `true` once [`build`](Self::build) has been called.
    ///
    /// Set at the very start of `build()` so a seed with zero orbit norm
    /// (which produces no states) still marks the basis as built.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Add the non-identity powers of a cyclic subgroup: `g^1, g^2, …, g^{order-1}`.
    ///
    /// `char_fn(k)` supplies the character for `g^k`. For the k-th
    /// 1D representation of a cyclic group of order `n`, this is
    /// typically `exp(-2πi · k · m / n)` for some integer momentum `m`.
    ///
    /// Pure sugar over [`add_symmetry`](Self::add_symmetry): the helper
    /// populates the same three storage vectors. Requires `L: Compose`
    /// because it computes each power via
    /// [`SymElement::compose`](crate::SymElement::compose); local-op
    /// types without a `Compose` impl (e.g. [`SignedPermDitMask`]) must
    /// enumerate their group elements via direct
    /// [`add_symmetry`](Self::add_symmetry) calls.
    ///
    /// [`SignedPermDitMask`]: quspin_bitbasis::SignedPermDitMask
    pub fn add_cyclic(
        &mut self,
        generator: crate::SymElement<L>,
        order: usize,
        char_fn: impl Fn(usize) -> Complex<f64>,
    ) -> Result<(), QuSpinError>
    where
        L: Compose,
    {
        if order < 2 {
            return Err(QuSpinError::ValueError(format!(
                "add_cyclic requires order >= 2, got {order}"
            )));
        }
        let mut gk = generator.clone(); // g^1
        for k in 1..order {
            self.add_symmetry(char_fn(k), gk.clone())?;
            if k + 1 < order {
                gk = gk.compose(&generator);
            }
        }
        Ok(())
    }

    /// Local Hilbert-space size.
    pub fn lhss(&self) -> usize {
        self.lhss
    }

    /// Whether Jordan-Wigner signs are tracked.
    pub fn fermionic(&self) -> bool {
        self.fermionic
    }

    /// Return the representative state and orbit norm for index `i`.
    ///
    /// The norm is cast to `f64` at this boundary; internally it is stored as
    /// `N` (typically `u8`, `u16`, or `u32`) to reduce memory usage.
    #[inline]
    pub fn entry(&self, i: usize) -> (B, f64) {
        let (s, n) = self.states[i];
        (s, n.to_f64())
    }

    /// Total group order: `1 + lattice_only + local_only + composite` (the
    /// `1` is the implicit identity). Used by expand/project paths to
    /// compute the reduced-basis normalisation factor
    /// `√(|G|·norm)` that ties expansion and projection together.
    #[inline]
    pub fn group_order(&self) -> usize {
        1 + self.lattice_only.len() + self.local_only.len() + self.composite.len()
    }
}

// ---------------------------------------------------------------------------
// SymBasis: orbit methods and basis construction (require L: BitStateOp<B>)
// ---------------------------------------------------------------------------

impl<B: BitInt, L: FermionicBitStateOp<B>, N: NormInt> SymBasis<B, L, N> {
    /// Return the representative state (largest orbit element) and the
    /// accumulated group character for `state`.
    pub fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        super::orbit::get_refstate(
            &self.lattice_only,
            &self.local_only,
            &self.composite,
            self.fermionic,
            state,
        )
    }

    /// Return the representative state and the orbit norm.
    pub fn check_refstate(&self, state: B) -> (B, f64) {
        super::orbit::check_refstate(
            &self.lattice_only,
            &self.local_only,
            &self.composite,
            self.fermionic,
            state,
        )
    }

    /// Batch variant of [`get_refstate`](Self::get_refstate).
    ///
    /// # Panics
    ///
    /// Panics if `states.len() != out.len()`.
    pub fn get_refstate_batch(&self, states: &[B], out: &mut [(B, Complex<f64>)]) {
        super::orbit::get_refstate_batch(
            &self.lattice_only,
            &self.local_only,
            &self.composite,
            self.fermionic,
            states,
            out,
        );
    }

    /// Batch variant of [`check_refstate`](Self::check_refstate).
    ///
    /// # Panics
    ///
    /// Panics if `states.len() != out.len()`.
    pub fn check_refstate_batch(&self, states: &[B], out: &mut [(B, f64)]) {
        super::orbit::check_refstate_batch(
            &self.lattice_only,
            &self.local_only,
            &self.composite,
            self.fermionic,
            states,
            out,
        );
    }

    /// Build the symmetric subspace reachable from `seed` under `op`.
    ///
    /// Uses level-synchronous BFS with two parallel phases per wave:
    ///
    /// 1. **Fused BFS + orbit mapping** — amplitude-cancellation BFS and
    ///    `get_refstate` per surviving candidate run in one parallel fold/reduce,
    ///    yielding a `HashSet<B>` of orbit representatives directly.
    /// 2. **Parallel norm computation** — `par_check_refstate_batch` on the
    ///    unique new representatives only (the smallest possible batch),
    ///    retaining the SIMD-friendly batch orbit traversal for this step.
    /// 3. **Sequential registration** — dedup and index-map update on the main
    ///    thread; frontier for the next wave assembled here.
    pub fn build<G>(&mut self, seed: B, graph: &G)
    where
        G: StateTransitions,
        L: Sync,
    {
        self.built = true;
        let (ref_seed, _coeff) = self.get_refstate(seed);
        let (_ref2, norm_seed) = self.check_refstate(ref_seed);

        if norm_seed > 0.0
            && let std::collections::hash_map::Entry::Vacant(e) = self.index_map.entry(ref_seed)
        {
            e.insert(self.states.len());
            self.states.push((ref_seed, N::from_norm(norm_seed)));
        }

        let mut frontier: Vec<B> = vec![ref_seed];

        while !frontier.is_empty() {
            let lattice_only = &self.lattice_only;
            let local_only = &self.local_only;
            let composite = &self.composite;
            let fermionic = self.fermionic;

            // Phase 1+2a (fused): BFS discovery + orbit-representative mapping
            // in a single parallel fold — one fork/join instead of two.
            // Uses get_refstate (no norm computation) per surviving candidate.
            let candidate_reps: HashSet<B> = bfs_wave_with_reps(
                &frontier,
                graph,
                lattice_only,
                local_only,
                composite,
                fermionic,
            );

            if candidate_reps.is_empty() {
                frontier.clear();
                continue;
            }

            // Sequential dedup: keep only reps not already registered.
            let unique_reps: Vec<B> = candidate_reps
                .into_iter()
                .filter(|rep| !self.index_map.contains_key(rep))
                .collect();

            if unique_reps.is_empty() {
                frontier.clear();
                continue;
            }

            // Phase 2b: compute orbit norms for unique new representatives.
            // Batch call retains SIMD-friendly inner-loop structure.
            let mut norm_out: Vec<(B, f64)> = vec![(B::from_u64(0), 0.0); unique_reps.len()];
            par_check_refstate_batch(
                lattice_only,
                local_only,
                composite,
                fermionic,
                &unique_reps,
                &mut norm_out,
            );

            // Phase 3: register valid representatives, build next frontier.
            frontier.clear();
            for (&rep, &(_, norm)) in unique_reps.iter().zip(norm_out.iter()) {
                if norm > 0.0
                    && let std::collections::hash_map::Entry::Vacant(e) = self.index_map.entry(rep)
                {
                    e.insert(self.states.len());
                    self.states.push((rep, N::from_norm(norm)));
                    frontier.push(rep);
                }
            }
        }

        self.sort();
    }

    fn sort(&mut self) {
        self.states.sort_unstable_by_key(|&(s, _)| s);
        self.index_map.clear();
        for (i, &(s, _)) in self.states.iter().enumerate() {
            self.index_map.insert(s, i);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused BFS + orbit-representative helper
// ---------------------------------------------------------------------------

/// Fused BFS wave + orbit-representative mapping.
///
/// Combines amplitude-cancellation BFS discovery (the work previously done by
/// `bfs_wave`) with a per-candidate [`orbit::get_refstate`] call in a single
/// parallel fold/reduce.  This eliminates the separate
/// `par_check_refstate_batch` pass on every raw candidate that the old code
/// performed (which computed norms that were immediately discarded during
/// deduplication).
///
/// ## Returns
///
/// A [`HashSet<B>`] of orbit-representative states reached from `frontier` in
/// one BFS wave.  Representatives are already de-duplicated; the caller is
/// responsible for filtering out those already in `index_map`.
///
/// ## Parallelism
///
/// - Below [`PARALLEL_FRONTIER_THRESHOLD`]: sequential single-threaded path.
/// - Above threshold: rayon `par_iter().fold().reduce()` — each thread owns a
///   disjoint slice of `frontier`, builds a thread-local `HashSet<B>` of
///   representatives, and the sets are unioned in the reduce step.
#[allow(clippy::too_many_arguments)]
fn bfs_wave_with_reps<B, L, G>(
    frontier: &[B],
    graph: &G,
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[(Complex<f64>, L)],
    composite: &[(BenesLatticeElement<B>, L)],
    fermionic: bool,
) -> HashSet<B>
where
    B: BitInt,
    L: FermionicBitStateOp<B> + Sync,
    G: StateTransitions,
{
    if frontier.len() < PARALLEL_FRONTIER_THRESHOLD {
        let mut reps = HashSet::new();
        let mut contributions: HashMap<B, (Complex<f64>, f64)> = HashMap::new();
        for &state in frontier {
            contributions.clear();
            graph.neighbors::<B, _>(state, |amp, next_state| {
                if next_state != state {
                    let e = contributions.entry(next_state).or_default();
                    e.0 += amp;
                    e.1 += amp.norm();
                }
            });
            for (&candidate, &(net_amp, scale)) in &contributions {
                if net_amp.norm() > scale * AMP_CANCEL_TOL {
                    let (rep, _) = super::orbit::get_refstate(
                        lattice_only,
                        local_only,
                        composite,
                        fermionic,
                        candidate,
                    );
                    reps.insert(rep);
                }
            }
        }
        reps
    } else {
        frontier
            .par_iter()
            .fold(
                || {
                    (
                        HashSet::<B>::new(),
                        HashMap::<B, (Complex<f64>, f64)>::new(),
                    )
                },
                |(mut reps, mut contributions), &state| {
                    contributions.clear();
                    graph.neighbors::<B, _>(state, |amp, next_state| {
                        if next_state != state {
                            let e = contributions.entry(next_state).or_default();
                            e.0 += amp;
                            e.1 += amp.norm();
                        }
                    });
                    for (&candidate, &(net_amp, scale)) in &contributions {
                        if net_amp.norm() > scale * AMP_CANCEL_TOL {
                            let (rep, _) = super::orbit::get_refstate(
                                lattice_only,
                                local_only,
                                composite,
                                fermionic,
                                candidate,
                            );
                            reps.insert(rep);
                        }
                    }
                    (reps, contributions)
                },
            )
            .map(|(reps, _)| reps)
            .reduce(HashSet::new, |mut a, b| {
                a.extend(b);
                a
            })
    }
}

// ---------------------------------------------------------------------------
// Parallel orbit batch helper
// ---------------------------------------------------------------------------

/// Threshold-gated parallel wrapper around [`orbit::check_refstate_batch`].
///
/// Above [`PARALLEL_BATCH_THRESHOLD`], splits the work into chunks distributed
/// across rayon threads. Each chunk calls the existing batch function, which
/// retains its inner auto-vectorisation.
fn par_check_refstate_batch<B, L>(
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[(Complex<f64>, L)],
    composite: &[(BenesLatticeElement<B>, L)],
    fermionic: bool,
    states: &[B],
    out: &mut [(B, f64)],
) where
    B: BitInt,
    L: FermionicBitStateOp<B> + Sync,
{
    if states.len() >= PARALLEL_BATCH_THRESHOLD {
        let chunk_size = (states.len() / rayon::current_num_threads()).max(64);
        states
            .par_chunks(chunk_size)
            .zip(out.par_chunks_mut(chunk_size))
            .for_each(|(s, o)| {
                super::orbit::check_refstate_batch(
                    lattice_only,
                    local_only,
                    composite,
                    fermionic,
                    s,
                    o,
                );
            });
    } else {
        super::orbit::check_refstate_batch(
            lattice_only,
            local_only,
            composite,
            fermionic,
            states,
            out,
        );
    }
}

// ---------------------------------------------------------------------------
// BasisSpace impl (no L bound needed — no orbit computation)
// ---------------------------------------------------------------------------

impl<B: BitInt, L, N: NormInt> BasisSpace<B> for SymBasis<B, L, N> {
    #[inline]
    fn n_sites(&self) -> usize {
        self.n_sites
    }

    #[inline]
    fn lhss(&self) -> usize {
        self.lhss
    }

    #[inline]
    fn fermionic(&self) -> bool {
        self.fermionic
    }

    #[inline]
    fn size(&self) -> usize {
        self.states.len()
    }

    #[inline]
    fn state_at(&self, i: usize) -> B {
        self.states[i].0
    }

    #[inline]
    fn index(&self, state: B) -> Option<usize> {
        self.index_map.get(&state).copied()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use quspin_bitbasis::PermDitMask;
    use quspin_bitbasis::test_graphs::XAllSites;

    fn x_op(n_sites: u32) -> XAllSites {
        XAllSites::new(n_sites)
    }

    #[test]
    fn sym_basis_bitflip_2site() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 2, false);
        basis
            .add_symmetry(Complex::new(1.0, 0.0), crate::SymElement::lattice(&[0, 1]))
            .unwrap();
        // spin inversion: XOR mask at all sites
        let mask = PermDitMask::<u32>::new(0b11u32);
        basis
            .add_symmetry(Complex::new(1.0, 0.0), crate::SymElement::local(mask))
            .unwrap();
        basis.build(0u32, &x_op(2));

        assert_eq!(basis.size(), 2);
        assert!(basis.index(3u32).is_some());
        assert!(basis.index(2u32).is_some());
    }

    #[test]
    fn sym_basis_no_symmetry_matches_subspace() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 3, false);
        basis
            .add_symmetry(
                Complex::new(1.0, 0.0),
                crate::SymElement::lattice(&[0, 1, 2]),
            )
            .unwrap();
        basis.build(0u32, &x_op(3));
        assert_eq!(basis.size(), 8);
    }

    #[test]
    fn sym_basis_sorted_ascending() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 3, false);
        basis
            .add_symmetry(
                Complex::new(1.0, 0.0),
                crate::SymElement::lattice(&[0, 1, 2]),
            )
            .unwrap();
        basis.build(0u32, &x_op(3));
        for i in 1..basis.size() {
            assert!(basis.state_at(i) > basis.state_at(i - 1));
        }
    }

    #[test]
    fn sym_basis_u8_norm() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u8>::new_empty(2, 2, false);
        let mask = PermDitMask::<u32>::new(0b11u32);
        basis
            .add_symmetry(Complex::new(1.0, 0.0), crate::SymElement::local(mask))
            .unwrap();
        basis.build(0u32, &x_op(2));
        assert_eq!(basis.size(), 2);
        for i in 0..basis.size() {
            let (_, norm) = basis.entry(i);
            assert_eq!(norm, 1.0);
        }
    }

    #[test]
    fn sym_basis_entry_roundtrip() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u8>::new_empty(2, 2, false);
        let mask = PermDitMask::<u32>::new(0b11u32);
        basis
            .add_symmetry(Complex::new(1.0, 0.0), crate::SymElement::local(mask))
            .unwrap();
        basis.build(0u32, &x_op(2));
        for i in 0..basis.size() {
            let (_, norm) = basis.entry(i);
            assert_eq!(norm, 1.0);
        }
    }

    // --- Parallel path tests ---

    #[test]
    fn sym_basis_parallel_no_symmetry_matches_full() {
        // 12 sites, identity-only lattice, no local ops.
        // No symmetry reduction → should find all 2^12 = 4096 representatives.
        // Frontier grows large enough to trigger parallel BFS.
        let n_sites = 12u32;
        let mut basis =
            SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, n_sites as usize, false);
        let identity: Vec<usize> = (0..n_sites as usize).collect();
        basis
            .add_symmetry(
                Complex::new(1.0, 0.0),
                crate::SymElement::lattice(&identity),
            )
            .unwrap();
        basis.build(0u32, &x_op(n_sites));
        assert_eq!(basis.size(), 1 << n_sites);
        // Verify sorted ascending
        for i in 1..basis.size() {
            assert!(basis.state_at(i) > basis.state_at(i - 1));
        }
        // Verify index roundtrip
        for i in 0..basis.size() {
            assert_eq!(basis.index(basis.state_at(i)), Some(i));
        }
    }

    #[test]
    fn sym_basis_parallel_bitflip_reduces_size() {
        // 12 sites with spin-inversion symmetry (XOR all bits).
        // Full space: 2^12 = 4096. With Z2 symmetry: roughly half.
        let n_sites = 12u32;
        let mut basis =
            SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, n_sites as usize, false);
        let identity: Vec<usize> = (0..n_sites as usize).collect();
        basis
            .add_symmetry(
                Complex::new(1.0, 0.0),
                crate::SymElement::lattice(&identity),
            )
            .unwrap();
        let mask = PermDitMask::<u32>::new((1u32 << n_sites) - 1);
        basis
            .add_symmetry(Complex::new(1.0, 0.0), crate::SymElement::local(mask))
            .unwrap();
        basis.build(0u32, &x_op(n_sites));

        // With Z2 spin inversion (XOR all bits) on 12 sites:
        // No state equals its own complement, so every orbit has exactly 2
        // distinct elements. Representatives = 4096 / 2 = 2048.
        assert_eq!(basis.size(), 2048);
        for i in 1..basis.size() {
            assert!(basis.state_at(i) > basis.state_at(i - 1));
        }
    }
}
