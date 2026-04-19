/// Merged symmetry-group + basis type: [`SymBasis<B, L, N>`].
///
/// Replaces the two-type hierarchy (`SymGrpBase<B,L>` + `SymmetricSubspace<G,N>`)
/// with a single flat struct. `N` is an explicit type parameter — the B→N
/// pairing is encoded in `SpaceInner` variant definitions, not a runtime enum.
use super::bfs::{AMP_CANCEL_TOL, PARALLEL_FRONTIER_THRESHOLD};
use super::lattice::{BenesLatticeElement, LatEl, LocalOpItem};
use super::traits::BasisSpace;
use num_complex::Complex;
use quspin_bitbasis::{BenesPermDitLocations, BitInt, BitStateOp, StateGraph};
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
    pub(crate) lattice: Vec<BenesLatticeElement<B>>,
    pub(crate) local: Vec<(Complex<f64>, L)>,
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
    /// Call [`add_lattice`](Self::add_lattice) / [`add_local`](Self::add_local)
    /// to add symmetry elements, then [`build`](Self::build) to populate states.
    pub fn new_empty(lhss: usize, n_sites: usize, fermionic: bool) -> Self {
        SymBasis {
            lhss,
            fermionic,
            n_sites,
            lattice: Vec::new(),
            local: Vec::new(),
            states: Vec::new(),
            index_map: HashMap::new(),
            built: false,
        }
    }

    /// Add a lattice (site-permutation) symmetry element. Valid before [`build`](Self::build).
    ///
    /// `perm[src] = dst` maps source site `src` to destination `dst`.
    pub fn add_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]) {
        let op = BenesPermDitLocations::<B>::new(self.lhss, perm, self.fermionic);
        self.lattice
            .push(BenesLatticeElement::new(grp_char, op, self.n_sites));
    }

    /// Add a local symmetry element. Valid before [`build`](Self::build).
    pub fn add_local(&mut self, grp_char: Complex<f64>, local_op: L) {
        self.local.push((grp_char, local_op));
    }

    /// Returns `true` once [`build`](Self::build) has been called.
    ///
    /// Set at the very start of `build()` so a seed with zero orbit norm
    /// (which produces no states) still marks the basis as built.
    pub fn is_built(&self) -> bool {
        self.built
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
}

// ---------------------------------------------------------------------------
// SymBasis: orbit methods and basis construction (require L: BitStateOp<B>)
// ---------------------------------------------------------------------------

impl<B: BitInt, L: BitStateOp<B>, N: NormInt> SymBasis<B, L, N> {
    /// Return the representative state (largest orbit element) and the
    /// accumulated group character for `state`.
    pub fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        super::orbit::get_refstate(&self.lattice, &self.local, state)
    }

    /// Return the representative state and the orbit norm.
    pub fn check_refstate(&self, state: B) -> (B, f64) {
        super::orbit::check_refstate(&self.lattice, &self.local, state)
    }

    /// Batch variant of [`get_refstate`](Self::get_refstate).
    ///
    /// Calls `orbit::get_refstate_batch` directly — zero N dispatch.
    ///
    /// # Panics
    ///
    /// Panics if `states.len() != out.len()`.
    pub fn get_refstate_batch(&self, states: &[B], out: &mut [(B, Complex<f64>)]) {
        super::orbit::get_refstate_batch(&self.lattice, &self.local, states, out);
    }

    /// Batch variant of [`check_refstate`](Self::check_refstate).
    ///
    /// # Panics
    ///
    /// Panics if `states.len() != out.len()`.
    pub fn check_refstate_batch(&self, states: &[B], out: &mut [(B, f64)]) {
        super::orbit::check_refstate_batch(&self.lattice, &self.local, states, out);
    }

    /// Build the symmetric subspace reachable from `seed` under `op`.
    ///
    /// Uses level-synchronous BFS with two parallel phases per wave:
    ///
    /// 1. **Fused BFS + orbit mapping** — amplitude-cancellation BFS and
    ///    `get_refstate` per surviving candidate run in one parallel fold/reduce,
    ///    yielding a `HashSet<B>` of orbit representatives directly.
    ///    This replaces the former separate BFS pass and the subsequent
    ///    `check_refstate_batch` on every candidate (which computed norms that
    ///    were immediately discarded).
    /// 2. **Parallel norm computation** — `par_check_refstate_batch` on the
    ///    unique new representatives only (the smallest possible batch),
    ///    retaining the SIMD-friendly batch orbit traversal for this step.
    /// 3. **Sequential registration** — dedup and index-map update on the main
    ///    thread; frontier for the next wave assembled here.
    pub fn build<G>(&mut self, seed: B, graph: &G)
    where
        G: StateGraph,
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
            let lattice = &self.lattice;
            let local = &self.local;

            // Phase 1+2a (fused): BFS discovery + orbit-representative mapping
            // in a single parallel fold — one fork/join instead of two.
            // Uses get_refstate (no norm computation) per surviving candidate.
            let candidate_reps: HashSet<B> = bfs_wave_with_reps(&frontier, graph, lattice, local);

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
            par_check_refstate_batch(lattice, local, &unique_reps, &mut norm_out);

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
fn bfs_wave_with_reps<B, E, L, G>(
    frontier: &[B],
    graph: &G,
    lattice: &[E],
    local: &[L],
) -> HashSet<B>
where
    B: BitInt,
    E: LatEl<B> + Sync,
    L: LocalOpItem<B> + Sync,
    G: StateGraph,
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
                    let (rep, _) = super::orbit::get_refstate(lattice, local, candidate);
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
                            let (rep, _) = super::orbit::get_refstate(lattice, local, candidate);
                            reps.insert(rep);
                        }
                    }
                    (reps, contributions)
                },
            )
            .map(|(reps, _)| reps)
            .reduce(
                || HashSet::new(),
                |mut a, b| {
                    a.extend(b);
                    a
                },
            )
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
fn par_check_refstate_batch<B, E, L>(lattice: &[E], local: &[L], states: &[B], out: &mut [(B, f64)])
where
    B: BitInt,
    E: LatEl<B> + Sync,
    L: LocalOpItem<B> + Sync,
{
    if states.len() >= PARALLEL_BATCH_THRESHOLD {
        let chunk_size = (states.len() / rayon::current_num_threads()).max(64);
        states
            .par_chunks(chunk_size)
            .zip(out.par_chunks_mut(chunk_size))
            .for_each(|(s, o)| {
                super::orbit::check_refstate_batch(lattice, local, s, o);
            });
    } else {
        super::orbit::check_refstate_batch(lattice, local, states, out);
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

    /// X on every site, exposed as a `StateGraph` for BFS tests.
    struct XAllSites {
        n_sites: u32,
    }

    impl StateGraph for XAllSites {
        fn lhss(&self) -> usize {
            2
        }
        fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, mut visit: F) {
            for loc in 0..self.n_sites {
                let mask = B::from_u64(1u64 << loc);
                visit(Complex::new(1.0, 0.0), state ^ mask);
            }
        }
    }

    fn x_op(n_sites: u32) -> XAllSites {
        XAllSites { n_sites }
    }

    #[test]
    fn sym_basis_bitflip_2site() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 2, false);
        basis.add_lattice(Complex::new(1.0, 0.0), &[0, 1]);
        // spin inversion: XOR mask at all sites
        let mask = PermDitMask::<u32>::new(0b11u32);
        basis.add_local(Complex::new(1.0, 0.0), mask);
        basis.build(0u32, &x_op(2));

        assert_eq!(basis.size(), 2);
        assert!(basis.index(3u32).is_some());
        assert!(basis.index(2u32).is_some());
    }

    #[test]
    fn sym_basis_no_symmetry_matches_subspace() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 3, false);
        basis.add_lattice(Complex::new(1.0, 0.0), &[0, 1, 2]);
        basis.build(0u32, &x_op(3));
        assert_eq!(basis.size(), 8);
    }

    #[test]
    fn sym_basis_sorted_ascending() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 3, false);
        basis.add_lattice(Complex::new(1.0, 0.0), &[0, 1, 2]);
        basis.build(0u32, &x_op(3));
        for i in 1..basis.size() {
            assert!(basis.state_at(i) > basis.state_at(i - 1));
        }
    }

    #[test]
    fn sym_basis_u8_norm() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u8>::new_empty(2, 2, false);
        basis.add_lattice(Complex::new(1.0, 0.0), &[0, 1]);
        let mask = PermDitMask::<u32>::new(0b11u32);
        basis.add_local(Complex::new(1.0, 0.0), mask);
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
        basis.add_lattice(Complex::new(1.0, 0.0), &[0, 1]);
        let mask = PermDitMask::<u32>::new(0b11u32);
        basis.add_local(Complex::new(1.0, 0.0), mask);
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
        basis.add_lattice(Complex::new(1.0, 0.0), &identity);
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
        basis.add_lattice(Complex::new(1.0, 0.0), &identity);
        let mask = PermDitMask::<u32>::new((1u32 << n_sites) - 1);
        basis.add_local(Complex::new(1.0, 0.0), mask);
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
