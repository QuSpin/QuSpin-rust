/// Merged symmetry-group + basis type: [`SymBasis<B, L, N>`].
///
/// Replaces the two-type hierarchy (`SymGrpBase<B,L>` + `SymmetricSubspace<G,N>`)
/// with a single flat struct. `N` is an explicit type parameter — the B→N
/// pairing is encoded in the per-family dispatch enum variant definitions
/// (`BitBasisDefault`, `TritBasisDefault`, etc.), not a runtime enum.
use super::bfs::{AMP_CANCEL_TOL, PARALLEL_FRONTIER_THRESHOLD};
use super::dispatch::validate::validate_perm;
use super::lattice::{BenesLatticeElement, CompositeElement, LocalElement};
use super::traits::BasisSpace;
use num_complex::Complex;
use quspin_bitbasis::{
    BenesPermDitLocations, BitInt, BitStateOp, Compose, FermionicBitStateOp, StateTransitions,
    manip::DynamicDitManip,
};
use quspin_types::QuSpinError;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Minimum batch size before switching to parallel orbit computation.
const PARALLEL_BATCH_THRESHOLD: usize = 256;

/// Maximum allowed error on the 1D character condition `χ(g·h) = χ(g)·χ(h)`
/// during `validate_group`. Characters are complex unit-modulus scalars, so
/// `1e-10` absorbs floating-point rounding from repeated multiplications but
/// still catches a genuinely-wrong user-supplied character.
const CHAR_VALIDATION_TOL: f64 = 1e-10;

/// Uniform borrow of a group element for `validate_group`: `(char, perm, local)`
/// where the last two are `None` for the implicit identity / pure-lattice /
/// pure-local cases.
type ElementRef<'a, B, L> = (
    Complex<f64>,
    Option<&'a BenesPermDitLocations<B>>,
    Option<&'a L>,
);

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
/// This pairing is enforced statically by the per-family dispatch enum
/// variant definitions (e.g. [`BitBasisDefault`](crate::dispatch::BitBasisDefault)).
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
    pub(crate) local_only: Vec<LocalElement<L>>,
    /// Composite group elements: site permutation + local op applied
    /// atomically as one element with a single character.
    pub(crate) composite: Vec<CompositeElement<B, L>>,
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
    /// The site-permutation component (if present) is validated against
    /// `self.n_sites` here — every dispatch path through `SymBasis<B,L,N>`
    /// shares this check, so callers don't need to pre-validate.
    ///
    /// The walker dispatches each element into one of three typed
    /// storage vectors at insert time based on which components are
    /// present, so the orbit hot loop has zero variant-branching.
    pub fn add_symmetry(
        &mut self,
        grp_char: Complex<f64>,
        element: crate::SymElement<L>,
    ) -> Result<(), QuSpinError> {
        if self.built {
            return Err(QuSpinError::ValueError(
                "cannot add symmetry elements after basis is built".into(),
            ));
        }
        let (perm, local) = element.into_parts();
        if let Some(ref p) = perm {
            validate_perm(p, self.n_sites)?;
        }
        match (perm, local) {
            (Some(p), None) => {
                let op = BenesPermDitLocations::<B>::new(self.lhss, &p, self.fermionic);
                self.lattice_only
                    .push(BenesLatticeElement::new(grp_char, op, self.n_sites));
                Ok(())
            }
            (None, Some(l)) => {
                self.local_only.push(LocalElement::new(grp_char, l));
                Ok(())
            }
            (Some(p), Some(l)) => {
                let op = BenesPermDitLocations::<B>::new(self.lhss, &p, self.fermionic);
                let lat = BenesLatticeElement::new(grp_char, op, self.n_sites);
                self.composite.push(CompositeElement::new(lat, l));
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
        super::orbit::get_refstate(&self.lattice_only, &self.local_only, &self.composite, state)
    }

    /// Return the representative state and the orbit norm.
    pub fn check_refstate(&self, state: B) -> (B, f64) {
        super::orbit::check_refstate(&self.lattice_only, &self.local_only, &self.composite, state)
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
            states,
            out,
        );
    }

    /// Check that the supplied group elements are closed under composition
    /// and that the supplied characters form a consistent 1D representation
    /// (`χ(g·h) = χ(g)·χ(h)`).
    ///
    /// The walker assumes every element of the group is present in one of
    /// the three storage vectors (plus the implicit identity) — users who
    /// supply only generators, or who forget a product like `g · h`, will
    /// produce an incorrect basis. This probe-state check catches those
    /// mistakes at [`build`](Self::build) time with a clear error message.
    ///
    /// Runtime: `O(|G|² · probes)` with small constants — each composed
    /// action is matched against the element list in `O(1)` via a
    /// precomputed `HashMap` keyed on the action vector. `|G|` is
    /// bounded by any physical symmetry group so this is inexpensive
    /// compared to the basis-construction BFS that follows.
    pub fn validate_group(&self) -> Result<(), QuSpinError> {
        // Nothing to validate: only the implicit identity is in play.
        if self.group_order() == 1 {
            return Ok(());
        }

        // Unified element list, implicit identity at index 0.
        let mut elements: Vec<ElementRef<'_, B, L>> = Vec::with_capacity(self.group_order());
        elements.push((Complex::new(1.0, 0.0), None, None));
        for lat in &self.lattice_only {
            elements.push((lat.grp_char, Some(&lat.op), None));
        }
        for el in &self.local_only {
            elements.push((el.grp_char, None, Some(&el.op)));
        }
        for el in &self.composite {
            elements.push((el.lat.grp_char, Some(&el.lat.op), Some(&el.loc)));
        }

        let probes = probe_states::<B>(self.n_sites, self.lhss);

        // Precompute each element's action on every probe state.
        let actions: Vec<Vec<B>> = elements
            .iter()
            .map(|elem| probes.iter().map(|&s| apply_action(elem, s)).collect())
            .collect();

        // Index by action so composed-action lookups are O(1) average
        // time instead of O(|G|). Detect duplicates while building the
        // map: two elements with the same action would inflate `|G|`
        // and double-count orbit images. The implicit identity is
        // `actions[0]`, so this also catches users who add an identity
        // permutation explicitly.
        let mut action_to_index: HashMap<Vec<B>, usize> = HashMap::with_capacity(actions.len());
        for (i, action) in actions.iter().enumerate() {
            if let Some(&prev) = action_to_index.get(action) {
                let hint = if prev == 0 {
                    " (element 0 is the implicit identity — do not add \
                     an identity permutation or identity-action local op)"
                } else {
                    ""
                };
                return Err(QuSpinError::ValueError(format!(
                    "symmetry elements g_{prev} and g_{i} have the same action{hint}"
                )));
            }
            action_to_index.insert(action.clone(), i);
        }

        // Scratch buffer reused across the O(|G|²) loop so we don't
        // allocate a fresh Vec per (i, j) pair.
        let mut composed: Vec<B> = vec![B::from_u64(0); probes.len()];
        for i in 0..elements.len() {
            for j in 0..elements.len() {
                // (g_i · g_j)(s) = g_i(g_j(s)).
                for (slot, &s) in composed.iter_mut().zip(probes.iter()) {
                    let after_j = apply_action(&elements[j], s);
                    *slot = apply_action(&elements[i], after_j);
                }

                let Some(&k) = action_to_index.get(&composed) else {
                    return Err(QuSpinError::ValueError(format!(
                        "symmetry group is not closed: composition g_{i} · g_{j} \
                         is not in the supplied element list \
                         (index 0 = implicit identity, indices 1.. are user-added)"
                    )));
                };

                let expected = elements[i].0 * elements[j].0;
                let actual = elements[k].0;
                if (expected - actual).norm() > CHAR_VALIDATION_TOL {
                    return Err(QuSpinError::ValueError(format!(
                        "character table inconsistent for 1D representation: \
                         χ(g_{i}) · χ(g_{j}) = {expected}, \
                         but the matching element χ(g_{k}) = {actual}"
                    )));
                }
            }
        }
        Ok(())
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
    ///
    /// # Errors
    ///
    /// On the first call per basis, runs [`validate_group`](Self::validate_group)
    /// and returns its error if the supplied group is not closed, the
    /// characters are inconsistent, or two elements have the same action.
    /// Subsequent calls (e.g. additional seeds on an already-built basis)
    /// reuse the validated group.
    pub fn build<G>(&mut self, seed: B, graph: &G) -> Result<(), QuSpinError>
    where
        G: StateTransitions,
        L: Sync,
    {
        if graph.lhss() != self.lhss {
            return Err(QuSpinError::ValueError(format!(
                "graph.lhss()={} does not match basis lhss={}",
                graph.lhss(),
                self.lhss,
            )));
        }
        if !self.built {
            self.validate_group()?;
        }
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

            // Phase 1+2a (fused): BFS discovery + orbit-representative mapping
            // in a single parallel fold — one fork/join instead of two.
            // Uses get_refstate (no norm computation) per surviving candidate.
            let candidate_reps: HashSet<B> =
                bfs_wave_with_reps(&frontier, graph, lattice_only, local_only, composite);

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
        Ok(())
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
// validate_group helpers
// ---------------------------------------------------------------------------

/// Apply a unified `(char, Option<perm>, Option<local>)` element to a state.
///
/// The character is ignored — this helper exists for action comparison
/// inside [`SymBasis::validate_group`]. Perm is applied first, then local.
#[inline]
fn apply_action<B, L>(elem: &ElementRef<'_, B, L>, state: B) -> B
where
    B: BitInt,
    L: FermionicBitStateOp<B>,
{
    let mut s = state;
    if let Some(perm) = elem.1 {
        s = perm.apply(s);
    }
    if let Some(local) = elem.2 {
        s = local.apply(s);
    }
    s
}

/// Generate a probe-state set used by [`SymBasis::validate_group`].
///
/// Two element actions are considered equal iff they map every probe state
/// to the same output. All probes must be **valid** (every dit in
/// `0..lhss`) because a `PermDitValues`-family local op indexes into a
/// length-`lhss` permutation with the extracted dit — invalid dit values
/// would panic with out-of-bounds.
///
/// The returned set consists of valid states that:
/// - cover every full dit-value / site combination for small systems
///   (`lhss^n_sites ≤ 32`), enabling exhaustive discrimination;
/// - cover every single-non-zero-dit configuration (`dit=v` at exactly
///   one site, `0` elsewhere) — sufficient to distinguish any site
///   permutation or any per-site local op even on large systems.
fn probe_states<B: BitInt>(n_sites: usize, lhss: usize) -> Vec<B> {
    let manip = DynamicDitManip::new(lhss);
    let mut probes: Vec<B> = Vec::new();

    // All-zero state: distinguishes local ops that act non-trivially on
    // the vacuum.
    probes.push(B::from_u64(0));

    // Exhaustive small coverage: every valid state of the full Hilbert
    // space, as long as it's bounded. Catches multi-dit local-op
    // differences that single-dit probes might miss.
    const EXHAUSTIVE_CAP: usize = 32;
    if let Some(total) = lhss.checked_pow(n_sites as u32)
        && total <= EXHAUSTIVE_CAP
    {
        for dense in 1..total {
            let s = manip.state_from_dense::<B>(dense, n_sites);
            if !probes.contains(&s) {
                probes.push(s);
            }
        }
        return probes;
    }

    // Single-non-zero-dit states: `dit=v` at site `i`, zero elsewhere.
    // Distinguishes any permutation (compare for each `(i, v)`) and any
    // local op that acts differently on any single-site state.
    for i in 0..n_sites {
        for v in 1..lhss {
            let s = manip.set_dit::<B>(B::from_u64(0), v, i);
            if !probes.contains(&s) {
                probes.push(s);
            }
        }
    }
    probes
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
fn bfs_wave_with_reps<B, L, G>(
    frontier: &[B],
    graph: &G,
    lattice_only: &[BenesLatticeElement<B>],
    local_only: &[LocalElement<L>],
    composite: &[CompositeElement<B, L>],
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
                    let (rep, _) =
                        super::orbit::get_refstate(lattice_only, local_only, composite, candidate);
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
    local_only: &[LocalElement<L>],
    composite: &[CompositeElement<B, L>],
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
                super::orbit::check_refstate_batch(lattice_only, local_only, composite, s, o);
            });
    } else {
        super::orbit::check_refstate_batch(lattice_only, local_only, composite, states, out);
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
        // Spin inversion Z2: XOR mask at all sites. Identity is implicit.
        let mask = PermDitMask::<u32>::new(0b11u32);
        basis
            .add_symmetry(Complex::new(1.0, 0.0), crate::SymElement::local(mask))
            .unwrap();
        basis.build(0u32, &x_op(2)).unwrap();

        assert_eq!(basis.size(), 2);
        assert!(basis.index(3u32).is_some());
        assert!(basis.index(2u32).is_some());
    }

    #[test]
    fn sym_basis_no_symmetry_matches_subspace() {
        // Trivial group (only implicit identity) — should match the full
        // non-symmetric basis for 3 sites LHSS=2.
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 3, false);
        basis.build(0u32, &x_op(3)).unwrap();
        assert_eq!(basis.size(), 8);
    }

    #[test]
    fn sym_basis_sorted_ascending() {
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 3, false);
        basis.build(0u32, &x_op(3)).unwrap();
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
        basis.build(0u32, &x_op(2)).unwrap();
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
        basis.build(0u32, &x_op(2)).unwrap();
        for i in 0..basis.size() {
            let (_, norm) = basis.entry(i);
            assert_eq!(norm, 1.0);
        }
    }

    // --- Parallel path tests ---

    #[test]
    fn sym_basis_parallel_no_symmetry_matches_full() {
        // 12 sites, trivial group (only implicit identity).
        // No symmetry reduction → should find all 2^12 = 4096 representatives.
        // Frontier grows large enough to trigger parallel BFS.
        let n_sites = 12u32;
        let mut basis =
            SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, n_sites as usize, false);
        basis.build(0u32, &x_op(n_sites)).unwrap();
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
        let mask = PermDitMask::<u32>::new((1u32 << n_sites) - 1);
        basis
            .add_symmetry(Complex::new(1.0, 0.0), crate::SymElement::local(mask))
            .unwrap();
        basis.build(0u32, &x_op(n_sites)).unwrap();

        // With Z2 spin inversion (XOR all bits) on 12 sites:
        // No state equals its own complement, so every orbit has exactly 2
        // distinct elements. Representatives = 4096 / 2 = 2048.
        assert_eq!(basis.size(), 2048);
        for i in 1..basis.size() {
            assert!(basis.state_at(i) > basis.state_at(i - 1));
        }
    }

    // --- validate_group tests ---

    #[test]
    fn validate_group_rejects_missing_closure() {
        // Add a 3-cycle translation T on 3 sites but forget T² — the group
        // isn't closed because T · T = T² is not in the list.
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 3, false);
        basis
            .add_symmetry(
                Complex::new(1.0, 0.0),
                crate::SymElement::lattice(&[1, 2, 0]),
            )
            .unwrap();
        let err = basis.validate_group().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not closed"), "unexpected error: {msg}");
    }

    #[test]
    fn validate_group_rejects_wrong_characters() {
        // Full translation group {T, T²} with inconsistent characters:
        // χ(T) = -1 would require χ(T²) = (-1)² = 1, but we supply 2.
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 3, false);
        basis
            .add_symmetry(
                Complex::new(-1.0, 0.0),
                crate::SymElement::lattice(&[1, 2, 0]),
            )
            .unwrap();
        basis
            .add_symmetry(
                Complex::new(2.0, 0.0),
                crate::SymElement::lattice(&[2, 0, 1]),
            )
            .unwrap();
        let err = basis.validate_group().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("character"), "unexpected error: {msg}");
    }

    #[test]
    fn validate_group_rejects_identity_permutation() {
        // Explicit identity permutation duplicates the implicit identity.
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 3, false);
        basis
            .add_symmetry(
                Complex::new(1.0, 0.0),
                crate::SymElement::lattice(&[0, 1, 2]),
            )
            .unwrap();
        let err = basis.validate_group().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("same action"), "unexpected error: {msg}");
    }

    #[test]
    fn validate_group_accepts_add_cyclic() {
        // `add_cyclic` generates the full cyclic group from a single
        // generator, so validation must pass.
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 4, false);
        basis
            .add_cyclic(crate::SymElement::lattice(&[1, 2, 3, 0]), 4, |_| {
                Complex::new(1.0, 0.0)
            })
            .unwrap();
        basis.validate_group().expect("cyclic group must validate");
    }

    #[test]
    fn composite_pz_group_has_order_2() {
        // The motivating case for the refactor: neither P (reflection)
        // nor Z (spin-flip) alone is a symmetry, but PZ is. Supplied as
        // a single composite element the group is `{I, PZ}` of order 2 —
        // not the cartesian product `⟨P⟩ × ⟨Z⟩` of order 4 that a
        // naïve two-vector walker would produce.
        let mut basis = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, 2, false);
        basis
            .add_symmetry(
                Complex::new(1.0, 0.0),
                crate::SymElement::composite(&[1, 0], PermDitMask::<u32>::new(0b11u32)),
            )
            .unwrap();
        basis.validate_group().expect("{I, PZ} must validate");
        assert_eq!(
            basis.group_order(),
            2,
            "composite-only PZ group must have |G| = 2"
        );
        assert_eq!(basis.composite.len(), 1);
        assert!(basis.lattice_only.is_empty());
        assert!(basis.local_only.is_empty());
    }
}
