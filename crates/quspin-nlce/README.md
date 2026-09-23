# quspin-nlce

Numerical linked-cluster expansion (NLCE) for finite-temperature
thermodynamics of spin-1/2 lattice models, built on QuSpin-rust for
Hamiltonian construction, symmetry sectors, and exact diagonalization.

```text
P/N = Σ_c L(c) W(c),        W(c) = P(c) − Σ_{s ⊂ c} M(s, c) W(s)
```

Two expansions are implemented, both with full ED per cluster and bare
partial sums per order:

- **Rectangle expansion** (`RectangleGenerator`): all open `m × n`
  rectangles up to `m + n ≤ max_order`, optionally capped by site count.
- **Topological bond expansion** (`BondGenerator`): all connected bond sets
  up to `max_order` bonds, merged by graph isomorphism (see below).

```sh
cargo run --release -p quspin-nlce --example heisenberg_partial_sums 8          # rectangles
cargo run --release -p quspin-nlce --example heisenberg_partial_sums 12 bond    # bonds
cargo run --release -p quspin-nlce --example bond_census 12 bond12.clusters     # counts + save DAG
cargo run --release -p quspin-nlce --example heisenberg_partial_sums 10 load bond12.clusters
```

### Reusing cluster DAGs (`ClusterSet`)

The cluster DAG is model independent. `ClusterSet::from_generator(&gen, desc)`
captures it, `.save(path)` / `ClusterSet::load(path)` store it in a small
line-based text format (documented in `src/store.rs`), and `ClusterSet` is
itself a `ClusterGenerator`, so a loaded set goes straight into `run_nlce`.
`.truncated(n)` keeps orders `≤ n` (valid because sub-clusters never exceed
their parent's order). The 12-bond square-lattice DAG is 3.2 MB and loads in
≈0.15 s, versus ≈17–20 s to regenerate. Loading validates every graph,
automorphism, and sub-cluster reference.

Plots of the partial sums against exact results (Heisenberg, Ising vs
Onsager, XX chain vs free fermions) and of the bond vs rectangle expansion
(`bond_vs_rect_*.png`) are in [`plots/`](plots/); regenerate them with

```sh
cargo run --release -p quspin-nlce --example export_csv -- crates/quspin-nlce/plots
python crates/quspin-nlce/plots/make_plots.py crates/quspin-nlce/plots   # needs matplotlib
```

## Trait boundaries

```text
Lattice ──► ClusterGenerator ──► [ClusterType] ──► ClusterSolver<M: Model> ──► P: Property
                                       │                                          │
                                       └──────────────► combine ◄─────────────────┘
                                                          │
                                           NlceResult { weights, partial_sums }
                                                          │
                                                     Resummation
```

| Trait | Responsibility | Phase 1 impl |
|---|---|---|
| `Lattice` | Infinite lattice: sites, labelled neighbour bonds, point group, canonical translation. Provides `distinct_orientations` (→ `L(c)`) and `cluster_graph` (induced open cluster + automorphisms). | `SquareLattice`, `ChainLattice` |
| `ClusterGenerator` | Yields `ClusterType { key, order, graph, lattice_constant = L(c), subclusters = [(s, M(s,c))] }`, sorted by order, closed under sub-clusters. | `RectangleGenerator`, `BondGenerator` |
| `Model` | Hamiltonian on a `ClusterGraph` as a QuSpin `SpinOperatorInner`, plus the symmetries it has (S^z conservation, spin flip). | `Xxz` (Heisenberg / XX / Ising / field) |
| `ClusterSolver<M>` | `(ClusterGraph, &M) → Property`. | `ExactDiagSolver → Thermo` |
| `Property` | Anything with `zeros_like` + `axpy`: the combiner only forms linear combinations. | `Thermo` (arrays over T), `Vec<f64>` |
| `Resummation<P>` | Acts on the partial-sum sequence (`NlceResult::sums()`). | `Bare`, `Wynn { cycles }`, `Euler { start }` |

`combine` / `run_nlce` never look at geometry or physics: they walk the
`ClusterType` list in order, subtract `M(s,c) · W(s)`, and accumulate
`L(c) · W(c)` per order. The cache (`ClusterCache`) is keyed by
`ClusterKey { topology, bond_labels, site_labels }`. Phase 1 fills only the
topology, but the labels are there so disordered or inhomogeneous clusters
with the same shape get separate cache entries. `run_nlce_cached` reuses an
existing cache when extending to a higher order.

### How the ED solver block-diagonalizes

Everything goes through QuSpin-rust:
- One `SpaceKind::Symm` (or `Sub`) `GenericBasis` per block.
- Every configuration of an S^z sector is passed as a BFS seed, so sectors
  come out complete even for diagonal Hamiltonians.
- Lattice symmetry uses the largest elementary-abelian 2-subgroup of the
  cluster automorphisms with all its ±1 characters. For rectangles this is
  C2v. Square clusters use only this subgroup of D4, because QuSpin's
  symmetric bases are validated for one-dimensional characters.
- For spin-flip-symmetric models, spin flip is added at S^z = 0 and the
  −S^z sectors are skipped (they have the same spectrum as +S^z).
- The Hamiltonian is built with `QMatrixInner::build_spin` → `to_dense`, then
  diagonalized with `quspin_krylov::dense::eigvalsh`.
- The solver checks that the block dimensions add up to 2^N.

Thermodynamics use Boltzmann weights shifted by the ground-state energy.

## Topological bond expansion: how the clusters are built

Clusters are connected sets of lattice bonds (the cluster Hamiltonian has
exactly those bonds), the order is the bond count, and embeddings with
isomorphic graphs share one cluster type. `BondGenerator` builds them in
four steps (`src/generator/bond.rs`, `src/canon.rs`):

1. **Enumerate each embedding exactly once — Redelmeier's algorithm.** Bonds
   of a finite lattice patch get integer ids in a translation-invariant order
   (by endpoints). Every translation class of connected `n`-bond clusters has
   exactly one representative whose smallest bond starts in the unit cell.
   From each such root bond, a depth-first search on the line graph (bonds
   adjacent when they share a site) grows clusters: each stack frame pops a
   bond from an *untried* list, adds it, records the cluster, and recurses
   with the untried list enlarged by the new bond's neighbours (only bonds
   above the root, each offered at most once per path). After the subtree
   *with* that bond is explored, the bond stays excluded for the rest of the
   frame, so the search tree partitions the clusters and none is produced
   twice. No hash set of seen clusters is needed, and memory is tiny.
2. **Canonicalise each embedding.** Colour refinement plus
   individualisation (nauty's scheme, without pruning) yields the
   lexicographically smallest relabelled adjacency matrix — equal for two
   graphs iff they are isomorphic. The same search returns the automorphism
   group, which the ED solver uses for block-diagonalisation.
3. **Lattice constants.** `L(c)` = number of embeddings that produced code
   `c`, per unit-cell site.
4. **Multiplicities.** Redelmeier again on each topology's own line graph
   enumerates its connected proper sub-bond-sets once each; canonicalising
   them gives `M(s, c)`. The single site is the order-0 cluster.

**Round-off.** `W(c)` is an alternating-sign combination over all
sub-clusters, and the bond expansion multiplies it by lattice constants up
to ~10⁴ over thousands of topologies, so rounding in `P(c)` is amplified
enormously. At 12 bonds this leaves a floor of ~1e-9 (Ising, exact
eigenvalues; the thermal sums use compensated summation) to ~1e-8
(Heisenberg, dense-eigensolver rounding) in the per-site sums at high T —
irrelevant where the expansion is not already converged, but visible in the
high-T tails of `plots/bond_vs_rect_*.png`.

Every embedding must be visited to count `L(c)`, so enumeration dominates:
on the square lattice there are 20,971,920 embeddings up to 12 bonds but only
4,423 topologies, and the whole DAG takes ≈17 s on 4 cores
(`examples/bond_census.rs`). Possible further speed-ups: canonicalise only
one embedding per point-group orbit (weighting by orbit size), or prune the
canonicalisation search with automorphisms.

## Resummation

`Wynn { cycles }` (Wynn's ε on the last `2·cycles + 1` partial sums) and
`Euler { start }` (Euler transformation of the terms from position `start`)
act on each component through the `Componentwise` trait, implemented for
`Thermo` and `Vec<f64>`. As in Tang, Khatami & Rigol, trust a resummed value
only where several methods agree (`plots/resummed_*.png`):

- **Heisenberg:** bare bond sums are useless below T ≈ 0.7 J, but Wynn with
  2–5 cycles turns them into estimates that agree with each other and with
  Wynn-resummed rectangles to ~3e-3 down to T ≈ 0.5 J; the specific-heat peak
  comes out ≈ 0.45–0.47 near T ≈ 0.6 J.
- **Ising:** against Onsager, Wynn with 2 cycles cuts the rectangle error by
  10–1000× above T_c. The bond series must be thinned to even orders first
  (odd orders vanish exactly); it gains less.
- **Euler** needs an alternating tail: it helps the even/odd-alternating
  Heisenberg bond series at low T but is biased (~1e-5 at T = 2 J for
  `start = 3`) where the series has already converged, and makes the
  non-alternating Ising series worse.

## Adding another generator (e.g. site-based)

Only a new `ClusterGenerator` is needed; the solver, model, combiner, cache,
and resummation are unchanged. A site-based expansion would run the same
Redelmeier search over sites instead of bonds, take the *induced* subgraph
(`Lattice::cluster_graph`) of each site set, and reuse `canon` for
topologies and sub-cluster counting. `ClusterGraph::automorphisms` may be
left empty (correct, just slower ED).

Other solvers slot in the same way: a Lanczos/FTLM `ClusterSolver` for larger
clusters, or a time-evolution solver returning a `Property` over time.

## Tests

| Test | What it checks |
|---|---|
| `tests/ed_solver.rs` | Symmetry-blocked spectrum = brute-force full-space ED (several models, graphs, D4 clusters, every bond topology up to 6 bonds); analytic dimer thermodynamics. |
| `tests/identity.rs` | `P(R) = Σ_{s ⊆ R} M(s,R) W(s)` to 1e-12 relative, with `M` counted by brute force, for 3×4. The 4×4 version is `#[ignore]`d; run it with `cargo test -p quspin-nlce --release -- --ignored`. |
| `tests/xx_chain.rs` | 1×n clusters → free-fermion E, S, χ (errors ≤ 1e-12 at T ≥ 1). |
| `tests/ising_onsager.rs` | 2D Ising energy → Onsager above T_c, monotone in order; Wynn beats the bare sums near T_c. |
| `src/resum.rs` (unit) | Wynn exact on geometric series; Wynn and Euler accelerate the alternating series for ln 2; converged sequences stay finite; per-component handling of `Thermo`. |
| `tests/heisenberg.rs` | High-T series (`−3β/8 − 3β²/32`, entropy), order-to-order convergence at high T, qualitative C(T), and cache reuse. |
| `src/store.rs` (unit) | Save/load round-trips for both generators; truncation equals lower-order generation; malformed files rejected. |
| `tests/bond_expansion.rs` | Bond expansion: identical to the rectangle expansion on the chain, order-`n` contributions vanish at least as `βⁿ`, agreement with rectangles at high T, Ising → Onsager. Unit tests in `bond.rs` / `canon.rs` check Redelmeier against brute force, `L(c)`/`M(s,c)` by hand and by exhaustive subsets, and the canonicaliser against the counts of unlabelled graphs. |
