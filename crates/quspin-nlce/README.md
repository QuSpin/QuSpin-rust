# quspin-nlce

Numerical linked-cluster expansion (NLCE) for finite-temperature
thermodynamics of spin-1/2 lattice models, built on QuSpin-rust for
Hamiltonian construction, symmetry sectors, and exact diagonalization.

```text
P/N = Σ_c L(c) W(c),        W(c) = P(c) − Σ_{s ⊂ c} M(s, c) W(s)
```

Phase 1 implements the **rectangle expansion** (all open `m × n`
rectangles up to `m + n ≤ max_order`, optionally capped by site count), with
full ED per cluster, and reports the bare partial sums per order.

```sh
cargo run --release -p quspin-nlce --example heisenberg_partial_sums 8
```

Plots of the partial sums against exact results (Heisenberg, Ising vs
Onsager, XX chain vs free fermions) are in [`plots/`](plots/); regenerate
them with

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
| `ClusterGenerator` | Yields `ClusterType { key, order, graph, lattice_constant = L(c), subclusters = [(s, M(s,c))] }`, sorted by order, closed under sub-clusters. | `RectangleGenerator` |
| `Model` | Hamiltonian on a `ClusterGraph` as a QuSpin `SpinOperatorInner`, plus the symmetries it has (S^z conservation, spin flip). | `Xxz` (Heisenberg / XX / Ising / field) |
| `ClusterSolver<M>` | `(ClusterGraph, &M) → Property`. | `ExactDiagSolver → Thermo` |
| `Property` | Anything with `zeros_like` + `axpy`: the combiner only forms linear combinations. | `Thermo` (arrays over T), `Vec<f64>` |
| `Resummation<P>` | Acts on the partial-sum sequence. | `Bare` (Wynn ε / Euler later) |

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

## Phase 2: a site-based generator

A site-based (general graph) expansion only needs a new `ClusterGenerator`.
The solver, model, combiner, cache, and resummation are unchanged:

1. **Enumerate** connected clusters by growing site sets from
   `Lattice::unit_cell_sites()` with `Lattice::neighbors`, up to `N` sites.
2. **Canonicalize** each cluster's induced graph (from `Lattice::cluster_graph`)
   into `Topology::Canonical(code)`, e.g. a canonical adjacency matrix under
   graph isomorphism. For disorder, put the bond/site labels into the
   `ClusterKey` label fields.
3. **Lattice constants:** count the distinct embeddings per site (site sets
   modulo translation, per unit-cell site). `Lattice::distinct_orientations`
   covers the point-group part.
4. **Multiplicities:** for each cluster, enumerate its connected sub-site-sets,
   canonicalize them, and count occurrences, giving `M(s, c)` as
   `subclusters`.
5. **Order** each cluster by its site (or bond) count and sort ascending.

`ClusterGraph::automorphisms` may be left empty. The solver then uses only
S^z / spin-flip sectors, which is correct but slower. Supplying the graph's
automorphism group lets the ED solver use any elementary-abelian 2-subgroup
automatically.

Other solvers slot in the same way: a Lanczos/FTLM `ClusterSolver` for larger
clusters, or a time-evolution solver returning a `Property` over time.

## Tests

| Test | What it checks |
|---|---|
| `tests/ed_solver.rs` | Symmetry-blocked spectrum = brute-force full-space ED (several models, graphs, D4 clusters); analytic dimer thermodynamics. |
| `tests/identity.rs` | `P(R) = Σ_{s ⊆ R} M(s,R) W(s)` to 1e-12 relative, with `M` counted by brute force, for 3×4. The 4×4 version is `#[ignore]`d; run it with `cargo test -p quspin-nlce --release -- --ignored`. |
| `tests/xx_chain.rs` | 1×n clusters → free-fermion E, S, χ (errors ≤ 1e-12 at T ≥ 1). |
| `tests/ising_onsager.rs` | 2D Ising energy → Onsager above T_c, monotone in order. |
| `tests/heisenberg.rs` | High-T series (`−3β/8 − 3β²/32`, entropy), order-to-order convergence at high T, qualitative C(T), and cache reuse. |
