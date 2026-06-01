---
marp: true
theme: default
paginate: true
size: 16:9
header: "QuSpin-rust: design patterns for new contributors"
style: |
  section { font-size: 22px; padding: 40px 56px; max-width: none; }
  section > * { max-width: 100%; }
  section p, section ul, section ol, section table { max-width: 100%; }
  h1 { font-size: 34px; }
  h2 { font-size: 28px; }
  code { font-size: 0.85em; }
  pre { font-size: 0.75em; }
  table { font-size: 0.85em; }
  .small { font-size: 0.8em; }
  .muted { color: #666; }
---

# QuSpin-rust

## Design patterns for new contributors

A tour of the library aimed at physicists who would like to contribute code.

<br>

We will spend most of our hour on **how the basis is built**, because that is where QuSpin earns its keep and where most of the interesting mechanical moving parts live.

<!-- Speaker notes:
This deck assumes physics background (Fock states, Hamiltonians, symmetry
sectors) and zero Rust background. Goal: by the end, an attendee can read
the crate DAG, identify which crate owns a given change, and write a
failing test in the right place.
-->

---

# Overview

**Act 1 — The problem and motivating the architecture** *(~15 min)*

**Act 2 — Deep Dive: The basis layer component by component** *(~30 min)*

**Act 3 — Wrapping it up: Matrix, dynamics, contributing** *(~10 min)*

---

# Act 1 — The problem and motivating the architecture

We will spend ~15 minutes here. The aim is to motivate the package structure **before** looking at any code.

1. The exact-diagonalization (ED) problem
2. What we actually need to compute
3. Decomposing ED into discrete jobs
4. Why separation of concerns gives us crates
5. The crate DAG
6. The running example: Heisenberg XXZ

---

# The problem: exact diagonalization

For a quantum lattice model on `L` sites with local Hilbert-space size `d`:

$$
\mathcal{H} = \underbrace{d^L}_{\text{Hilbert dimension}} \quad\quad H \in \mathbb{C}^{d^L \times d^L}
$$

| L  | d=2 (spin-1/2) | d=3 (boson, n_max=2) |
|----|----------------|----------------------|
| 10 | 1,024          | 59,049               |
| 20 | ~10⁶           | ~3.5 × 10⁹           |
| 30 | ~10⁹           | (out of reach)       |

**We cannot store `H` as a dense matrix.** We cannot even *enumerate* the full basis past ~30 sites. Yet `H` is **sparse** and **structured** — most of the mass sits in tiny symmetry blocks. The whole library is built around exploiting that.

---

# What we actually need

Three observables, all reducing to one primitive:

- **Ground state / low-lying spectrum** → Lanczos / Krylov eigensolvers
- **Real-time dynamics** $|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle$ → Krylov / Taylor `expm`
- **Imaginary-time / thermal averages** → FTLM, LTLM

Every one of them only needs **one operation**:

```
        v  ↦  H v          (sparse matrix–vector product, "matvec")
```

So our problem reduces to: **produce a matvec on a symmetry-reduced sparse `H`, as cheaply as possible.**

---

# Decomposing the calculation

A matvec is one line. Producing it from a physics problem is five jobs:

|   | Job                                            | Concerns                              |
|---|------------------------------------------------|---------------------------------------|
| a | Represent Fock states                          | bit packing, multi-word integers      |
| b | Represent operators **without a basis**        | terms in H as data, not matrices      |
| c | Enumerate the connected, symmetry-reduced basis| BFS, orbits, characters               |
| d | Assemble the sparse `H` on that basis          | CSR layout, parallel fill             |
| e | Consume the matvec (eigensolver / propagator)  | Lanczos / Taylor expm                 |

Each job has **different inputs, different invariants, different code that changes for different reasons.** This is the engineering thesis of the project.

---

# Design pattern: separation of concerns → crates

Five jobs ⇒ (roughly) five crates. Why bother?

- **Local reasoning** *(single-responsibility principle).* A bug in BFS does not require reading the matrix-fill code. A new operator term does not touch the symmetry layer.
- **Parallel compilation.** Independent crates rebuild in parallel — minutes → seconds.
- **Swap-ability via traits** *(open/closed principle).* The basis layer doesn't import operators — it asks "give me anything that can tell me a state's neighbours." Spin / boson / fermion all plug in the same way; a future operator type extends the system without modifying basis, matrix, or krylov.
- **Static dispatch.** The trait boundaries are crossed by generics, not `dyn`. Rust monomorphises at link time — the abstractions compile away.

The rest of the deck makes those abstractions concrete.

---

# The crate DAG

```
              quspin-types          ← errors, dtypes, BitInt, StateTransitions,
                    │                 LinearOperator (the workspace trait floor)
              quspin-bitbasis       ← bit twiddling, Benes networks, DitManip
                    │
       ┌────────────┼─────────────┬─────────────┐
  quspin-operator  quspin-basis  quspin-expm  quspin-krylov
       │            │             │             │
       └────────────┴──── quspin-matrix ────────┘
                          │
                     quspin-core   ← pure facade, ~90 lines
                          │
                     quspin-py     ← PyO3 bindings → Python
```

- The four mid-level crates **build in parallel** off `quspin-bitbasis`.
- `quspin-expm` / `quspin-krylov` know nothing about operators or bases — they consume the generic `LinearOperator` trait.
- `quspin-core` has **zero logic**. Never put code there.

---

# The running example: Heisenberg XXZ

For the rest of the talk, we use this concrete model on `L = 8` spin-1/2:

$$
H \;=\; J \sum_{i=0}^{L-1}\Big( S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + \Delta\, S^z_i S^z_{i+1} \Big), \quad \text{PBC}
$$

with two symmetries:

- $U(1)$: total $S^z$ conserved (we pick the $S^z = 0$ sector — 70 states)
- $\mathbb{Z}_L$: translation by one site (we pick momentum $k = 0$)

In Python, the whole calculation looks like:

```python
group  = SymmetryGroup(n_sites=8, lhss=2)
group.add_cyclic(Lattice(translation_by_1), k=0)
basis  = SpinBasis.symmetric(group, H_terms, seeds=["00001111"])
mat    = QMatrix.build_bond(H_terms, basis, np.float64)
ham    = Hamiltonian(mat, [Static()])
E0, ψ0, _ = EigSolver(ham).solve(v0, k_krylov=30, k_wanted=1, which="SA")
```

**By the end you will know which crate every one of these lines touches.**

---

# Act 2 — Deep Dive: The basis layer component by component

For the next ~30 minutes we walk bottom-up through the three crates that actually build a basis:

- **`quspin-bitbasis`** — encoding states and permutations as bits (5 slides)
- **`quspin-operator`** — operators as basis-agnostic term lists (3 slides)
- **`quspin-basis`** — BFS, orbits, symmetry validation (8 slides)

This is the part of the codebase with the most moving parts and the clearest physics payoff. The rest of the library is plumbing around it.

---

# Job (a): encoding Fock states as integers

Packing convention (`quspin-bitbasis/src/manip.rs`):

- **Site 0 = least-significant bits.** Each site uses a *fixed* $b = \lceil\log_2\text{lhss}\rceil$ bits; site $i$ occupies bits $[i\cdot b,\; (i{+}1)\cdot b)$.

```
Spin-1/2  (lhss=2, b=1):                Boson  (lhss=3, b=2, n∈{0,1,2}):
  site:  7 6 5 4 3 2 1 0                  site:  3   2   1   0
  spin:  ↑ ↓ ↑ ↑ ↓ ↑ ↑ ↑                  n_i:   2   1   0   2
  bit:   1 0 1 1 0 1 1 1  → 0xB7          bits:  10  01  00  10  → 0x92
```

Buys us $O(1)$ equality / hashing / sort, symmetries as bit permutations, and a sorted `Vec<integer>` as the entire basis storage.

**Catch:** `L = 100` doesn't fit in `u64` → next slide.

---

# Design pattern: `BitInt` — generic over integer width

```rust
// quspin-types/src/bit_int.rs
pub trait BitInt: Copy + Eq + Ord + Hash
    + BitAnd + BitOr + BitXor + Not + Shl<usize> + Shr<usize> + ... {
    const BITS: u32;            // 32 / 64 / N
    fn from_u64(v: u64) -> Self;
    fn to_usize(self) -> usize;
    fn count_ones(self) -> u32;
}

impl BitInt for u32  { ... }
impl BitInt for u64  { ... }
impl<const N: usize, const LIMBS: usize> BitInt for Uint<N, LIMBS> { ... }
//                                            ↑ ruint multi-word integer
```

**Reading the syntax:**

- `T: Copy + BitAnd + ...` — `T` must satisfy *all* of these; `+` means *and*.
- Bit manipulation comes from the operator supertraits (`&`, `|`, `^`, `<<`, `>>`), not from named methods.
- `<B: BitInt>` on a struct/fn — works for any `B` implementing `BitInt`.

**Static polymorphism:** one specialised copy per `B` used, zero runtime cost.

---

# `DitManip` — site-level API hiding the bit packing

For `lhss > 2`, bits-per-site varies. Higher layers do **not** want to remember that. `DitManip` gives a uniform site-level API:

```rust
// quspin-bitbasis/src/manip.rs
pub struct DitManip<const LHSS: usize>;            // zero-sized marker

impl<const LHSS: usize> DitManip<LHSS> {
    pub fn get_dit<I: BitInt>(state: I, site: usize) -> usize;
    pub fn set_dit<I: BitInt>(state: I, val: usize, site: usize) -> I;
    pub fn state_from_dense<I: BitInt>(dense: usize, n_sites: usize) -> I;
    pub fn dense_from_state<I: BitInt>(state: I, n_sites: usize) -> Option<usize>;
}                                                  // + get_sub_state / set_sub_state
```

For Heisenberg (`LHSS = 2`), this collapses to a single bit get/set/xor. For a Bose-Hubbard model with `LHSS = 5`, it does the 3-bit packing automatically.

`LHSS` is a *const generic* — the per-site arithmetic is monomorphised and the multiplies / shifts become constants.

---

# Benes networks — site permutations in $O(\log L)$ time

A translation by 1 site on `L = 8` permutes bit positions:

```
in:    b7 b6 b5 b4 b3 b2 b1 b0
out:   b6 b5 b4 b3 b2 b1 b0 b7
```

Reflection, point group, sublattice permutations are all the same shape, all needed millions of times during BFS.

**Benes network:** any permutation factors into $2\log_2 L - 1$ stages of "swap masked pairs by a fixed shift." Precompute the masks **once**:

```rust
let net: BenesNetwork<u64> = gen_benes(&translation_targets);
let permuted = benes_fwd(&net, state);   // hot loop: O(log L) shifts + ANDs
```

**Cross-lhss reuse.** Because our packing uses a *fixed* `b` bits per site, a site permutation $\pi$ tiles trivially into a bit permutation ($i\cdot b + k \mapsto \pi(i)\cdot b + k$). **One Benes implementation handles spins, bosons, fermions, higher-spin alike** — the original C++ QuSpin needed separate permutation code per local Hilbert space.

---

# Design pattern: `StateTransitions` — the connectivity contract

The basis crate does not import the operator crate. It asks for **anyone who can answer "what states does this connect to?"**:

```rust
// quspin-types/src/state_transitions.rs
pub trait StateTransitions: Send + Sync {
    fn lhss(&self) -> usize;
    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, visit: F);
    //                                ↑ callback: visit(amplitude, new_state)
}
```

Every operator type (`HardcoreOperator`, `BondOperator`, …) implements `StateTransitions`; BFS takes `&impl StateTransitions`.

Two patterns at once: **dependency inversion** (basis defines the contract, operator crate satisfies it) and the **visitor pattern in callback form** (closure inlines into the caller's hot loop, no intermediate `Vec`).

---

# Job (b): operators without a basis

An *operator* in QuSpin-rust is a description of terms in `H`. It does **not know** what basis it will be applied on.

```rust
// crates/quspin-operator/
pub struct HardcoreOperator<C> { /* Pauli strings on lhss=2 (a.k.a.    */ }
//                                  PauliOperator in Python bindings)
pub struct SpinOperator<C>     { /* spin-S operator strings            */ }
pub struct BondOperator<C>     { /* (op_id, bond_sites, amp) tuples    */ }
pub struct MonomialOperator<C> { /* permutation × amplitude table      */ }
pub struct BosonOperator<C>    { /* b†, b, n products                  */ }
pub struct FermionOperator<C>  { /* c†, c, n products + signs          */ }
```

Why basis-agnostic? Because the same `H` is reused across many bases — full space, $S^z$ sector, $S^z$ + translation, $S^z$ + translation + parity. We write `H` **once**, apply it on whichever basis we need.

Heisenberg XXZ → one `BondOperator` carrying the SxSx + SySy + ΔSzSz terms indexed by bonds `(0,1), (1,2), …, (L-1,0)`.

---

# The `Operator<C>` trait

All operator types implement one trait:

```rust
// quspin-operator/src/lib.rs
pub trait Operator<C> {                  // C = term-index width (u8 / u16)
    fn max_site(&self)     -> usize;
    fn lhss(&self)         -> usize;
    fn num_cindices(&self) -> usize;     // ← number of terms in H
    fn apply<B: BitInt, F>(&self, state: B, emit: F)
    where F: FnMut(C, Complex<f64>, B);  // emit(α, amp, σ') per non-zero term
}
```

**`cindex` indexes the terms of** $H = \sum_\alpha c_\alpha \hat O_\alpha$. One `apply(σ, emit)` call walks every term, firing the callback per non-zero contribution with $(\alpha,\ c_\alpha\langle\sigma'|\hat O_\alpha|\sigma\rangle,\ \sigma')$.

The `C` type parameter is the **only** runtime dispatch in the codebase: few-term operators use `cindex: u8`, larger ones `u16`. The enum `*OperatorInner { Ham8(…), Ham16(…) }` is a **tagged union** — chosen once at construction, then `match`-dispatched (one predictable branch, no `dyn`). Everything else is static.

---

# Why `StateTransitions` is free

`Operator::apply` reports the term index $\alpha$; `StateTransitions::neighbors` doesn't. Same callback shape otherwise:

```rust
apply     (state, emit:  FnMut(C, amp, new_state))   // ← knows α
neighbors (state, visit: FnMut(   amp, new_state))   //   doesn't
```

So `neighbors` is `apply` with a closure that drops the `cindex`. Same body for every operator type — written **once** as a macro (`quspin-operator/src/state_transitions.rs`) and expanded per type:

```rust
impl<C> StateTransitions for SpinOperator<C> {           // ← via macro
    fn neighbors<B, F: FnMut(Complex<f64>, B)>(&self, state: B, mut visit: F) {
        self.apply(state, |_c, amp, ns| visit(amp, ns));
    }
}
// macro repeats for Bond / Boson / Fermion / Hardcore / Monomial
```

The basis layer consumes any of them via `&impl StateTransitions` — without ever importing `quspin-operator`.

---

# Job (c): the basis problem

Heisenberg XXZ on L=8:

| Sector                                  | dimension |
|-----------------------------------------|-----------|
| Full Hilbert space                      | 256       |
| $S^z = 0$ sector                        | 70        |
| $S^z = 0$, $k=0$ (translation)          | ~11       |
| $S^z = 0$, $k=0$, parity even           | ~6        |

The library's job: **enumerate only the states in the sector we asked for, indexed in a way we can look up `H|ψ⟩` cheaply.**

This is exactly what `quspin-basis` does, in two stages:

1. **BFS from seeds** to discover the connected sector under `H`
2. **Orbit reduction** under the symmetry group to collapse equivalent states into one representative + a normalization

---

# BFS from seeds

Algorithm — given a `StateTransitions` `H` and a seed state $|s\rangle$:

```
queue   ← [s]
visited ← {s}
while queue not empty:
    σ ← queue.pop()
    for each (amplitude, σ') in H.neighbors(σ):
        if |amplitude| > tol and σ' not in visited:
            visited.add(σ')
            queue.push(σ')
return sort(visited)              ← this is the basis
```

That's it. For Heisenberg with seed `|00001111⟩` (S^z = 0), BFS reaches exactly the 70 S^z=0 states, in some order, and we sort them so we can binary-search for indices later.

Implemented once in `crates/quspin-basis/src/bfs.rs`; **every** basis type calls it.

---

# `Space` — the storage primitive

A basis exposes two operations: `state_at(i)` (row index → state) and `index(state)` (state → row index, or `None` if it left the sector).

```rust
// quspin-basis/src/space.rs + traits.rs (simplified)
pub struct FullSpace<B: BitInt> { /* no states stored; computed on the fly */ }
pub struct Subspace<B: BitInt>  { states: Vec<B>, /* + HashMap index */ }

impl<B: BitInt> Subspace<B> {
    pub fn build<G: StateTransitions>(&mut self, seed: B, graph: &G) -> Result<...>;
}

pub trait BasisSpace<B: BitInt> {
    fn state_at(&self, i: usize) -> B;
    fn index(&self, state: B) -> Option<usize>;
}
```

- `FullSpace`: lhss=2 → state integer **is** the dense index; lhss=3,4 → `DitManip` arithmetic. O(1), zero storage.
- `Subspace`: BFS-discovered states sorted in a `Vec<B>` with a hash map for O(1) average-case lookup.

`index(σ')` is the bridge the matrix layer uses for the column lookup during assembly.

---

# Symmetries: from states to orbits

When we add a symmetry group $G$, abstract basis vectors become orbits:

$$
|[\sigma]\rangle \;\propto\; \frac{1}{\sqrt{|G_\sigma|}} \sum_{g \in G} \chi(g)\, g|\sigma\rangle
$$

We store **one representative state $\sigma$ per orbit**, plus a tiny integer encoding the orbit's normalization. The matrix layer later multiplies amplitudes by the appropriate character ratios as it walks neighbours.

For Heisenberg with $\mathbb{Z}_8$ translation: 70 states in $S^z=0$ break into ~11 orbits of size up to 8. The basis stores 11 representatives.

The interesting engineering: **how do we apply $g \in G$ fast**, and **how do we make sure the user didn't give us a bogus group**?

---

# Design pattern: `SymElement<L>` — three shapes, three storage vectors

A group element on a lattice basis can be one of three shapes; encoded as **two `Option`s**, with constructors enforcing the three legal combos:

```rust
// quspin-basis/src/sym_element.rs
pub struct SymElement<L> { perm: Option<Vec<usize>>, local: Option<L> }

impl<L> SymElement<L> {
    pub fn lattice(perm: &[usize]) -> Self;   // (Some, None)  — pure permutation
    pub fn local(op: L)            -> Self;   // (None, Some)  — on-site unitary
    pub fn composite(p: &[usize], op: L) -> Self;  // (Some, Some) — both
}
```

`SymBasis` then routes each element into **one of three typed vectors** (`lattice_only`, `local_only`, `composite` — `crates/quspin-basis/src/sym.rs:113`). The hot loop walks each vector in turn — no per-element `match`, no virtual call. Pure-lattice symmetries skip the local-op multiply entirely.

**Make illegal states unrepresentable** + **devirtualization via typed dispatch.**

---

# `SymBasis::build` — validate first, BFS second

A symmetry group with a broken closure or a wrong character will produce silently-wrong physics. We refuse to start BFS until we've checked:

```rust
// quspin-basis/src/sym.rs (sketch)
impl<B, L, N> SymBasis<B, L, N> {
    pub fn build<G: StateTransitions>(&mut self, seed: B, graph: &G)
        -> Result<(), QuSpinError>
    {
        if !self.built { self.validate_group()?; }  // ← one-shot probe check
        self.built = true;                          //   (closure + duplicate +
                                                    //    character consistency,
        // ... BFS using graph: StateTransitions    //    O(|G|² · probes))
    }
}
```

Build-time error → impossible to feed a bad group into the matrix assembly. The cost is one $|G|^2 \cdot k$ pass; the payoff is no silent zero eigenvalues from a mis-specified momentum.

---

# Building Heisenberg's symmetric sector

Putting it all together — the Python and the call graph:

```python
group = SymmetryGroup(n_sites=8, lhss=2)
group.add_cyclic(Lattice(translation_by_1), k=0)        # ← SymElement::lattice
basis = SpinBasis.symmetric(group, H_terms, seeds=["00001111"])
```

What happens underneath:

```
SpinBasis.symmetric
  └─→ SymBasis::build
        ├─ validate_closure / validate_characters
        └─ bfs::run( seeds, H_terms : StateTransitions )
              └─ for each visited σ:
                   compute orbit representative via SymElement vectors
                   store (rep, norm)  if rep not seen
```

Result: a `SymBasis<u64, SpinLocal, u8>` with ~11 orbits, ready to be fed to `QMatrix::build_*` in the next slide.

---

# Why the orbit hot loop matters

The orbit lookup — "which $g \in G$ maps this state to its representative, and with what character ratio?" — is the hot loop **for computing matrix elements**. It runs in exactly two places:

1. **Basis construction (BFS)** — once per emitted neighbour during build.
2. **Matrix-free matvec** (`OperatorDispatch::apply_and_project_to`) — once per emitted neighbour per matvec.

For path (2) the orbit walk is the inner loop of every Lanczos step, so its constant factor sets the wall-clock cost of large dynamics runs.

**Once a `QMatrix` is materialised, orbit work is baked into the CSR entries.** Lanczos / Krylov on `QMatrix` see only sparse matvec — no symmetry code touched on the hot path. The basis cost is amortised at build time.

The three-vector `SymElement` split is what keeps paths (1) and (2) fast: **lattice-only** → Benes forward pass; **local-only** → `DitManip`; **composite** → both. No `match` inside the loop, vectorisable by the compiler. **This is why the type design pays off.**

---

# Act 3 — Wrapping it up: Matrix, dynamics, contributing

We are out of the hard part. Six slides on how the basis feeds the rest of the library, plus how to find your way around when you start contributing.

- QMatrix: assembling H on a basis
- expm / krylov: consuming `LinearOperator`
- core + py: facade and bindings
- dev workflow
- which crate owns which kind of change
- how to start

---

# Job (d): assembling H — `QMatrix`

With a basis + an operator, build a CSR-style sparse matrix once (`crates/quspin-matrix/src/qmatrix/build.rs`):

```rust
for row in 0..basis.size() {                     // ← rayon-parallelised
    let σ = basis.state_at(row);
    ham.apply(σ, |cindex, amp, σ_new| {          // Operator::apply, not neighbors —
        if let Some(col) = basis.index(σ_new) {  //   we keep cindex per entry
            entries.push(Entry { value: amp, col, cindex });
        }
    });
}
```

```rust
pub struct QMatrix<M, I, C> { indptr: Vec<I>, data: Vec<Entry<M, I, C>> }
// M = scalar (f64 / c64 …)   I = row/col idx (i32 / i64)   C = cindex (u8 / u16)
```

**Why keep `cindex` on every nonzero?** A time-dependent $H(t) = \sum_\alpha c_\alpha(t)\hat O_\alpha$ then *re-uses the same `QMatrix`* at every time step, just re-weighting by the `c_\alpha(t)` vector at matvec time. No rebuild per step.

A parallel matrix-free path (`OperatorDispatch::apply_and_project_to`, in `quspin-matrix/src/dispatch.rs`) computes $y = Hx$ directly from operator + basis when materialising `QMatrix` is unnecessary.

---

# Job (e): consumers via `LinearOperator`

`quspin-expm` and `quspin-krylov` know nothing about bases, operators, or bits. They consume one trait:

```rust
// quspin-types/src/linear_operator/
pub trait LinearOperator<V: ExpmComputation>: Send + Sync {
    fn dim(&self) -> usize;
    fn trace(&self) -> V;
    fn onenorm(&self, shift: V) -> V::Real;
    fn dot(&self, overwrite: bool, input: &[V], output: &mut [V])
        -> Result<(), QuSpinError>;                  // output (±)= H · input
    // + dot_transpose / dot_many / dot_chunk / parallel_hint
}
```

Both `QMatrix` and `Hamiltonian` (time-dependent) implement it. So does any user-supplied closure (`FnLinearOperator`).

This means: **a future contributor adding, say, an MPS-backed operator gets Krylov and expm for free** — they only have to satisfy `LinearOperator`. The solvers don't need to be rewritten.

This is **type erasure**, but bounded: it only happens at the top of the DAG, where the cost of one virtual call per matvec is negligible.

---

# `quspin-core` and `quspin-py`

```
quspin-core   ← pure re-export facade, ~90 lines, zero logic.
                NEVER add code here. Add to the focused crate that owns
                the domain.

quspin-py     ← PyO3 bindings, builds the `_rs` Python extension.
                Maps QuSpinError → Python exceptions, ruint → bit-strings,
                wraps every public type with a #[pyclass].
```

**Project rule (CLAUDE.md):** any feature that lands in the Rust core **must** include the matching PyO3 binding and update `python/quspin_rs/_rs.pyi` in the **same PR**. We don't merge half-features.

If you add a new operator type to `quspin-operator`, the PR also touches `crates/quspin-py/src/operator/` and `_rs.pyi`. This stays simple to follow because every existing type already does it the same way.

---

# Dev workflow in 60 seconds

Requirements: Rust stable, Python ≥ 3.10, [uv](https://github.com/astral-sh/uv), [just](https://github.com/casey/just).

```sh
just sync             # uv sync --dev --all-extras  (Python deps)
just check            # cargo check                 (type-check, fast)
just develop          # uv run maturin develop      (build & install _rs)
just test-rust        # cargo test -p quspin-core
just test-python      # uv run --locked pytest python/tests/ -v
just test             # test-rust + test-python
just test-python-fast # skip @pytest.mark.slow tests

cargo test  -p quspin-basis       # one crate at a time — much faster
cargo check -p quspin-bitbasis    # ditto for type-checking
pre-commit run --all-files        # fmt + clippy + black + ruff + pyright
```

CI on every PR runs the full matrix. Pre-commit hooks catch fmt/clippy issues before you push.

---

# Where to contribute, by job

| If your change is about…                  | …it lives in        |
|-------------------------------------------|---------------------|
| Bit encoding, new integer width, Benes    | `quspin-bitbasis`   |
| A new operator term (e.g. four-body)      | `quspin-operator`   |
| A new symmetry shape, BFS variant         | `quspin-basis`      |
| Sparsity layout, parallel matrix fill    | `quspin-matrix`     |
| A new eigensolver, time-evolver           | `quspin-krylov`     |
| A new Taylor-series strategy for `expm`   | `quspin-expm`       |
| A workspace-wide trait, dtype, error      | `quspin-types`      |
| Python surface, type stubs                | `quspin-py` + `.pyi`|
| Documentation, design docs                | `docs/`             |

If you don't know which crate, **ask in the PR** — the boundaries are worth pushing back on if a change feels like it doesn't fit.

---

# How to actually start

A first contribution looks like this:

1. `just sync && just develop` — get a working build
2. Pick a crate, read its `lib.rs` top-to-bottom (it's the API)
3. Find an existing test in `crates/<crate>/src/...` or `tests/`
4. **Write the failing test first** — it pins down what success means
5. Implement until it passes; run `cargo test -p <crate>` locally
6. If the surface changes: update `crates/quspin-py/` and `_rs.pyi`
7. `pre-commit run --all-files`, then open a PR

Reference reading, in order of usefulness:

- `CLAUDE.md` — the architectural map (this deck is mostly an unpacking of it)
- `docs/superpowers/specs/2026-04-18-crate-split-design.md` — why the crate split
- `docs/superpowers/specs/2026-04-26-python-symmetry-group-api-design.md`
- Recent PRs on `main` — current style for new features

---

# Thank you

## Questions?

Three things to take away:

1. **The crate boundaries follow the five jobs.** When you have a change in mind, ask "which job is this?" — the answer names the crate.

2. **Traits are how the layers communicate without coupling.** `BitInt`, `StateTransitions`, `Operator<C>`, `LinearOperator` — each defines a contract that one crate produces and another consumes.

3. **The basis layer is where the physics lives.** BFS + symmetry orbits + the typed `SymElement` split are what make QuSpin-rust faster than a textbook ED loop. Read that code first.
