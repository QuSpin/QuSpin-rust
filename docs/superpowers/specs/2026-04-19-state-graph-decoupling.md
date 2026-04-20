# StateTransitions: Decouple `quspin-basis` from `quspin-operator`

**Date:** 2026-04-19
**Status:** Approved (implemented in PR #57)
**Relates to:** Issue #48 follow-up (spec §9 of `2026-04-18-crate-split-design.md`)

> **Note on naming:** this spec was drafted under the working name
> `StateGraph`. The trait was renamed to `StateTransitions` during
> review (see §8 / review comment #1) because the final contract
> requires amplitudes, not just connectivity. Every reference in the
> rest of this doc uses `StateTransitions`; the old name survives here
> in the title for stable linking from earlier PR discussion.

---

## 1. Motivation

After the 7-crate split (PR #52), `quspin-basis` still depends on `quspin-operator` because the basis types' `build_*` methods accept concrete `*OperatorInner` enums and match on their `Ham8` / `Ham16` variants to extract an apply-callback. This is the only coupling left between the two and prevents the parallel compilation that the spec's §3.1 DAG promised.

The basis BFS needs one-step successor enumeration — "given state `S`, which states can I reach by applying the operator once?" — plus the emitted amplitude for each successor, so that sector membership is decided correctly when multiple terms contribute to the same target (e.g. `XX + YY` applied to `|00⟩` produces two contributions whose amplitudes exactly cancel). The basis does not need to know the concrete operator type or the cindex tag; only the `(amplitude, new_state)` stream matters. The operator types already know how to enumerate neighbours with this signature; the basis types do not need to know which operator type they were handed.

This spec introduces a minimal trait — `StateTransitions` — that lets the basis BFS take `&impl StateTransitions` instead of a concrete operator type. The result:

- `quspin-basis` drops its runtime `quspin-operator` dep; they compile in parallel.
- Eight typed `build_*` methods collapse into one generic `build<G: StateTransitions>` per basis (four total).
- `quspin-py` updates five one-line call sites to `.build(&op.inner, seeds)`; the Python-facing method names stay unchanged.

**Primary goal:** complete the §3.1 parallel-compile DAG.
**Secondary goal:** eliminate the `SmallVec<(amp, state, cindex)>` allocation BFS currently performs per frontier node.
**Non-goal:** any user-observable behaviour change.

---

## 2. Current Structure

### 2.1 Coupling

Every basis `build_*` method in `crates/quspin-basis/src/{spin,boson,fermion,generic}.rs` takes a concrete `*OperatorInner` and matches on it:

```rust
// quspin-basis/src/spin.rs
pub fn build_spin(
    &mut self,
    ham: &SpinOperatorInner,       // ← operator-specific type
    seeds: &[Vec<u8>],
) -> Result<(), QuSpinError> {
    // … validation …
    match ham {
        SpinOperatorInner::Ham8(h)  => subspace.build(s, |state| h.apply_smallvec(state).into_iter()),
        SpinOperatorInner::Ham16(h) => subspace.build(s, |state| h.apply_smallvec(state).into_iter()),
    }
}
```

The three other basis types follow the same pattern, for a total of eight methods:

| Basis | Method | Operator argument |
|-------|--------|-------------------|
| `SpinBasis` | `build_spin` | `&SpinOperatorInner` |
| `SpinBasis` | `build_hardcore` | `&HardcoreOperatorInner` |
| `SpinBasis` | `build_bond` | `&BondOperatorInner` |
| `BosonBasis` | `build_boson` | `&BosonOperatorInner` |
| `BosonBasis` | `build_bond` | `&BondOperatorInner` |
| `FermionBasis` | `build_fermion` | `&FermionOperatorInner` |
| `FermionBasis` | `build_bond` | `&BondOperatorInner` |
| `GenericBasis` | `build_monomial` | `&MonomialOperatorInner` |

### 2.2 Wasted allocation

The BFS callback returns an iterator of `(Complex<f64>, B, u8)` triples — amplitude, new state, cindex — but BFS discards everything except the new state. `apply_smallvec` materialises a `SmallVec` per frontier node only for the basis to strip 2/3 of each tuple. Since `Operator::apply` already offers a callback-based iteration, the intermediate collection is pure overhead.

---

## 3. Target Design

### 3.1 `StateTransitions` trait

**Location:** `quspin-types`. Reachable as either `quspin_types::StateTransitions` (canonical) or `quspin_bitbasis::StateTransitions` (re-export, kept for import-path stability). The `BitInt` trait and its three `impl` blocks also live in `quspin-types`; the orphan rule forces the `impl BitInt for Uint<N, LIMBS>` block to be in the same crate as the trait definition, and `BitInt` has no bit-manipulation behaviour of its own — `quspin-bitbasis` owns the Benes network, dit manipulation, and permutation helpers that are genuinely bit-level.

```rust
// crates/quspin-bitbasis/src/state_transitions.rs
use num_complex::Complex;

use crate::int::BitInt;

/// State-to-neighbour mapping used by basis BFS, with amplitudes.
///
/// The callback `visit(amplitude, new_state)` is invoked once for every
/// non-zero term produced by applying `self` to `state`. Amplitudes are
/// a required part of the contract: the basis accumulates per-target
/// contributions and discards states whose summed amplitudes cancel
/// below tolerance (e.g. `XX + YY` on `|00⟩` cancels exactly and must
/// not enter the sector). Cindex tags are not part of the contract.
pub trait StateTransitions: Send + Sync {
    /// Local Hilbert space size this operator acts on.
    ///
    /// The basis checks `transitions.lhss() == basis.lhss()` before building.
    fn lhss(&self) -> usize;

    /// Call `visit(amplitude, new_state)` once for every non-zero term
    /// produced by applying `self` to `state`.
    ///
    /// Duplicate targets are permitted — BFS accumulates their amplitudes
    /// and detects cancellation.
    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, visit: F);
}
```

The trait has no associated types and no trait-level type parameters — `B` is a method-level generic — so it is not `dyn`-compatible. That is fine; BFS uses static dispatch.

### 3.2 `Operator::lhss`

`lhss()` is currently defined only on the `*OperatorInner` dispatch enums. Promote it to the per-cindex `Operator<C>` trait in `quspin-operator`:

```rust
pub trait Operator<C> {
    fn max_site(&self) -> usize;
    fn num_cindices(&self) -> usize;
    fn lhss(&self) -> usize;                     // ← new
    fn apply<B: BitInt, F>(&self, state: B, emit: F)
    where F: FnMut(C, Complex<f64>, B);
}
```

This makes the generic operator types (`SpinOperator<u8>`, `BondOperator<u16>`, etc.) self-sufficient — they can implement `StateTransitions` directly without going through a dispatch enum. Callers that want to test `build` with a single-cindex-width operator (no `*Inner` wrapper) can do so.

### 3.3 `StateTransitions` impls

Two layers of impl:

**Per-cindex generic types** (in the respective operator modules):

```rust
impl<C: Copy + Ord + Send + Sync> StateTransitions for SpinOperator<C> {
    fn lhss(&self) -> usize { Operator::<C>::lhss(self) }
    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, mut visit: F) {
        self.apply::<B, _>(state, |_c, amp, ns| visit(amp, ns));
    }
}
```

One each for `SpinOperator<C>`, `BondOperator<C>`, `BosonOperator<C>`, `FermionOperator<C>`, `HardcoreOperator<C>`, `MonomialOperator<C>`.

**Dispatch enums** (delegate to the variant):

```rust
impl StateTransitions for SpinOperatorInner {
    fn lhss(&self) -> usize {
        match self {
            Self::Ham8(h)  => StateTransitions::lhss(h),
            Self::Ham16(h) => StateTransitions::lhss(h),
        }
    }
    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, visit: F) {
        match self {
            Self::Ham8(h)  => h.neighbors(state, visit),
            Self::Ham16(h) => h.neighbors(state, visit),
        }
    }
}
```

`HardcoreOperatorInner::lhss` stays `const fn` on the enum as a no-trait-bound inherent method; the trait impl delegates.

### 3.4 Basis method collapse

Each basis type gets exactly one generic `build` method. The eight typed `build_*` methods are deleted outright.

```rust
// quspin-basis/src/spin.rs
impl SpinBasis {
    /// Build the subspace reachable from `seeds` under the connectivity
    /// described by `graph`.
    pub fn build<G: StateTransitions>(&mut self, graph: &G, seeds: &[Vec<u8>]) -> Result<(), QuSpinError> {
        if self.inner.space_kind() == SpaceKind::Full {
            return Err(QuSpinError::ValueError("Full basis requires no build step".into()));
        }
        if self.inner.is_built() {
            return Err(QuSpinError::ValueError("basis is already built".into()));
        }
        let lhss = self.inner.lhss();
        if graph.lhss() != lhss {
            return Err(QuSpinError::ValueError(format!(
                "graph.lhss()={} does not match basis lhss={}", graph.lhss(), lhss
            )));
        }
        // same space_kind dispatch as today; each arm calls
        //     subspace.build(seed, graph)
        // where `Subspace::build` takes `&impl StateTransitions` directly
        // (see §3.5).
    }
}
```

`BosonBasis`, `FermionBasis`, `GenericBasis` follow the same pattern.

`quspin-py`'s five basis files (`crates/quspin-py/src/basis/{spin,boson,fermion,generic}.rs`) each change one line per call site: `basis.build_spin(&op.inner, seeds)` → `basis.build(&op.inner, seeds)`. The Python-facing method names (`build_spin`, `build_bond`, …) are unchanged — they live as `#[pyo3(name = …)]` annotations inside `quspin-py`, not in the basis crate.

**Why not keep typed aliases for backwards compatibility?** Forwarding aliases would be performance-equivalent at runtime (Rust inlines them), but they force `quspin-basis` to keep `*OperatorInner` in its public signatures, which in turn keeps the `quspin-basis → quspin-operator` edge in `cargo tree`. Dropping the aliases is strictly better for compile-time parallelism (the whole motivation of the refactor) at the cost of five trivial call-site edits in `quspin-py`.

### 3.5 BFS signature change

`Subspace::build` / `SymBasis::build` in `quspin-basis` previously took:

```rust
pub fn build<F, I>(&mut self, seed: B, apply: F)
where F: Fn(B) -> I, I: IntoIterator<Item = (Complex<f64>, B, u8)>
```

which forced `apply(state)` to materialise an iterator (in practice a `SmallVec<(Complex<f64>, B, u8); 8>`) per frontier state.

The implemented replacement takes the trait directly:

```rust
pub fn build<G: StateTransitions>(&mut self, seed: B, graph: &G)
```

The BFS hot loop calls `graph.neighbors::<B, _>(state, |amp, next_state| { /* accumulate into contributions map */ })`. No intermediate collection, no iterator adaptation — the closure writes straight into the per-wave `HashMap<B, (Complex<f64>, f64)>` that was already used for amplitude cancellation.

> **Design note.** An earlier draft of this section proposed an intermediate visitor-style signature (`FnMut(B, &mut dyn FnMut(B))`) with an optional `StateEmitter<B>` fallback. That indirection turned out to be unnecessary once the amplitude argument was added to the `StateTransitions` callback — the trait directly is both simpler and monomorphises better. The visitor form was never implemented.

### 3.6 Dependency graph

Before:
```
quspin-bitbasis
     │
     ├──────────────┐
quspin-operator  quspin-expm, quspin-krylov
     │
quspin-basis                              ← depends on quspin-operator
     │
quspin-matrix
```

After:
```
quspin-bitbasis
     │
     ├─────────────┬────────────┬──────────────┐
quspin-operator quspin-basis  quspin-expm   quspin-krylov   ← four-way parallel
     └──────┬──────┘
       quspin-matrix
```

`quspin-basis/Cargo.toml` removes `quspin-operator` from `[dependencies]` entirely and moves it to `[dev-dependencies]` — tests still use real operator types to drive `build`, but the production crate no longer references any operator symbols. `cargo tree -p quspin-basis --edges=normal | grep quspin-operator` returns empty.

### 3.7 Correctness note: fermion flag

Fermionic sign handling lives in `SymBasis` orbit/representative logic and uses the `fermionic: bool` on the space constructor. It is **not** applied during BFS connectivity expansion and does **not** depend on the operator type. `StateTransitions` therefore correctly omits any fermion flag; pairing a `FermionOperator` with a `SpinBasis` is a user error by convention, not by type check, and the `lhss` check catches the common case (LHSS mismatch).

---

## 4. Migration Strategy

Single PR. The three mechanical steps are tightly coupled:

1. Add `StateTransitions` + `Operator::lhss`, impl across operator crate.
2. Refactor basis BFS (§3.5) and collapse `build_*` methods to aliases (§3.4).
3. Drop `quspin-basis → quspin-operator` from runtime deps; move to dev-deps.

Splitting 1 and 2 would leave `quspin-basis` in a transient state where it imports a trait it doesn't use, which complicates review more than it helps.

### Verification per step

- After step 1: `cargo check -p quspin-operator`, no changes to basis.
- After step 2: `cargo test --workspace`; all existing `build_*` tests still pass via the aliases.
- After step 3: `cargo tree -p quspin-basis | grep quspin-operator` returns nothing; `cargo tree -p quspin-operator | grep quspin-basis` returns nothing.
- End-to-end: `uv run pytest python/tests/ -m "not slow"` — 113 pass.

---

## 5. Testing & CI

- Existing Rust tests in `quspin-basis/src/{spin,boson,fermion,generic}.rs` continue to use real operator types and exercise the `build_*` aliases. They keep their existing shape; `quspin-operator` moves to `[dev-dependencies]` to support them.
- Python tests are untouched and remain the end-to-end backstop.
- Add no new tests in this PR — the refactor is a pure type-level change with existing coverage already exhaustive at the call sites.

### CI isolation check

Add to `.github/workflows/ci.yaml`:

```yaml
- name: Verify operator / basis compile independence
  run: |
    cargo check -p quspin-operator
    cargo check -p quspin-basis
    ! cargo tree -p quspin-operator 2>/dev/null | grep -q quspin-basis
    ! cargo tree -p quspin-basis 2>/dev/null | grep -q "quspin-operator "
```

The trailing space in `"quspin-operator "` excludes matches against `quspin-operator` listed as a dev-dependency (cargo tree formats `quspin-operator v0.1.0 (dev)` differently from runtime deps).

---

## 6. Documentation updates

- `CLAUDE.md`: update the DAG diagram so `quspin-operator` and `quspin-basis` sit at the same level (both children of `quspin-bitbasis`). Add one sentence noting `StateTransitions` is the abstraction layer.
- `docs/superpowers/specs/2026-04-18-crate-split-design.md`: mark §9 as completed, link to this spec.

---

## 7. Risks & fallbacks

| Risk | Mitigation |
|------|------------|
| `cargo tree` check false-positives on dev-deps | Use `cargo tree --edges=normal`, which excludes dev-deps. The CI step also runs `cargo tree` as a separate command (not piped into `grep 2>/dev/null`) so a tool failure surfaces as a CI failure. |
| A future operator type wants to emit something richer than `(amp, state)` during BFS | That caller can fall back to `Operator::apply` directly; `StateTransitions` is additive. |

---

## 8. Out of scope / known follow-ups

- Collapsing the four basis types (`SpinBasis`, `BosonBasis`, `FermionBasis`, `GenericBasis`) into one. Requires restructuring the symmetry-group API, which the user wants to address as a separate subsequent refactor.
- Any change to `quspin-matrix`, `quspin-expm`, or `quspin-krylov`.

### Open tension: amplitude is a basis-layer concern (review #2)

The `StateTransitions` trait lives in `quspin-bitbasis` but carries a `Complex<f64>` amplitude in its callback because the basis needs it for symbolic-cancellation detection (`XX + YY` on `|00⟩`). This is a layering concession: a hypothetical non-physics consumer of `StateTransitions` that only needs graph connectivity is forced to fabricate unit amplitudes.

Two cleaner designs exist but are out of scope here:

1. **Split the trait.** Expose a narrow `Connectivity` trait (no amplitude) for consumers that only care about reachable neighbours, and keep the wider `StateTransitions` for BFS-with-cancellation.
2. **Push cancellation to the operator side.** Let each operator expose a `canceled_neighbors` method that internally deduplicates targets whose amplitudes sum to zero, so the basis only sees unique connected targets.

Neither is needed for the current QuSpin use case; both are tracked for a future refactor once a second consumer appears.

### Open tension: compile-time `lhss` enforcement (review #9)

`graph.lhss() == self.inner.lhss()` is currently a runtime string-error check inside `build_inner`. Callers that pass a spin-½ operator to a spin-1 basis only find out at `build` time. A const-generic or type-level encoding of `lhss` would catch this at compile time but is a larger refactor — tracked as a follow-up after the symmetry-group API restructure lands.

### Performance benchmark (review #7)

The refactor eliminates the per-frontier `SmallVec<(amp, state, cindex)>` allocation by writing `StateTransitions::neighbors` output directly into the BFS contribution hashmap. The performance impact has **not been measured**. A follow-up task is to add a `criterion` bench for `Subspace::build` with representative Hamiltonians and land the before/after numbers in a separate PR. Until that lands, the PR descriptions and CHANGELOG describe the change as "allocation savings unmeasured" rather than asserting a speedup.
