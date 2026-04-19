# StateGraph: Decouple `quspin-basis` from `quspin-operator`

**Date:** 2026-04-19
**Status:** Approved
**Relates to:** Issue #48 follow-up (spec §9 of `2026-04-18-crate-split-design.md`)

---

## 1. Motivation

After the 7-crate split (PR #52), `quspin-basis` still depends on `quspin-operator` because the basis types' `build_*` methods accept concrete `*OperatorInner` enums and match on their `Ham8` / `Ham16` variants to extract an apply-callback. This is the only coupling left between the two and prevents the parallel compilation that the spec's §3.1 DAG promised.

The basis BFS only needs **connectivity** — it asks "given state `S`, which states can I reach by applying the operator once?" It does not use the amplitude or the cindex tag. The operator types already know how to enumerate neighbours; the basis types do not need to know which operator type they were handed.

This spec introduces a minimal trait — `StateGraph` — that lets the basis BFS take `&impl StateGraph` instead of a concrete operator type. The result:

- `quspin-basis` drops its `quspin-operator` dep; they compile in parallel.
- Eight typed `build_*` methods collapse into one generic `build<G: StateGraph>` per basis (four total).
- Existing `build_*` names survive as thin forwarding aliases so `quspin-py` compiles unchanged.

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

### 3.1 `StateGraph` trait

**Location:** `quspin-bitbasis` (the lowest crate that knows `BitInt`).

```rust
// crates/quspin-bitbasis/src/state_graph.rs
use crate::int::BitInt;

/// Abstract connectivity oracle used by basis BFS.
///
/// Operators that implement this trait describe how a state maps to its
/// reachable neighbours under one application. BFS uses the set of
/// neighbours only; amplitudes and cindex tags are irrelevant for basis
/// enumeration.
pub trait StateGraph: Send + Sync {
    /// Local Hilbert space size this operator acts on.
    ///
    /// The basis checks `graph.lhss() == self.lhss()` before building.
    fn lhss(&self) -> usize;

    /// Call `visit(new_state)` once for each state reachable from `state`
    /// under one application of `self`.
    ///
    /// Duplicates are permitted — BFS deduplicates via its own hash set.
    fn neighbors<B: BitInt, F: FnMut(B)>(&self, state: B, visit: F);
}
```

The trait has no associated types and no type parameters — `B` is a method-level generic — so it is not `dyn`-compatible. That is fine; BFS uses static dispatch.

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

This makes the generic operator types (`SpinOperator<u8>`, `BondOperator<u16>`, etc.) self-sufficient — they can implement `StateGraph` directly without going through a dispatch enum. Callers that want to test `build` with a single-cindex-width operator (no `*Inner` wrapper) can do so.

### 3.3 `StateGraph` impls

Two layers of impl:

**Per-cindex generic types** (in the respective operator modules):

```rust
impl<C: CIndex> StateGraph for SpinOperator<C> {
    fn lhss(&self) -> usize { self.lhss }
    fn neighbors<B: BitInt, F: FnMut(B)>(&self, state: B, mut visit: F) {
        self.apply::<B, _>(state, |_c, _amp, ns| visit(ns));
    }
}
```

One each for `SpinOperator<C>`, `BondOperator<C>`, `BosonOperator<C>`, `FermionOperator<C>`, `HardcoreOperator<C>`, `MonomialOperator<C>`.

**Dispatch enums** (delegate to the variant):

```rust
impl StateGraph for SpinOperatorInner {
    fn lhss(&self) -> usize {
        match self { Self::Ham8(h) => h.lhss(), Self::Ham16(h) => h.lhss() }
    }
    fn neighbors<B: BitInt, F: FnMut(B)>(&self, state: B, visit: F) {
        match self {
            Self::Ham8(h)  => h.neighbors(state, visit),
            Self::Ham16(h) => h.neighbors(state, visit),
        }
    }
}
```

`HardcoreOperatorInner::lhss` stays `const fn` on the enum but the trait method delegates.

### 3.4 Basis method collapse

Each basis type gets one generic `build` method and keeps the existing names as inlined forwarders:

```rust
// quspin-basis/src/spin.rs
impl SpinBasis {
    /// Build the subspace reachable from `seeds` under the connectivity
    /// described by `graph`.
    pub fn build<G: StateGraph>(&mut self, graph: &G, seeds: &[Vec<u8>]) -> Result<(), QuSpinError> {
        // (single impl replacing build_spin / build_hardcore / build_bond bodies)
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
        // … same space_kind dispatch as today, but calling
        //    subspace.build(s, |state, visit| graph.neighbors(state, visit))
    }

    #[inline] pub fn build_spin    (&mut self, h: &SpinOperatorInner,     s: &[Vec<u8>]) -> Result<(), QuSpinError> { self.build(h, s) }
    #[inline] pub fn build_hardcore(&mut self, h: &HardcoreOperatorInner, s: &[Vec<u8>]) -> Result<(), QuSpinError> { self.build(h, s) }
    #[inline] pub fn build_bond    (&mut self, h: &BondOperatorInner,     s: &[Vec<u8>]) -> Result<(), QuSpinError> { self.build(h, s) }
}
```

`BosonBasis`, `FermionBasis`, `GenericBasis` follow the same pattern. The aliases are `#[inline]` zero-cost forwarders — `quspin-py` compiles with zero source changes.

### 3.5 BFS visitor refactor

`Subspace::build` / `SymBasis::build` in `quspin-basis` currently take:

```rust
pub fn build<F, I>(&mut self, seed: B, apply: F)
where F: Fn(B) -> I, I: IntoIterator<Item = (Complex<f64>, B, u8)>
```

Replace with a visitor-style callback that receives an emitter:

```rust
pub fn build<F>(&mut self, seed: B, mut expand: F)
where F: FnMut(B, &mut dyn FnMut(B))
```

Frontier loops call `expand(state, &mut |ns| { /* push to next frontier */ })`. Inside the basis `build<G>` method we pass `|state, visit| graph.neighbors(state, visit)`. No `SmallVec` allocation, no iterator adaptation.

If the `&mut dyn FnMut(B)` indirection measures poorly, fall back to a custom trait:

```rust
pub trait StateEmitter<B> { fn emit(&mut self, state: B); }
```

and make `build` take `F: FnMut(B, &mut dyn StateEmitter<B>)` or an impl-trait equivalent. Measure first.

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

`quspin-basis/Cargo.toml` drops the `quspin-operator` entry from `[dependencies]` and adds it under `[dev-dependencies]` so tests that build with real operator types still compile.

### 3.7 Correctness note: fermion flag

Fermionic sign handling lives in `SymBasis` orbit/representative logic and uses the `fermionic: bool` on the space constructor. It is **not** applied during BFS connectivity expansion and does **not** depend on the operator type. `StateGraph` therefore correctly omits any fermion flag; pairing a `FermionOperator` with a `SpinBasis` is a user error by convention, not by type check, and the `lhss` check catches the common case (LHSS mismatch).

---

## 4. Migration Strategy

Single PR. The three mechanical steps are tightly coupled:

1. Add `StateGraph` + `Operator::lhss`, impl across operator crate.
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

- `CLAUDE.md`: update the DAG diagram so `quspin-operator` and `quspin-basis` sit at the same level (both children of `quspin-bitbasis`). Add one sentence noting `StateGraph` is the abstraction layer.
- `docs/superpowers/specs/2026-04-18-crate-split-design.md`: mark §9 as completed, link to this spec.

---

## 7. Risks & fallbacks

| Risk | Mitigation |
|------|------------|
| Visitor signature (`FnMut(B, &mut dyn FnMut(B))`) measures worse than the current iterator form | Fall back to a custom `StateEmitter<B>` trait for monomorphisable calls. Benchmark before committing. |
| `cargo tree` check false-positives on dev-deps | Use the trailing-space pattern in §5 or switch to `cargo tree --edges=normal`. |
| A future operator type wants to emit something richer than just `B` during BFS | That caller can fall back to `Operator::apply` directly; `StateGraph` is additive. |

---

## 8. Out of scope

- Collapsing the four basis types (`SpinBasis`, `BosonBasis`, `FermionBasis`, `GenericBasis`) into one. Requires restructuring the symmetry-group API, which the user wants to address as a separate subsequent refactor.
- Removing the eight typed `build_*` aliases. They are free at runtime; removal is a deprecation-cycle decision, not a technical one.
- Any change to `quspin-matrix`, `quspin-expm`, or `quspin-krylov`.
