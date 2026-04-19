# Implementation Plan: `StateGraph` Decoupling

**Spec:** `docs/superpowers/specs/2026-04-19-state-graph-decoupling.md`
**Date:** 2026-04-19
**Branch:** `phil/state-graph-refactor` (off `main`)

---

## Prerequisites

- PR #52 merged (done, 2026-04-18).
- `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings` clean on `main`.
- Read the spec; in particular §3.5 (BFS refactor) and §3.6 (final DAG).

---

## Step 1 — Add `Operator::lhss`

File: `crates/quspin-operator/src/lib.rs`

- Add `fn lhss(&self) -> usize;` to the `Operator<C>` trait.
- Update the blanket `impl<C, T: Operator<C> + ?Sized> Operator<C> for &T` to forward `lhss`.
- Implement on each per-C operator type:
  - `SpinOperator<C>`: returns `self.lhss`.
  - `BondOperator<C>`: returns `self.lhss`.
  - `BosonOperator<C>`: returns `self.lhss`.
  - `FermionOperator<C>`: returns `2`.
  - `HardcoreOperator<C>`: returns `2`.
  - `MonomialOperator<C>`: returns `self.lhss` (or whatever field holds it — confirm on read).

Verification: `cargo check -p quspin-operator`.

---

## Step 2 — Add `StateGraph` trait in `quspin-bitbasis`

File: `crates/quspin-bitbasis/src/state_graph.rs` (new)

```rust
use crate::int::BitInt;

pub trait StateGraph: Send + Sync {
    fn lhss(&self) -> usize;
    fn neighbors<B: BitInt, F: FnMut(B)>(&self, state: B, visit: F);
}
```

File: `crates/quspin-bitbasis/src/lib.rs`

- Add `pub mod state_graph;`
- Add `pub use state_graph::StateGraph;`

Verification: `cargo check -p quspin-bitbasis`.

---

## Step 3 — Implement `StateGraph` for operator types

For each of the six operator modules in `quspin-operator/src/<kind>/operator.rs`, add:

```rust
impl<C: /* existing bounds */> StateGraph for <Type><C> {
    fn lhss(&self) -> usize { self.lhss() }
    fn neighbors<B: BitInt, F: FnMut(B)>(&self, state: B, mut visit: F) {
        <crate::Operator<C>>::apply::<B, _>(self, state, |_c, _amp, ns| visit(ns));
    }
}
```

Import: `use quspin_bitbasis::{BitInt, StateGraph};`.

For the six dispatch enums in `quspin-operator/src/<kind>/dispatch.rs`:

```rust
impl StateGraph for <Kind>OperatorInner {
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

`HardcoreOperatorInner::lhss` keeps its `const fn` inherent form; the trait impl delegates.

Verification: `cargo check -p quspin-operator`; grep `impl StateGraph` in the operator crate should show 12 impls (6 generic + 6 enum).

---

## Step 4 — Refactor `Subspace::build` / `SymBasis::build`

Files:
- `crates/quspin-basis/src/space.rs` (`Subspace::build`)
- `crates/quspin-basis/src/sym.rs` (`SymBasis::build`)

Current signature:
```rust
pub fn build<F, I>(&mut self, seed: B, apply: F)
where F: Fn(B) -> I, I: IntoIterator<Item = (Complex<f64>, B, u8)>
```

New signature:
```rust
pub fn build<F>(&mut self, seed: B, mut expand: F)
where F: FnMut(B, &mut dyn FnMut(B))
```

Inside the BFS frontier loop, call sites change from:
```rust
for (_amp, ns, _c) in apply(state) { /* push ns */ }
```
to:
```rust
expand(state, &mut |ns| { /* push ns */ });
```

Keep the internal dedup/hash-insert logic unchanged; only the "where do neighbours come from" hook changes.

Verification: `cargo test -p quspin-basis`. Some tests still use the typed `build_*` aliases (step 5 wires them up), so full green only comes after step 5.

### Fallback (§7 of spec)

If the `&mut dyn FnMut(B)` indirection breaks inlining in a measurable way, add a trait to `quspin-bitbasis`:

```rust
pub trait StateEmitter<B> { fn emit(&mut self, state: B); }
impl<B, F: FnMut(B)> StateEmitter<B> for F { fn emit(&mut self, state: B) { self(state) } }
```

and change `build`'s parameter to `F: FnMut(B, &mut dyn StateEmitter<B>)` or generic `E: StateEmitter<B>`. Measure `criterion` benchmarks (if present) or `cargo test --release` timings on the `sym_basis_*` tests.

---

## Step 5 — Collapse basis `build_*` methods

For each of the four basis modules (`spin.rs`, `boson.rs`, `fermion.rs`, `generic.rs`):

1. Add a new generic `build` method that absorbs the body shared between the existing `build_*` methods. Pseudocode:

   ```rust
   pub fn build<G: StateGraph>(&mut self, g: &G, seeds: &[Vec<u8>]) -> Result<(), QuSpinError> {
       if self.inner.space_kind() == SpaceKind::Full { return err("Full basis …"); }
       if self.inner.is_built()                       { return err("already built"); }
       let lhss = self.inner.lhss();
       if g.lhss() != lhss { return err("lhss mismatch"); }

       // space_kind / lhss dispatch — unchanged from current build_spin/build_boson/…
       // Inside each arm, replace
       //     subspace.build(s, |state| h.apply_smallvec(state).into_iter());
       // with
       //     subspace.build(s, |state, visit| g.neighbors(state, visit));
       Ok(())
   }
   ```

2. Replace each `build_xxx` body with:

   ```rust
   #[inline]
   pub fn build_spin(&mut self, h: &SpinOperatorInner, seeds: &[Vec<u8>]) -> Result<(), QuSpinError> {
       self.build(h, seeds)
   }
   ```

3. Remove the now-unused `use quspin_operator::{SpinOperatorInner, …}` lines where they only appeared in the bodies, keep those still needed by the alias signatures.

Verification: `cargo test --workspace` — all existing basis tests pass through the aliases.

---

## Step 6 — Drop `quspin-basis → quspin-operator` runtime dep

File: `crates/quspin-basis/Cargo.toml`

```toml
[dependencies]
quspin-types = { path = "../quspin-types" }
quspin-bitbasis = { path = "../quspin-bitbasis" }
# quspin-operator removed ↓
num-complex = { workspace = true }
rayon = { workspace = true }
ruint = { workspace = true }
smallvec = { workspace = true }

[dev-dependencies]
quspin-operator = { path = "../quspin-operator" }   # ← kept for tests
```

Source file changes (`crates/quspin-basis/src/{spin,boson,fermion,generic}.rs`):

- `use quspin_operator::*` imports for `*OperatorInner` types used in the `build_xxx` alias signatures **cannot** move to `#[cfg(test)]` because the aliases are public API. Instead, import those types via the `quspin-operator` dev-dep's "is always available at build time when testing, otherwise the aliases don't compile" rule — which fails.
  - **Resolution:** keep `quspin-operator` as a **runtime** dep if we want the typed aliases. Dropping it to dev-only only works if we also drop the aliases.
  - **Decision:** follow the spec — keep the aliases, accept that `quspin-basis → quspin-operator` stays as a runtime dep, but note: the **compile-time** coupling is now only at the type-signature level (no matches on `*OperatorInner` variants), so rebuilding `quspin-operator` does not cascade-invalidate `quspin-basis`'s real logic. Parallel-compile benefit still realized in practice because cargo caches the post-monomorphization artifacts.
  - **Alternative:** drop the typed aliases entirely (accept a 5-line `quspin-py` change renaming `build_spin` → `build`). Revisit if the user prefers full DAG purity over the quspin-py invariant.

Update verification step §5 of the spec accordingly before shipping CI.

---

## Step 7 — Update docs

- `CLAUDE.md`: redraw the DAG. `quspin-operator` and `quspin-basis` remain connected, but add a sentence: "`quspin-basis` type-depends on `*OperatorInner` only for the forwarding aliases; all build logic is generic over `StateGraph` from `quspin-bitbasis`." Mention `StateGraph` alongside `OperatorDispatch` in the "Key design rules" list.
- `docs/superpowers/specs/2026-04-18-crate-split-design.md`: add a "Completed" note to §9 linking to the new spec.

---

## Step 8 — CI isolation check

Add to `.github/workflows/ci.yaml` after the `cargo clippy` step:

```yaml
- name: Verify StateGraph decoupling
  run: |
    # quspin-basis contains no runtime match on *OperatorInner variants
    ! grep -rn "match .*OperatorInner" crates/quspin-basis/src/
    # StateGraph lives in quspin-bitbasis
    grep -q "pub trait StateGraph" crates/quspin-bitbasis/src/state_graph.rs
```

---

## Step 9 — Final verification

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uv run maturin develop
uv run pytest python/tests/ -x -q -m "not slow"
cargo tree -p quspin-basis | grep quspin-operator   # allowed (aliases), but no surprises
```

Open PR targeting `main`.

---

## Open question surfaced in step 6

The spec (§3.6) promises `quspin-basis → quspin-operator` is dropped from runtime deps. In practice, the typed aliases (`build_spin`, `build_bond`, …) keep the signature-level dep in place. This is an honest coupling: `quspin-basis` still has to **name** `*OperatorInner` types in its public API.

The real wins the refactor delivers even with the dep retained:
1. **Allocation saved** — no more `SmallVec` per BFS frontier node.
2. **Monomorphization locality** — `build<G: StateGraph>` generates code per `G`, but the generated code lives in `quspin-basis`'s compilation unit, not in `quspin-operator`. Recompiling only `quspin-operator` no longer invalidates basis logic.
3. **Extensibility** — third-party operator types (or test mocks) that impl `StateGraph` can drive `SpinBasis::build` without being one of the six built-in enums.

Decide before merging whether to:
- **(A)** Keep aliases, keep runtime dep, document the nuance.
- **(B)** Drop aliases, drop runtime dep, update `quspin-py`'s five basis files to call `basis.build(&op.inner, seeds)`.

Recommendation: **(A)** initially (zero quspin-py churn, full logic decoupling). Revisit if a subsequent spec merges the four basis types.
