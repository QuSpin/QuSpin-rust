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

## Step 5 — Replace basis `build_*` with a single generic `build`

For each of the four basis modules (`spin.rs`, `boson.rs`, `fermion.rs`, `generic.rs`):

1. Replace all existing `build_*` methods with one generic `build`:

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

2. Delete `build_spin`, `build_hardcore`, `build_bond`, `build_boson`, `build_fermion`, `build_monomial`.

3. Remove all `use quspin_operator::*` imports from `src/` — only test modules still need them.

Verification: `cargo check -p quspin-basis` with `quspin-operator` removed from `[dependencies]` (step 6). Tests will break until step 7.

---

## Step 6 — Drop `quspin-basis → quspin-operator` runtime dep

Decision recorded: we drop the typed aliases in favour of the single `build` method. This gives full parallel compile (no signature-level dep on `*OperatorInner`) and is strictly equal on runtime performance (Rust would inline the aliases to the same machine code).

File: `crates/quspin-basis/Cargo.toml`

```toml
[dependencies]
quspin-types    = { path = "../quspin-types" }
quspin-bitbasis = { path = "../quspin-bitbasis" }
# quspin-operator ← deleted from runtime deps
num-complex = { workspace = true }
rayon       = { workspace = true }
ruint       = { workspace = true }
smallvec    = { workspace = true }

[dev-dependencies]
quspin-operator = { path = "../quspin-operator" }   # tests only
```

Verification:
```sh
cargo check -p quspin-basis                          # must pass
cargo tree -p quspin-basis --edges=normal | grep quspin-operator   # must be empty
```

---

## Step 7 — Update `quspin-basis` tests

Tests in `crates/quspin-basis/src/{spin,boson,fermion,generic}.rs` call the old `build_spin` / `build_bond` / etc. aliases. Each test must be updated to call `.build(&ham, &seeds)`.

Grep to find them:
```sh
grep -rn "\.build_\(spin\|hardcore\|bond\|boson\|fermion\|monomial\)\b" crates/quspin-basis/src/
```

Replace each with `.build(...)`. No other test logic changes.

Verification: `cargo test -p quspin-basis`.

---

## Step 8 — Update `quspin-py`

Five PyO3 binding files currently call the typed build methods:

| File | Line(s) | Old | New |
|------|---------|-----|-----|
| `crates/quspin-py/src/basis/spin.rs` | 26, 30 | `.build_hardcore(&op.borrow().inner, …)` | `.build(&op.borrow().inner, …)` |
| `crates/quspin-py/src/basis/spin.rs` | 30 | `.build_bond(&op.borrow().inner, …)` | `.build(&op.borrow().inner, …)` |
| `crates/quspin-py/src/basis/boson.rs` | 24, 28 | `.build_boson` / `.build_bond` | `.build` |
| `crates/quspin-py/src/basis/fermion.rs` | 24, 28 | `.build_fermion` / `.build_bond` | `.build` |
| `crates/quspin-py/src/basis/generic.rs` | 124, 157 | `.build_monomial` | `.build` |

Python-facing method names (`build_spin`, `build_bond`, etc. on the PyO3 classes) stay the same — those are `#[pyo3(...)]` method names in `quspin-py`, not Rust method names in `quspin-basis`.

Verification: `uv run maturin develop` builds; `uv run pytest python/tests/ -x -q -m "not slow"` green.

---

## Step 9 — Update docs

- `CLAUDE.md`: redraw the DAG so `quspin-operator` and `quspin-basis` sit at the same level under `quspin-bitbasis`. Add one sentence noting `StateGraph` as the connectivity abstraction. Mention it in "Key design rules" alongside `OperatorDispatch`.
- `docs/superpowers/specs/2026-04-18-crate-split-design.md`: add a "Completed 2026-04-19" note to §9 linking to the new spec.

---

## Step 10 — CI isolation check

Add to `.github/workflows/ci.yaml` after the `cargo clippy` step:

```yaml
- name: Verify StateGraph decoupling
  run: |
    # quspin-basis has no runtime dep on quspin-operator
    ! cargo tree -p quspin-basis --edges=normal 2>/dev/null | grep -q quspin-operator
    # StateGraph trait lives in quspin-bitbasis
    grep -q "pub trait StateGraph" crates/quspin-bitbasis/src/state_graph.rs
    # quspin-basis does not match on *OperatorInner variants
    ! grep -rn "match .*OperatorInner" crates/quspin-basis/src/
```

---

## Step 11 — Final verification

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uv run maturin develop
uv run pytest python/tests/ -x -q -m "not slow"

# DAG sanity:
cargo tree -p quspin-basis    --edges=normal | grep quspin-operator   # empty
cargo tree -p quspin-operator --edges=normal | grep quspin-basis      # empty
```

Open PR targeting `main`.

---

## Decision log

- **Aliases kept vs dropped:** dropped. Performance-equivalent at runtime (inlined forwarding), but aliases keep `*OperatorInner` in `quspin-basis`'s public signatures which defeats the compile-time-parallelism goal. Dropping aliases costs five one-line edits in `quspin-py` (step 8) and deletes the `quspin-basis → quspin-operator` runtime edge entirely.
- **StateGraph location:** `quspin-bitbasis` (needs `BitInt`, §3.1 of spec).
- **`lhss()` trait method:** added to both `Operator<C>` and `StateGraph`, not a free function.
- **BFS signature:** visitor-style `FnMut(B, &mut dyn FnMut(B))`. If benchmarks show inlining loss, fall back to the `StateEmitter<B>` trait (step 4 fallback).
