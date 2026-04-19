# Implementation Plan: `StateTransitions` Decoupling

**Spec:** `docs/superpowers/specs/2026-04-19-state-graph-decoupling.md`
**Date:** 2026-04-19
**Branch:** `phil/state-graph-refactor` (off `main`)

> **Note on naming.** This plan was drafted under the working name
> `StateGraph`. The trait was renamed to `StateTransitions` during
> review because its final callback takes `(amplitude, new_state)`
> rather than just connectivity. All code snippets below show the
> final signatures; the earlier drafts in git history use the old name.

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

## Step 2 — Add `StateTransitions` trait in `quspin-bitbasis`

File: `crates/quspin-bitbasis/src/state_transitions.rs` (new)

```rust
use num_complex::Complex;

use crate::int::BitInt;

pub trait StateTransitions: Send + Sync {
    fn lhss(&self) -> usize;
    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, visit: F);
}
```

File: `crates/quspin-bitbasis/src/lib.rs`

- Add `pub mod state_transitions;`
- Add `pub use state_transitions::StateTransitions;`

Verification: `cargo check -p quspin-bitbasis`.

---

## Step 3 — Implement `StateTransitions` for operator types

All twelve impls (six per-cindex types + six dispatch enums) are
placed in a single file `crates/quspin-operator/src/state_transitions.rs`
to keep the per-operator files focused on domain logic.

```rust
use num_complex::Complex;
use quspin_bitbasis::{BitInt, StateTransitions};
use crate::Operator;

// Per-cindex types. `Send + Sync` on C is inherited from the trait
// supertrait; in practice C is always u8 or u16 so this is a no-op.
impl<C: Copy + Ord + Send + Sync> StateTransitions for SpinOperator<C> {
    fn lhss(&self) -> usize { Operator::<C>::lhss(self) }
    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, mut visit: F) {
        self.apply::<B, _>(state, |_c, amp, ns| visit(amp, ns));
    }
}
// … BondOperator, BosonOperator, FermionOperator, HardcoreOperator, MonomialOperator …

// Dispatch enums.
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
// … BondOperatorInner, BosonOperatorInner, … (six total) …
```

In practice both pattern families are generated from a pair of `macro_rules!` macros (`impl_state_transitions_for_operator!` and `impl_state_transitions_for_inner!`).

`HardcoreOperatorInner::lhss` keeps its `const fn` inherent form as a no-trait-bound method; the trait impl delegates via `StateTransitions::lhss(h)`.

Verification: `cargo check -p quspin-operator`; grep `impl StateTransitions` in the operator crate should show 12 impls (6 generic + 6 enum).

---

## Step 4 — Refactor `Subspace::build` / `SymBasis::build`

Files:
- `crates/quspin-basis/src/space.rs` (`Subspace::build`)
- `crates/quspin-basis/src/sym.rs` (`SymBasis::build`, plus the private `bfs_wave_with_reps` helper)
- `crates/quspin-basis/src/bfs.rs` (`bfs_wave`, `bfs_wave_sequential`, `bfs_wave_parallel`, `discover_from_state`)

Current signature:
```rust
pub fn build<F, I>(&mut self, seed: B, apply: F)
where F: Fn(B) -> I, I: IntoIterator<Item = (Complex<f64>, B, u8)>
```

New signature:
```rust
pub fn build<G: StateTransitions>(&mut self, seed: B, graph: &G)
```

(`SymBasis::build` has the same shape plus its existing `L: Sync` bound.)

Inside the BFS frontier loop, the call site changes from:
```rust
for (amp, ns, _c) in apply(state) {
    let e = contributions.entry(ns).or_default();
    e.0 += amp;
    e.1 += amp.norm();
}
```
to:
```rust
graph.neighbors::<B, _>(state, |amp, ns| {
    let e = contributions.entry(ns).or_default();
    e.0 += amp;
    e.1 += amp.norm();
});
```

The per-target amplitude-cancellation bookkeeping (the `contributions` hashmap and the `AMP_CANCEL_TOL`-gated `discovered.insert(ns)`) is unchanged; only the "where do neighbours come from" hook moves from an iterator-returning closure to a direct `StateTransitions::neighbors` call.

The same transformation applies to the private `bfs_wave_with_reps` helper in `sym.rs`, which BFS-expands and then maps each survivor to its orbit representative.

> **Why not a visitor-style `FnMut(B, &mut dyn FnMut(B))`?**
>
> An earlier draft of this plan introduced a visitor indirection, with an optional `StateEmitter<B>` fallback trait if `dyn` inlining misbehaved. Once the amplitude was re-added to the callback (needed for cancellation detection), the trait-based form turned out to be both simpler and monomorphises better — the visitor draft was never implemented.

Verification: `cargo test -p quspin-basis`. All 77 basis tests must pass. The tests use shared `StateTransitions` mocks from `quspin_bitbasis::test_graphs` (added in step 7a below).

---

## Step 5 — Replace basis `build_*` with a single generic `build`

For each of the four basis modules (`spin.rs`, `boson.rs`, `fermion.rs`, `generic.rs`):

1. Replace all existing `build_*` methods with one generic `build`:

   ```rust
   pub fn build<G: StateTransitions>(&mut self, g: &G, seeds: &[Vec<u8>]) -> Result<(), QuSpinError> {
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

- `CLAUDE.md`: redraw the DAG so `quspin-operator` and `quspin-basis` sit at the same level under `quspin-bitbasis`. Add one sentence noting `StateTransitions` as the connectivity abstraction. Mention it in "Key design rules" alongside `OperatorDispatch`.
- `docs/superpowers/specs/2026-04-18-crate-split-design.md`: add a "Completed 2026-04-19" note to §9 linking to the new spec.

---

## Step 10 — CI isolation check

Add to `.github/workflows/ci.yaml` after the `cargo clippy` step:

```yaml
- name: Verify basis/operator dependency isolation
  run: |
    # Run cargo tree as its own command so a tool failure surfaces as a
    # CI failure, rather than being silently swallowed by grep.
    tree_output="$(cargo tree -p quspin-basis --edges=normal)"
    ! printf '%s\n' "$tree_output" | grep -q quspin-operator
```

The `--edges=normal` flag excludes dev-dependencies, so the legitimate `[dev-dependencies] quspin-operator = { … }` entry on `quspin-basis` does not false-positive the check. Running `cargo tree` as a separate command (not piped into `grep 2>/dev/null`) ensures a tool failure causes a CI failure — otherwise the leading `!` would silently invert `grep`'s non-zero exit and mark the step as green.

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
- **Trait naming:** finalised as `StateTransitions`. The working name in the original draft was `StateGraph`; renamed during review because the final callback carries amplitudes, not just connectivity.
- **StateTransitions location:** `quspin-bitbasis` (needs `BitInt`, §3.1 of spec).
- **Amplitude in callback:** `FnMut(Complex<f64>, B)`. A connectivity-only `FnMut(B)` was drafted first but proved incorrect — the basis needs amplitudes to detect symbolic cancellation (e.g. `XX + YY` on `|00⟩`).
- **`lhss()` trait method:** added to both `Operator<C>` and `StateTransitions`, not a free function.
- **BFS signature:** `pub fn build<G: StateTransitions>(&mut self, seed: B, graph: &G)`. An intermediate visitor-style `FnMut(B, &mut dyn FnMut(B))` was in an earlier draft of this plan but never implemented — passing `&impl StateTransitions` directly turned out both simpler and better-inlined.
