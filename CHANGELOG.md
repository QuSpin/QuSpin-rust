# Changelog

All notable changes to the QuSpin-rust workspace are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions track the workspace-level `version` field in the root `Cargo.toml`.

## [Unreleased]

### Changed — BREAKING

- **Unified symmetry-element API on `SymBasis`.** `add_lattice`, `add_local`,
  and `add_inv` (on `SymBasis` itself) are removed. Every user-supplied
  element goes through the single method

  ```rust
  basis.add_symmetry(chi, SymElement::lattice(&perm))?;   // pure permutation
  basis.add_symmetry(chi, SymElement::local(op))?;        // pure local op
  basis.add_symmetry(chi, SymElement::composite(&perm, op))?; // atomic P·L
  ```

  plus the convenience `add_cyclic(generator, order, char_fn)` which
  populates `g¹…g^{order-1}` via `SymElement::compose`.

  The identity element is always implicit — adding an `SymElement` with
  identity action (e.g. the identity permutation, or an empty element)
  is rejected.

  `SymBasis::build` now runs `validate_group` internally: an
  `O(|G|²·probes)` closure + 1D-character check
  (`χ(g·h) = χ(g)·χ(h)`). Missing group closure (e.g. supplying only a
  translation generator), inconsistent characters, or duplicate actions
  now error at build time instead of silently producing a wrong basis.

  The shim methods on `SpaceInner` (`add_lattice` / `add_local` /
  `add_inv`) and the user-facing `SpinBasis` / `BosonBasis` /
  `FermionBasis` / `GenericBasis` wrappers with the same names are
  preserved — they build a `SymElement` internally and call
  `add_symmetry`. Python API names in `quspin-py` are unchanged; the
  only user-visible change is that all elements of any non-trivial
  cyclic group must now be supplied (previously only generators were
  needed).

- **Basis types expose one generic `build` method instead of typed aliases.**
  The eight typed methods `SpinBasis::{build_spin, build_hardcore, build_bond}`,
  `BosonBasis::{build_boson, build_bond}`, `FermionBasis::{build_fermion,
  build_bond}`, `GenericBasis::build_monomial` have been removed. All four
  basis types now expose a single method:

  ```rust
  pub fn build<G: StateTransitions>(&mut self, graph: &G, seeds: &[Vec<u8>])
      -> Result<(), QuSpinError>
  ```

  Migration: replace each call site. Every `*OperatorInner` enum (plus every
  per-cindex operator type) implements `StateTransitions`, so existing
  call sites keep the same argument list:

  ```rust
  // before
  basis.build_spin(&op.inner, seeds)?;
  basis.build_bond(&bond_op.inner, seeds)?;

  // after
  basis.build(&op.inner, seeds)?;
  basis.build(&bond_op.inner, seeds)?;
  ```

  Python-facing method names (`build_spin`, `build_bond`, etc. on the PyO3
  classes) are unchanged — they live in `#[pyo3(name = …)]` annotations
  inside `quspin-py`, not in the basis crate's public API.

### Added

- `SymElement<L>` type in `quspin-basis` — typed group-element constructor
  consumed by `SymBasis::add_symmetry` / `add_cyclic`.
- `Compose` trait in `quspin-bitbasis` — group-composition primitive
  implemented for `PermDitMask`, `PermDitValues<N>`, and
  `DynamicPermDitValues`. Used by `SymElement::compose` and therefore by
  `SymBasis::add_cyclic`.
- `FermionicBitStateOp: BitStateOp` trait in `quspin-bitbasis` with
  a default `fermion_sign = 1.0`. `SymBasis` now bounds its local-op
  type `L: FermionicBitStateOp<B>` so the walker can thread
  Jordan-Wigner signs through orbit images without paying anything on
  bosonic bases (LLVM eliminates the constant branch).
- `SignedPermDitMask<B>` in `quspin-bitbasis` — a `PermDitMask` with a
  per-site sign array (product over occupied sites). Separate type so
  particle-hole and spin-inversion signs do not cost anything on plain
  spin-1/2 bases. Not yet exposed to Python; tracked as a follow-up.
- New `StateTransitions` trait in `quspin-types` (renamed from an earlier
  internal draft called `StateGraph`). Describes the operator-side input to
  basis BFS as `(amplitude, new_state)` pairs. Re-exported from
  `quspin-bitbasis` for existing import paths.
- `BitInt` trait and all three concrete impls (`u32`, `u64`,
  `ruint::Uint<BITS, LIMBS>`) move from `quspin-bitbasis` to `quspin-types`.
  Re-exported from `quspin-bitbasis` so `use quspin_bitbasis::BitInt` keeps
  working. The orphan rule forces the impls to live alongside the trait
  definition, and `BitInt` is a workspace-level abstraction with no
  bitbasis-specific behaviour, so this is a better home.
- New `Operator::lhss()` method on the `Operator<C>` trait in `quspin-operator`.
- `quspin-bitbasis` gains a `test-graphs` feature exposing reusable
  `StateTransitions` mocks (`XAllSites`, `XXYYNearestNeighbor`,
  `NearestNeighborSwap`) for downstream test suites.

### Removed

- Runtime Cargo dependency `quspin-basis → quspin-operator`. The two crates
  now compile in parallel off `quspin-bitbasis`. `quspin-operator` remains
  as a dev-dependency of `quspin-basis` for test support.
- `Subspace::build` and `SymBasis::build` no longer accept a
  `Fn(B) -> IntoIterator<Item = (Complex, B, u8)>` closure; they take
  `&impl StateTransitions` instead. The closure form was internal to
  `quspin-basis` and not part of the public API — mentioned here for
  completeness.

### Internal

- BFS no longer materialises a `SmallVec<(amp, state, cindex)>` per frontier
  node; the `StateTransitions::neighbors` callback writes contributions
  directly into the per-wave hashmap. Allocation savings are unmeasured —
  benchmark is on the to-do list (tracked as a follow-up).
