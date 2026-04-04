# MonomialOperator + GenericBasis Design

**Date:** 2026-04-04
**Branch:** phil/api-explore
**Status:** Approved

---

## Overview

Two new paired types, following the same conventions as the existing
`BosonOperator`/`BosonBasis` pairs:

- **`MonomialOperator`** — a new operator type defined by monomial matrices
  (one non-zero per row) rather than operator strings.
- **`GenericBasis`** — a new basis wrapper for any LHSS that exposes both
  lattice symmetries and masked local symmetries.

`MonomialOperator` is intended only for use with `GenericBasis`. Existing
basis types (`BosonBasis`, `SpinBasis`, `FermionBasis`) do not accept it.

---

## Motivation

The existing operator types (`BosonOperator`, `SpinOperator`, `FermionOperator`)
are tied to specific particle algebras. `BondOperator` supports arbitrary
2-site dense matrices. Neither covers the case of a general k-site operator
whose action on each input state produces **exactly one output state** with a
complex amplitude — i.e., a monomial matrix. Common examples: Potts/clock
model terms, cyclic permutation operators, particle-hole conjugation operators.

`GenericBasis` is the matching basis for any such model, supporting the same
lattice symmetries as existing bases plus **masked local symmetries** (a
permutation of local Hilbert space states applied only to a specified subset of
sites).

---

## 1. `MonomialOperator` — `quspin-core`

### Location

New module: `crates/quspin-core/src/operator/monomial/`

### Data Structures

```rust
/// One term: a monomial matrix applied to every bond in `bonds`.
pub struct MonomialTerm<C> {
    /// perm[i] = output joint-state index for input joint-state i.
    /// Length = lhss^k where k = bond size.
    perm: Vec<usize>,
    /// amp[i] = complex amplitude for input joint-state i.
    /// Length = lhss^k. Encodes the full per-entry value; no separate coeff.
    amp: Vec<Complex<f64>>,
    /// Each bond is a k-tuple of site indices. All bonds within a term must
    /// have the same k. k may differ across terms.
    bonds: Vec<SmallVec<[u32; 4]>>,
    cindex: C,
}

pub struct MonomialOperator<C> {
    terms: Vec<MonomialTerm<C>>,
    lhss: usize,
    max_site: usize,
    num_cindices: usize,
    // No cached DynamicDitManip — constructed inline in apply_dynamic,
    // matching the BondOperator pattern.
}
```

**No `coeff` field** on `MonomialTerm`. Unlike `BondTerm` (where the matrix
encodes operator shape and `coeff` is the physical coupling), `amp` already
carries the full complex amplitude. Any scaling is absorbed at construction
time.

### `apply` Implementation

Joint-state indexing is row-major over k dits:
`idx = d_0 * lhss^(k-1) + d_1 * lhss^(k-2) + ... + d_{k-1}`

For each bond `(i_0, i_1, ..., i_{k-1})`:
1. Extract k dit values from `state` → encode as joint index `idx`
2. Look up `perm[idx]` and `amp[idx]`
3. Decode `perm[idx]` back into k dit values
4. Insert new dit values into `state` → `new_state`
5. `emit(amp[idx], new_state, cindex)`

Exactly **one emit per bond** — no branching, no superposition.

### Static Dispatch

Matches the `BondOperator` pattern: dispatch on `lhss` at apply time using a
`match` in `Operator::apply`.

- LHSS = 2, 3, 4: `apply_impl::<L>()` using compile-time `DitManip<const L>`
  for dit extraction and insertion.
- LHSS ≥ 5: `apply_dynamic()` constructing `DynamicDitManip::new(lhss)` inline
  (not cached on the struct).

### Dispatch Enum

`MonomialOperatorInner` follows the same established bridge-type convention as
`BondOperatorInner` (which also lives in `quspin-core/src/operator/bond/dispatch.rs`).
This is the accepted pattern for type-erasing the cindex type at the
core/PyO3 boundary — it does not violate the no-runtime-dispatch rule, which
applies to business-logic dispatch.

```rust
pub enum MonomialOperatorInner {
    Ham8(MonomialOperator<u8>),
    Ham16(MonomialOperator<u16>),
}
```

---

## 2. `GenericBasis` — `quspin-core` + `quspin-py`

### New `GenLocalOp<B>` Type

A single new enum consolidates static local-op dispatch for all LHSS values,
keeping the `SpaceInner` variant count minimal:

```rust
pub enum GenLocalOp<B: BitInt> {
    Lhss2(PermDitMask<B>),           // XOR mask — fast path for LHSS=2
    Lhss3(PermDitValues<3>),         // static DitManip<3> — existing type
    Lhss4(PermDitValues<4>),         // static DitManip<4> — existing type
    Dynamic(DynamicPermDitValues),   // runtime — LHSS≥5
}
```

`PermDitValues<const LHSS>` already exists in
`crates/quspin-core/src/bitbasis/transform.rs`. No new types needed for the
static LHSS=3 and LHSS=4 paths.

`GenLocalOp<B>` implements `BitStateOp<B>` by dispatching on the variant.
The LHSS dispatch is a single branch (enum variant check), and the inner
`PermDitValues::<3>::apply` / `PermDitValues::<4>::apply` use `DitManip<3>`
and `DitManip<4>` statically.

### New `SpaceInner` Variants

A single new `GenSym_*` family covers all LHSS values via `GenLocalOp<B>`:

| Variant | B type | Always compiled |
|---|---|---|
| `GenSym32` | `u32` | yes |
| `GenSym64` | `u64` | yes |
| `GenSym128` | `Uint<128,2>` | yes |
| `GenSym256` | `Uint<256,4>` | yes |
| `GenSym512` | `Uint<512,8>` | `large-int` feature |
| `GenSym1024` | `Uint<1024,16>` | `large-int` feature |
| `GenSym2048` | `Uint<2048,32>` | `large-int` feature |
| `GenSym4096` | `Uint<4096,64>` | `large-int` feature |
| `GenSym8192` | `Uint<8192,128>` | `large-int` feature |

All 9 use `SymBasis<B, GenLocalOp<B>, N>`.

**Note:** The `large-int` feature (`#[cfg(feature = "large-int")]`) gates
`Uint<512..8192>` variants. This feature lives on branch
`phil/build-improvements`. The `{512..8192}` variants should be sequenced
after that branch merges. Always compiled: `u32`, `u64`, `Uint<128>`,
`Uint<256>`.

This adds 9 new `SpaceInner` variants (vs 27 with three separate families).

### `add_local` Construction

`add_local(perm: &[usize], char: Complex<f64>, mask: &[usize])` dispatches on
`lhss` to select the right `GenLocalOp<B>` variant:

- LHSS = 2: construct `PermDitMask<B>` with bits set at the sites in `mask`.
  For LHSS=2 the only non-trivial permutation of `{0,1}` is `[1,0]` (flip);
  applying XOR at the mask sites is correct and matches this permutation
  independently at each site. If `perm == [0,1]` (identity), no local op is
  added.
- LHSS = 3: construct `PermDitValues::<3>` with `perm: [u8; 3]` and
  `locs: mask.to_vec()`
- LHSS = 4: construct `PermDitValues::<4>` with `perm: [u8; 4]` and
  `locs: mask.to_vec()`
- LHSS ≥ 5: construct `DynamicPermDitValues::new(lhss, perm_bytes, mask.to_vec())`

When `mask` is omitted (Python-side default), it defaults to all sites
`0..n_sites`.

### `build_monomial`

Calls the existing BFS machinery with `MonomialOperatorInner` as the
Hamiltonian, same as `build_boson` / `build_bond`.

---

## 3. Python API and `.pyi` Stubs

All stubs are added to `python/quspin_rs/_rs.pyi`.

### `MonomialOperator`

Mirrors `BondOperator`'s `(matrix, bonds, cindex)` tuple format:

```python
class MonomialOperator:
    def __init__(
        self,
        terms: list[tuple[
            npt.NDArray[np.intp],            # perm: shape (lhss^k,)
            npt.NDArray[np.complexfloating], # amp:  shape (lhss^k,) — no coeff
            list[tuple[int, ...]],           # bonds: k-tuples; k consistent within a term
            int,                             # cindex
        ]],
        lhss: int,
    ) -> None: ...
    @property
    def max_site(self) -> int: ...
    @property
    def num_cindices(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    def __repr__(self) -> str: ...
```

**Key documentation note:** `amp` carries the full per-entry amplitude; there
is no `coeff` parameter. This differs from `BondOperator` where a separate
coefficient scales the matrix. Bond k (number of sites per bond) must be
consistent within each term but may differ across terms.

### `GenericBasis`

```python
class GenericBasis:
    @classmethod
    def full(cls, n_sites: int, lhss: int) -> GenericBasis:
        """Full Hilbert space (no projection, no build step required)."""
        ...

    @classmethod
    def subspace(
        cls,
        n_sites: int,
        lhss: int,
        ham: MonomialOperator,
        seeds: list[str],
    ) -> GenericBasis:
        """Subspace reachable from seeds under ham."""
        ...

    @classmethod
    def symmetric(
        cls,
        n_sites: int,
        lhss: int,
        ham: MonomialOperator,
        seeds: list[str],
        symmetries: list[tuple[list[int], tuple[float, float]]],
        local_symmetries: list[
            tuple[list[int], tuple[float, float]]               # all sites
            | tuple[list[int], tuple[float, float], list[int]]  # explicit mask
        ],
    ) -> GenericBasis:
        """Symmetry-projected subspace.

        Args:
            symmetries: List of ``(perm, (re, im))`` lattice symmetry tuples
                where ``perm`` is a site-permutation (acts on all sites).
            local_symmetries: List of local symmetry tuples. Each is either
                ``(perm, (re, im))`` — apply to all sites — or
                ``(perm, (re, im), mask)`` — apply only to the listed sites.
                ``perm`` is a permutation of ``{0, ..., lhss-1}`` (single-site
                local states), applied independently at each site in the mask.
        """
        ...

    @property
    def n_sites(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def is_built(self) -> bool: ...
    def state_at(self, i: int) -> str: ...
    def index(self, state: str) -> int | None: ...
    def __repr__(self) -> str: ...
```

### `QMatrix` Extension

New static method added to `QMatrix` in `_rs.pyi`:

```python
@staticmethod
def build_monomial(
    op: MonomialOperator,
    basis: GenericBasis,
    dtype: np.dtype[Any],
) -> QMatrix: ...
```

---

## 4. Error Handling

Validated at construction / call time:

| Check | Error |
|---|---|
| `terms` must not be empty | `ValueError` |
| `len(perm) == len(amp) == lhss^k` for each term | `ValueError` |
| All bonds within a single term must have the same `k` | `ValueError` |
| `perm` values in `{0,...,lhss^k - 1}` | `ValueError` |
| `add_local` perm is a valid permutation of `{0,...,lhss-1}` | `ValueError` |
| `mask` site indices in `{0,...,n_sites-1}` | `ValueError` |
| `symmetric` called before `build` in BFS | `QuSpinError` (existing pattern) |

---

## 5. Testing

**Rust unit tests** (`quspin-core`):
- `MonomialTerm::apply` correctness for LHSS = 2, 3, 4, 5 (covers static and
  dynamic paths)
- Single-emit invariant: verify exactly one output per (term, bond) application
- `GenLocalOp` apply matches expected per-variant behaviour

**Python integration tests** (`python/tests/`):
- Round-trip: construct a `MonomialOperator` equivalent to a known
  `BondOperator`, verify `QMatrix` matrices match
- `GenericBasis.symmetric` with lattice + masked local symmetries: verify
  basis size matches expected sector dimension
- `local_symmetries` with and without explicit mask produce consistent results

---

## 6. Implementation Sequencing

1. `MonomialTerm` / `MonomialOperator` + `MonomialOperatorInner` in
   `quspin-core/src/operator/monomial/`
2. `GenLocalOp<B>` enum in `quspin-core/src/bitbasis/` (or `basis/`)
3. New `GenSym_*` variants (`u32`, `u64`, `Uint<128>`, `Uint<256>`) added to
   `SpaceInner` in `quspin-core/src/basis/dispatch.rs`
4. `GenericBasis` struct + `add_local` + `build_monomial` in `quspin-core`
5. PyO3 bindings: `PyMonomialOperator`, `PyGenericBasis` in `quspin-py`
6. `QMatrix::build_monomial` PyO3 binding — update QMatrix build dispatch
   (`quspin-py`) to handle `GenSym_*` variants
7. `.pyi` stub additions to `python/quspin_rs/_rs.pyi`:
   - `MonomialOperator` class
   - `GenericBasis` class
   - `QMatrix.build_monomial` static method
8. Rust + Python tests
9. `GenSym_{512..8192}` variants — after `phil/build-improvements` merges
