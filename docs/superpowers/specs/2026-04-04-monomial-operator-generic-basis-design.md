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
    /// Each bond is a k-tuple of site indices.
    bonds: Vec<SmallVec<[u32; 4]>>,
    cindex: C,
}

pub struct MonomialOperator<C> {
    terms: Vec<MonomialTerm<C>>,
    lhss: usize,
    max_site: usize,
    num_cindices: usize,
    /// Cached for dit extraction/insertion; constructed once.
    manip: DynamicDitManip,
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

Matches the `BondOperator` pattern: dispatch on `lhss` at apply time.

- LHSS = 2, 3, 4: `apply_impl::<L>()` using compile-time `DitManip<const L>`
  for dit extraction and insertion.
- LHSS ≥ 5: `apply_dynamic()` using the cached `DynamicDitManip`.

### Dispatch Enum

```rust
pub enum MonomialOperatorInner {
    Ham8(MonomialOperator<u8>),
    Ham16(MonomialOperator<u16>),
}
```

---

## 2. `GenericBasis` — `quspin-core` + `quspin-py`

### New `SpaceInner` Variants

`GenericBasis` needs static compiled local-op types for LHSS = 3 and 4 (LHSS
= 2 already uses `PermDitMask<B>` in the existing `Sym*` variants).

New variant families added to `SpaceInner`:

| Family | LHSS | Local op type | Always compiled |
|---|---|---|---|
| `GenSym3_{32,64,128,256}` | 3 | `StaticLocalPerm<3, B>` | yes |
| `GenSym3_{512..8192}` | 3 | `StaticLocalPerm<3, B>` | `large-int` feature |
| `GenSym4_{32,64,128,256}` | 4 | `StaticLocalPerm<4, B>` | yes |
| `GenSym4_{512..8192}` | 4 | `StaticLocalPerm<4, B>` | `large-int` feature |
| `GenDitSym_{32,64,128,256}` | ≥5 | `DynamicPermDitValues` | yes |
| `GenDitSym_{512..8192}` | ≥5 | `DynamicPermDitValues` | `large-int` feature |

`StaticLocalPerm<const LHSS, B>` uses `DitManip<LHSS>` for the per-site
permutation at each masked site.

**Note:** The `large-int` feature (`#[cfg(feature = "large-int")]`) gates
`Uint<512..8192>` variants. This feature lives on branch
`phil/build-improvements`; implementation of the `512..8192` variants should
be sequenced after that branch merges.

Always compiled (no feature gate): `u32`, `u64`, `Uint<128>`, `Uint<256>`.

### `add_local` Construction

`add_local(perm: &[usize], char: Complex<f64>, mask: &[usize])` dispatches on
`lhss`:

- LHSS = 2: construct `PermDitMask<B>` with bits set at `mask` sites (requires
  `perm == [1, 0]`)
- LHSS = 3: construct `StaticLocalPerm<3, B>` applying `perm` at each masked
  site, identity elsewhere
- LHSS = 4: construct `StaticLocalPerm<4, B>` as above
- LHSS ≥ 5: construct `DynamicPermDitValues` as above

When `mask` is omitted (Python-side default), it defaults to all sites
`0..n_sites`.

### `build_monomial`

Calls the existing BFS machinery with `MonomialOperatorInner` as the
Hamiltonian, same as `build_boson` / `build_bond`.

---

## 3. Python API and `.pyi` Stubs

### `MonomialOperator`

Mirrors `BondOperator`'s `(matrix, bonds, cindex)` tuple format.

```python
class MonomialOperator:
    def __init__(
        self,
        terms: list[tuple[
            npt.NDArray[np.intp],            # perm: shape (lhss^k,)
            npt.NDArray[np.complexfloating], # amp:  shape (lhss^k,) — no coeff
            list[tuple[int, ...]],           # bonds: k-tuples of site indices
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
coefficient scales the matrix.

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

A new static method is added to `QMatrix`:

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
| `len(perm) == len(amp) == lhss^k` | `ValueError` |
| All bonds in a term have the same `k` | `ValueError` |
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

**Python integration tests** (`python/tests/`):
- Round-trip: construct a permutation operator equivalent to a known
  `BondOperator`, verify `QMatrix` matrices match
- `GenericBasis.symmetric` with lattice + masked local symmetries: verify
  basis size matches expected sector dimension
- `local_symmetries` with and without explicit mask produce consistent results

---

## 6. Implementation Sequencing

1. `MonomialTerm` / `MonomialOperator` in `quspin-core` (no external deps)
2. `StaticLocalPerm<const LHSS, B>` new local op type in `quspin-core`
3. New `SpaceInner` variants (`GenSym3_*`, `GenSym4_*`, `GenDitSym_*`) for
   `u32`, `u64`, `Uint<128>`, `Uint<256>`
4. `GenericBasis` struct in `quspin-core`
5. PyO3 bindings: `PyMonomialOperator`, `PyGenericBasis` in `quspin-py`
6. `QMatrix::build_monomial` binding
7. `.pyi` stub updates
8. Rust + Python tests
9. `Uint<512..8192>` variants — after `phil/build-improvements` merges
