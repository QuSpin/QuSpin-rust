# Python Symmetry-Group API for `*Basis.symmetric(...)`

**Status:** spec
**Date:** 2026-04-26
**Author:** Phillip Weinberg (with Claude)
**Tracks:** quspin-py user-facing symmetry construction

## Summary

Replace the loose `symmetries=[(perm, (re, im)), …]` and `local_symmetries=[…]`
keyword arguments on `*Basis.symmetric(...)` with a single first-class
`SymmetryGroup` object. Add element constructors (`Lattice`, `Local`,
`Composite`) for the three group-element shapes already supported in Rust by
[`SymElement<L>`](../../../crates/quspin-basis/src/sym_element.rs), plus group
helpers (`add`, `add_cyclic`, `close`, `product`) that match common physics use
cases without forcing the user to enumerate every non-identity element by hand.

After this change every symmetric-basis constructor takes the same three-argument
form:

```python
basis = SpinBasis.symmetric(group, ham, seeds)
```

with `group.n_sites` and `group.lhss` driving the basis dimensions and
LHSS. Old keyword tuples are removed (no public release yet, breaking change
acceptable).

## Goals

- Cover the three group-element shapes the Rust core already supports:
  pure-lattice, pure-local, and composite (e.g. `PZ` parity-times-spin-flip,
  used when neither `P` nor `Z` is a symmetry on its own).
- Cover the common 1-D-irrep selection patterns with one-line helpers
  (`add_cyclic` with momentum index `k`, plus `eta=±1` sugar for Z₂
  generators).
- Provide an explicit closure helper (`SymmetryGroup.close`) for
  non-abelian / general group cases, with a user-supplied character function
  on the resulting elements.
- Make wrong inputs fail loudly with actionable messages: closure failures and
  inconsistent characters surface from `SymBasis::validate_group` at first
  build; same check is exposed eagerly via `group.validate()`.
- Migrate the `*Basis.symmetric(...)` signatures to a single `group=`
  argument (no `symmetries` / `local_symmetries`); pull `n_sites` / `lhss`
  off the group.

## Non-goals

- **Hamiltonian invariance check (`group.validate_hamiltonian(ham)`).** Deferred
  to a follow-up. Doing it correctly requires symbolic transforms on operator
  terms (rewriting each monomial under each generator and comparing the result
  back to the original term list); a probe-state implementation would have
  false negatives on accidental cancellations. Rust-side operators do not yet
  support the necessary algebraic manipulation.
- **Public composition arithmetic on element handles** (`a * b`, `g ** k` in
  Python). Not needed for the user-facing patterns covered here; can be added
  later with a clear use case.
- **Field-level accessors on element handles** (`elem.perm`,
  `elem.perm_vals`, `elem.locs`). Same reasoning.
- **Sub-cyclic-group enumeration.** A cyclic generator's group always has
  order equal to that generator's true order — there is no way to ask for a
  proper subset. If a user wants only powers of `g²`, they pass `g²` as the
  generator.
- **Non-1-D irreps** (full character tables, induced representations). Not
  required by the existing `SymBasis::validate_group` walker.

## User-facing surface

### Imports

```python
from quspin_rs import (
    SymmetryGroup,
    Lattice, Local, Composite,
    SpinBasis, FermionBasis, BosonBasis, GenericBasis,
)
```

### Element constructors

Three free functions in the `_rs` extension, each returning an opaque
`SymElement` handle. The handle is LHSS-agnostic — typing happens at
basis-construction time.

| Constructor | Signature | Internals |
|---|---|---|
| `Lattice(perm)` | `perm: list[int]` length `n_sites` | `(Some(perm), None, None)` |
| `Local(perm_vals, locs=None)` | `perm_vals: list[int]` length `lhss`; `locs: list[int] \| None` | `(None, Some(perm_vals), locs)` |
| `Composite(perm, perm_vals, locs=None)` | both | `(Some(perm), Some(perm_vals), locs)` |

**`locs=None` semantics.** `None` is preserved as `Option::None` all the way
down to the per-family inner enum's `add_local`, which expands it to the
"all sites" mask at the point of constructing the typed local op. Python /
the dispatch layers do not pre-expand it.

The `SymElement` `#[pyclass]` exposes only `__repr__`, `__eq__`, `__hash__`.
No `__mul__` / field readers.

### `SymmetryGroup`

Pure-Python class in `python/quspin_rs/`. Owns `(n_sites, lhss, [(element,
character), …])`. The list excludes the implicit identity (matching
`SymBasis::add_symmetry`'s contract).

```python
class SymmetryGroup:
    def __init__(self, n_sites: int, lhss: int): ...

    # --- Element insertion ---
    def add(self, element: SymElement, character: complex) -> None:
        """Add a single non-identity element with its character."""

    def add_cyclic(
        self,
        generator: SymElement,
        *,
        k: int | None = None,
        eta: int | None = None,        # ±1, sugar for order=2
        char: complex | None = None,   # explicit override
    ) -> None:
        """Add g, g², …, g^(N-1) where N is g's computed cyclic order.

        Exactly one of {k, eta, char} must be supplied:
        - `k=int`:    χ(g^a) = exp(-2πi · k · a / N)         any cyclic
        - `eta=±1`:   χ(g^a) = η^a, requires N == 2
        - `char=z`:   χ(g) = z;     χ(g^a) = z^a (user picks any consistent rep)
        """

    def close(
        self,
        generators: list[SymElement],
        char: Callable[[SymElement], complex],
    ) -> None:
        """BFS-close the orbit under composition; user supplies a 1-D
        character function indexed on the resulting element."""

    # --- Group operations ---
    def product(self, other: "SymmetryGroup") -> "SymmetryGroup":
        """Direct product. Both groups must share (n_sites, lhss).

        **Out-of-place:** returns a new `SymmetryGroup`; neither `self` nor
        `other` is mutated. Unlike `add` / `add_cyclic` / `close` (which
        build up a single group in place), `product` combines two
        already-built groups, so the factor objects must remain reusable.

        Cartesian enumeration: for every (a, χ_A) in self and every
        (b, χ_B) in other, the result contains (_compose(a, b), χ_A · χ_B).
        Plus the two factor groups themselves (paired with their
        characters) since the identity in the other factor is implicit.

        Caller asserts the two factor groups commute; validate_group
        catches a non-commuting product at first build."""

    # --- Validation (opt-in) ---
    def validate(self) -> None:
        """Run SymBasis::validate_group early. Raises ValueError on
        non-closure or 1-D-rep inconsistency."""

    # --- Introspection ---
    def __len__(self) -> int:
        """Number of explicit (non-identity) elements."""

    def __iter__(self) -> Iterator[tuple[SymElement, complex]]: ...
    def __repr__(self) -> str: ...
```

### `*Basis.symmetric(...)`

Single uniform signature across all four basis wrappers:

```python
basis = SpinBasis.symmetric(group, ham, seeds)
basis = FermionBasis.symmetric(group, ham, seeds)
basis = BosonBasis.symmetric(group, ham, seeds)
basis = GenericBasis.symmetric(group, ham, seeds)
```

`n_sites` and `lhss` are read off `group`. Compatibility table:

| basis | required of `group` |
|---|---|
| `SpinBasis` | `group.lhss == basis_lhss` (positional or default 2) — but since `lhss` no longer appears as an arg, the basis just adopts `group.lhss` |
| `FermionBasis` | `group.lhss == 2`; basis sets `fermionic=True` internally |
| `BosonBasis` | `group.lhss == basis_lhss` (same as Spin — taken from group) |
| `GenericBasis` | `group.lhss` taken from group |

`FermionBasis.symmetric(group, …)` raises `TypeError` if `group.lhss != 2`.

## Examples

### Translation only, momentum `k=1`

```python
group = SymmetryGroup(n_sites=4, lhss=2)
group.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
basis = SpinBasis.symmetric(group, ham, seeds=["0000"])
```

### Translation × spin-flip (abelian product)

```python
group = SymmetryGroup(n_sites=4, lhss=2)
group.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
group.add_cyclic(Local([1, 0]),       eta=-1)
basis = SpinBasis.symmetric(group, ham, seeds=["0000"])
```

### `PZ` composite (single non-trivial generator)

```python
group = SymmetryGroup(n_sites=4, lhss=2)
PZ = Composite(perm=[3, 2, 1, 0], perm_vals=[1, 0])
group.add_cyclic(PZ, eta=-1)
basis = SpinBasis.symmetric(group, ham, seeds=["0000"])
```

### Dihedral D₄ (non-abelian, trivial rep)

```python
group = SymmetryGroup(n_sites=4, lhss=2)
T = Lattice([1, 2, 3, 0])
P = Lattice([3, 2, 1, 0])
group.close(generators=[T, P], char=lambda elem: 1.0)
```

## Architecture

### Layout

```
quspin-py (Rust crate)
├── basis/sym_element.rs       NEW — PySymElement #[pyclass], frozen, opaque;
│                                    Lattice / Local / Composite #[pyfunction]
│                                    constructors; _compose (internal); _order
│                                    (internal helper for SymmetryGroup)
└── basis/sym_group.rs         NEW — bridge that ships an
                                     iter[(element, character)] list to the
                                     dispatch enum's add_symmetry_raw.

quspin-basis (Rust)
└── dispatch.rs (and per-family enums) — add a single `add_symmetry_raw`
    entry on GenericBasis / BitBasis / DitBasis that takes the untyped
    triple (perm, perm_vals, locs) plus a character. Replaces today's three
    separate add_lattice / add_inv / add_local calls at the call site;
    the existing methods stay for direct Rust users.

python/quspin_rs/
└── symmetry.py               NEW — pure-Python SymmetryGroup class
                                    (constructor, add, add_cyclic, close,
                                    product, validate, dunders).

python/quspin_rs/_rs.pyi      UPDATED — type stubs for the new surface.
```

### Data flow

```
Python user                     PySymElement              Rust SymBasis
─────────────────────────       ──────────────            ─────────────
Lattice([1,2,3,0])  ─────►      perm=Some,
                                perm_vals=None,
                                locs=None
                                 │
SymmetryGroup.add(elem, χ)       │
SymmetryGroup.add_cyclic(...)    │  (Python; uses
SymmetryGroup.close(...)         │   _order, _compose
                                 │   helpers in Rust)
                                 ▼
SymmetryGroup -> [(elem, χ), …]
                                 │  shipped at
                                 │  *Basis.symmetric(...)
                                 ▼
                          add_symmetry_raw(χ, perm, perm_vals, locs)
                          (per-family enum dispatch — validates &
                           constructs the typed local op, then calls
                           SymBasis::add_symmetry which validates
                           is_built + perm length / range / bijection)
                                 │
                                 ▼
                          first build() → SymBasis::validate_group
                          (closure + 1-D character check)
```

### `_compute_order` semantics

Computed from cycle structure of the element's typed components:

- Pure `Lattice(perm)`: LCM of cycle lengths in `perm`'s cycle decomposition
  over `0..n_sites`.
- Pure `Local(perm_vals, locs)`: LCM of cycle lengths of `perm_vals` over
  `0..lhss`. (`locs` only changes which sites the op affects, not the
  per-site value-permutation's cycle structure.)
- `Composite`: LCM of the lattice and local orders. Lattice and local
  components commute, so `(P, L)` raised to `k` is `(P^k, L^k)`; the smallest
  `k` returning identity is `lcm(order(P), order(L))`.

`order < 2` (identity element) → `add_cyclic` raises.

### `_compose` semantics

Mirrors `SymElement::compose`:

- Perm component: `(a · b)[src] = a[b[src]]`. Perm length must match.
- Local component (composing two `Local`s): merge `perm_vals` by composition
  on `0..lhss`, take the union of `locs` with conflict resolution. (The
  identity-locs case — both elements act on all sites — produces a single
  effective composed perm_vals applied to all sites.)
- Composite × Lattice or Composite × Local promotes to Composite; Lattice ×
  Local also promotes to Composite. Lattice × Lattice stays Lattice; Local
  × Local stays Local.

Used internally by `SymmetryGroup.add_cyclic` (to produce `g, g², …`) and
`SymmetryGroup.close` (to BFS-close the orbit). Not exposed to users.

### `add_symmetry_raw` on the dispatch enums

New method on `GenericBasis`, `BitBasis`, `DitBasis`:

```rust
pub fn add_symmetry_raw(
    &mut self,
    grp_char: Complex<f64>,
    perm: Option<&[usize]>,
    perm_vals: Option<Vec<u8>>,
    locs: Option<Vec<usize>>,
) -> Result<(), QuSpinError>;
```

Routes to the right per-family inner-enum method depending on the shape of
the triple and the family:

| `(perm, perm_vals)` | family | calls |
|---|---|---|
| `(Some, None)` | any | inner-enum's macro-generated `add_lattice` |
| `(None, Some)` | Bit | `BitBasisDefault::add_local` — Bit is the only family that further restricts `perm_vals`: it must equal `[1, 0]` (the only non-trivial LHSS=2 permutation) and otherwise raises `ValueError`. Trit/Quat/Dyn accept any valid `perm_vals` permutation. |
| `(None, Some)` | Trit/Quat/Dyn | `*BasisDefault::add_local` |
| `(Some, Some)` | any | new inner-enum method `add_composite`, mirrors `SymElement::composite` then `b.add_symmetry(...)` |
| `(None, None)` | any | error (identity element, rejected by `SymBasis::add_symmetry`) |

`add_composite` is the only new inner-enum method; perm + perm_vals + locs
validation already lives where the typed local op is constructed (per the
PR #59 / #61 design).

## Validation flow

### `group.validate()`

Builds a throwaway `BitBasis::Symm` / `DitBasis::Symm` at `(n_sites, lhss)`,
replays `(element, character)` via `add_symmetry_raw`, calls
`SymBasis::validate_group`, propagates the error. Cost
`O(|G|² · probes)`.

### Implicit closure check at first build

`SymBasis::build` already calls `validate_group` on its first invocation —
this is the safety net that catches mistakes from users who skip
`group.validate()`.

### Compatibility check at `*Basis.symmetric(...)`

Always-on, runs in pure Python before any Rust dispatch:

| basis | required |
|---|---|
| `FermionBasis` | `group.lhss == 2` |
| `SpinBasis` / `BosonBasis` / `GenericBasis` | none beyond the basis adopting `group.lhss` |

Mismatch → `TypeError`. `n_sites` always taken from `group`.

## Error handling

| condition | when checked | exception |
|---|---|---|
| `Lattice(perm)` contains negative ints | eagerly in Python `Lattice` constructor | `ValueError` with hint pointing to `Composite` |
| `Lattice` perm wrong length / out of range / not a bijection | at `*Basis.symmetric(...)`, propagated from `SymBasis::add_symmetry` | `ValueError` |
| `Local` / `Composite` `perm_vals` wrong length / not a bijection | at `*Basis.symmetric(...)`, from per-family inner enum's `add_local` | `ValueError` |
| `Local` / `Composite` `locs` out of range | same | `ValueError` |
| `add_cyclic` got more than one of `{k, eta, char}` (or none) | eagerly in Python | `ValueError` |
| `add_cyclic` got `eta` with computed order ≠ 2 | eagerly | `ValueError` |
| `add_cyclic` generator has computed order < 2 (identity) | eagerly | `ValueError` |
| `add_cyclic` `k` out of range `0 ≤ k < N` | eagerly | `ValueError` (no implicit `k mod N` — reduces footguns) |
| Two added elements have the same action | at first build, from `validate_group` | `ValueError` |
| Group not closed under composition | same | `ValueError` |
| Character table inconsistent (1-D rep violation) | same | `ValueError` |
| `*Basis.symmetric(group)` with mismatched `lhss` | eagerly in Python | `TypeError` |

## Testing

### Rust unit tests

| target | what it tests |
|---|---|
| `PySymElement` constructors | `Lattice / Local / Composite` round-trip via `__repr__` / `__eq__` / `__hash__`; equal elements hash equal |
| `_compute_order` | LCM correctness for representative perms (4-cycle → 4, two disjoint 2-cycles → 2, mixed 3+2 → 6); composite = LCM of components |
| `_compose` (internal) | Matches `SymElement::compose` semantics on lattice·lattice, local·local, composite·composite, lattice·local promotes to composite; identity composed with `g` returns `g` |
| `add_symmetry_raw` (new) | Accepts all three element shapes; routes to the right `SymBasis::add_symmetry`; validation errors propagate |

### Python tests (`python/tests/test_symmetry_group.py`)

1. **Constructor surface.** `Lattice` rejects negative ints with the migration
   hint; constructors round-trip via `__repr__` / `__eq__`.
2. **`add_cyclic`.** Exactly-one-of `{k, eta, char}` enforced; `eta` only on
   order-2 generators; `eta=±1` ↔ `k=0/1` produce the same element list and
   characters; out-of-range `k` raises; identity generator rejected.
3. **`product` / `close`.** `Cyclic(T, k=…) × Cyclic(Z, eta=…)` reproduces
   the hand-enumerated `T^a · Z^b` element/character pairs; `close` on a
   non-abelian D_L produces `2L` elements with the user's character function.
4. **`group.validate()`.** Closure-violating group raises; identity-character
   on a non-identity element raises; well-formed translation × parity
   validates clean.
5. **End-to-end basis match.** Build the same problem two ways:
   `*Basis.symmetric(group=...)` with explicit symmetry, and
   `*Basis.subspace(...)` (no symmetry); confirm spectrum / size relations
   match in the way the existing
   `apply_symmetric_basis_matches_qmatrix_dot` test does.
6. **Compatibility errors.** `FermionBasis.symmetric(group)` with
   `group.lhss != 2` raises `TypeError`.

### Type stubs

`python/quspin_rs/_rs.pyi` updated as part of the same PR:

- New entries: `SymElement` (opaque), `Lattice` / `Local` / `Composite`
  constructors, `SymmetryGroup`.
- Old `symmetries=…` / `local_symmetries=…` parameters removed from every
  `*Basis.symmetric(...)` signature.

## Migration

Library is pre-1.0; breaking change is acceptable. Existing in-tree call sites
to update:

- `python/tests/test_rs.py` — every `*Basis.symmetric(..., symmetries=[...])`
  call gets rewritten to `SymmetryGroup` form.
- `python/tests/test_monomial_generic.py` — same.
- `python/quspin_rs/_rs.pyi` — drop the old kwargs.
- The `apply_symmetries` helper in `crates/quspin-py/src/basis/mod.rs` and the
  `apply_local_symmetries` helper in `basis/generic.rs` go away.

## Future work

- **`group.validate_hamiltonian(ham)`.** Symbolic invariance check at the
  operator level. Requires the Rust operator types to support transforming
  each monomial under a `SymElement` (rename sites and apply the local op),
  then comparing the resulting term list back to `ham`'s. Belongs in a
  follow-up PR after the operator algebra is extended.
- **Public composition arithmetic on element handles.** If a use case emerges
  for `a * b` / `g ** k` outside of `add_cyclic` / `close`, expose `__mul__`
  / `__pow__` on `SymElement`; the underlying `_compose` is already in place.
- **Sector-string parsing.** Old QuSpin accepted strings like `"k=0"` /
  `"pz=+1"` to specify sectors; if users want that conveying-sectors-as-strings
  ergonomic, layer it on top of `add_cyclic` later.
