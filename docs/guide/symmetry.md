# Symmetry groups

`SymmetryGroup` is the user-facing handle for building the symmetry group you
pass to `*Basis.symmetric(...)`. Construct elements via `Lattice`, `Local`, and
`Composite`; collect them via `add` / `add_cyclic` / `close` / `product`; then
hand the group to a basis constructor.

```python
basis = SpinBasis.symmetric(group, ham, seeds)
```

`n_sites` and `lhss` come from the group; `FermionBasis` rejects
`group.lhss != 2` with a `TypeError`.

## Element constructors

| Constructor | Use it for |
|---|---|
| `Lattice(perm)` | site permutation only — translation, parity, reflection |
| `Local(perm_vals, locs=None)` | per-site value-permutation only — spin-flip, dit-permutation |
| `Composite(perm, perm_vals, locs=None)` | the two acting as a single tied element (e.g. `PZ`) |

Each returns an opaque `SymElement` handle.

## Common patterns

### Cyclic — translation, parity, spin-flip

```python
from quspin_rs import SymmetryGroup, Lattice, Local

# Z_L translation, momentum k.
group = SymmetryGroup(n_sites=4, lhss=2)
group.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
```

`add_cyclic(generator, k=…)` enumerates `g, g², …, g^(N-1)` automatically and
attaches the matching characters `χ(gᵃ) = exp(-2πi · k · a / N)`. The cyclic
order `N` is computed from the generator.

For `order = 2` factors (parity, spin-flip) the kwarg `eta=±1` is sugar:

```python
group.add_cyclic(Local([1, 0]), eta=-1)   # parity-odd Z₂ sector
```

For arbitrary 1-D characters, use `char=z`:

```python
group.add_cyclic(Local([1, 0]), char=-1.0 + 0j)   # same as eta=-1
```

### Direct product — abelian factor groups

Chain `add_cyclic` calls **don't** Cartesian-product. To combine commuting
factor groups, build each separately and call `.product(other)`:

```python
T = SymmetryGroup(n_sites=4, lhss=2)
T.add_cyclic(Lattice([1, 2, 3, 0]), k=0)

Z = SymmetryGroup(n_sites=4, lhss=2)
Z.add_cyclic(Local([1, 0]), eta=-1)

group = T.product(Z)   # Z_4 × Z_2
```

`product` is **out-of-place** — `T` and `Z` stay reusable. The factors must
commute; `validate_group` (run automatically at build time) catches
non-commuting products.

### Composite generators — PZ, particle-hole, …

When a single tied generator (`PZ`, particle-hole, etc.) is the symmetry but
neither `P` nor `Z` alone is:

```python
PZ = Composite(perm=[3, 2, 1, 0], perm_vals=[1, 0])
group = SymmetryGroup(n_sites=4, lhss=2)
group.add_cyclic(PZ, eta=-1)
```

The `perm` and `perm_vals` components are applied atomically as one element
with a single character.

### Non-abelian — dihedral, point groups

Use `close` with a user-supplied 1-D character function. The walker BFS-closes
the orbit under composition:

```python
T = Lattice([1, 2, 3, 0])
P = Lattice([3, 2, 1, 0])

group = SymmetryGroup(n_sites=4, lhss=2)
group.close(generators=[T, P], char=lambda elem: 1.0)   # trivial rep on D_4
```

For non-trivial 1-D reps the `char(elem)` callable distinguishes elements by
their action — see [`SymBasis::validate_group`][validate-group] for the
checks the closure must satisfy.

[validate-group]: ../api/symmetry.md#quspin_rs.symmetry.SymmetryGroup.validate

## Validating early

```python
group.validate()   # opt-in: closure + 1-D character consistency
```

`*Basis.symmetric(...)` runs the same check implicitly on first build, so
calling `validate()` directly is just for early feedback when iterating on
group construction.

## Compatibility check

`*Basis.symmetric(group, ham, seeds)` enforces:

| Basis | Required of `group` |
|---|---|
| `SpinBasis` | `group.lhss` becomes the basis LHSS |
| `FermionBasis` | `group.lhss == 2` (TypeError otherwise); `fermionic = True` set internally |
| `BosonBasis` | `group.lhss` becomes the basis LHSS |
| `GenericBasis` | `group.lhss` becomes the basis LHSS |

`group.n_sites` always becomes the basis `n_sites`; there is no separate
keyword.

## See also

- `scripts/symmetry_group_demo.py` — runnable walkthrough of the patterns above.
- [Symmetry API reference](../api/symmetry.md) — full method listings.
- Spec: `docs/superpowers/specs/2026-04-26-python-symmetry-group-api-design.md`.
