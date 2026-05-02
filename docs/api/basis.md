# Basis types

Four user-facing basis wrappers, each with three constructors:

- `full(...)` — full Hilbert space, no projection.
- `subspace(...)` — particle-number / energy sector built by BFS.
- `symmetric(group, ham, seeds)` — symmetry-projected subspace.

`n_sites` and `lhss` come from the `SymmetryGroup` for the symmetric
constructor; the full / subspace constructors take them positionally.

## SpinBasis

::: quspin_rs._rs.SpinBasis

## FermionBasis

::: quspin_rs._rs.FermionBasis

## BosonBasis

::: quspin_rs._rs.BosonBasis

## GenericBasis

::: quspin_rs._rs.GenericBasis
