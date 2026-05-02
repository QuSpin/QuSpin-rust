# Symmetry API

Pass a `SymmetryGroup` to `*Basis.symmetric(group, ham, seeds)`. See the
[guide](../guide/symmetry.md) for usage patterns.

## SymmetryGroup

::: quspin_rs.symmetry.SymmetryGroup

## Element constructors

The three element constructors produce opaque `SymElement` handles that
the `SymmetryGroup` consumes via `add` / `add_cyclic` / `close` / `product`.

### Lattice

::: quspin_rs._rs.Lattice

### Local

::: quspin_rs._rs.Local

### Composite

::: quspin_rs._rs.Composite

### SymElement

::: quspin_rs._rs.SymElement
