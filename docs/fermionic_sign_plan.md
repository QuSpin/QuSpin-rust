# Fermionic Sign Convention and Implementation Plan

## Overview

Fermionic creation/annihilation operators obey canonical anticommutation
relations. When a lattice permutation (symmetry operation) is applied to a
fermionic state, the result picks up a state-dependent sign from reordering
the creation operators. This sign must be computed alongside the bit
permutation itself.

Importantly, **no specific sign convention is required** for correctness —
only internal consistency matters. The convention is therefore chosen to
maximise computational efficiency.

---

## Operator application (Jordan-Wigner string)

For `c†_i` / `c_i` acting on site `i`, the JW sign is:

```
sign = (-1)^popcount(state & ((1 << i) - 1))
```

i.e. the parity of the number of occupied sites to the left of `i`.

For a multi-operator term (e.g. `c†_i c_j`), operators are applied
right-to-left. Each operator computes its JW sign against the *current*
intermediate state. The signs accumulate as a running product.

---

## Symmetry group sign (fermionic permutation sign)

Applying a site permutation `σ` to a fermionic state generates an additional
sign beyond the group character. This sign is the parity of the permutation
restricted to the occupied sites — equivalently, the number of inversions in
the sequence `σ(i₁), σ(i₂), ..., σ(iₖ)` where `i₁ < ... < iₖ` are the
occupied sites.

### Naive approach (avoided)

A dedicated O(k) loop over occupied sites counting inversions. This requires
a separate pass and is incompatible with SIMD batch processing.

### Chosen approach: per-stage accumulation via Benes network

Moving a fermion from site `i` to site `j` costs sign
`(-1)^popcount(state & mask(i, j))` where `mask(i, j)` has bits set strictly
between `i` and `j`. This is exactly the JW string between those sites.

A Benes permutation network decomposes any permutation into a fixed sequence
of butterfly stages. Each stage swaps a set of bit pairs described by a
`swap_mask`. The fermionic sign contribution from one stage is:

```
stage_parity = popcount(state & swap_mask) & 1
```

The total sign is the XOR of all stage parities:

```
total_parity = stage_0_parity ^ stage_1_parity ^ ... ^ stage_k_parity
fermionic_sign = if total_parity == 1 { -1.0 } else { +1.0 }
```

This means the fermionic sign is computed **in the same pass as the bit
permutation itself** — one `popcnt` and one XOR per butterfly stage, with no
separate loop over occupied sites.

### Overhead

Essentially zero relative to the bosonic case. The Benes network already
processes each stage; the fermionic path adds one `popcnt` and one XOR per
stage. Both are single-cycle instructions on modern hardware.

This is also naturally compatible with the batch SIMD path described in
`orbit_simd_plan.md`: the `popcnt` and XOR accumulate across stages in the
same vectorised loop that computes the permuted states.

---

## Basis representation

Fermionic states are stored as bit strings, identical to `HardcoreBasis`.
Multi-orbital or spin-full systems (e.g. spin-1/2 fermions on `L` physical
sites) are represented by mapping orbitals to computational sites:

```
site 2*i     → spin-down orbital at physical site i
site 2*i + 1 → spin-up   orbital at physical site i
```

The orbital-to-site mapping is a convention imposed by the Hamiltonian terms,
not the basis. **No new basis type is needed** — `HardcoreBasis` /
`PyHardcoreBasis` is reused directly.

---

## Implementation plan

### New types (quspin-core)

| Type | File | Notes |
|------|------|-------|
| `FermionOp` | `hamiltonian/fermion/op.rs` | `Plus` (c†), `Minus` (c), `N` (n̂) |
| `FermionOpEntry<C>` | `hamiltonian/fermion/op.rs` | Like `OpEntry<C>` but accumulates JW sign |
| `FermionHamiltonian<C>` | `hamiltonian/fermion/hamiltonian.rs` | Structurally identical to `HardcoreHamiltonian<C>` |
| `FermionHamiltonianInner` | `hamiltonian/fermion/dispatch.rs` | `Ham32` / `Ham64` / … variants |

### New types (quspin-py)

| Type | Notes |
|------|-------|
| `PyFermionHamiltonian` | Same term API as `PyHardcoreHamiltonian` |
| `PyQMatrix::build_fermion_hamiltonian` | Same pattern as `build_hardcore_hamiltonian` |

### Changes to existing types

`FermionicLatticeElement` (or a fermionic flag/method on `LatticeElement`)
must store the raw permutation array so the per-stage sign can be computed.
This requires the Benes network to be in place first — the fermionic symmetry
group implementation is therefore **blocked on the Benes network**.

The operator application (JW string for `c†`/`c`) is independent of the
Benes network and can be implemented immediately.

### Order of work

1. Implement `FermionOp` / `FermionOpEntry` / `FermionHamiltonian` with JW
   sign for operator terms — no dependency on Benes.
2. Add `PyFermionHamiltonian` and `PyQMatrix::build_fermion_hamiltonian`.
3. Implement Benes permutation network in `bitbasis`.
4. Add fermionic sign accumulation to the Benes apply path.
5. Add `FermionicLatticeElement` (or extend `LatticeElement`) using the Benes
   sign computation.
6. Expose fermionic symmetry groups via `SpinSymGrp` / a new
   `FermionSymGrp` as appropriate.
