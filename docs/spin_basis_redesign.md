# SpinBasis Redesign Plan

## Goal

Remove `SpinSymGrp`, `DitSymGrp`, and `SymmetryGrpInner` from the `SpinBasis` builder
path entirely. `SpinBasis` becomes:

```rust
pub struct SpinBasis {
    pub n_sites: usize,
    pub lhss:    usize,
    space_kind:  SpaceKind,
    pub inner:   BasisInner,
}
```

## Key Insight

`SymmetryGrpInner` is redundant. `BasisInner::Sym*` variants hold
`SymBasis<B, PermDitMask<B>, N>` and `BasisInner::DitSym*` hold
`SymBasis<B, DynamicPermDitValues, N>`. `SymBasis` already stores:

```
lattice: Vec<BenesLatticeElement<B>>   ← same field as SymGrpBase
local:   Vec<(Complex<f64>, L)>        ← same field as SymGrpBase
states:  Vec<(B, N)>                   ← populated during build()
index_map: HashMap<B, usize>           ← populated during build()
```

`SymGrpBase` / `SymmetryGrpInner` is only used as a pre-build staging area for
`lattice` and `local` before `SymBasis::from_grp` copies them in. We can skip this
entirely by starting with an empty `SymBasis` and adding group elements directly to it.

The same logic applies to `Subspace`: it is `Vec<B>` + `HashMap<B, usize>`, which
starts empty and gets populated during `build`.

So `BasisInner` already naturally represents both the pre-build (empty `states`) and
post-build (populated `states`) states. No extra enum needed.

---

## Required Additions to Existing Types

### 1. `SymBasis<B, L, N>` — add builder methods

Currently constructed only via `from_grp(SymGrpBase)`. Add:

```rust
impl<B: BitInt, L, N: NormInt> SymBasis<B, L, N> {
    /// Construct an empty basis with no group elements and no states.
    pub fn new_empty(lhss: usize, n_sites: usize, fermionic: bool) -> Self;

    /// Add a lattice (site-permutation) element. Valid before build().
    pub fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]);

    /// True once build() has been called and states are populated.
    pub fn is_built(&self) -> bool { !self.states.is_empty() }
}

// lhss=2 only (L = PermDitMask<B>)
impl<B: BitInt, N: NormInt> SymBasis<B, PermDitMask<B>, N> {
    /// Add a spin-inversion element. Valid before build().
    pub fn push_inverse(&mut self, grp_char: Complex<f64>, locs: &[usize]);
}
```

`push_lattice` computes a `BenesLatticeElement<B>` from `perm` and appends it to
`self.lattice` — the same logic currently in `SymGrpBase::push_lattice`.

`push_inverse` appends a `PermDitMask<B>` to `self.local` — same logic as
`SymGrpBase::push_inverse`.

### 2. `Subspace<B>` — add `is_built`

```rust
impl<B: BitInt> Subspace<B> {
    pub fn new_empty(n_sites: usize, lhss: usize) -> Self;
    pub fn is_built(&self) -> bool { !self.states.is_empty() }
}
```

### 3. `BasisInner` — add dispatch methods for builder phase

```rust
impl BasisInner {
    /// Push a lattice element. Errors if not a Sym*/DitSym* variant.
    pub fn push_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
    ) -> Result<(), QuSpinError>;

    /// Push a spin-inversion element. Errors if not a Sym* (lhss=2) variant.
    pub fn push_inverse(
        &mut self,
        grp_char: Complex<f64>,
        locs: &[usize],
    ) -> Result<(), QuSpinError>;

    /// True if states have been populated (i.e. build() has been called).
    pub fn is_built(&self) -> bool;
}
```

`push_lattice` matches on all 18 `Sym*`/`DitSym*` variants and calls
`sym_basis.push_lattice(...)`. Returns an error for `Full*` / `Sub*`.

`is_built` returns:
- `Full*` → always `true`
- `Sub*` → `subspace.is_built()`
- `Sym*` / `DitSym*` → `sym_basis.is_built()`

---

## `SpinBasis` Construction

### `new(n_sites, lhss, space_kind)`

| space_kind | lhss | `inner` at construction |
|------------|------|-------------------------|
| `Full`     | 2    | `select_b_for_n_sites!` → `BasisInner::Full32/64(FullSpace::new(...))` |
| `Full`     | >2   | same, using `DynamicDitManip` |
| `Sub`      | any  | `select_b_for_n_sites!` → `BasisInner::Sub*(Subspace::new_empty(...))` |
| `Symm`     | 2    | `select_b_for_n_sites!` → `BasisInner::Sym*(SymBasis::new_empty(lhss=2, ...))` |
| `Symm`     | >2   | `select_b_for_n_sites!` → `BasisInner::DitSym*(SymBasis::new_empty(lhss, ...))` |

Validation at `new()`:
- `lhss < 2` → error
- `Full` with `n_sites > 64` → error
- `Sub`/`Symm` with `n_sites > 8192` → error

---

## `SpinBasis` Methods

### `add_lattice`

```rust
pub fn add_lattice(
    &mut self,
    grp_char: Complex<f64>,
    perm: &[usize],
) -> Result<(), QuSpinError>
```

- Errors if `space_kind != Symm`
- Errors if `inner.is_built()` (too late, build already called)
- Errors if `perm.len() != self.n_sites`
- Delegates to `self.inner.push_lattice(grp_char, perm)`

### `add_inversion`

```rust
pub fn add_inversion(
    &mut self,
    grp_char: Complex<f64>,
    locs: &[usize],
) -> Result<(), QuSpinError>
```

- Errors if `space_kind != Symm`
- Errors if `inner.is_built()`
- Errors if `lhss != 2`
- Delegates to `self.inner.push_inverse(grp_char, locs)`

### `build_hardcore` / `build_boson`

```rust
pub fn build_hardcore(
    &mut self,
    ham: &HardcoreHamiltonianInner,
    seeds: &[Vec<u8>],
) -> Result<(), QuSpinError>

pub fn build_boson(
    &mut self,
    ham: &BosonHamiltonianInner,
    seeds: &[Vec<u8>],
) -> Result<(), QuSpinError>
```

- Errors if `space_kind == Full` ("Full basis requires no build")
- Errors if `inner.is_built()` (already built)

The build step mutates `self.inner` in place:

```
Sub* variant:
    with_plain_basis_mut!(&mut self.inner, B, subspace, {
        enumerate ham from seeds → populate subspace.states + subspace.index_map
    })

Sym* variant (build_hardcore, lhss=2):
    with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })

DitSym* variant (build_boson, lhss>2):
    with_dit_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })
```

This requires `_mut` variants of the existing dispatch macros, or `BasisInner` methods
that expose mutable access (see below).

---

## Mutable Dispatch

The existing macros (`with_sym_basis!`, etc.) take `&BasisInner`. We need `&mut`
variants, or equivalent methods on `BasisInner`. The simplest approach is adding
`_mut` macro variants that mirror the existing ones but bind `mut`:

```rust
macro_rules! with_sym_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            BasisInner::Sym32(ref mut $basis) => { type $B = u32; $body }
            BasisInner::Sym64(ref mut $basis) => { type $B = u64; $body }
            // ... 9 variants
            _ => unreachable!()
        }
    }
}
```

---

## What Gets Deleted / Demoted

| Type | Action |
|------|--------|
| `SpinSymGrp` (public) | Delete or make private; no longer needed |
| `DitSymGrp` (public) | Delete or make private; no longer needed |
| `SymmetryGrpInner` | No longer used in SpinBasis builder path; keep only if FermionicSymGrp still needs it |
| Old `SpinBasis` with `grp: Option<SpinSymGrp>` | Replace entirely |

`FermionicSymGrp` still uses `SymmetryGrpInner` (deferred), so it is not deleted yet.

---

## What Does NOT Change

- `BasisInner` enum shape (29 variants) — unchanged, just gets new methods
- `SymBasis<B, L, N>` struct shape — unchanged, just gets new builder methods
- `Subspace<B>` struct shape — unchanged, just gets `new_empty` + `is_built`
- `with_sym_basis!`, `with_dit_sym_basis!`, `with_plain_basis!` macros — unchanged
- `select_b_for_n_sites!` macro — unchanged
- `HardcoreHamiltonianInner`, `BosonHamiltonianInner` — unchanged

---

## Summary of Steps

1. Add `SymBasis::new_empty`, `push_lattice`, `push_inverse`, `is_built`
2. Add `Subspace::new_empty`, `is_built`
3. Add `BasisInner::push_lattice`, `push_inverse`, `is_built`
4. Add `with_sym_basis_mut!`, `with_dit_sym_basis_mut!`, `with_plain_basis_mut!` macros
5. Rewrite `spin_basis.rs` using the new struct and methods
6. Remove `SpinSymGrp` / `DitSymGrp` from public exports in `mod.rs`
